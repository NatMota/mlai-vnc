"""Render recorded golden trajectories from t=0, using the deployment camera and encoder code.

python render_golden.py --source /path/to/run_1000/baseline --split train --jobs 2
Set CAMERA_RENDERER=tiny (default) or egl consistently during training AND evaluation.
Only recorded poses position the scene; simulator truth never supplies a detection or tracker input.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
OUT = Path(os.environ.get('GOLDEN_CAMERA_ROOT', HERE / 'data/camera_golden'))


def frame_steps(n):
    """Exact schedule used in full_run.py, including the rounded endpoint."""
    steps = np.array([round(i * 1000 / 30) for i in range((n * 30 + 999) // 1000)], np.int64)
    return steps[steps < n]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png(path, rgb):
    """Small RGB PNG writer; rendering does not need Torch, plotting or Pillow."""
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    h, w, _ = rgb.shape
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w, h, 8, 2, 0, 0, 0))
                          + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def provenance(source):
    import wrist_camera as wc
    import sensors
    from camera_loop import MOUNT
    try:
        commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except subprocess.CalledProcessError:
        commit = os.environ.get('CAMERA_CODE_COMMIT', 'unversioned-copy')
    files = [Path(__file__), HERE.parent / 'driving_v2/camera_loop.py', Path(wc.__file__), Path(sensors.__file__)]
    return dict(schema_version=1, renderer=wc.RENDERER, pybullet=importlib.metadata.version('pybullet'), numpy=np.__version__,
                git_commit=commit, code_sha256={p.name: sha(p) for p in files}, source_sha256=sha(source),
                urdf_sha256=sha(wc.URDF), mount_robot_xyz_m=MOUNT.tolist(),
                camera_axes_robot=[[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
                image_size=wc.SIZE, cube_face_size=wc.FACE, near_m=wc.NEAR, far_m=wc.FAR,
                dot_radius_m=wc.DOT_RADIUS, dot_alpha=1.0 if wc.RENDERER == 'egl' else wc.DOT_RGBA[3],
                egl_pixel_alignment_xy=[0.5, 0.5] if wc.RENDERER == 'egl' else [0.0, 0.0],
                wheel_counts_per_rev=sensors.COUNTS_PER_REV, encoder_dt_s=sensors.DT,
                camera_schedule='round(i * 1000 / 30) < steps', source_state_alignment='pre-step')


def camera_inputs(split, dot, steps, root=OUT):
    """Held, causal float32 inputs. drive=(range,sin,cos,visible), arm=(x,y,z).

    A hidden frame dead-reckons a previously seen dot; between frames hold that estimate.
    Before any sighting the model receives zeros. NaNs are retained in the stored arm arrays.
    """
    query = np.atleast_1d(steps)
    if query.ndim != 1 or not np.issubdtype(query.dtype, np.integer):
        raise ValueError('steps must be a one-dimensional integer sequence')
    with np.load(Path(root) / split / f'camera_{dot}.npz', allow_pickle=False) as a:
        if np.any(query < 0) or np.any(query >= int(a['n_steps'])):
            raise ValueError('requested step outside the recorded episode')
        idx = np.searchsorted(a['step'], query, side='right') - 1
        return dict(drive=np.c_[a['drive'][idx], a['seen'][idx]].astype(np.float32),
                    arm=np.nan_to_num(a['arm'][idx], nan=0.0).astype(np.float32),
                    wheel=a['wheel_features'][idx].astype(np.float32))


def replay_trackers(detections, wheel_angles):
    from camera_loop import Tracker
    from sensors import WheelEncoders, wheel_features
    enc = WheelEncoders(); enc.reset(wheel_angles[0])
    drive, arm = Tracker(), Tracker(keep_height=True)
    ds, arms, features, speeds = [], [], [], []
    for detection, q in zip(detections, wheel_angles, strict=True):
        reading = enc.read(q)
        point = detection if np.isfinite(detection).all() else None
        xy = drive.update(point, reading['wheel_speed'])
        xyz = arm.update(point, reading['wheel_speed'])
        if xy is None:
            ds.append([0., 0., 0.])
        else:
            angle = np.arctan2(xy[1], xy[0]); ds.append([np.linalg.norm(xy), np.sin(angle), np.cos(angle)])
        arms.append(np.full(3, np.nan) if xyz is None else xyz.copy())
        features.append(wheel_features(reading['wheel_angle'], reading['wheel_speed'])[0])
        speeds.append(reading['wheel_speed'])
    return (np.asarray(x, np.float32) for x in (ds, arms, features, speeds))


def one(source_root, output_root, split, dot):
    import pybullet as p
    from camera_loop import Scene, MOUNT, see_dot, camera_to_robot, INSIDE
    started = time.perf_counter()
    source = Path(source_root) / f'run_{dot}.npz'
    out = Path(output_root) / split; out.mkdir(parents=True, exist_ok=True)
    dest = out / f'camera_{dot}.npz'
    prov = provenance(source)
    if dest.exists():
        with np.load(dest, allow_pickle=False) as a:
            if json.loads(str(a['provenance'])) != prov:
                raise ValueError(f'{dest}: provenance changed; use a new output directory')
            np.testing.assert_array_equal(a['step'], frame_steps(int(a['n_steps'])))
            return json.loads(str(a['summary']))
    with np.load(source, allow_pickle=False) as a:
        m = json.loads(str(a['manifest']))
        pos, quat, q, times = a['base_position'], a['base_quaternion'], a['q'], a['time_s']
    n = int(m['steps']); steps = frame_steps(n)
    if n <= 0 or q.shape != (n, 17) or pos.shape != (n, 3) or quat.shape != (n, 4):
        raise ValueError(f'Invalid state shapes in {source}')
    np.testing.assert_allclose(times, np.arange(n) * .001, atol=1e-8, rtol=0)
    names = m['joint_names']; target = np.array(m['target_position_world_m'], float)
    if len(names) != 17 or any('wheel' not in name for name in names[:4]):
        raise ValueError('Expected four wheels followed by seven arm and six finger joints')
    if not all(np.isfinite(x).all() for x in (pos, quat, q, target)):
        raise ValueError('Non-finite source state')
    park = int(m.get('parking_step', -1)); valid_park = 0 <= park < n
    shot = max(0, np.searchsorted(steps, park, side='right') - 1) if valid_park else len(steps) - 1
    scene = Scene(); detections = []; false_red = []
    try:
        if any(name not in scene.joint for name in names):
            raise ValueError('Source joint names do not match renderer URDF')
        for j, k in enumerate(steps):
            scene.pose(np.r_[pos[k], quat[k]], names, q[k], target)  # ignore returned WRIST transform
            B = np.array(p.getMatrixFromQuaternion(quat[k])).reshape(3, 3)
            rgb, depth, seg = scene.fisheye(pos[k] + B @ MOUNT, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
            detection = see_dot(rgb, depth)
            detections.append(np.full(3, np.nan) if detection is None else camera_to_robot(detection))
            r, g, b = (rgb[..., c].astype(int) for c in range(3))
            red = INSIDE & (r > 100) & (r > 1.7 * g) & (r > 1.7 * b)
            false_red.append(int((red & (seg != scene.dot)).sum()))  # diagnostics only; NEVER used by see_dot
            if j == shot:
                png(out / f'camera_{dot}.png', rgb)
    finally:
        p.disconnect(scene.cid)
    detection = np.asarray(detections, np.float32); seen = np.isfinite(detection).all(axis=1)
    drive, arm, features, speeds = replay_trackers(detection, q[steps, :4].astype(float))
    error = None
    if valid_park and np.isfinite(arm[shot]).all():
        B = np.array(p.getMatrixFromQuaternion(quat[park])).reshape(3, 3)
        error = float(1000 * np.linalg.norm(arm[shot] - B.T @ (target - pos[park])))
    post = steps >= park if valid_park else np.zeros(len(steps), bool)
    summary = dict(split=split, dot=int(dot), n_steps=n, frames=len(steps), seen_frames=int(seen.sum()),
                   first_sighting_step=int(steps[seen][0]) if seen.any() else None, parking_step=park,
                   park_error_mm=error, post_park_seen_fraction=float(seen[post].mean()) if post.any() else None,
                   termination=m.get('termination'), stalled=m.get('termination') != 'reached',
                   false_red_pixels_max=max(false_red), seconds=time.perf_counter() - started,
                   renderer=prov['renderer'])
    # Atomic publish; an interrupted write is not mistaken for a completed run.
    tmp = dest.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, step=steps, seen=seen, detection=detection, drive=drive, arm=arm,
                            wheel_features=features, wheel_speed=speeds, false_red_pixels=np.array(false_red),
                            dot=int(dot), target=target, parking_step=park, n_steps=n, joint_names=np.array(names),
                            provenance=json.dumps(prov), summary=json.dumps(summary, allow_nan=False))
    tmp.replace(dest)
    return summary


def summarise(root):
    rows = []
    for path in sorted(Path(root).glob('*/camera_*.npz')):
        with np.load(path, allow_pickle=False) as a:
            rows.append(json.loads(str(a['summary'])))
    errors = [r['park_error_mm'] for r in rows if r['park_error_mm'] is not None]
    report = dict(runs=rows, completed=len(rows),
                  error_mm=None if not errors else dict(zip(('median', 'p95', 'max'), np.percentile(errors, [50, 95, 100]).tolist())),
                  outliers=[r for r in rows if r['park_error_mm'] is None or r['park_error_mm'] > 100
                            or (r['post_park_seen_fraction'] is not None and r['post_park_seen_fraction'] < .8)])
    tmp = Path(root) / 'summary.json.tmp'; tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
    tmp.replace(Path(root) / 'summary.json')
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--split', choices=['train', 'val'], default='train')
    ap.add_argument('--selection', type=Path, default=ROOT / 'collection/native/effects/dots_spread.json')
    ap.add_argument('--splits', type=Path, default=ROOT / 'collection/native/effects/split_1000.json')
    ap.add_argument('--ids', nargs='+', type=int)
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args()
    split_plan = json.loads(a.splits.read_text())
    ids = a.ids if a.ids is not None else (json.loads(a.selection.read_text())['100'] if a.split == 'train' else split_plan['val'])
    if len(ids) != len(set(ids)) or not set(ids) <= set(split_plan[a.split]):
        raise ValueError('Duplicate IDs or requested IDs outside the named split')
    if a.jobs < 1:
        raise ValueError('jobs must be positive')
    for dot in ids:
        if not (a.source / f'run_{dot}.npz').is_file():
            raise FileNotFoundError(a.source / f'run_{dot}.npz')
    started = time.perf_counter()
    with ProcessPoolExecutor(a.jobs, mp_context=multiprocessing.get_context('spawn'), max_tasks_per_child=1) as pool:
        pending = [pool.submit(one, str(a.source), str(a.out), a.split, dot) for dot in ids]
        for i, f in enumerate(as_completed(pending), 1):
            row = f.result(); summarise(a.out)
            print(json.dumps(dict(completed=i, requested=len(ids), elapsed_s=time.perf_counter()-started, **row)), flush=True)
    print(json.dumps(dict(batch_seconds=time.perf_counter()-started, requested=len(ids), output=str(a.out))), flush=True)


if __name__ == '__main__':
    main()
