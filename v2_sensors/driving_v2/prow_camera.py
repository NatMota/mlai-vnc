"""180-degree fisheye RGB-D camera on the chassis nose (prow): render saved expert poses, find the dot.

Run with any Python that has PyBullet:  python prow_camera.py OUT.npz
Optional mount (base frame, metres) via the environment, so worker processes see it too:
    PROW_MOUNT="0.29,-0.13,0.36" python prow_camera.py OUT.npz
Same fisheye model, detector and checks as wrist_camera.py; only the mount changes.
Robot and dot unchanged from robotnik_arm_task.py. No physics is stepped.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pybullet as p
from wrist_camera import DATA, STRIDE, Scene, detect, planar

MOUNT = np.array([float(v) for v in os.environ.get('PROW_MOUNT', '0.29,-0.08,0.36').split(',')])  # base frame (x forward, y left, z up): on the chassis nose, in front of and right of the arm base (surface 0.33 m + 3 cm)


def prow_camera(base_pose):
    B = np.array(p.getMatrixFromQuaternion(base_pose[3:].tolist())).reshape(3, 3)
    eye = base_pose[:3] + B @ MOUNT
    z, y = B[:, 0], -B[:, 2]          # camera z = robot forward, camera y = down
    return eye, np.c_[np.cross(y, z), y, z]


def episode(r):
    scene = Scene()
    names = r['manifest']['joint_names']; target = np.array(r['manifest']['target_position_world_m'])
    with np.load(DATA / r['archive']) as a:
        poses, joints, times = a['base_pose'], a['joint_positions'], a['time_s']
    n = len(poses); idx = sorted(set(range(0, n, STRIDE)) | {n // 2, n - 1})
    key = {0: 'start', n // 2: 'middle', n - 1: 'park'}
    rows, images = [], {}
    for i in idx:
        scene.pose(poses[i], names, joints[i], target)
        eye, C = prow_camera(poses[i])
        rgb, depth, seg = scene.fisheye(eye, C)
        found, ray, dist, npx = detect(rgb, depth)
        red = (rgb[..., 0] > 100) & (rgb[..., 0] > 1.7 * rgb[..., 1]) & (rgb[..., 0] > 1.7 * rgb[..., 2])
        true_d, true_a = planar(target, poses[i])
        row = dict(bearing=r['bearing_deg'], index=i, time_s=float(times[i]), true_visible_px=int((seg == scene.dot).sum()),
                   found=found, dot_px=npx, false_red_px=int((red & (seg != scene.dot)).sum()),
                   true_dist_m=true_d, true_dir_deg=true_a, est_dist_m=np.nan, est_dir_deg=np.nan)
        if found:
            row['est_dist_m'], row['est_dir_deg'] = planar(eye + C @ (ray * dist), poses[i])
        rows.append(row)
        if i in key:
            images[key[i]] = rgb
    return rows, images


if __name__ == '__main__':
    out = Path(sys.argv[1])
    records = [json.loads(line) for line in open(DATA / 'episodes.jsonl')]
    unique = {}
    for r in records:
        unique.setdefault(r['trajectory_hash'], r)
    eps = sorted(unique.values(), key=lambda r: r['bearing_deg'])
    with Pool(6) as pool:
        results = pool.map(episode, eps)
    rows = [row for rs, _ in results for row in rs]
    imgs = {k: np.stack([im[k] for _, im in results]) for k in ('start', 'middle', 'park')}
    scene = Scene(); e0 = eps[0]
    with np.load(DATA / e0['archive']) as a:
        pose0 = a['base_pose'][-1]
        scene.pose(pose0, e0['manifest']['joint_names'], a['joint_positions'][-1], e0['manifest']['target_position_world_m'])
    eye, _ = prow_camera(pose0)
    marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 0.8, 0, 1], physicsClientId=scene.cid)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker, basePosition=eye.tolist(), physicsClientId=scene.cid)
    view = p.computeViewMatrix((pose0[:3] + np.array([0.2, -1.6, 1.0])).tolist(), (pose0[:3] + np.array([0.4, 0, 0.45])).tolist(), [0, 0, 1])
    _, _, ext, _, _ = p.getCameraImage(512, 384, view, p.computeProjectionMatrixFOV(45, 512 / 384, 0.01, 10),
                                       renderer=p.ER_TINY_RENDERER, physicsClientId=scene.cid)
    np.savez_compressed(out, rows=json.dumps(rows), bearings=np.array([e['bearing_deg'] for e in eps]),
                        outside=np.asarray(ext, np.uint8).reshape(384, 512, 4)[..., :3], camera_mount=MOUNT,
                        **{f'img_{k}': v for k, v in imgs.items()})
    print('frames', len(rows), '->', out)
