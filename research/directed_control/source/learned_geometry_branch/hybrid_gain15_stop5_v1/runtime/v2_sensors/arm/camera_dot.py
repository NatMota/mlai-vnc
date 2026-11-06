"""The camera's dot estimate along the expert's recorded runs, for training the arm on what it sees in testing.

Same pipeline as test_arm.py --camera, run on the recorded states instead of the network's: every 1/30 s from the
start of the run to the expert's stop, the corner fisheye is rendered from the recorded pose, the red dot found,
placed in the robot frame, and given to the height-keeping tracker (wheel encoders while hidden). The estimate is
held between frames, as in testing. Output per bearing: data/camera_dot/camera_{bearing}.npz with the frame steps,
the estimate per frame (NaN before the first sighting) and whether the dot was seen.

    python camera_dot.py [--bearings B ...]      (WORKERS=6, CAMERA_RENDERER=tiny by default)
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
OUT = HERE / 'data/camera_dot'


def episode(bearing):
    import pybullet as p
    import sensors
    from camera_loop import Scene, Tracker, see_dot, camera_to_robot, MOUNT
    from train_arm import DATA, table
    run = DATA / f'run_{bearing // 12 - 1}'
    m = json.loads((run / 'manifest.json').read_text()); target = m['target_position_world_m']
    names = json.loads((HERE.parent / 'driving_v2/closed_loop_native/result_wheels_live_dot.json').read_text())['joint_names']
    js, base = np.load(run / 'joint_state.npy'), table(run, 'base_state')
    pos = np.c_[base['position_m_x'], base['position_m_y'], base['position_m_z']]
    quat = np.c_[base['quaternion_x'], base['quaternion_y'], base['quaternion_z'], base['quaternion_w']]
    steps = [k for k in (round(i * 1000 / 30) for i in range(10 ** 5)) if k < m['steps']]
    scene, tracker, wheels = Scene(), Tracker(keep_height=True), sensors.WheelEncoders()
    wheels.reset(js[0, 2:6])
    est = np.full((len(steps), 3), np.nan); visible = np.zeros(len(steps), bool)
    for i, k in enumerate(steps):
        q = js[k, 2:19]
        B = np.array(p.getMatrixFromQuaternion(quat[k].tolist())).reshape(3, 3)
        scene.pose(np.r_[pos[k], quat[k]], names, q, target)
        rgb, depth, _ = scene.fisheye(pos[k] + B @ MOUNT, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
        seen = see_dot(rgb, depth)
        e = tracker.update(camera_to_robot(seen) if seen is not None else None, wheels.read(q[:4])['wheel_speed'])
        visible[i] = seen is not None
        if e is not None:
            est[i] = e
    np.savez(OUT / f'camera_{bearing}.npz', step=np.array(steps), est=est, visible=visible, parking_step=m['parking_step'])
    return bearing, len(steps), int(visible.sum())


def estimate_per_step(bearing, steps):
    """The latest estimate at each 1 ms step (zeros before the first sighting), as the network gets it in testing."""
    z = np.load(OUT / f'camera_{bearing}.npz')
    idx = np.searchsorted(z['step'], steps, side='right') - 1
    out = z['est'][idx]
    return np.where(np.isnan(out), 0.0, out)


if __name__ == '__main__':
    from train_arm import SPLIT
    ap = argparse.ArgumentParser(); ap.add_argument('--bearings', type=int, nargs='*'); a = ap.parse_args()
    bearings = a.bearings or [b for b in sorted(SPLIT) if SPLIT[b] != 'holdout']
    OUT.mkdir(parents=True, exist_ok=True); t0 = time.time()
    with Pool(int(os.environ.get('WORKERS', 6))) as pool:
        for b, n, seen in pool.imap_unordered(episode, bearings):
            print(f'{b}: {n} frames, {seen} with the dot seen, {time.time() - t0:.0f} s', flush=True)
