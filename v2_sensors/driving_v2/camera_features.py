"""Camera + tracker inputs for every saved expert row (train and test only; holdout untouched).

Same pipeline as camera_loop.py, run on the expert's recorded states instead of the model's:
render the corner fisheye from the recorded pose -> red threshold -> ray + depth -> robot frame
(fixed mount) -> tracker (wheel encoders when hidden). Output per row: dist, sin, cos, visible.
Before the first sighting the dot inputs are 0, 0, 0 (no information).

    python camera_features.py OUT_DIR      (CAMERA_RENDERER=tiny by default)
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pybullet as p
from wrist_camera import DATA, Scene
from camera_loop import MOUNT, see_dot, camera_to_robot, Tracker


def episode(r):
    scene, tracker = Scene(), Tracker()
    names = r['manifest']['joint_names']; target = np.array(r['manifest']['target_position_world_m'])
    with np.load(DATA / r['archive']) as a:
        poses, joints, x = a['base_pose'], a['joint_positions'], a['x']
    feats = np.zeros((len(poses), 4), np.float32); est_err = []
    for i, pose in enumerate(poses):
        scene.pose(pose, names, joints[i], target)
        B = np.array(p.getMatrixFromQuaternion(pose[3:].tolist())).reshape(3, 3)
        rgb, depth, _ = scene.fisheye(pose[:3] + B @ MOUNT, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
        seen = see_dot(rgb, depth)
        dot = tracker.update(camera_to_robot(seen) if seen is not None else None, x[i, 4:8])
        if dot is not None:
            a = np.arctan2(dot[1], dot[0]); feats[i, :3] = [np.hypot(*dot), np.sin(a), np.cos(a)]
        feats[i, 3] = float(seen is not None)
    return r['bearing_deg'], feats


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in open(DATA / 'episodes.jsonl')]
    split = {q['bearing_deg']: q['split'] for q in json.loads((DATA / 'split_plan.json').read_text())}
    unique = {}
    for r in records:
        unique.setdefault(r['trajectory_hash'], r)
    eps = [r for r in sorted(unique.values(), key=lambda r: r['bearing_deg']) if split[r['bearing_deg']] != 'holdout']
    with Pool(int(__import__('os').environ.get('WORKERS', 6))) as pool:
        for bearing, f in pool.imap_unordered(episode, eps):
            np.save(out / f'camera_{bearing}.npy', f)
            print(bearing, len(f), 'rows, visible', int(f[:, 3].sum()), flush=True)
