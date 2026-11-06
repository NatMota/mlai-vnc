"""Rows after the expert parks: replay the expert past its PARK command and keep the next N decisions (30 Hz).

    python expert_tail.py extract     park_tail/result_expert_tail.json -> park_tail/tail_<bearing>.npz (wheel inputs, poses)
    python expert_tail.py camera      adds the camera inputs (same camera, detector and tracker as camera_features.py)

The replay comes from collection/native policy_replay with "expert_tail_steps": 3400 (3.4 s after PARK).
Before PARK the replay is checked against the saved expert data.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'driving_30/data/expert_30hz'
TAIL = HERE / 'park_tail'
N = 100


def wheel_features(q, qd, q0):
    t = q - q0
    r, l = t[:, [0, 3]].mean(1), t[:, [1, 2]].mean(1); vr, vl = qd[:, [0, 3]].mean(1), qd[:, [1, 2]].mean(1)
    return np.c_[t, qd, (r + l) / 2, r - l, (vr + vl) / 2, vr - vl].astype(np.float32)


def extract():
    res = json.loads((TAIL / 'result_expert_tail.json').read_text())
    records = {}
    for l in open(DATA / 'episodes.jsonl'):
        r = json.loads(l); records.setdefault(r['bearing_deg'], r)
    worst = 0.0
    for run in res['runs']:
        b, e = run['bearing_deg'], run['expert']
        fr = e['frames']; park = e['park']['step']
        q = np.array([f['joints'][:4] for f in fr]); qd = np.array([f['wheel_velocity'] for f in fr]); q0 = q[0]
        grid = [i for i, f in enumerate(fr) if f['step'] == round(round(f['step'] * 30 / 1000) * 1000 / 30)]
        before = [i for i in grid if fr[i]['step'] <= park]; after = [i for i in grid if fr[i]['step'] > park][:N]
        assert len(after) == N, (b, len(after))
        with np.load(DATA / records[b]['archive']) as a:
            x_saved, pose_saved = a['x'][:, :12], a['base_pose']
        k = min(len(before), len(x_saved))
        x_re = wheel_features(q[before[:k]], qd[before[:k]], q0)
        pose_re = np.array([fr[i]['base'] + fr[i]['quaternion'] for i in before[:k]])
        worst = max(worst, float(np.abs(x_re - x_saved[:k]).max()), float(np.abs(pose_re - pose_saved[:k]).max()))
        idx = after
        np.savez(TAIL / f'tail_{b}.npz', x=wheel_features(q[idx], qd[idx], q0), wheel_q=q[idx], wheel_q0=q0,
                 base_pose=np.array([fr[i]['base'] + fr[i]['quaternion'] for i in idx]), joints=np.array([fr[i]['joints'] for i in idx]),
                 time_since_park_s=np.array([fr[i]['time_s'] - e['park']['time_s'] for i in idx]), target=np.array(run['target']))
    print(f'{len(res["runs"])} runs, {N} rows each after PARK. Largest difference from the saved expert data before PARK: {worst:.2e}')


def camera():
    from multiprocessing import Pool
    files = sorted(TAIL.glob('tail_*.npz'))
    with Pool(int(__import__('os').environ.get('WORKERS', 6))) as pool:
        for b, n in pool.imap_unordered(_camera_one, files):
            print(b, n, 'rows', flush=True)


def _camera_one(path):
    import pybullet as p
    from wrist_camera import Scene
    from camera_loop import MOUNT, see_dot, camera_to_robot, Tracker
    t = dict(np.load(path)); b = int(path.stem.split('_')[1])
    records = {}
    for l in open(DATA / 'episodes.jsonl'):
        r = json.loads(l); records.setdefault(r['bearing_deg'], r)
    names = records[b]['manifest']['joint_names']
    last = np.load(HERE / 'camera_features' / f'camera_{b}.npy')[-1]     # tracker carries on from the last expert row
    tracker = Tracker()
    if last[0] > 0:
        tracker.dot = last[0] * np.array([last[2], last[1]])
    scene, feats = Scene(), np.zeros((len(t['x']), 4), np.float32)
    for i, pose in enumerate(t['base_pose']):
        scene.pose(pose, names, t['joints'][i], t['target'])
        B = np.array(p.getMatrixFromQuaternion(pose[3:].tolist())).reshape(3, 3)
        rgb, depth, _ = scene.fisheye(pose[:3] + B @ MOUNT, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
        seen = see_dot(rgb, depth)
        dot = tracker.update(camera_to_robot(seen) if seen is not None else None, t['x'][i, 4:8])
        if dot is not None:
            a = np.arctan2(dot[1], dot[0]); feats[i, :3] = [np.hypot(*dot), np.sin(a), np.cos(a)]
        feats[i, 3] = float(seen is not None)
    t['camera'] = feats
    np.savez(path, **t)
    return b, len(feats)


if __name__ == '__main__':
    {'extract': extract, 'camera': camera}[sys.argv[1]]()
