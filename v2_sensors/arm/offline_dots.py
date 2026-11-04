"""How far arm networks are from the expert along the expert's own paths, on packed collector runs.

For each packed run (collection/native/tools/pack_runs.py) the network's inputs are built as in training
(train_arm.load_run): arm encoder readings every 1 ms from the true joint angles, the dot in the robot frame from the
simulator (not the camera), time inputs if the network uses them, from the expert's PARK to its stop. It reports the
torque error per joint and the stop network's error against the true gripper-to-dot distance. This measures how well
the network matches the expert's commands; it is not a test of the network driving the arm.

usage: python offline_dots.py MODEL [MODEL ...] --packed ROOT --conditions a b ... --split SPLIT_JSON --out OUT_JSON
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import sensors
from train_arm import dot_in_robot, yaw_of, select_inputs, policy, DIST_CLIP


def rows(z):
    m = json.loads(str(z['manifest'])); park, stop = m['parking_step'], m['steps'] - 1
    if park < 0 or park >= stop:
        return None
    q = z['q'][:, 4:11].astype(np.float64)
    angle, speed = sensors.ArmEncoders.readings_for_run(q, 1e-3)
    d = np.arange(park, stop + 1); t = d / 1000
    qx, qy, qz, qw = z['base_quaternion'][d].T
    dot = dot_in_robot(np.array(m['target_position_world_m']), z['base_position'][d].astype(np.float64), yaw_of(qx, qy, qz, qw))
    X = np.c_[angle[d], speed[d], dot, t - park / 1000, t]
    return X, z['arm_torque_command'][d], z['hand_distance'][d], m


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('models', nargs='+'); ap.add_argument('--packed', required=True)
    ap.add_argument('--conditions', nargs='+', required=True); ap.add_argument('--split', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--split-name', default='test')
    a = ap.parse_args()
    dots = json.loads(Path(a.split).read_text())[a.split_name]
    W = {name: dict(np.load(HERE / f'models/{name}.npz')) for name in a.models}
    result = {}
    for cond in a.conditions:
        per = {name: dict(err=[], stop_err=[]) for name in a.models}; used = 0; skipped = 0
        for dot in dots:
            f = Path(a.packed) / cond / f'run_{dot}.npz'
            if not f.exists():
                continue
            r = rows(np.load(f))
            if r is None:
                skipped += 1; continue
            X, T, D, m = r; used += 1
            for name, w in W.items():
                n_in = w['shared.0.weight'].shape[1]
                x = select_inputs(X, False, True, bool(w['info_time']) if 'info_time' in w else n_in == 19)
                out = [policy(w, row) for row in x[::10]]                       # every 10 ms
                P = np.array([o[0] for o in out]); S = np.array([o[1] for o in out])
                per[name]['err'].append(np.sqrt(((P - T[::10]) ** 2).mean(0)))
                per[name]['stop_err'].append(float(np.sqrt(np.mean((S - np.clip(D[::10], 0, DIST_CLIP)) ** 2))))
        result[cond] = dict(runs=used, skipped_no_park=skipped, **{name: dict(
            torque_rmse_per_joint_Nm=np.median(v['err'], 0).round(3).tolist() if v['err'] else None,
            torque_rmse_Nm=float(np.median([np.sqrt((e ** 2).mean()) for e in v['err']])) if v['err'] else None,
            stop_rmse_mm=float(1000 * np.median(v['stop_err'])) if v['stop_err'] else None) for name, v in per.items()})
        print(cond, json.dumps(result[cond]))
    Path(a.out).write_text(json.dumps(result, indent=1))
