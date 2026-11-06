"""Export the expert's runs for a set of dots from packed collector runs (collection/native/tools/pack_runs.py).

Writes one JSON with, for each dot: its target, the expert's run length, termination, parking step and final
gripper-to-dot distance (true), and its frames every 1/30 s for the 3D player (base pose, joints, hand, torques).
test_arm.py --cases uses the targets and run lengths; replay_arm.py uses the frames as the expert ghost.

usage: python export_dot_cases.py PACKED_CONDITION_FOLDER SPLIT_JSON OUT_JSON [--split test]
"""
import argparse, json
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('packed'); ap.add_argument('split'); ap.add_argument('out'); ap.add_argument('--split-name', default='test')
    a = ap.parse_args()
    dots = json.loads(Path(a.split).read_text())[a.split_name]
    out = {}
    for d in dots:
        f = Path(a.packed) / f'run_{d}.npz'
        if not f.exists():
            continue
        z = np.load(f); m = json.loads(str(z['manifest']))
        steps, park = m['steps'], m['parking_step']
        idx = [k for k in (round(i * 1000 / 30) for i in range(10 ** 5)) if k < steps]
        frames = [dict(time_s=k / 1000, base=z['base_position'][k].tolist(), quaternion=z['base_quaternion'][k].tolist(),
                       joints=z['q'][k].tolist(), ee=z['hand_position'][k].tolist(), parked=bool(park >= 0 and k >= park),
                       tau=np.round(z['arm_torque_command'][k], 3).tolist(), ee_mm=float(1000 * z['hand_distance'][k])) for k in idx]
        out[str(d)] = dict(target=m['target_position_world_m'], steps=steps, termination=m['termination'], parking_step=park,
                           final_mm=float(1000 * z['hand_distance'][-1]), park_s=park / 1000, stop_s=(steps - 1) / 1000,
                           stop_mm=float(1000 * z['hand_distance'][-1]), frames=frames)
    Path(a.out).write_text(json.dumps(out, separators=(',', ':')))
    print(f'{len(out)} dots exported to {a.out}')
