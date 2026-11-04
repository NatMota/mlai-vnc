"""Pack raw collector runs (one folder of NPY tables per run, about 17 MB) into one compressed .npz per run.

Each packed run keeps what training and testing use, every 1 ms, as float32 (time in float64):
  time_s, true state (q and qd for the 17 joints, base position, quaternion, linear and angular velocity),
  measured state (what the expert saw, only when noise or delay was on: q, qd, base position, quaternion),
  the expert's commands (7 arm torques, 4 wheel speeds), the torques sent to the 11 motors, its targets (q_des, qd_des),
  the hand pose and gripper-to-dot distance the expert computed (from what it measured),
  the true hand pose and true gripper-to-dot distance (recomputed from the true state when measured and true differ),
  reach_active and navigation_done flags, and columns.json naming every column.
The manifest (target, termination, parking step, effects, seeds, hashes) is stored as JSON text.

The vincent32 export (vincent_npy) writes Vincent's 32 columns per 1 ms row for the team's tools:
  base position (3), bearing, arm q (7), arm qd (7), dot (3), wheel commands (4), arm torques (7).

usage: python tools/pack_runs.py RAW_FOLDER OUT_FOLDER [--delete] [--condition NAME]
"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np


def table(folder, name):
    cols = json.loads((folder / f'{name}.schema.json').read_text())['columns']
    return dict(zip(cols, np.load(folder / f'{name}.npy').T))


def cols(t, prefix, labels):
    return np.stack([t[f'{prefix}_{l}'] for l in labels], 1)


class TruePose:
    """True hand position and rotation from the true state, with the robot model in PyBullet (loaded on first use)."""
    def __init__(self, urdf):
        self.urdf, self.p = urdf, None

    def __call__(self, pos, quat, q):
        if self.p is None:
            import pybullet as p
            self.p = p; self.cid = p.connect(p.DIRECT)
            self.robot = p.loadURDF(str(self.urdf), useFixedBase=True, flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=self.cid)
            info = [p.getJointInfo(self.robot, j, physicsClientId=self.cid) for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid))]
            self.ids = [i[0] for i in info if i[2] != p.JOINT_FIXED]
            self.hand = next(i[0] for i in info if i[12].decode() == 'robot_j2s7s300_end_effector')
        p, out_p, out_r = self.p, [], []
        for k in range(len(pos)):
            p.resetBasePositionAndOrientation(self.robot, pos[k].tolist(), quat[k].tolist(), physicsClientId=self.cid)
            for j, v in zip(self.ids, q[k]):
                p.resetJointState(self.robot, j, float(v), physicsClientId=self.cid)
            s = p.getLinkState(self.robot, self.hand, computeForwardKinematics=True, physicsClientId=self.cid)
            out_p.append(s[4]); out_r.append(p.getMatrixFromQuaternion(s[5]))
        return np.array(out_p), np.array(out_r)


def pack(run, out, pose, condition):
    m = json.loads((run / 'manifest.json').read_text())
    js, bs, ac, bc, kin, cmd = (table(run, n) for n in ('joint_state', 'base_state', 'arm_controller', 'base_controller', 'kinematics', 'commands'))
    J = [f'j{j:02}' for j in range(1, 18)]; A = [f'arm{j}' for j in range(1, 8)]; xyz, quat = list('xyz'), list('xyzw')
    rot = [f'r{r}{c}' for r in (1, 2, 3) for c in (1, 2, 3)]
    target = np.array(m['target_position_world_m'])
    arrays = dict(
        time_s=js['time_s'],
        q=cols(js, 'q_rad', J), qd=cols(js, 'qd_rad_s', J),
        base_position=cols(bs, 'position_m', xyz), base_quaternion=cols(bs, 'quaternion', quat),
        base_linear_velocity=cols(bs, 'linear_velocity_world_m_s', xyz), base_angular_velocity=cols(bs, 'angular_velocity_world_rad_s', xyz),
        arm_torque_command=cols(ac, 'tau_command_Nm', A), arm_q_des=cols(ac, 'q_des_rad', A), arm_qd_des=cols(ac, 'qd_des_rad_s', A),
        wheel_command=cols(bc, 'wheel_velocity_command_rad_s', ['front_right', 'front_left', 'back_left', 'back_right']),
        sent_torque=cols(cmd, 'sent_torque_Nm', J[:11]),
        hand_position_expert=cols(kin, 'ee_world_m', xyz), hand_rotation_expert=cols(kin, 'ee_rotation', rot),
        hand_distance_expert=kin['ee_distance_error_m'],
        reach_active=ac['reach_active'], navigation_done=bc['navigation_done'])
    measured = (run / 'measured_state.npy').exists()
    if measured:
        ms = table(run, 'measured_state')
        arrays.update(measured_q=cols(ms, 'q_rad', J), measured_qd=cols(ms, 'qd_rad_s', J),
                      measured_base_position=cols(ms, 'position_m', xyz), measured_base_quaternion=cols(ms, 'quaternion', quat))
        hp, hr = pose(arrays['base_position'], arrays['base_quaternion'], arrays['q'])
        arrays.update(hand_position=hp, hand_rotation=hr, hand_distance=np.linalg.norm(hp - target, axis=1))
    else:                                    # the expert measured the true state: its hand pose is the true one
        arrays.update(hand_position=arrays['hand_position_expert'], hand_rotation=arrays['hand_rotation_expert'], hand_distance=arrays['hand_distance_expert'])
    columns = {k: (list(v.shape[1:]) if v.ndim > 1 else []) for k, v in arrays.items()}
    store = {k: (v.astype(np.float64) if k == 'time_s' else v.astype(np.float32)) for k, v in arrays.items()}
    m.update(condition=condition, measured_state_saved=measured, packed_from=run.name)
    np.savez_compressed(out, manifest=np.array(json.dumps(m)), columns=np.array(json.dumps(columns)), **store)


def vincent_npy(npz_path, out_path):
    """Vincent's 32-column layout, one row per 1 ms step (true state)."""
    d = np.load(npz_path); m = json.loads(str(d['manifest']))
    qx, qy, qz, qw = d['base_quaternion'].T
    bearing = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    rows = np.c_[d['base_position'], bearing, d['q'][:, 4:11], d['qd'][:, 4:11], np.tile(m['target_position_world_m'], (len(bearing), 1)),
                 d['wheel_command'], d['arm_torque_command']]
    np.save(out_path, rows.astype(np.float64))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('raw'); ap.add_argument('out'); ap.add_argument('--delete', action='store_true')
    ap.add_argument('--condition', default=''); ap.add_argument('--urdf', default=str(Path(__file__).resolve().parents[1] / 'assets/summit_xl_arm_description/summit_xl_complete.urdf'))
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out); out.mkdir(parents=True, exist_ok=True); pose = TruePose(a.urdf); n = 0
    runs = sorted([p for p in raw.rglob('manifest.json')], key=lambda p: str(p))
    for mf in runs:
        run = mf.parent
        dest = out / f'{run.name}.npz'
        if not dest.exists():
            pack(run, dest, pose, a.condition); n += 1
        if a.delete:
            shutil.rmtree(run)
    print(f'packed {n} runs into {out}', file=sys.stderr)
