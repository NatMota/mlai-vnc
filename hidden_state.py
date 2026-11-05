"""Rebuild the expert's hidden arm reference (q_des, qd_des) for saved data_dart runs, offline.

The expert arm torque is kp*(q_des - q) + kd*(qd_des - qd) + gravity, where q_des is integrated
over time and was never saved. Replaying arm_controller on the saved states recovers it exactly,
up to two approximations: base roll/pitch are taken as 0, fingers as their fixed 0.1 rad.
This is labelling only; no expert code runs inside the learned policy.

    python hidden_state.py --verify data_dart2/run_1.npz    (compare with a recorded run)
    python hidden_state.py --workers 20                     (convert all of data_dart)

Converted runs go to data_dart_mem/run_<name>.npz in the same format as collect_memory.py.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_dart_mem"


def reach_start(labels):
    """First row after the last nonzero wheel label."""
    moving = np.nonzero(np.any(labels[:, :4] != 0, axis=1))[0]
    return int(moving[-1] + 1) if len(moving) else 0


def rebuild(robot, state, labels, with_torque=False):
    import pinocchio as pin
    from robotnik_arm_control import ArmControllerParams, arm_controller
    from sim import ARM_REACH_LIMIT
    params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    T = len(state)
    target = state[0, 18:21]
    q_hold = state[0, 4:11].copy()
    start = reach_start(labels)
    q_des_all = np.repeat(q_hold[None], T, axis=0)
    qd_des_all = np.zeros((T, 7))
    ee_all = np.zeros((T, 3))
    torque_all = np.full((T, 7), np.nan)
    q_des = q_hold.copy()
    fingers = np.full(6, 0.1)
    for i in range(T):
        s = state[i]
        yaw = s[3]
        ori = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
        q17 = np.concatenate((np.zeros(4), s[4:11], fingers))
        q_pin = robot.dyn.ReoderJoints2PinVec(np.concatenate((s[0:3], ori, q17)), "pos")
        pin.forwardKinematics(robot.dyn.pin_model, robot.dyn.pin_data, q_pin)
        pin.updateFramePlacements(robot.dyn.pin_model, robot.dyn.pin_data)
        ee = robot.dyn.pin_data.oMf[robot.ee_id].translation.copy()
        ee_all[i] = ee
        if i < start and not with_torque:
            continue
        torque, q_des, qd_des = arm_controller(
            robot.dyn, q_pin, s[4:11], s[11:18], q_des, q_hold, target, ee, robot.ee_id,
            robot.shoulder_id, robot.arm_lower, robot.arm_upper, robot.dt, params, reach_active=i >= start)
        q_des_all[i], qd_des_all[i], torque_all[i] = q_des, qd_des, torque
    reach = (np.arange(T) >= start).astype(float)
    return {"ee": ee_all, "q_des": q_des_all, "qd_des": qd_des_all, "reach": reach, "torque": torque_all}


def convert_one(name):
    from sim import Robot
    data = np.load(ROOT / name)
    state, labels = data[:, :21], data[:, 21:32]
    robot = Robot(state[0, 18:21])
    out = rebuild(robot, state, labels)
    final = float(np.linalg.norm(out["ee"][-1] - state[-1, 18:21]) * 100)
    tmp = OUT / f"run_{Path(name).stem}.tmp.npz"
    np.savez_compressed(tmp, state=state, ee=out["ee"], reach=out["reach"], label=labels,
                        q_des=out["q_des"], qd_des=out["qd_des"], final_error_cm=final)
    tmp.replace(OUT / f"run_{Path(name).stem}.npz")
    print(f"DONE {name} final_cm={final:.2f}", flush=True)


def verify(path):
    """Compare the rebuild with a run whose hidden reference was recorded during collection."""
    from sim import Robot
    d = np.load(path)
    robot = Robot(d["state"][0, 18:21])
    out = rebuild(robot, d["state"], d["label"], with_torque=True)
    reach = d["reach"] > 0
    report = {
        "rows": int(len(reach)), "reach_rows": int(reach.sum()),
        "reach_start_match": bool(np.argmax(reach) == np.argmax(out["reach"] > 0)),
        "ee_error_mm_median": float(np.median(np.linalg.norm(out["ee"] - d["ee"], axis=1)) * 1000),
        "ee_error_mm_max": float(np.max(np.linalg.norm(out["ee"] - d["ee"], axis=1)) * 1000),
        "q_des_error_mrad_max": float(np.abs(out["q_des"] - d["q_des"])[reach].max() * 1000),
        "qd_des_error_max": float(np.abs(out["qd_des"] - d["qd_des"])[reach].max()),
        "torque_error_Nm_median": float(np.median(np.abs(out["torque"] - d["label"][:, 4:]))),
        "torque_error_Nm_p99": float(np.percentile(np.abs(out["torque"] - d["label"][:, 4:]), 99)),
    }
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--one")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.verify:
        return verify(args.verify)
    OUT.mkdir(exist_ok=True)
    if args.one:
        return convert_one(args.one)
    split = json.loads((ROOT / "data_dart/split.json").read_text())
    names = [n for group in ("train", "val") for n in split[group]
             if not (OUT / f"run_{Path(n).stem}.npz").exists()]
    env = dict(os.environ, ROBOENV_HEADLESS="1", OMP_NUM_THREADS="1")

    def job(name):
        out = subprocess.run([sys.executable, "-B", __file__, "--one", name], env=env,
                             capture_output=True, text=True)
        done = [line for line in out.stdout.splitlines() if line.startswith("DONE")]
        return done[0] if done else f"FAILED {name} {out.stderr[-300:]}"

    with ThreadPoolExecutor(args.workers) as pool:
        for n, line in enumerate(pool.map(job, names), 1):
            print(f"[{n}/{len(names)}] {line}", flush=True)


if __name__ == "__main__":
    main()
