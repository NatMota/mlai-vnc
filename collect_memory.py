"""Collect extra DART expert runs in parallel, with the expert's hidden arm reference recorded.

    python collect_memory.py --runs 2000 --first-seed 100000 --workers 20

Each run is saved as data_dart2/run_<seed>.npz (seed-based names, so parallel runs never collide):
    state   (T, 21)  base_pos 3, bearing, arm q 7, arm qd 7, target 3   (same as data_dart columns 0-20)
    label   (T, 11)  clean expert wheel commands 4 + arm torques 7       (same as data_dart columns 21-31)
    ee      (T, 3)   end-effector position
    q_des   (T, 7)   expert arm joint reference used for that step's torque
    qd_des  (T, 7)   expert arm joint velocity reference
    reach   (T,)     1 once the expert's base navigation is done
Pushes match robotnik_arm_task_dart.py (wheel 0.5 rad/s while driving, arm 1.0 Nm), targets the same box.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data_dart2"
WHEEL_PUSH = 0.5
ARM_PUSH = 1.0


class Recorder:
    """Labeler that copies the expert's clean command and hidden reference into each row."""

    def reset(self, robot, target, obs0):
        pass

    def label_row(self, obs, active, policy):
        wheels, torque = policy.last_clean
        return np.concatenate((wheels, torque, policy.q_des, policy.qd_des))


def collect_one(seed, max_steps):
    from sim import Expert, run_episode
    rng = np.random.default_rng(seed)
    target = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0.1, 1.25)])
    rows = []
    result = run_episode(target, Expert(WHEEL_PUSH, ARM_PUSH, seed=seed + 1), max_steps=max_steps,
                         record=rows, labeler=Recorder())
    rows = np.asarray(rows)
    # Row layout from run_episode: 21 state, 3 ee, 1 reach, 4+7 executed, then 11 label + 7 q_des + 7 qd_des.
    tmp = OUT / f"run_{seed}.tmp.npz"
    np.savez_compressed(tmp, state=rows[:, :21], ee=rows[:, 21:24], reach=rows[:, 24],
                        label=rows[:, 36:47], q_des=rows[:, 47:54], qd_des=rows[:, 54:61],
                        final_error_cm=result["final_error_cm"])
    tmp.replace(OUT / f"run_{seed}.npz")
    print(f"DONE seed={seed} reached={result['reached']} final_cm={result['final_error_cm']:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2000)
    parser.add_argument("--first-seed", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=40000)
    parser.add_argument("--seed", type=int, help="internal: collect one run in this process")
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True)
    if args.seed is not None:
        return collect_one(args.seed, args.max_steps)
    env = dict(os.environ, ROBOENV_HEADLESS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    seeds = [s for s in range(args.first_seed, args.first_seed + args.runs)
             if not (OUT / f"run_{s}.npz").exists()]   # resumable

    def job(seed):
        out = subprocess.run([sys.executable, "-B", __file__, "--seed", str(seed),
                              "--max-steps", str(args.max_steps)], env=env, capture_output=True, text=True)
        done = [line for line in out.stdout.splitlines() if line.startswith("DONE")]
        return done[0] if done else f"FAILED seed={seed} {out.stderr[-300:]}"

    with ThreadPoolExecutor(args.workers) as pool:
        for n, line in enumerate(pool.map(job, seeds), 1):
            print(f"[{n}/{len(seeds)}] {line}", flush=True)


if __name__ == "__main__":
    main()
