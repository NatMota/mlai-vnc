"""DAgger data collection: the learned policy drives, a shadow expert labels every state it visits.

    python chester_dagger.py memory:outputs/memory-seed0/policy.pt --runs 600 --first-seed 300000 \
        --out data_dagger/iter1

Labels (all from the unchanged expert functions in robotnik_arm_control.py):
  wheels       the shadow expert's base state machine, stepped on the learner's own poses
               (BaseShadow: restarts when the learner drifts away from a parked pose).
               Once the shadow says "parked" the label is 0, which teaches the learner to stop.
  phase_label  1 once the shadow expert's base navigation is done (teaches when to switch).
  arm          the expert arm controller started from the learner's own memory q_ref, so the torque
               label matches the memory the learner actually carries; qd_des is the expert velocity.
Targets are random in the same box as the expert data, never split.json targets.
Saved like chester_collect.py, plus phase_label and weight (1 where the wheel label is usable).
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


class BaseShadow:
    """The expert's base state machine, stepped on the learner's poses, with restarts.

    The state machine only moves forward (align -> drive -> orient -> done), so once it says "parked"
    it would keep labelling "stop" even after the learner drifts off. It therefore restarts from
    align when the learner is clearly away from the spot, or when it is "done" but the learner's pose
    no longer meets a parked pose. No navigation timeout: the label always says where to go.
    """
    RESTART_DISTANCE_M = 0.15
    PARKED_DISTANCE_M = 0.10
    PARKED_HEADING_RAD = 0.10
    # Within this distance of the spot the label is always "turn in place to the final bearing".
    # The expert itself may still be heading for the spot there (align/drive) or already turning to
    # the final bearing (orient); from the pose alone the two look identical, and averaging opposite
    # turn labels taught the learner to stall ~5-10 degrees short of facing the dot.
    ORIENT_DISTANCE_M = 0.10

    def __init__(self, target):
        from robotnik_arm_control import BaseControllerParams, BaseNavState
        from chester_pool import parking_goal
        self.goal_pos, self.goal_bearing = parking_goal(target)
        self.params = BaseControllerParams(nav_timeout=float("inf"))
        self.state = BaseNavState()
        self.time = 0.0

    def step(self, base_pos, bearing, learner_reaching, dt):
        from robotnik_arm_control import BaseNavState, base_controller
        dist = np.linalg.norm(np.asarray(base_pos)[:2] - self.goal_pos)
        heading = abs(np.arctan2(np.sin(self.goal_bearing - bearing), np.cos(self.goal_bearing - bearing)))
        drifted = (self.state.phase in ("orient", "done") and dist > self.RESTART_DISTANCE_M) or (
            self.state.done and (dist > self.PARKED_DISTANCE_M or heading > self.PARKED_HEADING_RAD))
        if drifted and not learner_reaching:
            self.state = BaseNavState()
        if dist < self.ORIENT_DISTANCE_M and not self.state.done:
            self.state.phase = "orient"
        wheels = base_controller(self.state, base_pos, bearing, self.goal_pos, self.goal_bearing,
                                 self.time, self.params)
        self.time += dt
        return wheels, self.state.done


class ShadowExpert:
    """Labeler for chester_sim.run_episode."""

    def reset(self, robot, target, obs0):
        from chester_sim import Expert
        self.expert = Expert()
        self.expert.reset(robot, target, obs0)
        self.base = BaseShadow(target)

    def label_row(self, obs, learner_reaching, policy):
        from robotnik_arm_control import arm_controller
        e = self.expert
        wheels, done = self.base.step(obs["base_pos"], obs["bearing"], learner_reaching, e.robot.dt)
        torque, q_des, qd_des = arm_controller(
            e.robot.dyn, obs["q_pin"], obs["q"], obs["qd"], policy.q_ref_prev, e.q_hold, e.target,
            obs["ee"], e.robot.ee_id, e.robot.shoulder_id, e.robot.arm_lower, e.robot.arm_upper,
            e.robot.dt, e.arm_params, reach_active=learner_reaching)
        weight = 0.0 if learner_reaching else 1.0
        return np.concatenate((wheels, torque, q_des, qd_des, [float(done), weight]))


def relabel_base(src, dst, dt=0.001):
    """Recompute wheel and phase labels of a saved DAgger run with the current BaseShadow rules.

    Only base labels depend on the shadow's state machine, and they are a function of the saved
    poses alone, so no simulation is needed. Arm labels are copied unchanged.
    """
    from chester_pool import save_npz_atomic
    with np.load(src) as d:
        arrays = {k: d[k] for k in d.files}
    shadow = BaseShadow(arrays["state"][0, 18:21])
    label, phase = arrays["label"].copy(), arrays["phase_label"].copy()
    for i, s in enumerate(arrays["state"]):
        label[i, :4], done = shadow.step(s[0:3], s[3], arrays["reach"][i] > 0, dt)
        phase[i] = float(done)
    arrays.update(label=label, phase_label=phase)
    save_npz_atomic(dst, **arrays)


def collect_one(spec, seed, out, max_steps):
    from chester_parallel_eval import make_policy
    from chester_pool import EE, LABEL_START, REACH, STATE, sample_target, save_npz_atomic
    from chester_sim import run_episode
    rows = []
    result = run_episode(sample_target(seed), make_policy(spec), max_steps=max_steps, record=rows,
                         labeler=ShadowExpert())
    rows = np.asarray(rows)
    lab = rows[:, LABEL_START:]   # ShadowExpert.label_row: 11 label, 7 q_des, 7 qd_des, phase_label, weight
    save_npz_atomic(out / f"run_{seed}.npz", state=rows[:, STATE], ee=rows[:, EE], reach=rows[:, REACH],
                    label=lab[:, 0:11], q_des=lab[:, 11:18], qd_des=lab[:, 18:25], phase_label=lab[:, 25],
                    weight=lab[:, 26], final_error_cm=result["final_error_cm"], reached=result["reached"])
    print(f"DONE seed={seed} reached={result['reached']} final_cm={result['final_error_cm']:.2f}", flush=True)


def main():
    from chester_pool import run_pool, write_manifest
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--runs", type=int, default=600)
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--seed", type=int, help="internal: one run in this process")
    parser.add_argument("--relabel-from", type=Path, help="recompute base labels of this DAgger folder into --out")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        return collect_one(args.policy, args.seed, out, args.max_steps)
    if args.relabel_from:
        src = args.relabel_from if args.relabel_from.is_absolute() else ROOT / args.relabel_from
        files = sorted(f for f in src.glob("run_*.npz") if not f.name.endswith(".tmp.npz"))
        for n, f in enumerate(files, 1):
            relabel_base(f, out / f.name)
            if n % 50 == 0:
                print(f"relabelled {n}/{len(files)}", flush=True)
        write_manifest(out, script="chester_dagger.py --relabel-from", source=str(src), runs=len(files))
        return
    write_manifest(out, script="chester_dagger.py", policy=args.policy, runs=args.runs,
                   first_seed=args.first_seed, max_steps=args.max_steps)
    seeds = [s for s in range(args.first_seed, args.first_seed + args.runs)
             if not (out / f"run_{s}.npz").exists()]   # resumable
    jobs = [(s, [__file__, args.policy, "--seed", str(s), "--first-seed", "0", "--out", str(out),
                 "--max-steps", str(args.max_steps)]) for s in seeds]
    count = [0]

    def on_result(seed, text, error):
        count[0] += 1
        print(f"[{count[0]}/{len(jobs)}] seed={seed} " + (text or f"FAILED {error[-300:]}"), flush=True)

    failed = run_pool(jobs, args.workers, "DONE ", on_result)
    print(f"FINISHED {len(jobs) - failed} ok, {failed} failed", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
