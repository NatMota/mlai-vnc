"""Evaluate a policy on many split targets in parallel, one subprocess per target.

    python parallel_eval.py expert --group val --out outputs/eval/expert_val.jsonl
    python parallel_eval.py memory:outputs/memory-seed0/policy.pt --group val --n 120

Each line of the JSONL output is one target's result (see sim.run_episode).
A target whose process crashed or timed out counts as a miss, and the script then exits with 1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from pool import run_pool, sample_target
from iteration_io import output_lock, write_json

ROOT = Path(__file__).resolve().parent
FRESH_FIRST_SEED = 900000   # far from every seed used for data collection or DAgger
OUTSIDE_FIRST_SEED = 800000
# name, smallest and biggest max(|x|, |y|), lowest and highest z; the expert's box is 1.5 m and 0.1-1.25 m.
OUTSIDE_REGIONS = (("far", 1.5, 2.0, 0.1, 1.25), ("farther", 2.0, 3.0, 0.1, 1.25),
                   ("higher", 0.0, 1.5, 1.25, 1.45), ("lower", 0.0, 1.5, 0.0, 0.1))
SOURCES = ("parallel_eval.py", "fused_policy.py", "memory_policy.py", "sim.py",
           "features.py", "pool.py", "iteration_io.py", "robotnik_arm_control.py",
           "configs/robotnik_arm.json")


def split_files(group):
    return json.loads((ROOT / "data_dart/split.json").read_text())[group]


def target_of(name):
    return np.load(ROOT / name, mmap_mode="r")[0, 18:21].tolist()


def outside_target(index):
    """A goal outside the expert's target box; index % 4 picks the region in OUTSIDE_REGIONS."""
    _, inner, outer, low_z, high_z = OUTSIDE_REGIONS[index % len(OUTSIDE_REGIONS)]
    rng = np.random.default_rng(OUTSIDE_FIRST_SEED + index)
    while True:
        x, y = rng.uniform(-outer, outer, 2)
        if max(abs(x), abs(y)) >= inner:
            return [float(x), float(y), float(rng.uniform(low_z, high_z))]


def target_for(group, index):
    """'fresh' goals are new random draws from the expert's target box, not in any data folder."""
    if group == "fresh":
        return sample_target(FRESH_FIRST_SEED + index).tolist()
    if group == "outside":
        return outside_target(index)
    return target_of(split_files(group)[index])


def make_policy(spec):
    """'expert', 'memory:<checkpoint>[+...]', or 'memory30:<checkpoint>[+...]' (vel_net at about 30 Hz)."""
    if spec == "expert":
        from sim import Expert
        return Expert()
    kind, _, path = spec.partition(":")
    if kind in ("memory", "memory30"):
        from memory_policy import MemoryPolicy
        return MemoryPolicy.load(path, vel_every=33 if kind == "memory30" else 1)
    if kind in ("team", "fused", "fused30"):
        from fused_policy import make_fused
        return make_fused(kind, path)
    raise SystemExit(f"Unknown policy spec: {spec}")


class GoalError:
    """Simulates a biased goal estimate (e.g. from a camera): the policy sees a goal offset by a fixed
    length in a random direction, chosen per goal. Success is still measured at the true goal."""

    def __init__(self, policy, length_m, seed):
        direction = np.random.default_rng(seed).normal(size=3)
        self.offset = length_m * direction / np.linalg.norm(direction)
        self.policy = policy
        self.act = policy.act

    def reset(self, robot, target, obs0):
        self.policy.reset(robot, np.asarray(target) + self.offset, obs0)


def run_one(args):
    from sim import run_episode
    target = target_for(args.group, args.index)
    policy = make_policy(args.policy)
    if args.goal_error_m:
        policy = GoalError(policy, args.goal_error_m, args.index)
    result = run_episode(target, policy, max_steps=args.max_steps)
    result.update({"group": args.group, "target": args.index, "target_xyz": target, "policy": args.policy})
    print("JSON " + json.dumps(result), flush=True)


def summarize(rows):
    """Success over ALL rows: crashed targets are counted as misses, never dropped."""
    reached = [r for r in rows if r["reached"]]
    final = np.array([r["final_error_cm"] for r in rows if "error" not in r])
    return {"targets": len(rows), "successes": len(reached),
            "success_rate": len(reached) / max(1, len(rows)),
            "mean_final_error_cm": float(final.mean()) if len(final) else None,
            "median_final_error_cm": float(np.median(final)) if len(final) else None,
            "mean_success_time_s": float(np.mean([r["time_s"] for r in reached])) if reached else None,
            "process_errors": sum("error" in r for r in rows)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provenance(policy):
    info = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "source_sha256": {name: sha256(ROOT / name) for name in SOURCES}}
    if ":" in policy:
        info["checkpoint_sha256"] = [sha256(p) for p in policy.partition(":")[2].split("+")]
    if policy.startswith(("team:", "fused")):
        from fused_policy import TEAM_DIR
        info["team_checkpoint_sha256"] = sha256(TEAM_DIR / "checkpoint.pt")
        info["team_scaler_sha256"] = sha256(TEAM_DIR / "scaler.npz")
        info["team_limits_sha256"] = sha256(TEAM_DIR / "limits.npz")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--group", choices=("train", "val", "test", "fresh", "outside"), default="val")
    parser.add_argument("--n", type=int, default=120, help="first n targets of the group")
    parser.add_argument("--index", type=int, help="internal: run one target in this process")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--goal-error-m", type=float, default=0.0, help="bias added to the goal the policy sees")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true", help="continue only a matching evaluation manifest")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.index is not None:
        return run_one(args)
    if args.out is None:
        raise SystemExit("--out is required")
    if args.workers < 1 or args.max_steps < 1 or args.goal_error_m < 0:
        raise ValueError("workers and max-steps must be positive; goal error must be non-negative")
    available = 100000 if args.group in ("fresh", "outside") else len(split_files(args.group))
    if not 0 < args.n <= available:
        raise SystemExit(f"--n must be between 1 and {available} for group {args.group}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(args.out.with_suffix(".lock")):
        evaluate(args)


def evaluate(args):
    manifest = args.out.with_suffix(".manifest.json")
    summary_path = args.out.with_suffix(".summary.json")
    exists = any(p.exists() for p in (args.out, manifest, summary_path))
    if exists and not args.resume:
        raise ValueError("Output exists; use --resume or a new --out path")
    if args.resume and not manifest.exists():
        raise ValueError("Cannot resume legacy results without a manifest")
    info = provenance(args.policy)
    # Commit/working-tree state is informative; byte hashes define executable identity.
    config = {"policy": args.policy, "group": args.group, "n": args.n,
              "max_steps": args.max_steps, "goal_error_m": args.goal_error_m,
              "provenance": {k: v for k, v in info.items() if k not in ("commit", "dirty")}}
    if args.group in ("train", "val", "test"):
        config["target_sha256"] = {name: sha256(ROOT / name) for name in split_files(args.group)[:args.n]}
        config["split_sha256"] = sha256(ROOT / "data_dart/split.json")
    rows = []
    if args.resume:
        if json.loads(manifest.read_text(encoding="utf-8"))["config"] != config:
            raise ValueError("Evaluation config does not match the manifest")
        if args.out.exists():
            rows = [json.loads(line) for line in args.out.read_text(encoding="utf-8").splitlines()]
    else:
        write_json(manifest, {"config": config, "execution": info})
    seen = set()
    for row in rows:
        index = row.get("target")
        if index in seen:
            raise ValueError("Duplicate target in saved results")
        if (type(index) is not int or index not in range(args.n) or row.get("group") != args.group
                or row.get("policy", args.policy) != args.policy or type(row.get("reached")) is not bool):
            raise ValueError("Invalid saved result; preserve it and use a new output path")
        seen.add(index)
    if summary_path.exists():
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(seen) != args.n or old.get("results_sha256") != sha256(args.out):
            raise ValueError("Summary does not match saved results")
        print("SKIP matching completed evaluation", flush=True)
        return
    jobs = [(i, [__file__, args.policy, "--group", args.group, "--index", str(i),
                 "--max-steps", str(args.max_steps), "--goal-error-m", str(args.goal_error_m)])
            for i in range(args.n) if i not in seen]
    started = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as fh:
        def on_result(i, text, error):
            row = json.loads(text) if text else {"target": i, "group": args.group, "error": error,
                                                 "reached": False}
            if i in seen or row.get("target") != i or row.get("group") != args.group:
                raise ValueError("Duplicate or mismatched worker result")
            seen.add(i)
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if error:
                print(f"ERROR target {i:3d}: {error[-300:]}", flush=True)
            else:
                flag = "OK  " if row["reached"] else "MISS"
                print(f"{flag} target {i:3d}  final {row['final_error_cm']:.2f} cm", flush=True)
        run_pool(jobs, args.workers, "JSON ", on_result)
    if len(seen) != args.n:
        raise ValueError("Incomplete evaluation; use --resume to continue missing targets")
    summary = summarize(rows)
    summary.update({"policy": args.policy, "group": args.group, "n": args.n, "max_steps": args.max_steps, "goal_error_m": args.goal_error_m,
                    "wall_s": time.perf_counter() - started, "wall_s_scope": "this invocation",
                    "complete": True, "results_sha256": sha256(args.out), **info})
    if args.group in ("fresh", "outside"):
        summary["first_seed"] = FRESH_FIRST_SEED if args.group == "fresh" else OUTSIDE_FIRST_SEED
    write_json(summary_path, summary)
    print("SUMMARY " + json.dumps(summary), flush=True)
    from local_tracking import log_evaluation
    log_evaluation(args.policy, summary, files=[args.out, args.out.with_suffix(".summary.json")])
    if summary["process_errors"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
