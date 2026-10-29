"""Evaluate a policy on many split targets in parallel, one subprocess per target.

    python chester_parallel_eval.py expert --group val --out outputs/eval/expert_val.jsonl
    python chester_parallel_eval.py memory:outputs/memory-seed0/policy.pt --group val --n 120

Each line of the JSONL output is one target's result (see chester_sim.run_episode).
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

from chester_pool import run_pool

ROOT = Path(__file__).resolve().parent


def split_files(group):
    return json.loads((ROOT / "data_dart/split.json").read_text())[group]


def target_of(name):
    return np.load(ROOT / name, mmap_mode="r")[0, 18:21].tolist()


def make_policy(spec):
    """'expert', or 'memory:<checkpoint>[+<checkpoint>...]' for the learned memory policy."""
    if spec == "expert":
        from chester_sim import Expert
        return Expert()
    kind, _, path = spec.partition(":")
    if kind == "memory":
        from chester_memory_policy import MemoryPolicy
        return MemoryPolicy.load(path)
    raise SystemExit(f"Unknown policy spec: {spec}")


def run_one(args):
    from chester_sim import run_episode
    target = target_of(split_files(args.group)[args.index])
    result = run_episode(target, make_policy(args.policy), max_steps=args.max_steps)
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


def provenance(policy):
    info = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip()}
    if ":" in policy:
        info["checkpoint_sha256"] = [hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                     for p in policy.partition(":")[2].split("+")]
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("--group", choices=("train", "val", "test"), default="val")
    parser.add_argument("--n", type=int, default=120, help="first n targets of the group")
    parser.add_argument("--index", type=int, help="internal: run one target in this process")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.index is not None:
        return run_one(args)
    if args.out is None:
        raise SystemExit("--out is required")
    available = len(split_files(args.group))
    if not 0 < args.n <= available:
        raise SystemExit(f"--n must be between 1 and {available} for group {args.group}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(i, [__file__, args.policy, "--group", args.group, "--index", str(i),
                 "--max-steps", str(args.max_steps)]) for i in range(args.n)]
    started = time.perf_counter()
    rows = []
    with args.out.open("w", encoding="utf-8") as fh:
        def on_result(i, text, error):
            row = json.loads(text) if text else {"target": i, "group": args.group, "error": error,
                                                 "reached": False}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if error:
                print(f"ERROR target {i:3d}: {error[-300:]}", flush=True)
            else:
                flag = "OK  " if row["reached"] else "MISS"
                print(f"{flag} target {i:3d}  final {row['final_error_cm']:.2f} cm", flush=True)
        failed = run_pool(jobs, args.workers, "JSON ", on_result)
    summary = summarize(rows)
    summary.update({"policy": args.policy, "group": args.group, "n": args.n, "max_steps": args.max_steps,
                    "wall_s": time.perf_counter() - started, **provenance(args.policy)})
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary), flush=True)
    from chester_tracking import log_evaluation
    log_evaluation(args.policy, summary, files=[args.out, args.out.with_suffix(".summary.json")])
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
