"""Evaluate the v2 runs on Nat's 25 bearings (21 train, 4 test; the 5 holdout bearings are never used).

    python v2_eval.py A B C D E1 E2 E3 --workers 12

Runs (driving, arm, arm inputs):
    A   Nat iteration 13 (seed 7), arm held all run, -        parity of Nat's driving in this simulator
    B   expert,                    two-rate arm, sensors      arm on sensors without driving errors
    C   Nat iteration 13 (seed 7), two-rate arm, sensors      full v2
    D   Nat iteration 13 (seed 7), two-rate arm, true state   C without arm sensor errors
    E1-E3  as C with Nat's driving seeds 1, 2, 3
One subprocess per episode (see pool). Results: outputs/v2/RUN.jsonl and RUN.summary.json.
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

from pool import run_pool

ROOT = Path(__file__).resolve().parent
NAT_DIR = ROOT / "nat_v2"
ARM_CHECKPOINT = ROOT / "outputs/ablation/all_world/policy.pt"
GOAL_RADIUS_M, GOAL_HEIGHT_M = 1.1, 0.5            # Nat's targets: [1.1 cos b, 1.1 sin b, 0.5]
RUNS = {
    "A": ("iter13", "true", False),
    "B": ("expert", "sensor", True),
    "C": ("iter13", "sensor", True),
    "D": ("iter13", "true", True),
    "E1": ("iter13_seed1", "sensor", True),
    "E2": ("iter13_seed2", "sensor", True),
    "E3": ("iter13_seed3", "sensor", True),
}


def bearings():
    """{bearing in degrees: split} for the 25 bearings Nat tests (train and test only)."""
    ref = json.loads((NAT_DIR / "expert_reference.json").read_text())["bearings"]
    return {int(b): v["split"] for b, v in ref.items() if v["split"] in ("train", "test")}


def goal_of(bearing):
    b = np.deg2rad(bearing)
    return [GOAL_RADIUS_M * np.cos(b), GOAL_RADIUS_M * np.sin(b), GOAL_HEIGHT_M]


def run_one(run, bearing):
    digest = code_digest(file_hashes())             # the files as this episode imports them
    from sim import run_episode
    from v2_sensor import NatDriver, V2Policy
    driver_name, arm_inputs, use_arm = RUNS[run]
    driver = "expert" if driver_name == "expert" else NatDriver(NAT_DIR / "models" / f"{driver_name}.npz")
    policy = V2Policy.build(ARM_CHECKPOINT, driver, arm_inputs=arm_inputs, use_arm=use_arm)
    result = run_episode(goal_of(bearing), policy)
    expert = json.loads((NAT_DIR / "expert_reference.json").read_text())["bearings"][str(bearing)]
    yaw_error = (result["final_bearing"] - expert["final_yaw"] + np.pi) % (2 * np.pi) - np.pi
    result.update(policy.report, run=run, bearing=bearing, split=bearings()[bearing], driver=driver_name,
                  arm_inputs=arm_inputs, use_arm=use_arm,
                  base_error_vs_expert_cm=float(np.linalg.norm(np.subtract(result["final_base"],
                                                                           expert["final_base_xy"])) * 100),
                  yaw_error_vs_expert_deg=float(abs(np.rad2deg(yaw_error))),
                  code_digest=digest)
    print("JSON " + json.dumps(result), flush=True)


def summarize(rows):
    """Counts per split, with crashed episodes counted as misses."""
    out = {}
    for split in ("train", "test", "all"):
        part = [r for r in rows if split == "all" or r.get("split") == split]
        good = [r for r in part if "error" not in r]
        def median(key):
            values = [r[key] for r in good if r.get(key) is not None]
            return float(np.median(values)) if values else None
        out[split] = {"episodes": len(part), "reached_1cm": sum(bool(r.get("reached")) for r in part),
                      "parked": sum(bool(r.get("parked")) for r in part),
                      "median_final_error_cm": median("final_error_cm"),
                      "median_base_error_vs_expert_cm": median("base_error_vs_expert_cm"),
                      "median_camera_xy_error_at_park_cm": median("camera_xy_error_at_park_cm"),
                      "median_camera_height_error_at_park_cm": median("camera_height_error_at_park_cm"),
                      "median_camera_xy_error_last_cm": median("camera_xy_error_last_cm"),
                      "median_camera_height_error_last_cm": median("camera_height_error_last_cm"),
                      "process_errors": len(part) - len(good)}
    return out


SOURCES = ("v2_sensor.py", "v2_eval.py", "sim.py", "memory_policy.py",
           "fused_policy.py", "pool.py", "features.py", "robotnik_arm_control.py",
           "configs/robotnik_arm.json")


def file_hashes():
    """sha256 of every file that decides a result."""
    files = [*(ROOT / name for name in SOURCES), ARM_CHECKPOINT, *sorted(NAT_DIR.glob("*.py")),
             *sorted(NAT_DIR.glob("*.json")), *sorted((NAT_DIR / "models").glob("*.npz"))]
    return {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def code_digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def stamp(row, digest):
    """Stamp a finished row made by this batch's files; mark one made by other files (edited mid-batch)."""
    if "error" in row:
        return row
    if row.get("code_digest") == digest:
        row["provenance_sha256"] = digest
    else:
        row["provenance"] = "mismatch"
    return row


def provenance():
    """Taken before any episode starts: the commit, the uncommitted changes and every file that decides a result."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout
    return {"commit": git("rev-parse", "HEAD").strip(),
            "uncommitted_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "untracked": git("ls-files", "-o", "--exclude-standard").split(),
            "sha256": file_hashes()}


def finished_rows(path, digest, keep_unstamped=False):
    """Saved rows that can be kept when a run continues: no process error, made by the same code and files
    (provenance_sha256 == digest), one per bearing. Rows from before stamping are kept, marked unverified, only
    with keep_unstamped. A torn last line from an interrupted write is skipped. Returns (rows, dropped)."""
    if not path.exists():
        return [], 0
    rows, dropped = {}, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        saved = row.get("provenance_sha256")
        # Unstamped rows from before stamping may be kept; rows known to come from other files never are.
        from_before = saved is None and "code_digest" not in row and keep_unstamped
        if "error" in row or (saved != digest and not from_before):
            dropped += 1
            continue
        if saved is None:
            row["provenance"] = "unverified"
        rows[row["bearing"]] = row                  # the last row per bearing wins
    return list(rows.values()), dropped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help=" ".join(RUNS))     # checked below: choices rejects an empty list
    parser.add_argument("--one", nargs=2, metavar=("RUN", "BEARING"), help="internal: one episode in this process")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/v2")
    parser.add_argument("--fresh", action="store_true", help="discard saved rows instead of continuing them")
    parser.add_argument("--keep-unstamped", action="store_true",
                        help="also keep rows saved before provenance stamping (marked unverified)")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    os.chdir(ROOT)
    if args.one:
        run, bearing = args.one
        if run not in RUNS or int(bearing) not in bearings():
            raise SystemExit(f"Unknown run or bearing: {args.one}")
        return run_one(run, int(bearing))
    if not args.runs or not set(args.runs) <= set(RUNS):
        raise SystemExit("Name one or more runs from: " + " ".join(RUNS))
    args.runs = list(dict.fromkeys(args.runs))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    info = provenance()
    digest = code_digest(info["sha256"])
    rows, files, dropped = {}, {}, 0
    for run in args.runs:
        path = args.out_dir / f"{run}.jsonl"
        rows[run], lost = ([], 0) if args.fresh else finished_rows(path, digest, args.keep_unstamped)
        dropped += lost
        (args.out_dir / f"{run}.summary.json").unlink(missing_ok=True)      # never leave a stale summary
        tmp = path.with_suffix(".tmp.jsonl")                                # kept rows first, then swap in
        tmp.write_text("".join(json.dumps(row) + "\n" for row in rows[run]), encoding="utf-8")
        tmp.replace(path)
        files[run] = path.open("a", encoding="utf-8")
    done = {(run, r["bearing"]) for run in args.runs for r in rows[run]}
    jobs = [((run, b), [__file__, "--one", run, str(b)]) for run in args.runs for b in sorted(bearings())
            if (run, b) not in done]
    print(f"{len(done)} episodes kept, {dropped} saved rows dropped, {len(jobs)} to run", flush=True)
    started = time.perf_counter()

    def on_result(key, text, error):
        run, bearing = key
        row = json.loads(text) if text else {"run": run, "bearing": bearing, "split": bearings()[bearing],
                                             "error": error, "reached": False, "parked": False}
        rows[run].append(stamp(row, digest))
        files[run].write(json.dumps(row) + "\n")
        files[run].flush()
        status = "ERROR" if error else ("OK  " if row["reached"] else "MISS")
        detail = error[-200:] if error else f"final {row['final_error_cm']:.2f} cm, parked {row['parked']}"
        print(f"{run:3s} {bearing:3d} {status} {detail}", flush=True)

    try:
        failed = run_pool(jobs, args.workers, "JSON ", on_result)
    finally:
        for fh in files.values():
            fh.close()
    for run in args.runs:
        summary = {"run": run, "driving": RUNS[run][0], "arm_inputs": RUNS[run][1], "arm": RUNS[run][2],
                   **summarize(rows[run]), "batch_wall_s": time.perf_counter() - started,
                   "unverified_rows": sum(r.get("provenance") == "unverified" for r in rows[run]),
                   "mismatched_rows": sum(r.get("provenance") == "mismatch" for r in rows[run]),
                   "provenance_sha256": digest, **info}
        tmp = args.out_dir / f"{run}.summary.tmp.json"
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(args.out_dir / f"{run}.summary.json")
        print("SUMMARY", run, json.dumps({k: summary[k] for k in ("train", "test", "all")}), flush=True)
    mismatched = sum(r.get("provenance") == "mismatch" for run in args.runs for r in rows[run])
    if mismatched:
        print(f"WARNING: {mismatched} rows were made by files edited during this batch", flush=True)
    if failed or mismatched:
        sys.exit(1)


if __name__ == "__main__":
    main()
