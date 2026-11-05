"""One subprocess per golden case, with durable receipts and no automatic reruns.

Run S, freeze protocol.json from its validation summary, then run P and G.
The G ledger lives in outputs/golden: an attempted case is never run again.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import math
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
import tempfile
import time
import traceback

ENVIRONMENT = {"KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONIOENCODING": "utf-8",
               "ROBOENV_HEADLESS": "1", "CAMERA_RENDERER": "tiny",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
os.environ.update(ENVIRONMENT)

from pool import run_pool

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs/golden"
NAT_DIR = ROOT / "nat_golden"
ARM_CHECKPOINT = ROOT / "outputs/v2_arm/arm_no_world_inputs.pt"
DRIVING_CHECKPOINT = NAT_DIR / "models/driving_seed7.npz"
VERSION = "golden-eval-v1"
SOURCES = ("golden_eval.py", "golden_sensor.py", "v2_sensor.py", "v2_eval.py",
           "sim.py", "memory_policy.py", "fused_policy.py",
           "pool.py", "features.py", "robotnik_arm_control.py",
           "configs/robotnik_arm.json", "pixi.lock")
STATUSES = ("success", "stop_outside_10mm", "no_stop_timeout", "no_park_timeout", "process_error")


def utc_now() -> str:
    """A timestamp independent of the host's local timezone."""
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    """Hash a file without decoding or inspecting dataset values."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_new_json(path: Path, value: dict) -> None:
    """Publish a complete, fsynced JSON file without replacing an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, name = tempfile.mkstemp(prefix=".golden-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    """Read a JSON object, rejecting malformed input."""
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def select_ids(cases: dict, run: str) -> list[int]:
    """The smallest 12 numeric IDs for S/P, all 100 for G."""
    if run not in ("S", "P", "G"):
        raise ValueError(f"Unknown run: {run}")
    identifiers = sorted(int(key) for key in cases)
    if len(identifiers) != 100 or len(set(identifiers)) != 100:
        raise ValueError("Expected exactly 100 distinct case IDs")
    return identifiers if run == "G" else identifiers[:12]


def cases_for(run: str) -> dict:
    """Load the fixed split used by a run; V denotes a validation replicate."""
    split = "val" if run == "S" or run.startswith("V") else "test"
    return read_json(NAT_DIR / f"expert_{split}.json")


def summarize(rows: list[dict]) -> dict:
    """Counts and medians over observed results; missing values remain null."""
    good = [row for row in rows if "error" not in row]
    counts = Counter(row.get("status", "process_error") for row in rows)
    result = {"episodes": len(rows), "success_10mm": sum(bool(r.get("success")) for r in good),
              "parked": sum(bool(r.get("parked")) for r in good),
              "stopped": sum(r.get("stop_s") is not None for r in good),
              "process_errors": len(rows) - len(good),
              "status_counts": {status: counts[status] for status in STATUSES}}
    for field in ("at_stop_mm", "closest_mm", "park_error_cm"):
        values = [float(r[field]) for r in good if r.get(field) is not None]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite metric: {field}")
        result[f"median_{field}"] = median(values) if values else None
    return result


def file_hashes() -> dict[str, str]:
    """Behavioral sources, frozen models/data, and all robot rendering assets."""
    files = {ROOT / name for name in SOURCES}
    files.add(ARM_CHECKPOINT)
    files.update(path for path in NAT_DIR.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.update(path for path in (ROOT / "nat_v2").glob("*.py"))
    asset_dir = ROOT / "models/summit_xl_arm_description"
    files.update(path for path in asset_dir.rglob("*") if path.is_file())
    hashes = {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(files)}
    package = util.find_spec("simulation_and_control")
    if package is None or package.origin is None:
        raise RuntimeError("simulation_and_control cannot be located for provenance")
    package_dir = Path(package.origin).parent
    hashes.update({"runtime/simulation_and_control/" + path.relative_to(package_dir).as_posix(): sha256(path)
                   for path in sorted(package_dir.rglob("*.py"))})
    return hashes


def code_digest(hashes: dict[str, str]) -> str:
    """Stable identity of the files consumed by the experiment."""
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def provenance() -> dict:
    """Capture files and runtime without changing Git configuration or branches."""
    def git(*arguments: str) -> str:
        result = subprocess.run(["git", f"--work-tree={ROOT}", *arguments], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", check=True)
        return result.stdout.strip()

    versions = {}
    for package in ("numpy", "torch", "pybullet", "pin", "simulation-and-control"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"version": VERSION, "created_utc": utc_now(), "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "uncommitted_diff_sha256": hashlib.sha256(git("diff", "HEAD").encode()).hexdigest(),
            "untracked": git("ls-files", "-o", "--exclude-standard").splitlines(),
            "sha256": file_hashes(), "python": sys.version, "executable": Path(sys.executable).name,
            "platform": platform.platform(), "packages": versions, "environment": ENVIRONMENT}


def validate_protocol(protocol: dict, threshold: float, hold_steps: int, validation_hash: str) -> None:
    """Only the prespecified 5/8 mm rule selected from S may enter P/G."""
    if (protocol.get("version") != 1 or protocol.get("selected_from") != "S"
            or not protocol.get("reason") or protocol.get("stop_distance_m") != threshold
            or threshold not in (0.005, 0.008) or hold_steps != 100
            or protocol.get("stop_hold_steps") != hold_steps
            or protocol.get("validation_summary_sha256") != validation_hash):
        raise ValueError("Frozen protocol is missing, differs from the code, or is not based on S")


def frozen_protocol(run: str, out_dir: Path) -> dict | None:
    """Require the completed validation evidence before touching the test split."""
    if run not in ("P", "G"):
        return None
    from golden_sensor import STOP_DISTANCE_M, STOP_HOLD_STEPS
    summary_path = out_dir / "S.summary.json"
    summary = read_json(summary_path)
    if summary.get("episodes") != 12 or not summary.get("complete") or summary.get("process_errors"):
        raise ValueError("S must finish all 12 validation cases without process errors")
    protocol = read_json(out_dir / "protocol.json")
    validate_protocol(protocol, STOP_DISTANCE_M, STOP_HOLD_STEPS, sha256(summary_path))
    if run == "G":
        parity = read_json(out_dir / "P.summary.json")
        if (parity.get("episodes") != 12 or not parity.get("complete") or parity.get("process_errors")
                or parity.get("protocol") != protocol):
            raise ValueError("P must finish all 12 driving-only cases using this frozen protocol before G")
    return protocol


def load_receipts(directory: Path, run: str, identifiers: list[int], digest: str) -> dict[int, dict]:
    """Recover durable child results, refusing mixed versions or changed sources."""
    rows = {}
    for path in sorted(directory.glob("*.json")):
        row = read_json(path)
        identifier = row.get("case_id")
        if (identifier not in identifiers or str(identifier) != path.stem or identifier in rows
                or row.get("run") != run or row.get("version") != VERSION
                or row.get("provenance_sha256") != digest or row.get("bearing") != identifier
                or row.get("status") not in STATUSES or not isinstance(row.get("success"), bool)
                or (row["status"] == "process_error") != ("error" in row)):
            raise ValueError(f"Incompatible or duplicate receipt: {path}")
        rows[identifier] = row
    return rows


def pending_ids(identifiers: list[int], rows: dict[int, dict], attempts: Path) -> list[int]:
    """Never rerun a started case whose final receipt is missing."""
    unresolved = [identifier for identifier in identifiers
                  if identifier not in rows and (attempts / f"{identifier}.json").exists()]
    if unresolved:
        raise RuntimeError(f"Attempted cases without receipts; automatic rerun forbidden: {unresolved}")
    return [identifier for identifier in identifiers if identifier not in rows]


def run_one(run: str, identifier: int, out_dir: Path, digest: str) -> None:
    """Internal child: claim this case once, persist its receipt, then print it."""
    manifest = read_json(out_dir / f"{run}.manifest.json")
    if manifest["provenance_sha256"] != digest or identifier not in manifest["case_ids"]:
        raise ValueError("Child arguments do not match the prepared manifest")
    if code_digest(file_hashes()) != digest:
        raise ValueError("Sources changed since the parent prepared this run")
    frozen_protocol(run, out_dir)
    case = cases_for(run)[str(identifier)]
    attempt_dir, receipt_dir = out_dir / "attempts" / run, out_dir / "receipts" / run
    write_new_json(attempt_dir / f"{identifier}.json",
                   {"run": run, "case_id": identifier, "started_utc": utc_now(), "digest": digest})
    row = {"run": run, "case_id": identifier, "bearing": identifier,
           "split": "val" if run == "S" or run.startswith("V") else "test", "parity": run == "P",
           "version": VERSION, "code_digest": digest, "provenance_sha256": digest}
    try:
        from golden_sensor import GoldenPolicy, run_golden_episode
        policy = GoldenPolicy.build_golden(ARM_CHECKPOINT, DRIVING_CHECKPOINT, use_arm=run != "P")
        row.update(run_golden_episode(case, policy, parity=run == "P"))
        row["stopped"] = row.get("stop_s") is not None
        if code_digest(file_hashes()) != digest:
            raise RuntimeError("Experiment files changed while this episode was running")
    except Exception:
        row.update(error=traceback.format_exc(), status="process_error", success=False)
    row["completed_utc"] = utc_now()
    write_new_json(receipt_dir / f"{identifier}.json", row)
    print("JSON " + json.dumps(row, allow_nan=False), flush=True)


def write_outputs(out_dir: Path, run: str, rows: dict[int, dict], manifest: dict,
                  elapsed: float) -> dict:
    """Rebuild derived reports from immutable receipts after each batch."""
    ordered = [rows[key] for key in sorted(rows)]
    jsonl = out_dir / f"{run}.jsonl"
    temporary = out_dir / f"{run}.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in ordered), encoding="utf-8")
    temporary.replace(jsonl)
    summary = {**manifest, **summarize(ordered), "complete": len(rows) == len(manifest["case_ids"]),
               "batch_wall_s": elapsed, "updated_utc": utc_now(), "results_sha256": sha256(jsonl)}
    temporary = out_dir / f"{run}.summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(out_dir / f"{run}.summary.json")
    return summary


def reports_match(out_dir: Path, run: str, rows: dict[int, dict], manifest: dict) -> bool:
    """Check saved derived reports without rewriting a frozen validation summary."""
    result_path, summary_path = out_dir / f"{run}.jsonl", out_dir / f"{run}.summary.json"
    if not result_path.exists() or not summary_path.exists():
        return False
    saved_rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    if saved_rows != [rows[key] for key in sorted(rows)]:
        return False
    summary = read_json(summary_path)
    expected = {**manifest, **summarize(saved_rows), "complete": len(rows) == len(manifest["case_ids"]),
                "results_sha256": sha256(result_path)}
    return all(summary.get(key) == value for key, value in expected.items())


def run_batch(run: str, out_dir: Path, workers: int, validation_case: int | None = None) -> None:
    """Resume unstarted cases only; an existing completed run performs no simulations."""
    protocol = frozen_protocol(run, out_dir)
    cases = cases_for(run)
    identifiers = select_ids(cases, run) if validation_case is None else [validation_case]
    if any(str(identifier) not in cases for identifier in identifiers):
        raise ValueError("Requested validation case is absent from expert_val.json")
    info = provenance()
    digest = code_digest(info["sha256"])
    if run == "G" and read_json(out_dir / "P.summary.json").get("provenance_sha256") != digest:
        raise ValueError("P used different experiment files; refusing to start G")
    manifest = {**info, "run": run, "case_ids": identifiers, "provenance_sha256": digest,
                "workers": workers, "protocol": protocol, "split": "val" if run.startswith(("S", "V")) else "test"}
    manifest_path = out_dir / f"{run}.manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if (manifest.get("provenance_sha256") != digest or manifest.get("case_ids") != identifiers
                or manifest.get("version") != VERSION or manifest.get("protocol") != protocol):
            raise ValueError("Existing run has different sources, cases or protocol; refusing to replace it")
    else:
        write_new_json(manifest_path, manifest)
    rows = load_receipts(out_dir / "receipts" / run, run, identifiers, digest)
    pending = pending_ids(identifiers, rows, out_dir / "attempts" / run)
    print(f"{run}: {len(rows)} receipts retained, {len(pending)} cases to run", flush=True)
    if not pending and reports_match(out_dir, run, rows, manifest):
        summary = summarize(list(rows.values()))
        print("SUMMARY " + json.dumps(summary), flush=True)
        if summary["process_errors"]:
            raise RuntimeError("Saved run contains process errors; automatic rerun forbidden")
        return
    if run == "S" and (out_dir / "protocol.json").exists():
        raise RuntimeError("Frozen S reports cannot be changed; inspect the existing evidence")
    jobs = [(identifier, [__file__, "--one", run, str(identifier), "--out-dir", str(out_dir),
                          "--digest", digest]) for identifier in pending]
    started = time.perf_counter()

    def on_result(identifier: int, text: str | None, error: str | None) -> None:
        path = out_dir / "receipts" / run / f"{identifier}.json"
        if path.exists():
            rows.update(load_receipts(path.parent, run, identifiers, digest))
            row = rows[identifier]
            print(f"{run} {identifier}: {row['status']}", flush=True)
        else:
            print(f"{run} {identifier}: PROCESS ERROR {error or 'missing durable receipt'}", flush=True)
        write_outputs(out_dir, run, rows, manifest, time.perf_counter() - started)

    failed = run_pool(jobs, workers, "JSON ", on_result) if jobs else 0
    summary = write_outputs(out_dir, run, rows, manifest, time.perf_counter() - started)
    print("SUMMARY " + json.dumps(summarize(list(rows.values()))), flush=True)
    if failed or summary["process_errors"] or not summary["complete"]:
        raise RuntimeError("Run incomplete or contains process errors; no cases will be silently rerun")


def main() -> None:
    """CLI entry; test runs use a fixed output directory to preserve their ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", choices=("S", "P", "G"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--validation-case", type=int)
    parser.add_argument("--one", nargs=2, metavar=("RUN", "CASE"), help=argparse.SUPPRESS)
    parser.add_argument("--digest", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("workers must be between 1 and 12")
    out_dir = args.out_dir.resolve()
    os.chdir(ROOT)
    if args.one:
        run, identifier = args.one
        if not args.digest:
            parser.error("--one requires a prepared batch digest")
        if run in ("P", "G") and out_dir != OUT_DIR.resolve():
            parser.error("P/G must use outputs/golden to preserve the once-only ledger")
        run_one(run, int(identifier), out_dir, args.digest)
        return
    if (args.run is None) == (args.validation_case is None):
        parser.error("Choose one of S/P/G or --validation-case ID")
    run = args.run or f"V{args.validation_case}"
    if run in ("P", "G") and out_dir != OUT_DIR.resolve():
        parser.error("P/G must use outputs/golden to preserve the once-only ledger")
    if not ARM_CHECKPOINT.exists():
        from v2_eval import ensure_arm
        ensure_arm()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_batch(run, out_dir, args.workers, args.validation_case)


if __name__ == "__main__":
    main()
