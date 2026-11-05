"""Join golden results with Nat's published per-point metrics, without simulation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
VERSION = "golden-compare-v1"
METRICS = ("park_s", "stop_s", "park_error_cm", "at_stop_mm", "closest_mm")
NAT_FIELDS = ("dot", "success", "parked", *METRICS)


def sha256(path: Path) -> str:
    """Return the byte identity of one comparison input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_bool(value: str) -> bool:
    """Parse CSV booleans explicitly so the string False never counts as true."""
    normalized = value.strip().lower()
    if normalized not in ("true", "false", "1", "0"):
        raise ValueError(f"Invalid CSV boolean: {value!r}")
    return normalized in ("true", "1")


def number(value: str | float | None) -> float | None:
    """Convert a missing metric to null and reject nonfinite measured values."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "none", "null", "nan")):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite metric: {value!r}")
    return result


def keyed(rows: list[dict], key: str) -> dict[int, dict]:
    """Index one row per integer ID, refusing duplicates."""
    result = {}
    for row in rows:
        identifier = int(row[key])
        if identifier in result:
            raise ValueError(f"Duplicate point ID: {identifier}")
        result[identifier] = row
    return result


def measured_median(rows: list[dict], field: str, absolute: bool = False) -> float | None:
    """Median of defined observations, with an explicit null for no observations."""
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if absolute:
        values = [abs(value) for value in values]
    return median(values) if values else None


def compare(golden: list[dict], native: list[dict]) -> tuple[list[dict], dict]:
    """Pair identical IDs and compute success and parking comparisons."""
    ours, theirs = keyed(golden, "case_id"), keyed(native, "dot")
    if not ours or set(ours) != set(theirs):
        raise ValueError("Comparison requires nonempty, identical point-ID sets")
    rows = []
    table = dict(both_success=0, two_rate_only=0, nat_only=0, neither_success=0)
    for identifier in sorted(ours):
        left, right = ours[identifier], theirs[identifier]
        if "error" in left:
            raise ValueError(f"Point {identifier} has a process error, not a measured result")
        for key in ("parked", "success"):
            if not isinstance(left[key], bool):
                raise ValueError(f"Point {identifier}: {key} must be a JSON boolean")
        row = {"case_id": identifier, "two_rate_status": left.get("status"),
               "nat_status": right.get("status")}
        for key in ("parked", "success"):
            row[f"two_rate_{key}"], row[f"nat_{key}"] = left[key], csv_bool(right[key])
        for field in METRICS:
            a, b = number(left.get(field)), number(right.get(field))
            row[f"two_rate_{field}"], row[f"nat_{field}"] = a, b
            row[f"delta_{field}"] = None if a is None or b is None else a - b
        pair = (row["two_rate_success"], row["nat_success"])
        table[{(True, True): "both_success", (True, False): "two_rate_only",
               (False, True): "nat_only", (False, False): "neither_success"}[pair]] += 1
        rows.append(row)
    both = [row for row in rows if row["two_rate_parked"] and row["nat_parked"]]
    paired = {"points": len(both)}
    for side in ("two_rate", "nat"):
        count = sum(row[f"{side}_success"] for row in both)
        paired[f"{side}_successes"] = count
        paired[f"{side}_success_rate"] = count / len(both) if both else None
    summary = {"points": len(rows), "success_2x2": table, "both_parked": paired,
               "park_agreement": sum(row["two_rate_parked"] == row["nat_parked"] for row in rows),
               "median_abs_park_time_difference_s_both_parked":
                   measured_median(both, "delta_park_s", absolute=True),
               "max_abs_park_time_difference_s_both_parked":
                   max((abs(row["delta_park_s"]) for row in both if row["delta_park_s"] is not None),
                       default=None)}
    for side in ("two_rate", "nat"):
        summary[side] = {
            "success_10mm": sum(row[f"{side}_success"] for row in rows),
            "parked": sum(row[f"{side}_parked"] for row in rows),
            "stopped": sum(row[f"{side}_stop_s"] is not None for row in rows),
            **{f"median_{field}": measured_median(
                [r for r in rows if r[f"{side}_parked"]] if field == "park_error_cm" else rows,
                f"{side}_{field}")
               for field in ("at_stop_mm", "closest_mm", "park_error_cm")},
        }
    return rows, summary


def compare_parking(golden: list[dict], native: list[dict]) -> tuple[list[dict], dict]:
    """P is a driving check: never read or report either arm's outcomes."""
    ours, theirs = keyed(golden, "case_id"), keyed(native, "dot")
    if not ours or set(ours) != set(theirs):
        raise ValueError("Parking comparison requires identical nonempty point-ID sets")
    rows = []
    for identifier in sorted(ours):
        left, right = ours[identifier], theirs[identifier]
        if "error" in left or not isinstance(left["parked"], bool):
            raise ValueError(f"Invalid parking result for point {identifier}")
        row = {"case_id": identifier, "two_rate_parked": left["parked"],
               "nat_parked": csv_bool(right["parked"])}
        for field in ("park_s", "park_error_cm"):
            a, b = number(left.get(field)), number(right.get(field))
            row[f"two_rate_{field}"], row[f"nat_{field}"] = a, b
            row[f"delta_{field}"] = None if a is None or b is None else a - b
        rows.append(row)
    both = [row for row in rows if row["two_rate_parked"] and row["nat_parked"]]
    summary = {"points": len(rows), "purpose": "driving parity only; arm held",
               "park_agreement": sum(row["two_rate_parked"] == row["nat_parked"] for row in rows),
               "both_parked_points": len(both),
               "median_abs_park_time_difference_s_both_parked":
                   measured_median(both, "delta_park_s", absolute=True),
               "max_abs_park_time_difference_s_both_parked":
                   max((abs(row["delta_park_s"]) for row in both if row["delta_park_s"] is not None),
                       default=None)}
    for side in ("two_rate", "nat"):
        parked = [row for row in rows if row[f"{side}_parked"]]
        summary[side] = {"parked": len(parked),
                         "median_park_error_cm": measured_median(parked, f"{side}_park_error_cm")}
    return rows, summary


def reference_scores(rows: list[dict]) -> dict:
    """Read test results for seed 7 at iterations 43 and 57."""
    result = {}
    for iteration in (43, 57):
        selected = [row for row in rows if int(row["iteration"]) == iteration
                    and row["split"] == "test" and int(row["seed"]) == 7]
        if len(selected) != 1:
            raise ValueError(f"Expected one iteration {iteration} test/seed7 reference")
        row = selected[0]
        result[f"iteration_{iteration}"] = {
            "name": row["name"], "iteration": iteration, "split": "test", "seed": 7,
            **{key: int(row[key]) for key in ("attempts", "parked", "stopped", "success_10mm")},
        }
    return result


def read_csv(path: Path) -> list[dict]:
    """Read headers and values without lossy numeric type inference."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def result_provenance(record: dict) -> str:
    """Identify original execution evidence separately from publication artifacts."""
    value = record.get("source_provenance_sha256", record.get("provenance_sha256"))
    if not isinstance(value, str) or not value:
        raise ValueError("Missing execution provenance")
    return value


def load_completed(out_dir: Path, run: str) -> tuple[list[dict], dict]:
    """Reject partial, changed, or malformed batch results before comparison."""
    summary = json.loads((out_dir / f"{run}.summary.json").read_text(encoding="utf-8"))
    path = out_dir / f"{run}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    expected = 100 if run == "G" else 12
    if (not summary.get("complete") or summary.get("process_errors") or len(rows) != expected
            or summary.get("results_sha256") != sha256(path)
            or len(keyed(rows, "case_id")) != expected
            or any(result_provenance(row) != result_provenance(summary) for row in rows)):
        raise ValueError(f"{run} results are incomplete or have incompatible provenance")
    return rows, summary


def main() -> None:
    """Write per-point CSV and aggregate JSON for G, or the driving-only P check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("P", "G"), default="G")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/golden")
    args = parser.parse_args()
    rows, batch = load_completed(args.out_dir, args.run)
    metrics_path = ROOT / "nat_golden/seed7_metrics.csv"
    native = read_csv(metrics_path)
    identifiers = {row["case_id"] for row in rows}
    if args.run == "G" and identifiers != set(keyed(native, "dot")):
        raise ValueError("G must cover all and only Nat's 100 test IDs")
    native = [row for row in native if int(row["dot"]) in identifiers]
    comparisons, summary = (compare_parking(rows, native) if args.run == "P" else compare(rows, native))
    sources = [Path(__file__), args.out_dir / f"{args.run}.jsonl",
               args.out_dir / f"{args.run}.summary.json", metrics_path]
    if args.run == "G":
        scores_path = ROOT / "nat_golden/publication_scores.csv"
        summary["publication_references"] = reference_scores(read_csv(scores_path))
        sources.append(scores_path)
    summary.update(run=args.run, version=VERSION, source_provenance_sha256=result_provenance(batch),
                   input_sha256={str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                                 sha256(path) for path in sources},
                   delta_definition="Two-rate arm minus Nat",
                   note="Iteration 57 uses a different driving network; reference only.")
    prefix = "" if args.run == "G" else "P_"
    with (args.out_dir / f"{prefix}compare.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    (args.out_dir / f"{prefix}compare_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
