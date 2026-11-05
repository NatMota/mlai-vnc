"""Review and publish one verified golden experiment; no existing run is changed.

Without --upload this script writes a local plan only. The explicit upload path
checks both Git remotes, snapshots existing runs, creates at most one new run,
and verifies all metrics, tags and artifact bytes before marking it finished.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from statistics import median
import subprocess
import tempfile
import time
from typing import Iterator
import unicodedata
import uuid

TRACKING_URI = "https://nm-linmint-mlflow.taile381c0.ts.net"
EXPERIMENT_ID = "3"
EXPERIMENT_NAME = "v2-sensors"
RUN_NAME = "golden-tworate-arm-iter43"
BRANCH = "experiments/v2-golden-arm"
DESCRIPTION = "Compared with Nat iteration 43, the median distance at STOP decreased from 169 mm to 17 mm."
STOP_RULE = "estimated distance <=5 mm for 100 consecutive 1 ms steps"
NAT_COMMIT = "d89051b4f971c54d59c81bf71f9f92cb4f8e2833"
METRIC_NAMES = ("success_10mm", "parked", "stopped", "median_at_stop_mm",
                "median_closest_mm", "median_park_error_cm")
TEXT_EXTENSIONS = {".py", ".md", ".json", ".jsonl", ".csv", ".txt", ".patch", ".sha256"}
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | {".npz", ".pt"}
TRANSACTION_TAG = "publication.transaction_id"
PLAN_TAG = "publication.plan_sha256"


def canonical(value: object) -> bytes:
    """A deterministic, standards-compliant representation for hashing."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("A publication state file is not a JSON object")
    return value


def atomic_json(path: Path, value: dict) -> None:
    """Persist publication state before each remote transition."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".publication-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def publication_lock(state_dir: Path) -> Iterator[None]:
    """An OS lock is released after a crash; simultaneous publishers cannot create duplicates."""
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "publication.lock").open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def reject_forbidden(text: str, tokens: list[str]) -> None:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if any(unicodedata.normalize("NFKC", token).casefold() in normalized for token in tokens):
        raise ValueError("A prohibited personal token remains in publication content")


def bundle_hashes(bundle: Path, tokens: list[str]) -> dict[str, str]:
    """Only regular, approved artifact types are accepted; no linked files escape the bundle."""
    files = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("Publication bundles cannot contain symbolic links")
        if not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        reject_forbidden(relative, tokens)
        if path.suffix.lower() not in ALLOWED_EXTENSIONS or path.name.startswith("."):
            raise ValueError("Unexpected or hidden artifact type in publication bundle")
        payload = path.read_bytes()
        for token in tokens:
            if any(token.encode(encoding).lower() in payload.lower() for encoding in ("utf-8", "utf-16-le")):
                raise ValueError("A prohibited personal token remains in artifact bytes")
        if path.suffix.lower() in TEXT_EXTENSIONS:
            text = payload.decode("utf-8-sig")
            reject_forbidden(text, tokens)
            if any("\u3400" <= character <= "\u9fff" for character in text):
                raise ValueError("A text artifact contains non-English CJK content")
        files[relative] = hashlib.sha256(payload).hexdigest()
    if not files:
        raise ValueError("Publication bundle is empty")
    if "publication_manifest.json" in files:
        raise ValueError("publication_manifest.json is reserved for the generated upload manifest")
    return files


def metrics_from_rows(rows: list[dict], expected: int) -> dict[str, float]:
    """Recompute the reported metrics without trusting precomputed summaries."""
    identifiers = [row.get("case_id") for row in rows]
    if len(rows) != expected or any(not isinstance(item, int) for item in identifiers):
        raise ValueError("Wrong episode count or missing case IDs")
    if len(set(identifiers)) != expected or any("error" in row for row in rows):
        raise ValueError("Duplicate cases or failed simulation processes")
    if any(not isinstance(row.get("success"), bool) or not isinstance(row.get("parked"), bool) for row in rows):
        raise ValueError("Episode success and parked fields must be booleans")
    for row in rows:
        if "stop_s" not in row or "at_stop_mm" not in row:
            raise ValueError("Required STOP evidence is missing")
        stopped = row["stop_s"] is not None
        if stopped != (row["at_stop_mm"] is not None):
            raise ValueError("STOP timestamp and distance disagree")
        success = row["parked"] and stopped and float(row["at_stop_mm"]) <= 10.0
        if row["success"] != success:
            raise ValueError("Episode success disagrees with the fixed 10 mm STOP criterion")
    result = {"success_10mm": float(sum(row["success"] for row in rows)),
              "parked": float(sum(row["parked"] for row in rows)),
              "stopped": float(sum(row.get("stop_s") is not None for row in rows))}
    for name in ("at_stop_mm", "closest_mm", "park_error_cm"):
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        if not values or not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Missing or invalid metric values in episode rows")
        result["median_" + name] = float(median(values))
    return result


def validated_metrics(bundle: Path) -> dict[str, float]:
    metrics = {}
    for group, count, prefix in (("G", 100, ""), ("S", 12, "val12_")):
        folder = bundle / "outputs/golden"
        rows_path = folder / f"{group}.jsonl"
        rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
        if any(row.get("run") != group or row.get("split") != ("test" if group == "G" else "val") for row in rows):
            raise ValueError("Episode rows do not belong to the required run and split")
        computed = metrics_from_rows(rows, count)
        summary = read_json(folder / f"{group}.summary.json")
        if summary.get("episodes") != count or not summary.get("complete") or summary.get("process_errors"):
            raise ValueError("Summary is incomplete or contains process errors")
        if summary.get("results_sha256") != file_hash(rows_path):
            raise ValueError("Summary does not identify the supplied result file")
        for name, value in computed.items():
            if not math.isclose(float(summary[name]), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Summary metrics disagree with recomputed episode metrics")
            metrics[prefix + name] = value
    return metrics


def git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(["git", "-C", str(repo), f"--work-tree={repo}", *arguments], capture_output=True,
                            text=True, encoding="utf-8", check=False)
    if result.returncode:
        raise RuntimeError("A read-only Git verification command failed")
    return result.stdout.strip()


def repository_info(repo: Path) -> dict:
    branch, commit = git(repo, "branch", "--show-current"), git(repo, "rev-parse", "HEAD")
    if branch != BRANCH or len(commit) != 40:
        raise ValueError("The publication checkout is not on the required branch")
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked repository files contain unpublished changes")
    return {"git_commit": commit, "git_branch": branch, "git_tree": git(repo, "rev-parse", "HEAD^{tree}")}


def git_blob(repo: Path, relative: str) -> bytes:
    """Read exact committed bytes without newline or text decoding conversions."""
    result = subprocess.run(["git", "-C", str(repo), f"--work-tree={repo}", "show", "HEAD:" + relative],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("An uploaded code or report file is absent from the recorded commit")
    return result.stdout


def verify_committed_artifacts(bundle: Path, repo: Path, files: dict[str, str]) -> dict[str, str]:
    """Uploaded code/ files and docs/ reports must match their committed Git blobs."""
    mapping = {}
    for name, expected in files.items():
        relative = name.removeprefix("code/") if name.startswith("code/") else name
        if not name.startswith(("code/", "docs/")):
            continue
        content = git_blob(repo, relative)
        if hashlib.sha256(content).hexdigest() != expected or (bundle / name).read_bytes() != content:
            raise ValueError("Uploaded code or report bytes do not match the recorded Git commit")
        mapping[name] = relative
    if not any(name.startswith("code/") for name in mapping):
        raise ValueError("Publication bundle contains no committed code artifacts")
    return mapping


def verify_remote_heads(repo: Path, commit: str) -> None:
    for remote in ("origin", "upstream"):
        rows = git(repo, "ls-remote", "--heads", remote, "refs/heads/" + BRANCH).splitlines()
        if len(rows) != 1 or rows[0].split() != [commit, "refs/heads/" + BRANCH]:
            raise ValueError("Both remote publication branches must match the recorded commit")


def verify_local_files(plan: dict, bundle: Path) -> None:
    """Catch a changed bundle before creating a new run, then recheck each upload."""
    for name, expected in plan["files"].items():
        path = bundle / name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(bundle.resolve()):
            raise ValueError("A reviewed artifact is missing or became a link")
        if file_hash(path) != expected:
            raise ValueError("Local artifact changed after the plan was reviewed")


def build_plan(bundle: Path, repo: Path, tokens: list[str]) -> dict:
    """Build a reviewable local plan; no network writes happen here."""
    files = bundle_hashes(bundle, tokens)
    information = repository_info(repo)
    committed = verify_committed_artifacts(bundle, repo, files)
    metrics = validated_metrics(bundle)
    comparison = read_json(bundle / "outputs/golden/compare_summary.json")
    reference = comparison["nat"]
    later = comparison["publication_references"]["iteration_57"]
    if round(float(reference["median_at_stop_mm"])) != 169 or round(metrics["median_at_stop_mm"]) != 17:
        raise ValueError("The requested description does not match the supplied STOP medians")
    protocol = read_json(bundle / "outputs/golden/protocol.json")
    if protocol.get("stop_distance_m") != 0.005 or protocol.get("stop_hold_steps") != 100:
        raise ValueError("The bundle does not contain the specified frozen STOP rule")
    comparison_note = (f"STOP medians use the stopped subsets: {reference['stopped']} runs for Nat iteration 43 "
                       f"and {metrics['stopped']:g} for this arm. Success is {metrics['success_10mm']:g}/100 "
                       f"versus {reference['success_10mm']}/100. Nat iteration 57 reached "
                       f"{later['success_10mm']}/{later['attempts']} with a different driving network.")
    tags = {"mlflow.runName": RUN_NAME, "mlflow.user": "experiment-contributor",
            "mlflow.note.content": DESCRIPTION + "\n\n" + comparison_note, "driving": "Nat iteration 43 basics_standard",
            "arm": "Exported two-rate arm, source run 540bdcf9", "nat_source_commit": NAT_COMMIT,
            "git_commit": information["git_commit"], "git_branch": BRANCH, "stop_rule": STOP_RULE}
    plan = {"schema": 1, "tracking_uri": TRACKING_URI, "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME, "run_name": RUN_NAME, **information,
            "metrics": metrics, "tags": tags, "files": files,
            "committed_artifacts": committed,
            "code_files": {name: value for name, value in files.items() if name.endswith(".py")},
            "comparison": {"nat_iteration43_median_at_stop_mm": reference["median_at_stop_mm"],
                           "arm_median_at_stop_mm": metrics["median_at_stop_mm"],
                           "nat_iteration43_stopped": reference["stopped"], "arm_stopped": metrics["stopped"]},
            "comparison_note": comparison_note}
    reject_forbidden(canonical(plan).decode(), tokens)
    return plan


def make_client():
    """Use only the explicitly configured service, with no automatic POST retries."""
    os.environ.update(MLFLOW_DISABLE_AGENT_HINT="1", MLFLOW_HTTP_REQUEST_MAX_RETRIES="0",
                      MLFLOW_HTTP_REQUEST_TIMEOUT="60")
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.MlflowClient(tracking_uri=TRACKING_URI)


def verify_experiment(client) -> None:
    experiment = client.get_experiment(EXPERIMENT_ID)
    if (experiment is None or experiment.name != EXPERIMENT_NAME
            or experiment.lifecycle_stage != "active"):
        raise ValueError("The expected existing experiment is unavailable")


def snapshot_runs(client) -> dict:
    """Read metadata of every existing run, including deleted runs, with pagination."""
    runs, token = {}, None
    while True:
        page = client.search_runs([EXPERIMENT_ID], run_view_type=3, max_results=1000, page_token=token)
        for run in page:
            runs[run.info.run_id] = run.to_dictionary()
        token = getattr(page, "token", None)
        if not token:
            return runs


def assert_existing_unchanged(before: dict, after: dict) -> None:
    if any(identifier not in after or metadata != after[identifier] for identifier, metadata in before.items()):
        raise RuntimeError("Existing run metadata changed; verification cannot claim it was preserved")


def owned_run(client, identifier: str, transaction: dict, before: dict):
    """Every mutation is limited to the new run identified by this transaction."""
    if identifier in before:
        raise RuntimeError("Refusing to modify a pre-existing run")
    run = client.get_run(identifier)
    if (str(run.info.experiment_id) != EXPERIMENT_ID
            or run.data.tags.get(TRANSACTION_TAG) != transaction["transaction_id"]
            or run.data.tags.get(PLAN_TAG) != transaction["plan_sha256"]):
        raise RuntimeError("Run ownership does not match this publication transaction")
    return run


def obtain_run(client, plan: dict, state_dir: Path, before: dict, transaction: dict) -> tuple[str, dict]:
    """Recover a potentially committed create request before considering a new one."""
    state_path = state_dir / "transaction.json"
    matches = [run for run in snapshot_runs(client).values()
               if run["data"]["tags"].get(TRANSACTION_TAG) == transaction["transaction_id"]]
    if len(matches) > 1:
        raise RuntimeError("Duplicate transaction runs detected; no run will be modified")
    found = matches[0]["info"]["run_id"] if matches else None
    identifier = transaction.get("run_id") or found
    if identifier:
        if found is not None and identifier != found:
            raise RuntimeError("Receipt and server disagree about the new run")
        owned_run(client, identifier, transaction, before)
    else:
        if transaction["phase"] != "prepared":
            raise RuntimeError("Run creation outcome is uncertain; refusing a duplicate creation request")
        transaction = {**transaction, "phase": "creation_started"}
        atomic_json(state_path, transaction)
        tags = {**plan["tags"], TRANSACTION_TAG: transaction["transaction_id"],
                PLAN_TAG: transaction["plan_sha256"]}
        created = client.create_run(EXPERIMENT_ID, tags=tags, run_name=RUN_NAME)
        identifier = created.info.run_id
    transaction = {**transaction, "run_id": identifier, "phase": "uploading"}
    atomic_json(state_path, transaction)
    return identifier, transaction


def remote_files(client, identifier: str, directory: str | None = None) -> list[str]:
    files = []
    for item in client.list_artifacts(identifier, directory):
        path = PurePosixPath(item.path)
        if path.is_absolute() or ".." in path.parts or "\\" in item.path:
            raise ValueError("Unsafe remote artifact path")
        if item.is_dir:
            files.extend(remote_files(client, identifier, item.path))
        else:
            files.append(item.path)
    return files


def remote_hash(client, identifier: str, artifact: str, directory: Path) -> str:
    local = Path(client.download_artifacts(identifier, artifact, str(directory))).resolve()
    if not local.is_relative_to(directory.resolve()) or not local.is_file():
        raise ValueError("Downloaded artifact escaped the verification directory")
    return file_hash(local)


def upload_metrics(client, identifier: str, plan: dict, transaction: dict, before: dict) -> None:
    current = owned_run(client, identifier, transaction, before)
    if set(current.data.metrics) - set(plan["metrics"]):
        raise ValueError("Unexpected metrics in the publication run")
    for name, value in plan["metrics"].items():
        existing = current.data.metrics.get(name)
        if existing is not None:
            if not math.isclose(existing, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Existing new-run metric differs from the publication plan")
            continue
        owned_run(client, identifier, transaction, before)
        client.log_metric(identifier, name, value, timestamp=int(time.time() * 1000), step=0)


def upload_files(client, identifier: str, plan: dict, bundle: Path, state_dir: Path,
                 transaction: dict, before: dict) -> dict[str, str]:
    manifest_path = state_dir / "publication_manifest.json"
    atomic_json(manifest_path, {"plan_sha256": digest(plan), **plan})
    expected = {**plan["files"], "publication_manifest.json": file_hash(manifest_path)}
    present = set(remote_files(client, identifier))
    if present - set(expected):
        raise ValueError("Unexpected artifacts in the new publication run")
    with tempfile.TemporaryDirectory(prefix="artifact-verify-", dir=state_dir) as folder:
        for name, wanted in expected.items():
            if name in present:
                if remote_hash(client, identifier, name, Path(folder)) != wanted:
                    raise ValueError("Existing new-run artifact differs from the publication plan")
                continue
            local = manifest_path if name == "publication_manifest.json" else bundle / name
            if file_hash(local) != wanted:
                raise ValueError("Local artifact changed after the plan was reviewed")
            owned_run(client, identifier, transaction, before)
            parent = str(PurePosixPath(name).parent)
            client.log_artifact(identifier, str(local), artifact_path=None if parent == "." else parent)
    return expected


def verify_new_run(client, identifier: str, plan: dict, expected: dict[str, str], state_dir: Path,
                   transaction: dict, before: dict) -> dict:
    run = owned_run(client, identifier, transaction, before)
    required_tags = {**plan["tags"], TRANSACTION_TAG: transaction["transaction_id"], PLAN_TAG: digest(plan)}
    if run.data.tags != required_tags:
        raise ValueError("Read-back tags differ from the reviewed plan")
    if run.data.metrics != plan["metrics"]:
        raise ValueError("Read-back metrics differ from the reviewed plan")
    files = remote_files(client, identifier)
    if set(files) != set(expected):
        raise ValueError("Read-back artifact list differs from the reviewed plan")
    with tempfile.TemporaryDirectory(prefix="artifact-readback-", dir=state_dir) as folder:
        actual = {name: remote_hash(client, identifier, name, Path(folder)) for name in files}
    if actual != expected:
        raise ValueError("Read-back artifact hashes differ from the reviewed plan")
    return {"run_id": identifier, "metrics": run.data.metrics, "tags": run.data.tags,
            "artifact_sha256": actual, "verified_utc_ms": int(time.time() * 1000)}


def publish(client, plan: dict, bundle: Path, state_dir: Path, repo: Path) -> dict:
    """Create or safely resume this publication only; existing runs are read-only."""
    verify_local_files(plan, bundle)
    verify_remote_heads(repo, plan["git_commit"])
    verify_experiment(client)
    before_path, state_path = state_dir / "existing_runs.before.json", state_dir / "transaction.json"
    if state_path.exists():
        transaction = read_json(state_path)
        if transaction["plan_sha256"] != digest(plan):
            raise ValueError("This state directory belongs to another publication plan")
        before = read_json(before_path)
        if transaction["existing_snapshot_sha256"] != digest(before):
            raise ValueError("Existing-run snapshot no longer matches the saved receipt")
    else:
        before = snapshot_runs(client)
        if any(run["data"]["tags"].get("mlflow.runName") == RUN_NAME for run in before.values()):
            raise ValueError("The intended run name already exists; no duplicate will be created")
        atomic_json(before_path, before)
        transaction = {"transaction_id": uuid.uuid4().hex, "plan_sha256": digest(plan),
                       "existing_snapshot_sha256": digest(before), "phase": "prepared", "run_id": None}
        atomic_json(state_path, transaction)
    assert_existing_unchanged(before, snapshot_runs(client))
    identifier, transaction = obtain_run(client, plan, state_dir, before, transaction)
    upload_metrics(client, identifier, plan, transaction, before)
    expected = upload_files(client, identifier, plan, bundle, state_dir, transaction, before)
    verification = verify_new_run(client, identifier, plan, expected, state_dir, transaction, before)
    after = snapshot_runs(client)
    atomic_json(state_dir / "existing_runs.after.json", after)
    assert_existing_unchanged(before, after)
    owned_run(client, identifier, transaction, before)
    client.set_terminated(identifier, status="FINISHED")
    finished = owned_run(client, identifier, transaction, before)
    if finished.info.status != "FINISHED":
        raise RuntimeError("The newly created run did not enter FINISHED status")
    final_snapshot = snapshot_runs(client)
    atomic_json(state_dir / "existing_runs.after.json", final_snapshot)
    assert_existing_unchanged(before, final_snapshot)
    verification.update(status="FINISHED", existing_runs_unchanged=True,
                        existing_run_count=len(before), run_url=f"{TRACKING_URI}/#/experiments/{EXPERIMENT_ID}/runs/{identifier}")
    atomic_json(state_dir / "verification.json", verification)
    atomic_json(state_path, {**transaction, "phase": "complete"})
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--forbid-token", action="append", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    bundle, repo, state_dir = (value.resolve() for value in (args.bundle, args.repo, args.state_dir))
    if not bundle.is_dir() or not repo.is_dir() or state_dir.is_relative_to(bundle):
        parser.error("Bundle/repository must exist and state must be outside the uploaded bundle")
    if any(len(token) < 3 for token in args.forbid_token):
        parser.error("Prohibited tokens must contain at least three characters")
    with publication_lock(state_dir):
        plan = build_plan(bundle, repo, args.forbid_token)
        plan_path = state_dir / "plan.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("Plan changed; use a new state directory after reviewing the changes")
        atomic_json(plan_path, plan)
        print(json.dumps({"plan_sha256": digest(plan), "files": len(plan["files"]),
                          "git_commit": plan["git_commit"], "metrics": plan["metrics"]}, indent=2))
        if args.upload:
            result = publish(make_client(), plan, bundle, state_dir, repo)
            print(json.dumps({"run_url": result["run_url"], "status": result["status"],
                              "artifacts_verified": len(result["artifact_sha256"]),
                              "existing_runs_unchanged": result["existing_runs_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
