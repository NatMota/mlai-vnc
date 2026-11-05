"""Synthetic publication tests; no network or MLflow server is contacted."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import upload_golden as u


class FakeRun:
    def __init__(self, identifier, tags=None):
        self.info = SimpleNamespace(run_id=identifier, experiment_id="3", status="RUNNING")
        self.data = SimpleNamespace(tags=dict(tags or {}), metrics={}, params={})

    def to_dictionary(self):
        return deepcopy({"info": vars(self.info), "data": vars(self.data)})


class FakeClient:
    def __init__(self):
        self.runs = {"old": FakeRun("old", {"mlflow.runName": "existing-run"})}
        self.artifacts = {"old": {}}
        self.mutations = []
        self.creations = 0
        self.lose_create_response = False

    def get_experiment(self, identifier):
        return SimpleNamespace(name="v2-sensors", lifecycle_stage="active")

    def search_runs(self, *args, **kwargs):
        return list(self.runs.values())

    def get_run(self, identifier):
        return self.runs[identifier]

    def create_run(self, experiment_id, tags, run_name):
        self.creations += 1
        identifier = "created-" + str(self.creations)
        run = FakeRun(identifier, tags)
        self.runs[identifier] = run
        self.artifacts[identifier] = {}
        self.mutations.append(("create", identifier))
        if self.lose_create_response:
            self.lose_create_response = False
            raise TimeoutError("Synthetic lost response")
        return run

    def log_metric(self, identifier, name, value, **kwargs):
        self.mutations.append(("metric", identifier))
        self.runs[identifier].data.metrics[name] = value

    def list_artifacts(self, identifier, directory=None):
        prefix = "" if directory is None else directory + "/"
        immediate = {}
        for name in self.artifacts[identifier]:
            if name.startswith(prefix):
                suffix = name[len(prefix):]
                first = suffix.split("/")[0]
                immediate[prefix + first] = "/" in suffix
        return [SimpleNamespace(path=name, is_dir=directory) for name, directory in sorted(immediate.items())]

    def log_artifact(self, identifier, filename, artifact_path=None):
        self.mutations.append(("artifact", identifier))
        name = Path(filename).name if artifact_path is None else artifact_path + "/" + Path(filename).name
        self.artifacts[identifier][name] = Path(filename).read_bytes()

    def download_artifacts(self, identifier, name, directory):
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.artifacts[identifier][name])
        return str(path)

    def set_terminated(self, identifier, status):
        self.mutations.append(("finish", identifier))
        self.runs[identifier].info.status = status


class MetricsTests(unittest.TestCase):
    def row(self, identifier, distance=17.0):
        return {"case_id": identifier, "parked": True, "stop_s": 4.0,
                "success": distance <= 10, "at_stop_mm": distance,
                "closest_mm": 12.0, "park_error_cm": 5.0}

    def test_metrics_are_computed_with_only_stopped_distances(self):
        rows = [self.row(1, 8), self.row(2, 30),
                {**self.row(3), "stop_s": None, "at_stop_mm": None, "success": False}]
        metrics = u.metrics_from_rows(rows, 3)
        self.assertEqual(metrics["success_10mm"], 1)
        self.assertEqual(metrics["stopped"], 2)
        self.assertEqual(metrics["median_at_stop_mm"], 19)

    def test_invalid_results_rejected(self):
        for rows in ([self.row(1), self.row(1)],
                     [{**self.row(1), "success": True}, self.row(2)],
                     [{**self.row(1), "error": "failure"}, self.row(2)],
                     [{**self.row(1), "closest_mm": float("nan")}, self.row(2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                u.metrics_from_rows(rows, 2)


class BundleTests(unittest.TestCase):
    def test_names_and_bytes_are_scanned_without_echoing_token(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "report.md").write_text("private-name", encoding="utf-8")
            with self.assertRaises(ValueError) as raised:
                u.bundle_hashes(root, ["private-name"])
            self.assertNotIn("private-name", str(raised.exception))
            (root / "report.md").write_text("An English result.", encoding="utf-8")
            self.assertEqual(set(u.bundle_hashes(root, ["private-name"])), {"report.md"})

    def test_text_must_be_english_and_hidden_files_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "report.md").write_text(chr(0x4E2D), encoding="utf-8")
            with self.assertRaises(ValueError):
                u.bundle_hashes(root, ["private-name"])
            (root / "report.md").unlink()
            (root / ".secret.json").write_text("{}")
            with self.assertRaises(ValueError):
                u.bundle_hashes(root, ["private-name"])


class CommitEvidenceTests(unittest.TestCase):
    def test_git_blob_keeps_binary_bytes_and_explicit_worktree(self):
        with patch.object(u.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b"line\r\n")) as command:
            self.assertEqual(u.git_blob(Path("repo"), "run.py"), b"line\r\n")
        self.assertIn("--work-tree=repo", command.call_args.args[0])
        self.assertNotIn("text", command.call_args.kwargs)

    def test_code_and_report_must_match_exact_git_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "code").mkdir()
            (root / "docs").mkdir()
            (root / "code/run.py").write_bytes(b"print(1)\r\n")
            (root / "docs/golden_results.md").write_bytes(b"Result.\n")
            files = {"code/run.py": u.file_hash(root / "code/run.py"),
                     "docs/golden_results.md": u.file_hash(root / "docs/golden_results.md")}
            with patch.object(u, "git_blob", side_effect=[b"print(1)\r\n", b"Result.\n"]):
                mapping = u.verify_committed_artifacts(root, root, files)
            self.assertEqual(mapping["code/run.py"], "run.py")
            self.assertEqual(mapping["docs/golden_results.md"], "docs/golden_results.md")
            with patch.object(u, "git_blob", return_value=b"print(1)\n"):
                with self.assertRaises(ValueError):
                    u.verify_committed_artifacts(root, root, files)

    def test_description_includes_comparison_caveat(self):
        metrics = {"median_at_stop_mm": 16.95189884006594, "stopped": 68.0, "success_10mm": 1.0}
        comparison = {"nat": {"median_at_stop_mm": 168.80902936971373, "stopped": 91, "success_10mm": 0},
                      "publication_references": {"iteration_57": {"success_10mm": 4, "attempts": 100}}}
        protocol = {"stop_distance_m": 0.005, "stop_hold_steps": 100}
        with patch.object(u, "bundle_hashes", return_value={"code/run.py": "hash"}), \
                patch.object(u, "repository_info", return_value={"git_commit": "1" * 40}), \
                patch.object(u, "validated_metrics", return_value=metrics), \
                patch.object(u, "read_json", side_effect=[comparison, protocol]), \
                patch.object(u, "verify_committed_artifacts", return_value={"code/run.py": "run.py"}):
            plan = u.build_plan(Path("bundle"), Path("repo"), ["private-name"])
        note = plan["tags"]["mlflow.note.content"]
        self.assertTrue(note.startswith(u.DESCRIPTION))
        self.assertIn("91 runs", note)
        self.assertIn("68 for this arm", note)
        self.assertIn("1/100 versus 0/100", note)
        self.assertIn("different driving network", note)


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bundle, self.state = self.root / "bundle", self.root / "state"
        self.bundle.mkdir()
        self.state.mkdir()
        (self.bundle / "code").mkdir()
        (self.bundle / "code/run.py").write_text('"""Example code."""\n', encoding="utf-8")
        self.plan = {"schema": 1, "git_commit": "1" * 40,
                     "metrics": {"success_10mm": 1.0, "parked": 100.0},
                     "tags": {"mlflow.runName": u.RUN_NAME, "mlflow.user": "experiment-contributor"},
                     "files": {"code/run.py": u.file_hash(self.bundle / "code/run.py")}}
        self.client = FakeClient()
        self.before = u.snapshot_runs(self.client)

    def tearDown(self):
        self.temp.cleanup()

    def publish(self):
        with patch.object(u, "verify_remote_heads"):
            return u.publish(self.client, self.plan, self.bundle, self.state, self.root)

    def test_upload_verifies_bytes_and_never_mutates_existing_run(self):
        result = self.publish()
        self.assertEqual(result["status"], "FINISHED")
        self.assertEqual(result["existing_run_count"], 1)
        self.assertEqual(self.client.creations, 1)
        self.assertFalse(any(identifier == "old" for _, identifier in self.client.mutations))
        self.assertEqual(self.before["old"], u.snapshot_runs(self.client)["old"])
        self.assertEqual(result["artifact_sha256"]["code/run.py"], self.plan["files"]["code/run.py"])

    def test_resume_creates_no_duplicate_or_duplicate_metric_points(self):
        first = self.publish()
        metric_calls = sum(kind == "metric" for kind, _ in self.client.mutations)
        second = self.publish()
        self.assertEqual(first["run_id"], second["run_id"])
        self.assertEqual(self.client.creations, 1)
        self.assertEqual(sum(kind == "metric" for kind, _ in self.client.mutations), metric_calls)

    def test_lost_creation_response_recovers_only_owned_run(self):
        self.client.lose_create_response = True
        with self.assertRaises(TimeoutError):
            self.publish()
        transaction = u.read_json(self.state / "transaction.json")
        self.assertEqual(transaction["phase"], "creation_started")
        result = self.publish()
        self.assertEqual(result["run_id"], "created-1")
        self.assertEqual(self.client.creations, 1)

    def test_uncertain_creation_without_server_match_is_not_retried(self):
        self.client.lose_create_response = True
        with self.assertRaises(TimeoutError):
            self.publish()
        del self.client.runs["created-1"]
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assertEqual(self.client.creations, 1)

    def test_preexisting_name_blocks_creation(self):
        self.client.runs["old"].data.tags["mlflow.runName"] = u.RUN_NAME
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(self.client.creations, 0)
        self.assertEqual(self.client.mutations, [])

    def test_receipt_cannot_authorize_mutating_a_preexisting_run(self):
        transaction = {"transaction_id": "owned", "plan_sha256": "hash"}
        with self.assertRaises(RuntimeError):
            u.owned_run(self.client, "old", transaction, self.before)
        self.assertEqual(self.client.mutations, [])

    def test_existing_metadata_changes_are_detected_on_resume(self):
        self.publish()
        self.client.runs["old"].data.tags["changed_elsewhere"] = "yes"
        mutation_count = len(self.client.mutations)
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assertEqual(len(self.client.mutations), mutation_count)

    def test_changed_local_artifact_is_not_uploaded(self):
        (self.bundle / "code/run.py").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse(any(kind == "artifact" for kind, _ in self.client.mutations))
        self.assertEqual(self.client.creations, 0)

    def test_remote_corruption_is_detected_before_completion(self):
        self.publish()
        self.client.artifacts["created-1"]["code/run.py"] = b"corrupt"
        count = len(self.client.mutations)
        with self.assertRaises(ValueError):
            self.publish()
        self.assertEqual(len(self.client.mutations), count)

    def test_wrong_remote_heads_block_all_mlflow_mutations(self):
        with patch.object(u, "verify_remote_heads", side_effect=ValueError("wrong remote")):
            with self.assertRaises(ValueError):
                u.publish(self.client, self.plan, self.bundle, self.state, self.root)
        self.assertEqual(self.client.mutations, [])


if __name__ == "__main__":
    unittest.main()
