"""Evaluation output safety and resumption without real simulation jobs."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import parallel_eval as e


def row(index):
    return {"target": index, "group": "fresh", "policy": "expert", "reached": True,
            "final_error_cm": 0.5, "time_s": 1.0}


class EvaluationIterationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        old = Path.cwd()
        self.addCleanup(os.chdir, old)
        self.out = self.root / "eval.jsonl"

    def invoke(self, pool, *extra):
        argv = ["parallel_eval.py", "expert", "--group", "fresh", "--n", "2",
                "--out", str(self.out), *extra]
        with patch.object(e, "ROOT", self.root), patch.object(sys, "argv", argv), \
                patch.object(e, "provenance", return_value={"source_sha256": {"sim.py": "source-v1"}}), \
                patch.object(e, "run_pool", side_effect=pool), \
                patch("local_tracking.log_evaluation") as log, contextlib.redirect_stdout(io.StringIO()):
            e.main()
        return log

    @staticmethod
    def complete(jobs, workers, prefix, callback):
        for index, _ in jobs:
            callback(index, json.dumps(row(index)), None)
        return 0

    @staticmethod
    def interrupt(jobs, workers, prefix, callback):
        index, _ = jobs[0]
        callback(index, json.dumps(row(index)), None)
        raise KeyboardInterrupt

    def test_existing_results_are_preserved_without_resume(self):
        self.out.write_text('{"old": true}\n')
        summary = self.out.with_suffix(".summary.json")
        summary.write_text('{"old_summary": true}\n')
        with self.assertRaisesRegex((ValueError, SystemExit, FileExistsError), "exists|resume|overwrite"):
            self.invoke(self.complete)
        self.assertEqual(self.out.read_text(), '{"old": true}\n')
        self.assertEqual(summary.read_text(), '{"old_summary": true}\n')

    def test_interrupted_iteration_resumes_only_missing_targets(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(self.interrupt)
        self.assertFalse(self.out.with_suffix(".summary.json").exists())
        visited = []
        def finish(jobs, workers, prefix, callback):
            visited.extend(i for i, _ in jobs)
            return self.complete(jobs, workers, prefix, callback)
        self.invoke(finish, "--resume")
        self.assertEqual(visited, [1])
        summary = json.loads(self.out.with_suffix(".summary.json").read_text())
        self.assertEqual(summary["targets"], 2)
        self.assertEqual(summary["successes"], 2)
        self.assertTrue(summary["complete"])

    def test_complete_resume_does_not_run_or_log_again(self):
        self.invoke(self.complete)
        before = self.out.with_suffix(".summary.json").read_bytes()
        def forbidden(*args):
            self.fail("Completed evaluation must not execute jobs")
        log = self.invoke(forbidden, "--resume")
        log.assert_not_called()
        self.assertEqual(self.out.with_suffix(".summary.json").read_bytes(), before)

    def test_resume_rejects_changed_parameters(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(self.interrupt)
        before = self.out.read_bytes()
        with self.assertRaisesRegex(ValueError, "config|match"):
            self.invoke(self.complete, "--resume", "--goal-error-m", "0.001")
        self.assertEqual(self.out.read_bytes(), before)

    def test_resume_rejects_duplicate_targets(self):
        with self.assertRaises(KeyboardInterrupt):
            self.invoke(self.interrupt)
        with self.out.open("a") as handle:
            handle.write(json.dumps(row(0)) + "\n")
        with self.assertRaisesRegex(ValueError, "duplicate|Duplicate"):
            self.invoke(self.complete, "--resume")

    def test_legacy_results_cannot_be_adopted_as_resume_state(self):
        self.out.write_text(json.dumps(row(0)) + "\n")
        with self.assertRaisesRegex(ValueError, "manifest|legacy|Legacy"):
            self.invoke(self.complete, "--resume")

    def test_lock_blocks_a_second_writer(self):
        self.out.with_suffix(".lock").write_text("12345")
        with self.assertRaises(FileExistsError):
            self.invoke(self.complete)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.out.with_suffix(".lock").read_text(), "12345")

    def test_changed_completed_results_reject_resume(self):
        self.invoke(self.complete)
        self.out.write_text(self.out.read_text().replace('0.5', '0.9'))
        with self.assertRaisesRegex(ValueError, "Summary.*match"):
            self.invoke(self.complete, "--resume")

    def test_process_errors_stay_in_denominator(self):
        def failed(jobs, workers, prefix, callback):
            callback(0, json.dumps(row(0)), None)
            callback(1, None, "worker failed")
            return 1
        with self.assertRaises(SystemExit) as error:
            self.invoke(failed)
        self.assertEqual(error.exception.code, 1)
        summary = json.loads(self.out.with_suffix(".summary.json").read_text())
        self.assertEqual((summary["targets"], summary["successes"], summary["process_errors"]), (2, 1, 1))

    def test_wrong_worker_identity_is_not_written(self):
        def wrong(jobs, workers, prefix, callback):
            callback(0, json.dumps(row(1)), None)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.invoke(wrong)
        self.assertEqual(self.out.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
