"""Synthetic contracts for resumable golden evaluation; never runs test targets."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import golden_eval as e


class SelectionTests(unittest.TestCase):
    def test_numeric_order_and_group_sizes(self):
        cases = {str(i): {} for i in range(100, 0, -1)}
        self.assertEqual(e.select_ids(cases, "S"), list(range(1, 13)))
        self.assertEqual(e.select_ids(cases, "P"), list(range(1, 13)))
        self.assertEqual(e.select_ids(cases, "G"), list(range(1, 101)))

    def test_invalid_dataset_or_group_rejected(self):
        with self.assertRaises(ValueError):
            e.select_ids({"1": {}}, "G")
        with self.assertRaises(ValueError):
            e.select_ids({str(i): {} for i in range(100)}, "X")


class SummaryTests(unittest.TestCase):
    def test_counts_and_medians_include_misses(self):
        rows = [
            {"success": True, "parked": True, "stop_s": 3, "at_stop_mm": 8,
             "closest_mm": 4, "park_error_cm": 2, "status": "success"},
            {"success": False, "parked": True, "stop_s": 4, "at_stop_mm": 30,
             "closest_mm": 20, "park_error_cm": 8, "status": "stop_outside_10mm"},
            {"success": False, "parked": False, "stop_s": None,
             "at_stop_mm": None, "closest_mm": None, "park_error_cm": None,
             "status": "no_park_timeout"},
        ]
        summary = e.summarize(rows)
        self.assertEqual(summary["episodes"], 3)
        self.assertEqual(summary["success_10mm"], 1)
        self.assertEqual(summary["parked"], 2)
        self.assertEqual(summary["stopped"], 2)
        self.assertEqual(summary["median_at_stop_mm"], 19)
        self.assertEqual(summary["median_closest_mm"], 12)
        self.assertEqual(summary["median_park_error_cm"], 5)
        self.assertEqual(summary["status_counts"]["no_park_timeout"], 1)

    def test_empty_summary_has_null_medians(self):
        self.assertIsNone(e.summarize([])["median_at_stop_mm"])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.row = {"run": "G", "case_id": 11, "bearing": 11,
                    "version": e.VERSION, "provenance_sha256": "abc",
                    "status": "no_park_timeout", "success": False}

    def tearDown(self):
        self.temp.cleanup()

    def test_receipt_cannot_be_overwritten(self):
        path = self.root / "11.json"
        e.write_new_json(path, self.row)
        with self.assertRaises(FileExistsError):
            e.write_new_json(path, {"changed": True})
        self.assertEqual(json.loads(path.read_text()), self.row)

    def test_resume_keeps_matching_receipt(self):
        e.write_new_json(self.root / "11.json", self.row)
        rows = e.load_receipts(self.root, "G", [11, 12], "abc")
        self.assertEqual(rows, {11: self.row})

    def test_resume_rejects_changed_code_or_foreign_case(self):
        e.write_new_json(self.root / "11.json", self.row)
        with self.assertRaises(ValueError):
            e.load_receipts(self.root, "G", [11], "different")
        with self.assertRaises(ValueError):
            e.load_receipts(self.root, "G", [12], "abc")

    def test_resume_rejects_receipt_without_status(self):
        broken = {key: value for key, value in self.row.items() if key != "status"}
        e.write_new_json(self.root / "11.json", broken)
        with self.assertRaises(ValueError):
            e.load_receipts(self.root, "G", [11], "abc")

    def test_attempt_without_receipt_blocks_rerun(self):
        e.write_new_json(self.root / "11.json", {"started": True})
        with self.assertRaises(RuntimeError):
            e.pending_ids([11, 12], {}, self.root)
        self.assertEqual(e.pending_ids([11, 12], {11: self.row}, self.root), [12])


class ProtocolTests(unittest.TestCase):
    def test_freeze_requires_validation_and_exact_rule(self):
        protocol = {"version": 1, "selected_from": "S", "reason": "12 STOPs",
                    "stop_distance_m": 0.005, "stop_hold_steps": 100,
                    "validation_summary_sha256": "abc"}
        e.validate_protocol(protocol, 0.005, 100, "abc")
        for key, value in (("selected_from", "G"), ("stop_distance_m", 0.008),
                           ("stop_hold_steps", 99), ("validation_summary_sha256", "bad")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                e.validate_protocol({**protocol, key: value}, 0.005, 100, "abc")


class ChildAndBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.hashes = {"source.py": "fixed"}
        self.digest = e.code_digest(self.hashes)
        self.manifest = {"case_ids": [11], "provenance_sha256": self.digest,
                         "version": e.VERSION, "run": "G"}
        e.write_new_json(self.root / "G.manifest.json", self.manifest)
        self.result = {"success": True, "parked": True, "stop_s": 3.0,
                       "at_stop_mm": 5.0, "closest_mm": 4.0, "park_error_cm": 2.0,
                       "status": "success"}

    def tearDown(self):
        self.temp.cleanup()

    def test_child_receipt_precedes_print_and_repeat_cannot_simulate(self):
        calls = []
        module = SimpleNamespace(
            GoldenPolicy=SimpleNamespace(build_golden=lambda *a, **k: k),
            run_golden_episode=lambda *a, **k: calls.append(k) or self.result,
        )
        with patch.dict("sys.modules", {"golden_sensor": module}), \
                patch.object(e, "file_hashes", return_value=self.hashes), \
                patch.object(e, "frozen_protocol"), \
                patch.object(e, "cases_for", return_value={"11": {}}), \
                patch("builtins.print") as output:
            e.run_one("G", 11, self.root, self.digest)
            saved = e.read_json(self.root / "receipts/G/11.json")
            self.assertTrue(saved["success"])
            self.assertTrue(saved["stopped"])
            self.assertEqual(json.loads(output.call_args.args[0][5:]), saved)
            with self.assertRaises(FileExistsError):
                e.run_one("G", 11, self.root, self.digest)
        self.assertEqual(len(calls), 1)

    def test_child_failure_has_receipt_and_counts_as_process_error(self):
        def crash(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        module = SimpleNamespace(GoldenPolicy=SimpleNamespace(build_golden=crash))
        with patch.dict("sys.modules", {"golden_sensor": module}), \
                patch.object(e, "file_hashes", return_value=self.hashes), \
                patch.object(e, "frozen_protocol"), \
                patch.object(e, "cases_for", return_value={"11": {}}), patch("builtins.print"):
            e.run_one("G", 11, self.root, self.digest)
        saved = e.read_json(self.root / "receipts/G/11.json")
        self.assertEqual(saved["status"], "process_error")
        self.assertFalse(saved["success"])
        self.assertEqual(e.summarize([saved])["process_errors"], 1)

    def test_child_rejects_sources_changed_before_launch(self):
        with patch.object(e, "file_hashes", return_value={"changed": "source"}):
            with self.assertRaises(ValueError):
                e.run_one("G", 11, self.root, self.digest)
        self.assertFalse((self.root / "attempts/G/11.json").exists())

    def test_outputs_order_rows_and_preserve_incomplete_count(self):
        manifest = {**self.manifest, "case_ids": [11, 12]}
        row = {**self.result, "case_id": 11}
        summary = e.write_outputs(self.root, "G", {11: row}, manifest, 2.0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["results_sha256"], e.sha256(self.root / "G.jsonl"))
        self.assertEqual(json.loads((self.root / "G.jsonl").read_text()), row)

    def test_completed_batch_never_calls_pool(self):
        manifest = {**self.manifest, "protocol": None}
        (self.root / "G.manifest.json").write_text(json.dumps(manifest))
        row = {**self.result, "run": "G", "case_id": 11, "bearing": 11,
               "version": e.VERSION, "provenance_sha256": self.digest}
        e.write_new_json(self.root / "receipts/G/11.json", row)
        e.write_new_json(self.root / "P.summary.json", {"provenance_sha256": self.digest})
        e.write_outputs(self.root, "G", {11: row}, manifest, 20.0)
        initial = (self.root / "G.summary.json").read_bytes()
        with patch.object(e, "frozen_protocol", return_value=None), \
                patch.object(e, "cases_for", return_value={"11": {}}), \
                patch.object(e, "select_ids", return_value=[11]), \
                patch.object(e, "provenance", return_value={"sha256": self.hashes}), \
                patch.object(e, "run_pool") as pool, patch("builtins.print"):
            e.run_batch("G", self.root, 12)
        pool.assert_not_called()
        self.assertEqual((self.root / "G.summary.json").read_bytes(), initial)

    def test_changed_manifest_is_rejected_before_pool(self):
        e.write_new_json(self.root / "P.summary.json", {"provenance_sha256": self.digest})
        with patch.object(e, "frozen_protocol", return_value=None), \
                patch.object(e, "cases_for", return_value={"11": {}}), \
                patch.object(e, "select_ids", return_value=[11]), \
                patch.object(e, "provenance", return_value={"sha256": {"new": "source"}}), \
                patch.object(e, "run_pool") as pool:
            with self.assertRaises(ValueError):
                e.run_batch("G", self.root, 12)
        pool.assert_not_called()

    def test_frozen_s_cannot_rewrite_damaged_reports(self):
        manifest = {**self.manifest, "run": "S", "protocol": None}
        e.write_new_json(self.root / "S.manifest.json", manifest)
        row = {**self.result, "run": "S", "case_id": 11, "bearing": 11,
               "version": e.VERSION, "provenance_sha256": self.digest}
        e.write_new_json(self.root / "receipts/S/11.json", row)
        e.write_new_json(self.root / "protocol.json", {"frozen": True})
        with patch.object(e, "frozen_protocol", return_value=None), \
                patch.object(e, "cases_for", return_value={"11": {}}), \
                patch.object(e, "select_ids", return_value=[11]), \
                patch.object(e, "provenance", return_value={"sha256": self.hashes}), \
                patch.object(e, "run_pool") as pool, patch("builtins.print"):
            with self.assertRaises(RuntimeError):
                e.run_batch("S", self.root, 12)
        pool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
