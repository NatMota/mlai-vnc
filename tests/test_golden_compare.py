"""Synthetic comparison tests; no benchmark targets or outcomes are used."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile

import golden_compare as c


class CompareTests(unittest.TestCase):
    def fixtures(self):
        golden = [
            {"case_id": 1, "success": True, "parked": True, "park_s": 2.2,
             "park_error_cm": 3, "stop_s": 4, "at_stop_mm": 8, "closest_mm": 5},
            {"case_id": 2, "success": False, "parked": True, "park_s": 3,
             "park_error_cm": 5, "stop_s": None, "at_stop_mm": None, "closest_mm": 20},
            {"case_id": 3, "success": False, "parked": False, "park_s": None,
             "park_error_cm": None, "stop_s": None, "at_stop_mm": None, "closest_mm": None},
        ]
        nat = [
            {"dot": "1", "success": "False", "parked": "True", "park_s": "2",
             "park_error_cm": "2", "stop_s": "5", "at_stop_mm": "30", "closest_mm": "10"},
            {"dot": "2", "success": "True", "parked": "True", "park_s": "3",
             "park_error_cm": "4", "stop_s": "6", "at_stop_mm": "9", "closest_mm": "8"},
            {"dot": "3", "success": "False", "parked": "False", "park_s": "",
             "park_error_cm": "", "stop_s": "", "at_stop_mm": "", "closest_mm": ""},
        ]
        return golden, nat

    def test_point_join_and_success_table(self):
        golden, nat = self.fixtures()
        rows, summary = c.compare(golden, list(reversed(nat)))
        self.assertEqual([r["case_id"] for r in rows], [1, 2, 3])
        self.assertAlmostEqual(rows[0]["delta_park_s"], 0.2)
        self.assertEqual(summary["success_2x2"], {
            "both_success": 0, "two_rate_only": 1, "nat_only": 1, "neither_success": 1})
        self.assertEqual(summary["both_parked"]["points"], 2)
        self.assertEqual(summary["both_parked"]["two_rate_success_rate"], 0.5)
        self.assertEqual(summary["both_parked"]["nat_success_rate"], 0.5)

    def test_invalid_boolean_rejected(self):
        with self.assertRaises(ValueError):
            c.csv_bool("maybe")
        self.assertFalse(c.csv_bool("False"))

    def test_duplicate_and_missing_ids_rejected(self):
        golden, nat = self.fixtures()
        with self.assertRaises(ValueError):
            c.compare(golden + golden[:1], nat)
        with self.assertRaises(ValueError):
            c.compare(golden, nat[:-1])

    def test_process_error_cannot_be_scored(self):
        golden, nat = self.fixtures()
        golden[0]["error"] = "synthetic failure"
        with self.assertRaises(ValueError):
            c.compare(golden, nat)

    def test_no_both_parked_is_null_rate(self):
        golden, nat = self.fixtures()
        for row in golden:
            row["parked"] = False
            row["success"] = False
        _, summary = c.compare(golden, nat)
        self.assertIsNone(summary["both_parked"]["two_rate_success_rate"])


    def test_parking_report_never_reads_arm_outcomes(self):
        golden, nat = self.fixtures()
        for row in golden:
            row["success"] = "unreadable arm outcome"
        for row in nat:
            row["success"] = "unreadable arm outcome"
        rows, summary = c.compare_parking(golden, nat)
        self.assertEqual(summary["park_agreement"], 3)
        self.assertNotIn("success_2x2", summary)
        self.assertTrue(all("stop" not in key and "success" not in key for key in rows[0]))

    def test_parking_median_excludes_unparked_final_position(self):
        golden, nat = self.fixtures()
        nat[2]["park_error_cm"] = "1000"
        _, summary = c.compare(golden, nat)
        self.assertEqual(summary["nat"]["median_park_error_cm"], 3)

    def test_nonfinite_numeric_measurement_rejected(self):
        with self.assertRaises(ValueError):
            c.number(float("nan"))


    def test_archive_loader_checks_source_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"case_id": i, "source_provenance_sha256": "a" * 64} for i in range(12)]
            rows[-1]["source_provenance_sha256"] = "b" * 64
            result = root / "P.jsonl"
            result.write_text("".join(json.dumps(row) + chr(10) for row in rows), encoding="utf-8")
            summary = {"complete": True, "process_errors": 0,
                       "source_provenance_sha256": "a" * 64,
                       "results_sha256": hashlib.sha256(result.read_bytes()).hexdigest()}
            (root / "P.summary.json").write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(ValueError):
                c.load_completed(root, "P")

    def test_reference_selection_is_iteration_test_and_seed7(self):
        rows = [
            {"iteration": "43", "split": "test", "seed": "7", "success_10mm": "0",
             "attempts": "100", "parked": "90", "stopped": "80", "name": "a"},
            {"iteration": "57", "split": "test", "seed": "7", "success_10mm": "4",
             "attempts": "100", "parked": "87", "stopped": "81", "name": "b"},
            {"iteration": "57", "split": "val", "seed": "7", "success_10mm": "6",
             "attempts": "100", "parked": "92", "stopped": "89", "name": "b"},
        ]
        refs = c.reference_scores(rows)
        self.assertEqual(refs["iteration_57"]["success_10mm"], 4)
        self.assertEqual(refs["iteration_43"]["attempts"], 100)


if __name__ == "__main__":
    unittest.main()
