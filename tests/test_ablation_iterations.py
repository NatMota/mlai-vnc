"""Training iteration contracts, with tiny synthetic inputs and no model training."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import ablation as a


class AblationIterationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        old = Path.cwd()
        self.addCleanup(os.chdir, old)
        (self.root / "data").mkdir()
        np.savez(self.root / "data/run_1.npz", state=np.zeros((1, 21)))
        self.donor = {"settings": {"epochs": 20, "seed": 99}, "vel": {}, "torque": {}}
        torch.save(self.donor, self.root / "donor.pt")
        self.tables = {name: (np.zeros((2, 45)), np.zeros((2, 7))) for name in ("vel", "torque")}

    def invoke(self, *extra):
        argv = ["ablation.py", "--data", "data", "--from", "donor.pt", "--output", "runs/a",
                "--groups", "none", "--seed", "1", "--epochs", "3", *extra]
        with patch.object(a, "ROOT", self.root), patch.object(sys, "argv", argv), \
                patch.object(a, "load_runs", return_value=[]), \
                patch.object(a, "tables", return_value=self.tables), \
                patch.object(a, "train_net", return_value={"best_val": 0.1}) as train, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            a.main()
        return train.call_count, output.getvalue()

    def test_records_current_settings_and_separate_donor_settings(self):
        self.invoke()
        saved = torch.load(self.root / "runs/a/none/policy.pt", weights_only=False)
        self.assertEqual(saved["settings"]["epochs"], 3)
        self.assertEqual(saved["settings"]["seed"], 1)
        self.assertEqual(saved["donor_settings"], self.donor["settings"])
        config = json.loads((self.root / "runs/a/none/training_config.json").read_text())
        self.assertEqual(config, saved["training_config"])
        self.assertIn("data/run_1.npz", config["data_sha256"])
        self.assertTrue(config["donor_sha256"])

    def test_existing_checkpoint_is_not_silently_reused(self):
        self.invoke()
        old = (self.root / "runs/a/none/policy.pt").read_bytes()
        with self.assertRaisesRegex((ValueError, SystemExit), "resume|exists|overwrite"):
            self.invoke("--seed", "2")
        self.assertEqual((self.root / "runs/a/none/policy.pt").read_bytes(), old)

    def test_explicit_resume_skips_identical_completed_group(self):
        self.invoke()
        old = (self.root / "runs/a/training_summary.json").read_bytes()
        count, output = self.invoke("--resume")
        self.assertEqual(count, 0)
        self.assertIn("SKIP", output)
        self.assertEqual((self.root / "runs/a/training_summary.json").read_bytes(), old)

    def test_resume_rejects_changed_seed(self):
        self.invoke()
        with self.assertRaisesRegex(ValueError, "config|match"):
            self.invoke("--resume", "--seed", "2")

    def test_resume_rejects_changed_data_bytes(self):
        self.invoke()
        np.savez(self.root / "data/run_1.npz", state=np.ones((1, 21)))
        with self.assertRaisesRegex(ValueError, "config|match"):
            self.invoke("--resume")

    def test_resume_rejects_changed_donor_bytes(self):
        self.invoke()
        torch.save({"settings": {"seed": 7}}, self.root / "donor.pt")
        with self.assertRaisesRegex(ValueError, "config|match"):
            self.invoke("--resume")


if __name__ == "__main__":
    unittest.main()
