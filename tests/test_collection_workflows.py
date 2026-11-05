"""Keep the original collector and the memory-policy collector usable together."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import collect_memory
from memory_policy import vel_scale
from train_memory import load_runs, tables


ROOT = Path(__file__).resolve().parents[1]


class OriginalCollectionTests(unittest.TestCase):
    def run_collector(self, arguments):
        with patch.object(sys, "argv", ["collect.py", *arguments]), \
                patch("subprocess.run") as run, redirect_stdout(io.StringIO()):
            runpy.run_path(str(ROOT / "collect.py"), run_name="__main__")
        return run.call_args_list

    def test_original_normal_and_dart_commands_keep_their_dispatch(self):
        for kind, script in (("normal", "robotnik_arm_task.py"),
                             ("dart", "robotnik_arm_task_dart.py")):
            with self.subTest(kind=kind):
                calls = self.run_collector([kind, "3"])
                self.assertEqual(len(calls), 3)
                for call in calls:
                    self.assertEqual(call.args, ([sys.executable, script],))
                    self.assertEqual(call.kwargs["env"]["ROBOENV_HEADLESS"], "1")

    def test_original_default_still_collects_ten_normal_runs(self):
        calls = self.run_collector([])
        self.assertEqual(len(calls), 10)
        self.assertTrue(all(call.args[0][1] == "robotnik_arm_task.py" for call in calls))


class MemoryCollectionTests(unittest.TestCase):
    def test_seeded_npz_keeps_clean_labels_and_hidden_references_for_training(self):
        seed = 42
        expected_rng = np.random.default_rng(seed)
        expected_target = np.array([expected_rng.uniform(-1.5, 1.5),
                                    expected_rng.uniform(-1.5, 1.5),
                                    expected_rng.uniform(0.1, 1.25)])
        expected = {key: [] for key in ("state", "ee", "reach", "label", "q_des", "qd_des")}
        policy = SimpleNamespace()

        def record_episode(target, expert, max_steps, record, labeler):
            np.testing.assert_array_equal(target, expected_target)
            self.assertIs(expert, policy)
            self.assertEqual(max_steps, 123)
            for i in range(4):
                state = np.zeros(21)
                state[:3] = [-3.0, -3.0, 0.0]
                state[4:11] = np.arange(7) * 0.05 + i * 0.01
                state[11:18] = -0.1
                state[18:21] = target
                ee = target + [0.2, 0.0, 0.0]
                active = i >= 2
                wheels = np.zeros(4) if active else np.arange(4) + 1.0
                torque = np.arange(7) + 10.0 + i * 100.0
                expert.last_clean = wheels, torque
                expert.q_des = state[4:11] + 0.03 * (i + 1)
                expert.qd_des = (np.arange(7) + 1.0) * 0.02
                clean = np.concatenate((wheels, torque))
                # DART applies pushes to the executed command, not its training label.
                executed = clean + 1000.0
                record.append(np.concatenate((state, ee, [float(active)], executed,
                                              labeler.label_row(None, active, expert))))
                for key, value in (("state", state), ("ee", ee), ("reach", float(active)),
                                   ("label", clean), ("q_des", expert.q_des),
                                   ("qd_des", expert.qd_des)):
                    expected[key].append(np.array(value, copy=True))
            return {"reached": True, "final_error_cm": 0.5}

        fake_sim = SimpleNamespace(Expert=Mock(return_value=policy),
                                   run_episode=Mock(side_effect=record_episode))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            legacy = folder / "dart_123.npy"
            np.save(legacy, np.full((2, 32), 9.0))
            legacy_bytes = legacy.read_bytes()
            with patch.object(collect_memory, "OUT", folder), \
                    patch.dict(sys.modules, {"sim": fake_sim}), redirect_stdout(io.StringIO()):
                collect_memory.collect_one(seed, max_steps=123)
            fake_sim.Expert.assert_called_once_with(0.5, 1.0, seed=seed + 1)
            output = folder / "run_42.npz"
            self.assertTrue(output.is_file())
            self.assertFalse((folder / "run_42.tmp.npz").exists())
            self.assertEqual(legacy.read_bytes(), legacy_bytes)
            with np.load(output) as saved:
                self.assertEqual(set(saved.files), set(expected) | {"final_error_cm"})
                for key, values in expected.items():
                    np.testing.assert_array_equal(saved[key], np.asarray(values))
                self.assertEqual(float(saved["final_error_cm"]), 0.5)

            # An unfinished memory run and the original .npy data must both be ignored.
            (folder / "run_43.tmp.npz").write_bytes(b"unfinished")
            with redirect_stdout(io.StringIO()):
                loaded = load_runs([folder], stride=1, good_cm=2.0,
                                   rng=np.random.default_rng(0), early_rows=1)
            self.assertEqual(len(loaded), 1)
            run = loaded[0]
            state = np.asarray(expected["state"])
            np.testing.assert_allclose(run["z"], np.asarray(expected["q_des"]) - state[:, 4:11])
            np.testing.assert_allclose(run["ev"], np.asarray(expected["qd_des"]) - state[:, 11:18])
            np.testing.assert_array_equal(run["phase"], [0, 1, 1, 1])
            np.testing.assert_array_equal(run["reach"], [0, 0, 1, 1])
            np.testing.assert_array_equal(run["phase_valid"], np.ones(4))
            training = tables(loaded)
            np.testing.assert_allclose(training["base"][1], np.asarray(expected["label"])[:2, :4])
            np.testing.assert_allclose(training["torque"][1], np.asarray(expected["label"])[:, 4:])
            self.assertEqual(training["torque"][0].shape, (4, 49))
            np.testing.assert_allclose(training["vel"][1] * vel_scale(run["x"][2:])[:, None],
                                       np.asarray(expected["qd_des"])[2:], rtol=1e-6)

    def test_parallel_resume_uses_the_memory_collector_filename_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "run_43.npz").write_bytes(b"already collected")
            arguments = ["collect_memory.py", "--runs", "3", "--first-seed", "42",
                         "--workers", "2", "--max-steps", "123"]
            with patch.object(collect_memory, "OUT", folder), \
                    patch.object(collect_memory.os, "chdir"), \
                    patch.object(sys, "argv", arguments), \
                    patch.object(collect_memory.subprocess, "run", return_value=SimpleNamespace(
                        stdout="DONE collected\n", stderr="", returncode=0)) as run, \
                    redirect_stdout(io.StringIO()):
                collect_memory.main()
            self.assertEqual(run.call_count, 2)
            seeds = set()
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[:3], [sys.executable, "-B", collect_memory.__file__])
                self.assertEqual(command[3], "--seed")
                seeds.add(int(command[4]))
                self.assertEqual(command[5:], ["--max-steps", "123"])
                for key in ("ROBOENV_HEADLESS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                    self.assertEqual(call.kwargs["env"][key], "1")
            self.assertEqual(seeds, {42, 44})

    def test_single_seed_cli_does_not_start_another_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(collect_memory, "OUT", Path(directory)), \
                    patch.object(collect_memory.os, "chdir"), \
                    patch.object(sys, "argv", ["collect_memory.py", "--seed", "42", "--max-steps", "123"]), \
                    patch.object(collect_memory, "collect_one") as collect, \
                    patch.object(collect_memory.subprocess, "run") as run:
                collect_memory.main()
            collect.assert_called_once_with(42, 123)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
