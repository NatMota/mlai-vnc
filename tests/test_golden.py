"""Contracts for the golden test: copies of Nat's code, his scoring rules and the sensor-only STOP."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import numpy as np

import golden_sensor as g
from tests.test_v2_sensor import source_of

ROOT = Path(__file__).resolve().parents[1]
COPIED = ("MOUNT", "WHEEL_RADIUS", "DT", "camera_to_robot", "see_dot", "Tracker", "driving_inputs", "policy")
NAT_FILES = ("camera_loop.py", "sensors.py", "wrist_camera.py", "full_run.py", "models/driving_seed7.npz",
             "expert_test.json", "expert_val.json", "seed7_metrics.csv", "test_config.json", "publication_scores.csv")


class CopyTests(unittest.TestCase):
    def test_copied_code_matches_nat_camera_loop(self):
        for name in COPIED:             # TRACK is assigned together with WHEEL_RADIUS
            self.assertEqual(source_of(ROOT / "golden_sensor.py", name),
                             source_of(ROOT / "nat_golden/camera_loop.py", name), name)

    def test_nat_files_match_the_recorded_hashes(self):
        table = (ROOT / "nat_golden/SOURCE.md").read_text(encoding="utf-8")
        for name in NAT_FILES:
            digest = hashlib.sha256((ROOT / "nat_golden" / name).read_bytes()).hexdigest()
            row = next(line for line in table.splitlines() if line.startswith("| " + chr(96) + name + chr(96) + " |"))
            self.assertEqual(row.split("|")[-2].strip().strip(chr(96)), digest, name)


class RuleTests(unittest.TestCase):
    def test_driving_model_takes_eleven_inputs(self):
        with np.load(ROOT / "nat_golden/models/driving_seed7.npz", allow_pickle=False) as weights:
            w = dict(weights)
        x = g.driving_inputs(w, np.zeros(12), np.array([1.0, 0.2]), True, 0.4)
        self.assertEqual(x.shape, (11,))
        cmd, park = g.policy(w, x)
        self.assertTrue(np.all(np.abs(cmd) <= 3.0))
        self.assertIn(park, (0.0, 1.0))

    def test_arm_tracker_keeps_height_while_hidden(self):
        tracker = g.Tracker(keep_height=True)
        self.assertIsNone(tracker.update(None, np.zeros(4)))
        np.testing.assert_allclose(tracker.update(np.array([1.0, 0.2, 0.3]), np.zeros(4)), [1.0, 0.2, 0.3])
        moved = tracker.update(None, np.full(4, 1.0))
        self.assertLess(moved[0], 1.0)
        self.assertEqual(moved[2], 0.3)

    def test_time_limit_and_expert_park_position(self):
        case = {"stop_s": 8.0, "reference_valid": True,
                "frames": [{"base": [0, 0, 0], "parked": False}, {"base": [0.4, 0.1, 0], "parked": True}]}
        self.assertEqual(g.time_limit_s(case), 12.0)
        np.testing.assert_allclose(g.expert_park_xy(case), [0.4, 0.1])
        self.assertIsNone(g.expert_park_xy({**case, "reference_valid": False}))

    def test_cases_cover_one_hundred_test_and_validation_dots(self):
        for name in ("expert_test.json", "expert_val.json"):
            cases = json.loads((ROOT / "nat_golden" / name).read_text())
            self.assertEqual(len(cases), 100, name)
        self.assertFalse(set(json.loads((ROOT / "nat_golden/expert_test.json").read_text()))
                         & set(json.loads((ROOT / "nat_golden/expert_val.json").read_text())))


class StopTests(unittest.TestCase):
    def test_stop_needs_a_full_hold_and_latches(self):
        stop = g.SensorStop()
        for _ in range(99):
            self.assertFalse(stop.update(0.004))
        self.assertTrue(stop.update(0.004))
        self.assertTrue(stop.update(0.5))           # latched

    def test_one_step_above_the_threshold_restarts_the_count(self):
        stop = g.SensorStop()
        for _ in range(60):
            stop.update(0.004)
        stop.update(0.0051)
        for _ in range(99):
            self.assertFalse(stop.update(0.004))
        self.assertTrue(stop.update(0.004))


class ScoreTests(unittest.TestCase):
    def test_statuses_follow_full_run(self):
        self.assertEqual(g.score(False, None, None), ("no_park_timeout", False))
        self.assertEqual(g.score(True, None, None), ("no_stop_timeout", False))
        self.assertEqual(g.score(True, 9000, 0.010), ("success", True))
        self.assertEqual(g.score(True, 9000, 0.011), ("stop_outside_10mm", False))


class PolicyIntegrationTests(unittest.TestCase):
    def test_stop_cannot_start_before_parking_and_controller_continues(self):
        controller = object.__new__(g.GoldenPolicy)
        controller.robot = SimpleNamespace(dt=0.001, dyn=SimpleNamespace(pin_model=None))
        controller.pin_data = None
        controller.step = 0
        controller.next_decision = 10000
        controller.arm_enc = SimpleNamespace(read=lambda q: {"arm_angle": q, "arm_speed": q})
        controller.parked = False
        controller.use_arm = True
        controller.dot = np.zeros(3)
        controller.hold = np.zeros(7)
        controller.wheels = np.ones(4)
        controller.stop = g.SensorStop()
        controller.stop_step = None
        controller.report = {}
        controller._reach = Mock(return_value=np.zeros(7))
        controller.ee_robot_frame = Mock(return_value=np.zeros(3))
        obs = {"q": np.zeros(7), "qd": np.zeros(7), "q_pin": np.zeros(24)}
        with patch.object(g.pin, "computeGeneralizedGravity", return_value=np.zeros(24)):
            for _ in range(120):
                controller.act(obs)
        self.assertIsNone(controller.stop_step)
        self.assertEqual(controller.stop.count, 0)
        controller._reach.assert_not_called()
        controller.parked = True
        for _ in range(99):
            controller.act(obs)
            self.assertIsNone(controller.stop_step)
        controller.act(obs)
        self.assertEqual(controller.stop_step, 219)
        for _ in range(500):
            controller.act(obs)
        self.assertEqual(controller._reach.call_count, 600)
        self.assertEqual(controller.stop_step, 219)


class EpisodeIntegrationTests(unittest.TestCase):
    @staticmethod
    def fixture(park_at=2, stop_at=4, nan_terminal=False):
        class Robot:
            dt = 0.001
            def __init__(self):
                self.steps = 0
            def observe(self):
                distance = 0.001 if self.steps < 2 else 0.009
                if nan_terminal and self.steps == 504:
                    distance = float("nan")
                return {"base_pos": np.zeros(3), "q": np.zeros(7), "qd": np.zeros(7),
                        "ee": np.array([distance, 0.0, 0.0])}
            def step(self, wheels, torque):
                self.steps += 1

        class Policy:
            use_arm = True
            def reset(self, robot, target, obs):
                self.robot = robot
                self.report = {}
                self.stop_step = None
                self.calls = 0
            def act(self, obs):
                step = self.calls
                self.calls += 1
                parked = park_at is not None and step >= park_at
                if self.use_arm and stop_at is not None and step == stop_at:
                    self.stop_step = step
                return np.zeros(4), np.zeros(7), parked

        case = {"target": [0.0, 0.0, 0.0], "stop_s": 0.01, "park_s": 0.002,
                "frames": [{"base": [0.0, 0.0, 0.0], "parked": True}]}
        return Robot(), Policy(), case

    def test_stop_continues_500_steps_and_closest_is_post_park(self):
        robot, controller, case = self.fixture()
        with patch("sim.Robot", return_value=robot):
            result = g.run_golden_episode(case, controller)
        self.assertEqual(robot.steps, 504)
        self.assertEqual(result["final_s"], result["stop_s"] + 0.5)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["closest_mm"], 9.0)
        self.assertAlmostEqual(result["at_stop_mm"], 9.0)

    def test_no_stop_uses_expert_stop_plus_four(self):
        robot, controller, case = self.fixture(stop_at=None)
        with patch("sim.Robot", return_value=robot):
            result = g.run_golden_episode(case, controller)
        self.assertEqual(robot.steps, 4010)
        self.assertEqual(result["status"], "no_stop_timeout")

    def test_no_park_times_out(self):
        robot, controller, case = self.fixture(park_at=None, stop_at=None)
        with patch("sim.Robot", return_value=robot):
            result = g.run_golden_episode(case, controller)
        self.assertEqual(result["status"], "no_park_timeout")
        self.assertIsNone(result["closest_mm"])

    def test_parity_disables_arm_and_ends_half_second_after_park(self):
        robot, controller, case = self.fixture()
        with patch("sim.Robot", return_value=robot):
            result = g.run_golden_episode(case, controller, parity=True)
        self.assertFalse(controller.use_arm)
        self.assertIsNone(result["stop_s"])
        self.assertEqual(robot.steps, 502)

    def test_nonfinite_terminal_state_rejected(self):
        robot, controller, case = self.fixture(nan_terminal=True)
        with patch("sim.Robot", return_value=robot):
            with self.assertRaises(FloatingPointError):
                g.run_golden_episode(case, controller)


if __name__ == "__main__":
    unittest.main()
