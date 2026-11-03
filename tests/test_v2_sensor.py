"""Contracts for the v2 sensor policy: copies of Nat's code, timing, tracker, hold controller and robot-frame inputs."""
import ast
import hashlib
from pathlib import Path
import unittest

import numpy as np

import v2_sensor as v2
from ablation import GROUPS
from memory_policy import features

ROOT = Path(__file__).resolve().parents[1]
ARM_CHECKPOINT = ROOT / "outputs/ablation/all_world/policy.pt"
COPIED = ("camera_to_robot", "see_dot", "Tracker", "policy", "MOUNT", "WHEEL_RADIUS", "DT")
NAT_FILES = ("sensors.py", "wrist_camera.py", "camera_loop.py", "expert_reference.json", "split_plan.json",
             "models/iter13.npz", "models/iter13_seed1.npz", "models/iter13_seed2.npz", "models/iter13_seed3.npz")


def names_of(node):
    if isinstance(node, ast.Assign):
        return {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return {getattr(node, "name", None)}


def source_of(path, name):
    """Source text of the top-level function, class or assignment that defines name."""
    text = Path(path).read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body if name in names_of(n))
    return ast.get_source_segment(text, node)


class CopyTests(unittest.TestCase):
    def test_copied_code_matches_nat_camera_loop(self):
        for name in COPIED:             # TRACK is assigned together with WHEEL_RADIUS
            self.assertEqual(source_of(ROOT / "v2_sensor.py", name),
                             source_of(ROOT / "nat_v2/camera_loop.py", name), name)

    def test_nat_files_match_the_recorded_hashes(self):
        table = (ROOT / "nat_v2/SOURCE.md").read_text(encoding="utf-8")
        for name in NAT_FILES:
            digest = hashlib.sha256((ROOT / "nat_v2" / name).read_bytes()).hexdigest()
            self.assertIn(digest, table, name)


class TimingTests(unittest.TestCase):
    def test_decisions_follow_nat_simulator(self):
        self.assertEqual([v2.decision_step(k) for k in range(5)], [0, 33, 67, 100, 133])
        self.assertEqual(sum(v2.decision_step(k) < 1000 for k in range(100)), 30)


class TrackerTests(unittest.TestCase):
    def test_never_seen_returns_none(self):
        self.assertIsNone(v2.HeightTracker().update(None, np.zeros(4)))

    def test_seen_replaces_and_hidden_keeps_height(self):
        tracker = v2.HeightTracker()
        np.testing.assert_allclose(tracker.update(np.array([1.0, 0.2, 0.3]), np.zeros(4)), [1.0, 0.2, 0.3])
        moved = tracker.update(None, np.full(4, 1.0))         # driving forward: the dot comes closer
        self.assertLess(moved[0], 1.0)
        self.assertEqual(moved[2], 0.3)


class HoldTests(unittest.TestCase):
    def test_hold_is_pd_plus_gravity_and_clipped(self):
        q, qd, hold, g = np.zeros(7), np.full(7, 0.1), np.full(7, 0.01), np.full(7, 2.0)
        np.testing.assert_allclose(v2.hold_torque(q, qd, hold, g), 100 * 0.01 - 30 * 0.1 + 2.0)
        self.assertEqual(v2.hold_torque(q, qd, np.full(7, 5.0), g).max(), 30.0)


class DriverTests(unittest.TestCase):
    def test_iteration13_wheels_are_clipped_and_park_is_a_bool(self):
        driver = v2.NatDriver(ROOT / "nat_v2/models/iter13.npz")
        cmd, park = driver.decide(np.full(12, 50.0), np.array([1.0, 0.1]), True)
        self.assertTrue(np.all(np.abs(cmd) <= 3.0))
        self.assertIsInstance(park, (bool, np.bool_))
        cmd, _ = driver.decide(np.zeros(12), None, False)
        self.assertEqual(cmd.shape, (4,))


class RobotFrameTests(unittest.TestCase):
    """The robot-frame state gives the arm the same unmasked inputs as the true world-frame state."""

    def test_unmasked_features_match_the_world_frame(self):
        rng = np.random.default_rng(0)
        removed = set(GROUPS["all_world"])
        for _ in range(20):
            base, yaw = np.r_[rng.uniform(-2, 2, 2), 0.0], rng.uniform(-np.pi, np.pi)
            q, qd = rng.uniform(-1, 1, 7), rng.uniform(-1, 1, 7)
            goal_rf, ee_rf = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
            world = features(np.r_[base, yaw, q, qd, base + R @ goal_rf], base + R @ ee_rf)[0]
            robot = features(np.r_[np.zeros(4), q, qd, goal_rf], ee_rf)[0]
            keep = [i for i in range(34) if i not in removed]
            np.testing.assert_allclose(robot[keep], world[keep], atol=1e-9)


class ArmEncoderTests(unittest.TestCase):
    def test_coarse_tick_floor_and_speed(self):
        tick, dt = 2 * np.pi / 4096, 0.001
        enc = v2.ArmEncodersAt(4096, dt)
        self.assertEqual(enc.tick, tick)
        enc.reset(np.full(7, 0.25 * tick))
        for offset, angle, delta in [(1.75, 1, 1), (-0.25, -1, -2), (-0.5, -1, 0)]:
            reading = enc.read(np.full(7, offset * tick))
            np.testing.assert_array_equal(reading["arm_angle"], np.full(7, angle * tick))
            np.testing.assert_array_equal(reading["arm_speed"], np.full(7, delta * tick / dt))

    def test_kinova_counts_match_nat(self):
        native = v2.sensors.ArmEncoders(dt=0.001)
        custom = v2.ArmEncodersAt(v2.sensors.ARM_COUNTS_PER_REV, 0.001)
        q = np.linspace(-0.5, 0.5, 7)
        native.reset(q)
        custom.reset(q)
        for shift in (1e-6, 2e-5, -3e-4):
            expected, actual = native.read(q + shift), custom.read(q + shift)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key])

    def test_build_modes(self):
        with self.assertRaises(ValueError):
            v2.V2Policy.build(ARM_CHECKPOINT, "expert", arm_inputs="unknown")
        policy = v2.V2Policy.build(ARM_CHECKPOINT, "expert", arm_inputs="encoder", arm_counts=4096)
        self.assertEqual((policy.arm_inputs, policy.arm_counts), ("encoder", 4096))


class ResumeTests(unittest.TestCase):
    def test_finished_rows_keeps_same_code_rows_once_and_skips_torn_lines(self):
        import json
        import tempfile
        from v2_eval import finished_rows
        lines = [{"bearing": 12, "provenance_sha256": "new", "reached": False},
                 {"bearing": 12, "provenance_sha256": "new", "reached": True},       # later row wins
                 {"bearing": 24, "provenance_sha256": "old"},                         # other code: dropped
                 {"bearing": 36, "provenance_sha256": "new", "error": "timeout"},     # error: dropped
                 {"bearing": 60}]                                                     # before stamping
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "C.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in lines) + '{"bearing": 7', encoding="utf-8")
            rows, dropped = finished_rows(path, "new")
            self.assertEqual([(r["bearing"], r["reached"]) for r in rows], [(12, True)])
            self.assertEqual(dropped, 4)
            rows, _ = finished_rows(path, "new", keep_unstamped=True)
            self.assertEqual(sorted(r["bearing"] for r in rows), [12, 60])
            self.assertEqual(next(r for r in rows if r["bearing"] == 60)["provenance"], "unverified")

    def test_rows_from_edited_files_are_marked_and_never_kept(self):
        import json
        import tempfile
        from v2_eval import finished_rows, stamp
        same = stamp({"bearing": 12, "code_digest": "new"}, "new")
        other = stamp({"bearing": 24, "code_digest": "edited"}, "new")
        failed = stamp({"bearing": 36, "error": "timeout"}, "new")
        self.assertEqual(same["provenance_sha256"], "new")
        self.assertEqual(other["provenance"], "mismatch")
        self.assertNotIn("provenance_sha256", other)
        self.assertNotIn("provenance", failed)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "C.jsonl"
            path.write_text("".join(json.dumps(r) + "\n" for r in (same, other, failed)), encoding="utf-8")
            rows, dropped = finished_rows(path, "new", keep_unstamped=True)
            self.assertEqual([r["bearing"] for r in rows], [12])
            self.assertEqual(dropped, 2)


class BlankScene:
    """Camera stand-in that never sees the dot (no rendering)."""

    def pose(self, *args):
        pass

    def fisheye(self, eye, axes):
        return np.zeros((256, 256, 3), np.uint8), np.full((256, 256), np.inf), None


class ParkAfter(v2.NatDriver):
    """Driving stand-in: full speed ahead, PARK from the given decision on."""

    def __init__(self, decisions):
        self.decisions, self.calls = decisions, 0

    def decide(self, wheel_x, dot_xy, visible):
        self.calls += 1
        return np.ones(4), self.calls > self.decisions


@unittest.skipUnless(ARM_CHECKPOINT.exists(), "needs the final arm checkpoint")
class V2PolicyTests(unittest.TestCase):
    def test_decision_steps_park_latch_and_hold_without_a_dot(self):
        from sim import Robot
        target = np.array([1.0, 0.2, 0.5])
        robot = Robot(target)
        policy = v2.V2Policy.build(ARM_CHECKPOINT, ParkAfter(3))
        policy.scene = BlankScene()
        decided, decide = [], policy._decide
        policy._decide = lambda obs, t: (decided.append(policy.step), decide(obs, t))
        policy.reset(robot, target, robot.observe())
        out = []
        for _ in range(200):
            wheels, torque, active = policy.act(robot.observe())
            robot.step(wheels, torque)
            out.append((wheels, active))
        self.assertEqual(decided, [0, 33, 67, 100, 133, 167])
        first = next(i for i, (_, active) in enumerate(out) if active)
        self.assertEqual(first, 100)                     # the fourth decision parks
        self.assertTrue(all(active and not wheels.any() for wheels, active in out[first:]))
        self.assertTrue(np.all(out[first - 1][0] == 1.0))
        self.assertTrue(policy.report["dot_never_seen"])  # never seen: the arm holds and nothing crashes


if __name__ == "__main__":
    unittest.main()
