"""Contracts for the memory policy, its training tables and the hidden-state helpers (no simulator)."""
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from chester_hidden_state import reach_start
from chester_memory_policy import MAX_Q_LEAD, MemoryPolicy, NumpyMLP, features, torque_inputs
from chester_train_memory import mlp, tables


class Const:
    def __init__(self, value):
        self.value = np.atleast_2d(value)

    def __call__(self, x):
        return np.repeat(self.value, len(np.atleast_2d(x)), axis=0)


def make_policy(vel, phase_logit, scaled_vel=False):
    nets = {"base": [Const(np.ones(4))], "phase": [Const([phase_logit])],
            "vel": [Const(vel)], "torque": [Const(np.zeros(7))]}
    robot = SimpleNamespace(dt=0.001, arm_lower=np.full(7, -10.0), arm_upper=np.full(7, 10.0))
    policy = MemoryPolicy(nets, (np.full(4, -5.0), np.full(4, 5.0)), (np.full(7, -30.0), np.full(7, 30.0)),
                          scaled_vel=scaled_vel)
    obs = {"base_pos": np.zeros(3), "bearing": 0.0, "q": np.zeros(7), "qd": np.zeros(7), "ee": np.zeros(3)}
    policy.reset(robot, np.array([1.0, 0.0, 0.5]), obs)
    return policy, obs


class MemoryPolicyTests(unittest.TestCase):
    def test_numpy_mlp_matches_torch(self):
        rng = np.random.default_rng(0)
        torch.manual_seed(0)
        net = mlp(5, 3, 16, 3)
        saved = {"weights": {k: v.detach().clone() for k, v in net.state_dict().items()},
                 "x_mean": rng.normal(size=5).astype(np.float32), "x_std": np.full(5, 2.0, np.float32),
                 "y_mean": rng.normal(size=3).astype(np.float32), "y_std": np.full(3, 3.0, np.float32)}
        x = rng.normal(size=(4, 5)).astype(np.float32)
        with torch.no_grad():
            expected = net(torch.from_numpy((x - saved["x_mean"]) / saved["x_std"])).numpy()
        expected = expected * saved["y_std"] + saved["y_mean"]
        np.testing.assert_allclose(NumpyMLP.from_saved(saved)(x), expected, rtol=1e-5, atol=1e-5)

    def test_features_and_torque_inputs_shapes(self):
        state = np.zeros((3, 21))
        state[:, 18:21] = [1.0, 0.5, 0.6]
        x = features(state, np.zeros((3, 3)))
        self.assertEqual(x.shape, (3, 34))
        u = torque_inputs(x, np.zeros((3, 7)), np.ones((3, 7)), np.array([0.0, 1.0, 1.0]))
        self.assertEqual(u.shape, (3, 49))
        self.assertEqual(list(u[:, -1]), [0.0, 1.0, 1.0])

    def test_reach_start_is_first_row_after_last_wheel_command(self):
        labels = np.zeros((6, 11))
        labels[[0, 2], 0] = 1.0
        self.assertEqual(reach_start(labels), 3)
        self.assertEqual(reach_start(np.zeros((4, 11))), 0)

    def test_driving_keeps_start_pose_and_uses_wheels(self):
        policy, obs = make_policy(np.full(7, 100.0), phase_logit=-5.0)
        for _ in range(100):
            wheels, _, active = policy.act(obs)
        self.assertFalse(active)
        np.testing.assert_array_equal(wheels, np.ones(4))
        np.testing.assert_array_equal(policy.q_ref, np.zeros(7))

    def test_switches_after_50_steps_then_memory_is_clipped(self):
        policy, obs = make_policy(np.full(7, 100.0), phase_logit=5.0)
        for _ in range(49):
            self.assertFalse(policy.act(obs)[2])
        wheels, _, active = policy.act(obs)
        self.assertTrue(active)
        np.testing.assert_array_equal(wheels, np.zeros(4))
        np.testing.assert_allclose(policy.q_ref, np.full(7, 0.1))          # 100 rad/s * 1 ms
        policy.act(obs)
        np.testing.assert_allclose(policy.q_ref, np.full(7, MAX_Q_LEAD))   # 0.2 clipped to q + 0.15

    def test_memory_integrates_small_velocity_exactly(self):
        policy, obs = make_policy(np.full(7, 0.5), phase_logit=5.0)
        for _ in range(50 + 9):
            policy.act(obs)
        np.testing.assert_allclose(policy.q_ref, np.full(7, 0.5 * 0.001 * 10))

    def test_scaled_velocity_is_multiplied_by_the_gap(self):
        policy, obs = make_policy(np.full(7, 2.0), phase_logit=5.0, scaled_vel=True)
        obs["ee"] = np.array([1.0, 0.0, 0.3])        # 0.2 m below the dot at [1, 0, 0.5]
        for _ in range(50):
            policy.act(obs)
        np.testing.assert_allclose(policy.v, np.full(7, 2.0 * 0.2))

    def test_tables_route_rows_to_the_right_networks(self):
        run = {"x": np.zeros((4, 34), np.float32), "reach": np.array([0, 0, 1, 1], np.float32),
               "phase": np.array([0, 1, 1, 1], np.float32),
               "wheels": np.ones((4, 4), np.float32), "torque": np.zeros((4, 7), np.float32),
               "z": np.zeros((4, 7), np.float32), "ev": np.zeros((4, 7), np.float32),
               "qd_des": np.ones((4, 7), np.float32), "base_valid": np.array([1, 0, 1, 1], np.float32),
               "phase_valid": np.array([1, 1, 0, 1], np.float32)}
        run["x"][:, 33] = [0.5, 0.5, 0.2, 0.001]      # end-effector gap distance
        t = tables([run])
        self.assertEqual(len(t["base"][0]), 1)
        self.assertEqual(len(t["vel"][0]), 2)
        self.assertEqual(t["torque"][0].shape, (4, 49))
        self.assertEqual(t["phase"][0].shape, (3, 14))
        self.assertEqual(t["phase"][1][:, 0].tolist(), [0, 1, 1])   # DAgger row (phase_valid 0) dropped
        np.testing.assert_allclose(t["vel"][1][:, 0], [1 / 0.2, 1 / 0.005])  # gap-scaled, floor 5 mm


class BaseShadowTests(unittest.TestCase):
    def setUp(self):
        from chester_dagger import BaseShadow
        self.shadow = BaseShadow(np.array([1.5, 0.0, 0.5]))      # goal (0.85, 0), facing 0 rad

    def test_parked_pose_gives_done_and_zero_wheels(self):
        wheels, done = self.shadow.step(np.array([0.85, 0.0, 0.0]), 0.0, False, 0.001)
        self.assertTrue(done)
        np.testing.assert_array_equal(wheels, np.zeros(4))

    def test_done_restarts_when_heading_drifts(self):
        self.shadow.step(np.array([0.85, 0.0, 0.0]), 0.0, False, 0.001)
        wheels, done = self.shadow.step(np.array([0.85, 0.0, 0.0]), 0.17, False, 0.001)   # ~10 degrees off
        self.assertFalse(done)
        self.assertGreater(np.abs(wheels).max(), 0.5)            # it turns back instead of "stop here"

    def test_near_the_spot_always_turns_to_final_bearing(self):
        # 9 cm behind the spot, facing 10 degrees off: the label turns toward the final bearing (0 rad).
        wheels, done = self.shadow.step(np.array([0.76, 0.0, 0.0]), 0.17, False, 0.001)
        self.assertFalse(done)
        right_minus_left = wheels[0] - wheels[1]
        self.assertLess(right_minus_left, 0)                   # clockwise, back toward 0 rad

    def test_no_restart_while_learner_is_reaching(self):
        self.shadow.step(np.array([0.85, 0.0, 0.0]), 0.0, False, 0.001)
        _, done = self.shadow.step(np.array([0.85, 0.0, 0.0]), 0.17, True, 0.001)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
