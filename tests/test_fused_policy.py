"""Contracts for the fused policy and the goal-error wrapper (no simulator)."""
from types import SimpleNamespace
import unittest

import numpy as np

from fused_policy import FusedPolicy
from parallel_eval import FRESH_FIRST_SEED, GoalError, target_for
from pool import TARGET_HIGH, TARGET_LOW


class FakeTeam:
    """Counts calls and returns fixed outputs in place of the teammate's networks."""

    def __init__(self, p_reach):
        self.p, self.calls = p_reach, {"phase": 0, "commands": 0}

    def scale(self, x):
        return x.astype(np.float32)

    def p_reach(self, x_s):
        self.calls["phase"] += 1
        return self.p

    def commands(self, x_s, phase):
        self.calls["commands"] += 1
        return np.arange(11, dtype=float)


class Const:
    def __init__(self, value):
        self.value = np.atleast_2d(value)

    def __call__(self, x):
        return np.repeat(self.value, len(np.atleast_2d(x)), axis=0)


def make(p_reach, high_every=1, team_arm=False):
    nets = {"vel": [Const(np.full(7, 0.5))], "torque": [Const(np.full(7, 2.0))]}
    limits = (np.full(7, -30.0), np.full(7, 30.0))
    policy = FusedPolicy(nets, (np.full(4, -5.0), np.full(4, 5.0)), limits, vel_every=high_every)
    team = FakeTeam(p_reach)
    policy.configure(team, team_arm=team_arm, high_every=high_every)
    robot = SimpleNamespace(dt=0.001, arm_lower=np.full(7, -10.0), arm_upper=np.full(7, 10.0))
    obs = {"base_pos": np.zeros(3), "bearing": 0.0, "q": np.zeros(7), "qd": np.zeros(7), "ee": np.zeros(3)}
    policy.reset(robot, np.array([1.0, 0.0, 0.5]), obs)
    return policy, team, obs


class FusedPolicyTests(unittest.TestCase):
    def test_uses_team_wheels_and_memory_arm(self):
        policy, _, obs = make(p_reach=0.0)
        wheels, torque, active = policy.act(obs)
        np.testing.assert_array_equal(wheels, [0, 1, 2, 3])
        np.testing.assert_array_equal(torque, np.full(7, 2.0))
        self.assertFalse(active)

    def test_team_arm_uses_team_torques(self):
        policy, _, obs = make(p_reach=0.0, team_arm=True)
        np.testing.assert_array_equal(policy.act(obs)[1], [4, 5, 6, 7, 8, 9, 10])

    def test_switch_needs_two_consecutive_updates_at_30_hz(self):
        policy, team, obs = make(p_reach=1.0, high_every=33)
        for _ in range(33):
            self.assertFalse(policy.act(obs)[2])
        self.assertTrue(policy.act(obs)[2])              # second update, 33 ms after the first
        self.assertEqual(team.calls["phase"], 2)

    def test_wheels_stop_on_switch_and_high_level_is_held_between_updates(self):
        policy, team, obs = make(p_reach=0.0, high_every=33)
        for _ in range(33):
            policy.act(obs)
        self.assertEqual(team.calls["commands"], 1)
        policy.team.p = 1.0
        for _ in range(34):
            wheels, _, active = policy.act(obs)
        self.assertTrue(active)
        np.testing.assert_array_equal(wheels, np.zeros(4))


class GoalErrorTests(unittest.TestCase):
    def test_offset_has_the_requested_length_and_is_repeatable(self):
        a = GoalError(SimpleNamespace(act=None), 0.005, seed=3)
        b = GoalError(SimpleNamespace(act=None), 0.005, seed=3)
        self.assertAlmostEqual(np.linalg.norm(a.offset), 0.005)
        np.testing.assert_array_equal(a.offset, b.offset)

    def test_policy_sees_the_shifted_goal(self):
        seen = {}
        inner = SimpleNamespace(act=None, reset=lambda robot, target, obs: seen.update(target=target))
        wrapper = GoalError(inner, 0.01, seed=1)
        wrapper.reset(None, np.zeros(3), None)
        np.testing.assert_array_equal(seen["target"], wrapper.offset)


class FreshGoalTests(unittest.TestCase):
    def test_fresh_goals_are_repeatable_different_and_inside_the_box(self):
        first, second = target_for("fresh", 0), target_for("fresh", 1)
        self.assertEqual(first, target_for("fresh", 0))
        self.assertNotEqual(first, second)
        self.assertTrue(np.all(np.array(first) >= TARGET_LOW) and np.all(np.array(first) <= TARGET_HIGH))

    def test_fresh_seeds_stay_above_the_collection_and_dagger_seeds(self):
        self.assertGreater(FRESH_FIRST_SEED, 300399)


if __name__ == "__main__":
    unittest.main()
