"""Combine the teammate's base and switching networks with the memory arm policy.

    team      teammate's model unchanged: base, switch and arm (MLflow run f978e343)
    fused     teammate's base and switch + memory arm
    fused30   as fused, with base, switch and vel_net at about 30 Hz and torque_net at 1 kHz

Each side keeps its own input scaling. The teammate's networks are re-implemented in NumPy from their
checkpoint, with the same order of operations as their drive.py.
"""
import math
from pathlib import Path

import numpy as np

from memory_policy import MemoryPolicy, SWITCH_STEPS, features
from features import PHASE_COLS

ROOT = Path(__file__).resolve().parent
TEAM_DIR = ROOT / "outputs/teammate_f978e343/model"
HIGH_LEVEL_STEPS = 33        # 33 steps at 1 kHz = about 30 Hz


def _dense(state, prefix, x, last=False):
    y = x @ state[f"{prefix}.weight"].T + state[f"{prefix}.bias"]
    return y if last else np.maximum(y, 0.0)


class TeamNets:
    """phase_net and SplitControlNet from the teammate's checkpoint, float32 like their torch model."""

    def __init__(self, folder=TEAM_DIR):
        import torch
        saved = torch.load(Path(folder) / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.phase_w = {k: v.numpy() for k, v in saved["phase_net"].items()}
        self.ctrl = {k: v.numpy() for k, v in saved["control_net"].items()}
        scaler = np.load(Path(folder) / "scaler.npz")
        self.x_mean, self.x_std = scaler["X_mean"], scaler["X_std"]
        self.y_mean, self.y_std = scaler["Y_mean"], scaler["Y_std"]
        limits = np.load(Path(folder) / "limits.npz")
        self.y_min, self.y_max = limits["y_min"], limits["y_max"]

    def scale(self, x):
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def p_reach(self, x_s):
        h = _dense(self.phase_w, "0", x_s[:, PHASE_COLS])
        h = _dense(self.phase_w, "2", h)
        logit = float(_dense(self.phase_w, "4", h, last=True)[0, 0])
        return 1.0 / (1.0 + math.exp(-logit))

    def commands(self, x_s, phase):
        """The 11 commands (4 wheels, 7 arm torques) in real units, clipped like their drive.py."""
        u = np.concatenate((x_s, np.array([[phase]], np.float32)), axis=1)
        shared = _dense(self.ctrl, "body.2", _dense(self.ctrl, "body.0", u))
        wheels = _dense(self.ctrl, "base_end.2", _dense(self.ctrl, "base_end.0", shared), last=True)
        arm = _dense(self.ctrl, "arm_end.2", _dense(self.ctrl, "arm_end.0", shared), last=True)
        y = np.concatenate((wheels, arm), axis=1)[0] * self.y_std + self.y_mean
        return np.clip(y, self.y_min, self.y_max)


class FusedPolicy(MemoryPolicy):
    """Teammate's base and switch; arm from the teammate (team_arm=True) or the memory policy."""

    def configure(self, team, team_arm=False, high_every=1):
        self.team, self.team_arm, self.high_every = team, team_arm, high_every
        # 50 confirming steps at 1 kHz; at about 30 Hz this rounds up to two updates, 33 ms apart.
        self.confirm = max(1, math.ceil(SWITCH_STEPS / high_every))
        return self

    def reset(self, robot, target, obs0):
        super().reset(robot, target, obs0)
        self.step_no = 0
        self.wheels = np.zeros(4)

    def act(self, obs):
        state = np.concatenate((obs["base_pos"], [obs["bearing"]], obs["q"], obs["qd"], self.target))
        x = features(state, obs["ee"])
        x_s = self.team.scale(x)
        update_high = self.step_no % self.high_every == 0
        self.step_no += 1
        if self.phase == 0.0 and update_high:
            self.p_reach = self.team.p_reach(x_s)
            self.count = self.count + 1 if self.p_reach > 0.5 else 0
            if self.count >= self.confirm:
                self.phase = 1.0
        need_team_output = self.team_arm or (update_high and self.phase == 0.0)
        y = self.team.commands(x_s, self.phase) if need_team_output else None
        if self.phase == 1.0:
            self.wheels = np.zeros(4)
        elif y is not None and update_high:
            self.wheels = y[:4]
        torque = y[4:] if self.team_arm else self._arm(x, obs["q"], obs["qd"])
        return self.wheels, torque, self.phase == 1.0


def make_fused(kind, arm_checkpoint):
    """kind: team, fused or fused30. arm_checkpoint: memory policy file (its base/phase are not used)."""
    high_every = HIGH_LEVEL_STEPS if kind == "fused30" else 1
    policy = FusedPolicy.load(arm_checkpoint, vel_every=high_every)
    return policy.configure(TeamNets(), team_arm=kind == "team", high_every=high_every)
