"""Behaviour-cloning policy with an integrating memory, so the arm can remove its own steady-state error.

Why: the expert arm integrates a joint reference q_des from the end-effector position error, so any torque bias is
cancelled over time. A memoryless network has to output exact torques and settles 1-6 cm off.
This policy keeps the same kind of memory, all networks learned from expert data:

    wheels   = base_net(x)                                  x = the 34 features from features.py
    p_reach  = phase_net(x[PHASE_COLS])                     switch after 50 steps above 0.5, never back
    v        = vel_net(x)          while reaching, else 0   learned joint-velocity reference (7)
    q_ref   <- clip(q_ref + v*dt, q +- 0.15, joint limits)  the memory; q_ref = start pose while driving
    torque   = torque_net(x, q_ref - q, v - qd, phase)      learned joint torques (7)

Training is teacher-forced on the expert's recorded q_des/qd_des (train_memory.py).
Inference uses NumPy only, so a rollout step costs microseconds.
"""
import json
from pathlib import Path

import numpy as np

from features import PHASE_COLS, add_ee, add_relative

MAX_Q_LEAD = 0.15      # same anti-windup bound as the expert's reference
LIMIT_MARGIN = 0.05    # keep the reference this far inside the joint limits
SWITCH_STEPS = 50      # same switch confirmation as drive.py
VEL_SCALE_MIN = 0.005  # m; below this end-effector position error the velocity scale stops shrinking
POSITION_ERROR_COL = 33      # features(): straight-line end-effector-to-goal distance


def vel_scale(x34):
    """vel_net predicts qd_des / vel_scale. Near the goal the expert velocity is tiny (it is
    proportional to the position error), so an absolute error would swamp it; scaling by the position error makes
    the velocity error shrink with the position error and the arm converge onto the goal."""
    return np.maximum(np.atleast_2d(x34)[:, POSITION_ERROR_COL], VEL_SCALE_MIN)


def features(state21, ee):
    """34 inputs from the 21 raw state values and the end-effector position (batch or single row)."""
    state21 = np.atleast_2d(state21)
    ee = np.atleast_2d(ee)
    return add_ee(add_relative(state21), ee)


def torque_inputs(x34, z, ev, phase):
    """Inputs of torque_net: 34 features, memory error z = q_ref - q, velocity error ev = v - qd, phase."""
    return np.concatenate((x34, z, ev, np.reshape(phase, (-1, 1))), axis=1)


class NumpyMLP:
    """ReLU MLP with standardised inputs and outputs, loaded from a torch state dict."""

    def __init__(self, layers, x_mean, x_std, y_mean, y_std, x_mask=None):
        self.layers = [(np.asarray(w, dtype=np.float64), np.asarray(b, dtype=np.float64)) for w, b in layers]
        self.x_mean, self.x_std = np.asarray(x_mean), np.asarray(x_std)
        self.y_mean, self.y_std = np.asarray(y_mean), np.asarray(y_std)
        self.x_mask = None if x_mask is None else np.asarray(x_mask)   # 0 = input removed (ablation)

    @classmethod
    def from_saved(cls, saved):
        state = saved["weights"]
        names = sorted({k.rsplit(".", 1)[0] for k in state}, key=lambda n: int(n.split(".")[-1]))
        layers = [(state[f"{n}.weight"].numpy(), state[f"{n}.bias"].numpy()) for n in names]
        return cls(layers, saved["x_mean"], saved["x_std"], saved["y_mean"], saved["y_std"],
                   saved.get("x_mask"))

    def __call__(self, x):
        h = (np.atleast_2d(x) - self.x_mean) / self.x_std
        if self.x_mask is not None:
            h = h * self.x_mask
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h * self.y_std + self.y_mean


class MemoryPolicy:
    """Rollout policy for sim.run_episode."""

    def __init__(self, nets, wheel_limits, torque_limits, scaled_vel=True, vel_every=1):
        self.scaled_vel = scaled_vel        # False only for checkpoints trained before vel scaling
        if vel_every < 1:
            raise ValueError("vel_every must be at least 1")
        self.vel_every = vel_every          # steps between vel_net updates (33 at 1 kHz = about 30 Hz)
        self.nets = nets                    # dict name -> list of NumpyMLP (an ensemble is averaged)
        self.wheel_limits = wheel_limits    # (min, max) of expert wheel commands in training data
        self.torque_limits = torque_limits

    @classmethod
    def load(cls, paths, vel_every=1):
        """One checkpoint, or several joined with '+' to average them as an ensemble."""
        import torch
        nets, wheel_limits, torque_limits, scaled = {}, None, None, set()
        for path in str(paths).split("+"):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            for name in ("base", "phase", "vel", "torque"):
                nets.setdefault(name, []).append(NumpyMLP.from_saved(saved[name]))
            scaled.add(bool(saved.get("vel_scaled", False)))
            # An ensemble uses the union of its members' command ranges.
            w_lo, w_hi = (np.asarray(v) for v in saved["wheel_limits"])
            t_lo, t_hi = (np.asarray(v) for v in saved["torque_limits"])
            if wheel_limits is not None:
                w_lo, w_hi = np.minimum(w_lo, wheel_limits[0]), np.maximum(w_hi, wheel_limits[1])
                t_lo, t_hi = np.minimum(t_lo, torque_limits[0]), np.maximum(t_hi, torque_limits[1])
            wheel_limits, torque_limits = (w_lo, w_hi), (t_lo, t_hi)
        if len(scaled) != 1:
            raise ValueError("Cannot ensemble checkpoints with different velocity scaling")
        return cls(nets, wheel_limits, torque_limits, scaled_vel=scaled.pop(), vel_every=vel_every)

    def _run(self, name, x):
        return np.mean([net(x) for net in self.nets[name]], axis=0)[0]

    def reset(self, robot, target, obs0):
        self.robot = robot
        self.target = np.asarray(target, dtype=float)
        self.q_ref = obs0["q"].copy()
        self.phase = 0.0
        self.count = 0
        self.reach_steps = 0
        self.v = np.zeros(7)
        self.p_reach = 0.0
        lower, upper = robot.arm_lower, robot.arm_upper
        self.has_limits = lower < upper
        self.lower, self.upper = lower + LIMIT_MARGIN, upper - LIMIT_MARGIN

    def act(self, obs):
        state = np.concatenate((obs["base_pos"], [obs["bearing"]], obs["q"], obs["qd"], self.target))
        x = features(state, obs["ee"])
        q, qd = obs["q"], obs["qd"]
        if self.phase == 0.0:
            self.p_reach = float(1.0 / (1.0 + np.exp(-self._run("phase", x[:, PHASE_COLS])[0])))
            self.count = self.count + 1 if self.p_reach > 0.5 else 0
            if self.count >= SWITCH_STEPS:
                self.phase = 1.0
        wheels = np.clip(self._run("base", x), *self.wheel_limits) if self.phase == 0.0 else np.zeros(4)
        return wheels, self._arm(x, q, qd), self.phase == 1.0

    def _arm(self, x, q, qd):
        """One arm step: update the memory (while reaching), then compute the joint torques."""
        self.q_ref_prev = self.q_ref.copy()   # memory before this step's update (for DAgger labels)
        if self.phase == 1.0:
            # The velocity is held between vel_net updates; q_ref and torque_net still run every step.
            if self.reach_steps % self.vel_every == 0:
                self.v = self._run("vel", x) * (vel_scale(x)[0] if self.scaled_vel else 1.0)
            self.reach_steps += 1
            q_ref = np.clip(self.q_ref + self.v * self.robot.dt, q - MAX_Q_LEAD, q + MAX_Q_LEAD)
            self.q_ref = np.where(self.has_limits, np.clip(q_ref, self.lower, self.upper), q_ref)
        else:
            self.v = np.zeros(7)
        u = torque_inputs(x, (self.q_ref - q)[None], (self.v - qd)[None], self.phase)
        return np.clip(self._run("torque", u), *self.torque_limits)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
