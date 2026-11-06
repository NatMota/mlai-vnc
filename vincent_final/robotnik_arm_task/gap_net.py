# =============================================================================
# gap_net.py
# This is for: the angle gap network and its 23 inputs. train_gap_net.py and test_drive_net.py both import
#              this file, so training and testing always build the inputs in exactly the same way.
# How to run:  you do not run this file.
# Result:      nothing on its own. It only holds shared code and settings.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   tip gap    target minus end-effector, seen from the robot: ahead, left, up (m).
#   plan       the joint speeds the plan network wants the arm to have (rad/s).
#   angle gap  how far each joint should be from the plan (rad). This is the network's output.
#   window     a stretch of recent time. The normal version only reads the current moment.
# -----------------------------------------------------------------------------

import numpy as np
import torch.nn as nn

# SETTING: neurons in each hidden layer
HIDDEN = 128

# Where this network is saved. Kept apart from the other models, so nothing is overwritten.
MODEL_FOLDER = "arm_gap_model"

# SETTING: the stretch of time for the window version: WINDOW_STEPS past moments, WINDOW_MS apart (5 x 20 ms = the last 100 ms).
WINDOW_STEPS = 5
WINDOW_MS = 20


def make_inputs(tip_gap, q, qd, qd_plan):
    """The 23 inputs: 3 tip gap numbers, 6 joint angles, 7 joint speeds, 7 planned speeds.
    Input: 4 arrays with shapes (rows, 3), (rows, 7), (rows, 7), (rows, 7). Output: shape (rows, 23)."""
    return np.concatenate((tip_gap, q[:, :6], qd, qd_plan), axis=1).astype(np.float32)


def window_inputs(q_now, q_past, plan_past):
    """The extra inputs for the window version: how far each joint moved since each past moment, and the plan at that moment.
    Input: q_now (rows, 7); q_past and plan_past (rows, WINDOW_STEPS, 7). Output: shape (rows, 2 x WINDOW_STEPS x 7)."""
    moved = q_now[:, None, :] - q_past
    return np.concatenate((moved.reshape(len(q_now), -1), plan_past.reshape(len(q_now), -1)), axis=1).astype(np.float32)


class GapNet(nn.Module):
    """n_in inputs in (23, or 93 for the window version), 7 angle gaps out (scaled)."""
    def __init__(self, n_in=23):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 7))

    def forward(self, x):
        return self.layers(x)
