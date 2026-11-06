# =============================================================================
# input_nets.py
# This is for: the networks that replace analytical calculations in the inputs. train_input_nets.py and
#              test_drive_net.py both import this file, so training and testing build the inputs the same way.
# How to run:  you do not run this file.
# Result:      nothing on its own. It only holds shared code and settings.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   X21     one row of a run, columns 0 to 20 (see INPUT LAYOUT in features.py).
#   tip     the end of the gripper, seen from the robot (m).
# -----------------------------------------------------------------------------

import numpy as np
import torch.nn as nn

# SETTING: neurons in each hidden layer
HIDDEN = 256

# Where these networks are saved. Kept apart from the other models, so nothing is overwritten.
MODEL_FOLDER = "input_net_model"


def make_inputs(X21):
    """The 7 inputs: base x, y, sin and cos of the heading, target x, y, z.
    Input: X21, shape (rows, 21). Output: shape (rows, 7)."""
    X21 = np.asarray(X21, dtype=np.float64)
    return np.c_[X21[:, :2], np.sin(X21[:, 3]), np.cos(X21[:, 3]), X21[:, 18:21]].astype(np.float32)


class InputNet(nn.Module):
    """7 inputs in, n_out numbers out (scaled)."""
    def __init__(self, n_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                    nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_out))

    def forward(self, x):
        return self.layers(x)


def tip_from_net(weights, angle):
    """The tip's position in the robot's frame, from the 7 arm angles, using the saved tip network (NumPy file)."""
    out = (np.r_[angle, np.sin(angle), np.cos(angle)] - weights["x_mean"]) / weights["x_std"]
    for layer in ("0", "2", "4"):
        out = np.maximum(out @ weights[f"{layer}.weight"].T + weights[f"{layer}.bias"], 0)
    return (out @ weights["6.weight"].T + weights["6.bias"]) * weights["y_std"] + weights["y_mean"]
