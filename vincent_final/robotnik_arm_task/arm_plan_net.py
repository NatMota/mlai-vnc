# =============================================================================
# arm_plan_net.py
# This is for: the arm plan network and its 9 inputs. train_arm_plan.py and
#              test_drive_net.py both import this file, so training and testing
#              always build the inputs in exactly the same way.
# How to run:  you do not run this file.
# Result:      nothing on its own. It only holds shared code and settings.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   plan       the joint speeds the expert wants the arm to have (qd_des, rad/s).
#   tip gap    target minus end-effector, seen from the robot: ahead, left, up (m).
#   X21        one row of a run, columns 0 to 20 (see INPUT LAYOUT in features.py).
# -----------------------------------------------------------------------------

import numpy as np
import torch.nn as nn
from features import add_relative, add_ee


# SETTING: decide every N simulation steps. 33 = about 30 decisions per second, as in drive_net.py.
DECIDE_EVERY = 33

# SETTING: neurons in each hidden layer
HIDDEN = 128

# Where the arm plan network is saved. Kept apart from the other models, so nothing is overwritten.
MODEL_FOLDER = "arm_plan_model"

# Columns after add_ee (see INPUT LAYOUT in features.py).
GAP_COLS = [31, 32, 30]                 # tip gap ahead, left, up
ANGLE_COLS = list(range(4, 10))         # joint angles 1 to 6. Joint 7 hardly moves, so it is left out.


def make_inputs(X21, ee):
    """The 9 inputs: 3 tip gap numbers and 6 joint angles.
    Input: X21, shape (rows, 21); ee, shape (rows, 3). Output: shape (rows, 9)."""
    return add_ee(add_relative(X21), ee)[:, GAP_COLS + ANGLE_COLS].astype(np.float32)


class ArmPlanNet(nn.Module):
    """9 inputs in, 7 joint speeds out (scaled). The expert's formula for the plan uses no joint speeds."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 7),
        )

    def forward(self, x):
        return self.layers(x)
