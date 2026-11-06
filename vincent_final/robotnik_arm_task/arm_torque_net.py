# =============================================================================
# arm_torque_net.py
# This is for: the torque network and its 20 inputs. train_arm_torque.py and
#              test_drive_net.py both import this file, so training and testing
#              always build the inputs in exactly the same way.
# How to run:  you do not run this file.
# Result:      nothing on its own. It only holds shared code and settings.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   q, qd       measured joint angles (rad) and joint speeds (rad/s), 7 of each.
#   q_des       the planned joint angles: the plan's speeds, added up over time.
#   qd_des      the planned joint speeds, from the arm plan network.
#   angle gap   q_des - q. How far each joint is from the plan.
#   speed gap   qd_des - qd. How far each joint's speed is from the plan.
# -----------------------------------------------------------------------------

import numpy as np
import torch.nn as nn


# SETTING: neurons in each hidden layer
HIDDEN = 128

# Where the torque network is saved. Kept apart from the other models, so nothing is overwritten.
MODEL_FOLDER = "arm_torque_model"


def make_inputs(q_des, qd_des, q, qd):
    """The 20 numbers: 7 angle gaps, 7 speed gaps and joint angles 1 to 6 (for gravity).
    Joint 7's angle hardly moves in the data (spread 0.002 rad), so it is left out of the gravity part.
    Input: 4 arrays, each shape (rows, 7). Output: shape (rows, 20)."""
    return np.concatenate((q_des - q, qd_des - qd, q[:, :6]), axis=1).astype(np.float32)


class ArmTorqueNet(nn.Module):
    """20 numbers in, 7 arm torques out (scaled). It runs at every simulation step (1000 per second)."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 7),
        )

    def forward(self, x):
        return self.layers(x)
