# =============================================================================
# drive_net.py
# This is for: the driving network and its 4 inputs. train_drive_net.py and
#              test_drive_net.py both import this file, so training and testing
#              always build the inputs in exactly the same way.
# How to run:  you do not run this file.
# Result:      nothing on its own. It only holds shared code and settings.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   parking spot  where the expert parks the base before it starts reaching.
#   stop decision the network's answer to "is the robot parked yet?". Above 0 = yes.
#   decision      one time the network is asked for commands (30 times per second).
#   X21           one row of a run, columns 0 to 20 (see INPUT LAYOUT in features.py).
# -----------------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn as nn
from features import add_park, add_imu


# SETTING: decide every N simulation steps
#   The simulation runs 1000 steps per second. 33 = about 30 decisions per second.
#   Training uses every 33rd row, and the test holds the wheel commands in between.
DECIDE_EVERY = 33

# SETTING: neurons in each hidden layer
HIDDEN = 64

# Where the driving network is saved. Kept apart from the other models, so nothing is overwritten.
MODEL_FOLDER = "drive_net_model"

# SWITCH: IMU inputs for the driving network
#   ON  (True):  2 more inputs: forward speed (m/s) and turning rate (rad/s), so the network can see
#                how fast the base is moving. Needs the data_v2 folders. Adds "_imu" to the model name.
#   OFF (False): only the 4 parking inputs. This is the default, and the reported models use it.
#   Type it before the command to turn it ON, for example: ROBOENV_USE_IMU=1 python3 test_drive_net.py 20 val
USE_IMU = os.environ.get("ROBOENV_USE_IMU", "0") == "1"
MODEL_TAG = "_imu" if USE_IMU else ""
N_INPUTS = 6 if USE_IMU else 4
IMU_KEEP = [0, 2]   # from add_imu's 5 features: forward speed, turning rate


def make_inputs(X21, quat=None, lin_vel=None, ang_vel=None):
    """The inputs: parking spot ahead (m), parking spot left (m), sin, cos of the turn still needed
    at the spot, and with USE_IMU ON, forward speed and turning rate.
    Input: X21, shape (rows, 21); with USE_IMU ON also quat (rows, 4), lin_vel (rows, 3), ang_vel (rows, 3).
    Output: shape (rows, N_INPUTS)."""
    x = add_park(X21)[:, -4:]
    if USE_IMU:
        imu = add_imu(x, X21[:, 3], quat, lin_vel, ang_vel)[:, -5:]
        x = np.concatenate((x, imu[:, IMU_KEEP]), axis=1)
    return x.astype(np.float32)


class DriveNet(nn.Module):
    """N_INPUTS in. 3 numbers out: right wheels (rad/s, scaled), left wheels (rad/s, scaled),
    stop decision. The expert always sends right, left, left, right, so 2 wheel numbers are enough."""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(N_INPUTS, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 3),
        )

    def forward(self, x):
        return self.layers(x)
