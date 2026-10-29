# =============================================================================
# features.py
# This is for: making the extra inputs for the model.
# Used by train.py and drive.py, so both always build the inputs the same way.
# You do not run this file. You only change the setting below.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   X21        a table with the 21 original inputs (one row per moment).
#   X28        the same table with 28 inputs (after add_relative).
#   dx, dy     gap from the base to the dot, along the floor (x and y, in metres).
#   front      how far the dot is in front of the robot (negative = behind).
#   left       how far the dot is to the robot's left (negative = right).
#   dist       straight-line distance on the floor.
#   angle      how much the robot must turn to face the dot (radians).
#   bearing    which way the robot faces (radians).
#   c, s       cos(bearing) and sin(bearing), used to turn the gap into the robot's view.
#   ee         end-effector position (x, y, z).
#   gap        dot position minus end-effector position.
#   sin, cos   a way to write an angle with no jump at 180 degrees.
#   [:, 18:21] columns 18, 19 and 20 of every row (the dot x, y, z).
# -----------------------------------------------------------------------------

import numpy as np


# =============================================================================
# SWITCHES (True = ON, False = OFF)
# After changing a switch, run train.py again, then test.py.
# =============================================================================

# SWITCH: end-effector inputs
#   ON  (True):  34 inputs. Adds the gap from the end-effector to the dot.
#                Needs add_ee.py to be run first.
#   OFF (False): 28 inputs.
USE_EE = True

# SWITCH: only good runs
#   ON  (True):  learn only from runs where the expert reached the dot (within 2 cm).
#                Needs add_ee.py to be run first.
#   OFF (False): learn from all runs, including the ones where the expert failed.
ONLY_GOOD_RUNS = True

# Names of the saved model files. Each setting keeps its own files, so nothing is overwritten.
MODEL_NAME = ("ee" if USE_EE else "phase_rel") + ("_good" if ONLY_GOOD_RUNS else "")


# =============================================================================
# INPUTS 22 to 28: "look at the ball" (where is the dot, seen from the robot?)
# This helps the model: it can steer the base towards the dot more easily.
# =============================================================================

N_INPUTS = 28   # 21 original inputs + 7 new ones


def add_relative(X21):
    """Takes the 21 original inputs. Returns 28 inputs."""
    X21 = np.asarray(X21, dtype=np.float64)
    base_xy = X21[:, 0:2]        # base x, y
    bearing = X21[:, 3]          # which way the base faces
    target = X21[:, 18:21]       # red dot x, y, z

    # Gap from the base to the dot, on the floor.
    dx = target[:, 0] - base_xy[:, 0]
    dy = target[:, 1] - base_xy[:, 1]

    # Same gap, seen from the robot: how far in front, and how far to the left.
    c, s = np.cos(bearing), np.sin(bearing)
    front = c * dx + s * dy
    left = -s * dx + c * dy

    dist = np.hypot(dx, dy)              # distance on the floor
    angle = np.arctan2(left, front)      # how much to turn to face the dot

    # sin and cos are used for angles, so 179 and -179 degrees look close (as they should).
    new = np.stack([front, left, dist,
                    np.sin(angle), np.cos(angle),
                    np.sin(bearing), np.cos(bearing)], axis=1)
    return np.concatenate([X21, new], axis=1)


# =============================================================================
# Which inputs phase_net may see: only the base and the dot, NOT the arm.
# This helps the model: it decides "reach now" from "am I parked?",
# not from "is my arm moving?" (the arm only moves AFTER the switch).
# =============================================================================

PHASE_COLS = [0, 1, 2, 3, 18, 19, 20] + list(range(21, 28))   # 14 inputs


# =============================================================================
# INPUTS 29 to 34: end-effector to dot (only used when USE_EE = True)
# This helps the model: the arm can see how far the end-effector is from the dot.
# =============================================================================

N_INPUTS_EE = N_INPUTS + 6   # 34


def add_ee(X28, ee):
    """Takes the 28 inputs and the end-effector x, y, z. Returns 34 inputs."""
    X28 = np.asarray(X28, dtype=np.float64)
    ee = np.asarray(ee, dtype=np.float64)
    target = X28[:, 18:21]
    bearing = X28[:, 3]

    gap = target - ee                            # gap in world directions (x, y, z)
    c, s = np.cos(bearing), np.sin(bearing)
    gap_front = c * gap[:, 0] + s * gap[:, 1]    # same gap, seen from the robot
    gap_left = -s * gap[:, 0] + c * gap[:, 1]
    gap_dist = np.linalg.norm(gap, axis=1)       # straight-line distance to the dot

    new = np.stack([gap[:, 0], gap[:, 1], gap[:, 2], gap_front, gap_left, gap_dist], axis=1)
    return np.concatenate([X28, new], axis=1)
