# =============================================================================
# features.py
# This is for: the input features, settings (switches) and control network of the earlier single model.
#              train.py and drive.py both import this file, so training and driving always build the
#              inputs in exactly the same way.
#              The three-network model also uses some of its calculations: add_park (the parking inputs),
#              add_relative and add_ee (the target and tip seen from the robot), and the stand-off points.
# How to run:  you do not run this file. For the earlier single model, change the switches below,
#              then run train.py and test.py.
# Result:      nothing on its own. It only holds shared code and settings.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   world frame   fixed x, y, z axes of the simulation (metres). Positions saved in the data use it.
#   robot frame   axes that move with the robot: x = straight ahead, y = to its left.
#   bearing       the robot's heading (yaw angle) in the world frame, in radians.
#   target        the point the end-effector must reach (x, y, z, world frame, metres).
#   end-effector  the tip of the arm (ee for short).
#   feature       one input number given to the network.
#   X             input table, shape (rows, features). One row per time step (1 ms).
#   X21, X28      the same table with 21 or 28 features.
#   sin, cos      an angle is given as sin and cos, so +179 and -179 degrees are close
#                 numbers (as they are in reality), with no jump.
#   phase         0 = driving (base moves), 1 = reaching (arm moves, base stays still).
#   phase_net     small network that predicts the phase.
#   control_net   network that predicts the 11 commands: 4 wheel speeds + 7 arm torques.
#   head          the last layers of a network that produce one group of outputs.
#                 With SPLIT_HEADS ON, control_net has a wheel head and an arm head.
#   body          the layers shared by both heads.
#   IMU           inertial measurement unit: senses the base's speed, turning rate and tilt.
#   roll, pitch   tilt of the base sideways (roll) and forwards/backwards (pitch), radians.
#   quaternion    a 4-number way to store a 3D orientation (x, y, z, w).
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# INPUT LAYOUT (USE_EE ON, USE_IMU OFF = 34 features)
#   Index   Feature                                              Unit     Source
#   0-2     base position x, y, z (world frame)                  m        measured
#   3       base bearing                                         rad      measured
#   4-10    arm joint angles q, joints 1-7                       rad      measured
#   11-17   arm joint speeds qd, joints 1-7                      rad/s    measured
#   18-20   target position x, y, z (world frame)                m        given
#   21-22   target in the robot frame: ahead, left               m        add_relative()
#   23      distance from base to target on the floor            m        add_relative()
#   24-25   sin, cos of the turn needed to face the target       -        add_relative()
#   26-27   sin, cos of the bearing                              -        add_relative()
#   28-30   target minus end-effector, x, y, z (world frame)     m        add_ee()
#   31-32   that gap in the robot frame: ahead, left             m        add_ee()
#   33      that gap as a straight-line distance                 m        add_ee()
#   34-38   (USE_IMU ON) forward speed, sideways speed (m/s),
#           turning rate (rad/s), roll, pitch (rad)                       add_imu()
#   control_net also receives the phase as one extra input (the last column).
#   Every feature is standardised in train.py (mean 0, standard deviation 1).
# -----------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# SWITCHES (True = ON, False = OFF)
# After changing a switch, run train.py again, then test.py.
# =============================================================================

# SWITCH: end-effector inputs
#   ON  (True):  34 features. Adds the gap from the end-effector to the target (add_ee).
#                Needs add_ee.py to be run on the data folder first.
#   OFF (False): 28 features.
USE_EE = True

# SWITCH: only good runs
#   ON  (True):  train only on runs where the expert itself ended within 2 cm of the target.
#                Failed expert runs are left out, because the model copies everything it sees.
#                Needs add_ee.py to be run first.
#   OFF (False): train on all runs, including failed ones.
ONLY_GOOD_RUNS = True

# SWITCH: separate heads for the wheels and the arm (Nat's idea)
#   ON  (True):  control_net has a shared body (2 layers), then a wheel head (4 outputs)
#                and an arm head (7 outputs), each with its own hidden layer.
#   OFF (False): one network gives all 11 commands from the same last layer.
SPLIT_HEADS = True

# SWITCH: start/stop outputs (tested; did not help, keep OFF)
#   ON  (True):  the wheel head also outputs start/stop, which replaces phase_net.
#                Only the arm head is given the phase. Needs SPLIT_HEADS ON.
#   OFF (False): phase_net decides when to stop driving.
START_STOP = False
assert SPLIT_HEADS or not START_STOP, "START_STOP needs SPLIT_HEADS = True"

# SETTING: early switch label
#   phase_net is trained to switch to "reach" this many rows (1 row = 1 ms) BEFORE the expert did.
#   This makes the robot stop when it is nearly parked, instead of driving past the parking spot.
#   0 = same time as the expert. 200 (0.2 s) was best on the validation targets.
#   Any value other than 500 adds "_early<value>" to the model name, so models are not overwritten.
EARLY_ROWS = 200

# SWITCH: DAgger data (tested; did not help, keep OFF)
#   ON  (True):  also train on DAgger runs from dagger.py (the model drove, the expert labelled).
#   OFF (False): train only on the expert's own runs.
USE_DAGGER = False
DAGGER_FOLDER = "data_dagger_arm2"   # where dagger.py saves its runs

# SETTING: extra data folders
#   Extra runs used for TRAINING ONLY, on top of DATA_FOLDER in train.py.
#   Validation and test targets always come from DATA_FOLDER. Each folder needs add_ee.py first.
#   []                      = no extra data.
#   ["data_dart2"]          = also train on the medium-push runs.
#   ["data_v2/medium_push"] = the version 2 medium-push runs (with DATA_FOLDER = "data_v2/big_push").
#   The folder names are added to the model name, for example ..._plus_dart2.
EXTRA_FOLDERS = ["data_v2/balanced"]
#   Results on 20 validation targets (23 inputs, seed 0):
#     ["data_v2/balanced"] with 1000 runs: 17 of 20 (close targets 4 of 5, far 3 of 5)
#     ["data_v2/balanced"] with 1500 runs: 17 of 20 (close targets 2 of 5, far 5 of 5;
#        13 and 14 turned to face the target too early, too close, then got stuck)
#     ["data_v2/balanced", "data_v2/random_start"] (2000 runs, ROW_STEP 2): 13 of 20.
#        The random starts made the stop decision too careful (many late switches). Not used.
#   The best model so far: DATA_FOLDER = "data_dart" (train.py) and EXTRA_FOLDERS = ["data_dart2"].
#   For the data_v2 tests: DATA_FOLDER = "data_v2/big_push" and EXTRA_FOLDERS = ["data_v2/medium_push"].
#   For the balanced data: DATA_FOLDER = "data_dart", EXTRA_FOLDERS = ["data_v2/balanced"],
#   and ONLY_EXTRA_FOLDERS = True (below).

# SWITCH: train only on the extra folders
#   ON  (True):  train ONLY on the runs in EXTRA_FOLDERS (all of them). DATA_FOLDER (data_dart) is
#                then only used for the validation and test targets, so scores stay comparable.
#                Used for the balanced data: all 1000 runs are used for training, none are held back.
#                Adds "_extraonly" to the model name.
#   OFF (False): train on DATA_FOLDER's training runs AND the extra folders (as before).
ONLY_EXTRA_FOLDERS = True

# SWITCH: IMU inputs (Nat's idea; needs the data_v2 folders, which save the IMU readings)
#   ON  (True):  5 more features: forward speed, sideways speed, turning rate, roll, pitch.
#   OFF (False): no IMU features.
USE_IMU = False

# SETTING: number of inputs (lecturer feedback: use fewer inputs)
#   34 = all inputs (the best model so far).
#   20 = only what the robot needs to act. The map positions and the repeated inputs are removed.
#        Kept: the target seen from the robot (ahead, left, height), the end-effector-to-target
#        gap seen from the robot (ahead, left, up), the 7 joint angles and the 7 joint speeds.
#   13 = the same as 20, without the 7 joint speeds.
#   23 = the same as 20, plus 3 end-effector inputs: the gap on the map (x, y) and the gap distance.
#        (20 reached 8 of 20 with phase accuracy 98%, so the arm part was missing something.)
#        23 reached 16 of 20, the same as 34.
#   6  = the simple plan: the target seen from the robot (ahead, left, height) and the
#        end-effector-to-target gap seen from the robot (ahead, left, up). No joint angles or speeds.
#        (Tested with ACTION_MASK ON: learned nothing about the torques, and the arm threw the robot.)
#   17 = two separate models (no shared layers):
#        driving model, 7 inputs: target seen from the robot (ahead, left, height),
#                                 target x, y on the map, sin and cos of the heading. Gives the 4 wheel speeds.
#        arm model, 10 inputs:    end-effector-to-target gap seen from the robot (ahead, left, up),
#                                 7 joint angles, plus the phase. Gives the 7 arm torques.
#        The arm model runs all the time, so it also holds the arm still while driving.
#        (Tested: train MSE 0.28, val MSE got worse every epoch. The arm needs the joint speeds.)
#   24 = two separate models, chosen with check_inputs.py:
#        driving model, 21 inputs: the 7 above + 7 joint angles + 7 joint speeds. Gives the 4 wheel speeds.
#        arm model, 17 inputs:     tip gap seen from the robot (3) + 7 joint angles + 7 joint speeds,
#                                  plus the phase. Gives the 7 arm torques.
#        The joint numbers are shared, so the robot reads 24 different numbers in total.
#   19 and 22 = two separate models, picked by "check_inputs.py drop data_v2/balanced 400"
#        (backward elimination, starting from the 27 inputs of the 23-input model with parking).
#        Both need USE_PARK = True. (An older 22 option, without parking, was never tested and is replaced.)
#        driving model, 5 inputs:  parking spot ahead, left, turn sin, joint angle 3, joint speed 1.
#        arm model, 11 inputs:     tip gap up, joint angles 2, 4, 5, 6, joint speeds 1, 2, 4, 5, 6, 7.
#        phase_net, 7 inputs:      target ahead, left, height, and the 4 parking inputs.
#        19 = exactly what the check kept: the robot reads 19 different numbers.
#        22 = the same, but the arm also keeps tip gap ahead, tip gap left and tip distance (14 arm inputs).
#             The check scores the average copying error and hardly sees the last centimetre of reaching,
#             and removing the tip distance once dropped the score from 16 to 8 of 20. So test 22 first.
#   24, 23, 22, 20, 19, 17, 13 and 6 need USE_EE = True. Any value other than 34 adds "_in<value>" to the model name.
#   The exact columns are in INPUT SELECTION, at the end of this file.
INPUT_SIZE = 23

# SETTING: phase_net inputs (only used when INPUT_SIZE is 20 or 13)
#   3 = the target seen from the robot: ahead, left, height. (Tested: 10 of 20 reached. It switched
#       at the wrong time, because the expert parks on the line from the map centre to the target,
#       and 3 inputs cannot show where that line is.)
#   7 = the same 3, plus the target's x, y on the map and the sin, cos of the robot's heading.
#       With these, phase_net can work out where that parking line is.
#   Any value other than 3 adds "_phase<value>" to the model name.
PHASE_SIZE = 7

# SWITCH: action mask (part of the simple plan)
#   ON  (True):  while the expert is driving, the model only practises the 4 wheel speeds
#                (the arm torques are ignored in the error). While the expert is reaching,
#                it only practises the 7 arm torques (the wheel speeds are ignored).
#                Watch out: the arm torques while driving are then never practised,
#                but drive.py still sends them to the arm.
#   OFF (False): the model practises all 11 commands on every row.
#   (Tested with INPUT_SIZE 6: the untrained arm torques threw the robot. Keep OFF.)
ACTION_MASK = False

# SETTING: driving row step (smart thinning, to make the data smaller)
#   Keep only every Nth row in the middle of driving, where the rows are very repetitive.
#   All reaching rows are kept, because reaching needs the most precision.
#   The last NEAR_STOP_ROWS driving rows before the expert stopped are also all kept,
#   because phase_net learns when to stop from them.
#   (Thinning every row, with DECIDE_EVERY, lost the stop moment: 11 of 20 instead of 15.)
#   1 = keep every driving row (as before). 10 = keep every 10th driving row.
#   Any value other than 1 adds "_drivestep<value>" to the model name.
#   (Tested 10 with NEAR_STOP_ROWS 1000: 15 of 20 against 16 with every row, but only about
#   25% fewer rows, because most rows are reaching rows. Not worth it for now.)
DRIVE_ROW_STEP = 1
NEAR_STOP_ROWS = 1000   # 1000 rows = the last 1 second of driving, all kept (0 = none kept)

# SETTING: decide every N steps (Nat's lower-rate idea; also called the control rate)
#   The simulation runs 1000 steps per second. The model decides every N steps and the
#   robot holds the last commands in between (drive.py). Training uses every Nth row (train.py),
#   so training and driving work at the same rate.
#   1  = 1000 decisions per second (as before).
#   10 = 100 per second.   20 = 50 per second.   33 = about 30 per second (Nat's idea).
#   Any value other than 1 adds "_every<value>" to the model name, and train.py uses 3 x N epochs.
#   Results on 20 validation targets (23 inputs):
#     1  (every row):                    15 of 20
#     33 with HOLD_COMMANDS ON:           0 of 20 (the held torques made the arm swing, robot thrown)
#     33 with HOLD_COMMANDS OFF:          7 of 20 (too few rows, phase_net unsure when to stop)
#     10 with HOLD_COMMANDS OFF:         11 of 20 (fewer rows = worse). Keep 1.
DECIDE_EVERY = 1

# SWITCH: hold commands (only matters when DECIDE_EVERY is above 1)
#   ON  (True):  the robot decides every DECIDE_EVERY steps and holds the commands in between.
#   OFF (False): the robot decides at every step (1000 per second), even though the model
#                learned from every DECIDE_EVERY-th row. Only drive.py uses this switch,
#                so the same trained model can be tested both ways without training again.
HOLD_COMMANDS = False

# SETTING: repeat close-target runs (oversampling)
#   check_rare.py showed that only about 9% of the training runs have a target that starts
#   within CLOSE_DISTANCE of the robot, and 4 of the 5 failed validation targets were like that
#   (the robot wandered, because it had seen few examples of the expert's move for close targets).
#   Each close-target training run is used CLOSE_REPEAT times, so the model practises it more.
#   Only training runs are repeated, never validation or test runs.
#   1 = no repeats (as before). 5 = each close-target run 5 times.
#   Any value other than 1 adds "_close<value>" to the model name.
#   Results on 20 validation targets (23 inputs, close targets 9, 11, 13, 14, 15):
#     1:          15 of 20, close targets reached 1 of 5
#     5, seed 0:  16 of 20, close targets reached 3 of 5
#     5, seed 1:  15 of 20, close targets reached 2 of 5 (different ones: only half-learned)
#   With the balanced data (data_v2/balanced) close targets are no longer rare, so keep 1 there.
CLOSE_REPEAT = 1
CLOSE_DISTANCE = 0.5   # metres, floor distance from the robot to the target at the start

# SWITCH: parking-spot inputs (add_park, below)
#   check_expert_steps.py showed the expert drives straight to its parking spot first and only turns
#   on the spot when it is about 10 cm away. At 20 to 50 cm from the spot the data mixes "driving"
#   and "turning" (from different runs), so the model turned too early (validation target 13).
#   ON  (True):  4 inputs about the expert's parking spot, seen from the robot: how far ahead,
#                how far left, and sin, cos of the turn still needed once there.
#                They replace the target's map x, y and the heading sin, cos in phase_net,
#                and are added to control_net (the driving model with two models).
#                Adds "_park" to the model name (instead of "_phase<value>").
#   OFF (False): no parking-spot inputs (as before).
USE_PARK = True

# Name of the saved model files. It is built from the switches, so each setting keeps its own files.
MODEL_NAME = (("ee" if USE_EE else "phase_rel") + ("_good" if ONLY_GOOD_RUNS else "")
              + ("_split" if SPLIT_HEADS else "") + ("_startstop" if START_STOP else "")
              + ("" if EARLY_ROWS == 500 else f"_early{EARLY_ROWS}")
              + ("_dagger" if USE_DAGGER else "")
              + "".join("_plus_" + f.replace("data_", "").replace("/", "_") for f in EXTRA_FOLDERS)
              + ("_extraonly" if ONLY_EXTRA_FOLDERS else "")
              + ("_imu" if USE_IMU else "")
              + ("_mask" if ACTION_MASK else "")
              + ("" if DRIVE_ROW_STEP == 1 else f"_drivestep{DRIVE_ROW_STEP}")
              + ("" if DRIVE_ROW_STEP == 1 or NEAR_STOP_ROWS == 0 else f"_near{NEAR_STOP_ROWS}")
              + ("" if DECIDE_EVERY == 1 else f"_every{DECIDE_EVERY}")
              + ("" if CLOSE_REPEAT == 1 else f"_close{CLOSE_REPEAT}")
              + ("" if INPUT_SIZE == 34 else f"_in{INPUT_SIZE}")
              + ("_park" if USE_PARK else ("" if INPUT_SIZE == 34 or PHASE_SIZE == 3 else f"_phase{PHASE_SIZE}")))


# =============================================================================
# FEATURES 21 to 27: the target in the robot frame
# The data stores the target in the world frame. To steer, the model needs to know
# where the target is relative to the robot: ahead or behind, left or right.
# The network could learn this rotation itself, but giving it directly makes steering
# much easier to learn.
#
# Example: target at world (1, 0). Robot at (0, 0).
#   Robot faces +x:          world target (1, 0)    robot frame: 1 m ahead
#   Robot turned left 90°:   world target (1, 0)    robot frame: 1 m to the right
#   Robot faces -x:          world target (1, 0)    robot frame: 1 m behind
# =============================================================================

N_INPUTS = 28   # 21 measured or given features + 7 robot-frame features


def add_relative(X21):
    """Adds the 7 robot-frame target features.
    Input:  X21, shape (rows, 21), columns as in the INPUT LAYOUT above.
    Output: shape (rows, 28)."""
    X21 = np.asarray(X21, dtype=np.float64)
    base_xy = X21[:, 0:2]        # base x, y (world frame, m)
    bearing = X21[:, 3]          # robot heading (rad)
    target = X21[:, 18:21]       # target x, y, z (world frame, m)

    # Vector from the base to the target on the floor (world frame).
    dx = target[:, 0] - base_xy[:, 0]
    dy = target[:, 1] - base_xy[:, 1]

    # Rotate it into the robot frame: "ahead" and "left" of the robot.
    c, s = np.cos(bearing), np.sin(bearing)
    front = c * dx + s * dy
    left = -s * dx + c * dy

    dist = np.hypot(dx, dy)              # floor distance to the target (m)
    angle = np.arctan2(left, front)      # turn needed to face the target (rad)

    new = np.stack([front, left, dist,
                    np.sin(angle), np.cos(angle),
                    np.sin(bearing), np.cos(bearing)], axis=1)
    return np.concatenate([X21, new], axis=1)


# =============================================================================
# phase_net inputs: base and target only, NOT the arm.
# The arm only starts moving after the switch. If phase_net could see the arm,
# it would learn the shortcut "switch when the arm moves" and never switch.
# =============================================================================

PHASE_COLS = [0, 1, 2, 3, 18, 19, 20] + list(range(21, 28))   # 14 features


# =============================================================================
# FEATURES 28 to 33: gap from the end-effector to the target (only when USE_EE = True)
# Tells the arm how far its tip still has to move. The end-effector position comes
# from forward kinematics (Pinocchio robot model): add_ee.py for the saved runs,
# drive.py while driving.
# =============================================================================

N_INPUTS_EE = N_INPUTS + 6   # 34


def add_ee(X28, ee):
    """Adds the 6 end-effector-to-target features.
    Input:  X28, shape (rows, 28); ee, shape (rows, 3) = end-effector x, y, z (world frame, m).
    Output: shape (rows, 34)."""
    X28 = np.asarray(X28, dtype=np.float64)
    ee = np.asarray(ee, dtype=np.float64)
    target = X28[:, 18:21]
    bearing = X28[:, 3]

    gap = target - ee                            # gap in the world frame (m)
    c, s = np.cos(bearing), np.sin(bearing)
    gap_front = c * gap[:, 0] + s * gap[:, 1]    # the same gap in the robot frame
    gap_left = -s * gap[:, 0] + c * gap[:, 1]
    gap_dist = np.linalg.norm(gap, axis=1)       # straight-line distance (m)

    new = np.stack([gap[:, 0], gap[:, 1], gap[:, 2], gap_front, gap_left, gap_dist], axis=1)
    return np.concatenate([X28, new], axis=1)


# =============================================================================
# control_net: predicts the 11 commands (4 wheel speeds in rad/s, 7 arm torques in N m).
#
# SPLIT_HEADS OFF: one multilayer perceptron (MLP).
#   inputs -> 128 -> 128 -> 11
#
# SPLIT_HEADS ON: shared body, then two heads.
#   inputs -> 128 -> 128            (body, shared)
#     wheel head: 64 -> 4 wheel speeds   (+ start/stop if START_STOP is ON)
#     arm head:   64 -> 7 arm torques
#   Each head has its own hidden layer, so learning the wheels does not interfere
#   with learning the arm. (Splitting only the final linear layer would change nothing,
#   because each output already has its own weights there.)
#
# START_STOP ON: the phase input goes to the arm head only, not to the body or the
#   wheel head. Otherwise the stop output could simply copy the phase input.
# =============================================================================

class SplitControlNet(nn.Module):
    def __init__(self, n_in, start_stop=False):
        """n_in = number of inputs, with the phase as the last column."""
        super().__init__()
        self.start_stop = start_stop
        n_base = 6 if start_stop else 4             # 4 wheel speeds (+ start, stop)
        n_body = n_in - 1 if start_stop else n_in   # START_STOP ON: the body does not get the phase
        n_arm = 128 + 1 if start_stop else 128      # START_STOP ON: the arm head gets the phase
        self.body = nn.Sequential(
            nn.Linear(n_body, 128), nn.ReLU(),  # shared layer 1
            nn.Linear(128, 128), nn.ReLU(),     # shared layer 2
        )
        self.base_end = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),      # wheel head: hidden layer
            nn.Linear(64, n_base),              # wheel head: outputs
        )
        self.arm_end = nn.Sequential(
            nn.Linear(n_arm, 64), nn.ReLU(),    # arm head: hidden layer
            nn.Linear(64, 7),                   # arm head: 7 torques
        )

    def forward(self, x):
        """x: shape (batch, n_in). Returns shape (batch, 11), or (batch, 13) with START_STOP."""
        if self.start_stop:
            phase = x[:, -1:]                                 # last column = phase
            shared = self.body(x[:, :-1])                     # body without the phase
            base = self.base_end(shared)
            arm = self.arm_end(torch.cat([shared, phase], dim=1))   # arm head also gets the phase
        else:
            shared = self.body(x)          # body runs once
            base = self.base_end(shared)   # both heads read the same body output
            arm = self.arm_end(shared)
        # Output order: 4 wheel speeds, 7 arm torques (the 11 commands), then start, stop if present.
        return torch.cat([base[:, :4], arm, base[:, 4:]], dim=1)


class TwoModelNet(nn.Module):
    """INPUT_SIZE 17, 22 or 24: two separate models with no shared layers.
    Input x: shape (batch, n_drive + n_arm + 1) = driving model columns, arm model columns, phase.
    Output: shape (batch, 11) = 4 wheel speeds, then 7 arm torques."""
    def __init__(self, n_drive, n_arm):
        super().__init__()
        self.n_drive = n_drive
        self.drive_model = nn.Sequential(         # driving model: its inputs only, no phase
            nn.Linear(n_drive, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4),                     # 4 wheel speeds
        )
        self.arm_model = nn.Sequential(           # arm model: its inputs + the phase
            nn.Linear(n_arm + 1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 7),                     # 7 arm torques
        )

    def forward(self, x):
        wheels = self.drive_model(x[:, :self.n_drive])
        arm = self.arm_model(x[:, self.n_drive:])   # the arm columns and the phase (last column)
        return torch.cat([wheels, arm], dim=1)


def build_control_net(n_in):
    """Builds control_net for the current switches. n_in = number of inputs including the phase."""
    if TWO_MODELS:
        return TwoModelNet(N_DRIVE_IN, N_ARM_IN)
    if SPLIT_HEADS:
        return SplitControlNet(n_in, START_STOP)
    return nn.Sequential(
        nn.Linear(n_in, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 11),                # 4 wheel speeds + 7 arm torques
    )


# =============================================================================
# FEATURES 34 to 38: IMU (only when USE_IMU = True)
# Before this, the model could not see how fast the base was moving, so phase_net
# had to decide when to stop without knowing the speed.
# Training: read from the data_v2 columns (32-35 quaternion, 53-55 linear velocity,
# 56-58 angular velocity). Driving: drive.py reads the same values from the simulation.
# =============================================================================

N_IMU = 5   # forward speed, sideways speed, turning rate, roll, pitch

# The IMU features are the last 5 columns. phase_net is given them too, so it knows the speed.
_IMU_START = (N_INPUTS + 6) if USE_EE else N_INPUTS
if USE_IMU:
    PHASE_COLS = PHASE_COLS + list(range(_IMU_START, _IMU_START + N_IMU))


def add_imu(X, bearing, quat, lin_vel, ang_vel):
    """Adds the 5 IMU features.
    Input:  X (rows, n); bearing (rows,) rad; quat (rows, 4) base orientation (x, y, z, w);
            lin_vel (rows, 3) base velocity, world frame, m/s; ang_vel (rows, 3) rad/s.
    Output: shape (rows, n + 5)."""
    X = np.asarray(X, dtype=np.float64)
    bearing = np.asarray(bearing, dtype=np.float64).reshape(-1)
    quat = np.asarray(quat, dtype=np.float64).reshape(-1, 4)
    lin_vel = np.asarray(lin_vel, dtype=np.float64).reshape(-1, 3)
    ang_vel = np.asarray(ang_vel, dtype=np.float64).reshape(-1, 3)

    # Base velocity in the robot frame: forward and to the left (m/s).
    c, s = np.cos(bearing), np.sin(bearing)
    forward = c * lin_vel[:, 0] + s * lin_vel[:, 1]
    sideways = -s * lin_vel[:, 0] + c * lin_vel[:, 1]
    turning = ang_vel[:, 2]   # turning rate about the vertical axis (rad/s)

    # Roll and pitch from the quaternion (standard formulas), in radians.
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))

    new = np.stack([forward, sideways, turning, roll, pitch], axis=1)
    return np.concatenate([X, new], axis=1)


# =============================================================================
# PARKING-SPOT FEATURES (only when USE_PARK = True)
# The expert parks on the line from the map centre to the target, at a distance that depends
# on the target's height (the same numbers as in robotnik_arm_task_dart.py). Then it turns on the
# spot to face along that line. These 4 inputs show the robot where that spot is, from its own view.
# =============================================================================

STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]     # target height (m), the same as the expert's code
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]    # parking distance from the target (m)


def add_park(X):
    """Adds the 4 parking-spot features.
    Input:  X, shape (rows, n), with the INPUT LAYOUT columns 0-20 (base x, y, bearing, target x, y, z).
    Output: shape (rows, n + 4): spot ahead (m), spot left (m), sin and cos of the turn still
            needed once at the spot (to face the way the expert faces when it parks)."""
    X = np.asarray(X, dtype=np.float64)
    base_xy, bearing, target = X[:, 0:2], X[:, 3], X[:, 18:21]

    # Where the expert parks (world frame): back from the target, towards the map centre.
    standoff = np.interp(target[:, 2], STANDOFF_Z_POINTS, STANDOFF_POINTS)
    far_enough = np.hypot(target[:, 0], target[:, 1]) > 1e-4
    line = np.where(far_enough, np.arctan2(target[:, 1], target[:, 0]), 0.0)   # direction of the line
    park_x = target[:, 0] - standoff * np.cos(line)
    park_y = target[:, 1] - standoff * np.sin(line)

    # The same spot seen from the robot: ahead and left.
    dx, dy = park_x - base_xy[:, 0], park_y - base_xy[:, 1]
    c, s = np.cos(bearing), np.sin(bearing)
    park_ahead = c * dx + s * dy
    park_left = -s * dx + c * dy

    # The turn still needed at the spot: the expert ends up facing along the line.
    final_turn = line - bearing
    new = np.stack([park_ahead, park_left, np.sin(final_turn), np.cos(final_turn)], axis=1)
    return np.concatenate([X, new], axis=1)


# =============================================================================
# INPUT SELECTION (INPUT_SIZE, PHASE_SIZE)
# train.py and drive.py first build all the inputs (34, or 39 with the IMU).
# Then they store only the columns in KEEP_COLS (the ones either network needs).
#   CONTROL_COLS  which stored columns control_net reads (INPUT_SIZE of them, + IMU).
#   PHASE_COLS    which stored columns phase_net reads (PHASE_SIZE of them, + IMU).
# The column numbers below are the ones in the INPUT LAYOUT table at the top of this file.
# =============================================================================

assert INPUT_SIZE in (34, 24, 23, 22, 20, 19, 17, 13, 6), "INPUT_SIZE must be 34, 24, 23, 22, 20, 19, 17, 13 or 6"
assert PHASE_SIZE in (3, 7), "PHASE_SIZE must be 3 or 7"
assert INPUT_SIZE == 34 or USE_EE, "INPUT_SIZE 24, 23, 22, 20, 19, 17, 13 or 6 needs USE_EE = True"
PARK_SETS = INPUT_SIZE in (19, 22)          # input sets picked with the parking inputs included
assert USE_PARK or not PARK_SETS, "INPUT_SIZE 19 or 22 needs USE_PARK = True"
TWO_MODELS = INPUT_SIZE in (17, 19, 22, 24)   # separate driving and arm models
assert not (TWO_MODELS and START_STOP), "INPUT_SIZE 17, 19, 22 or 24 and START_STOP cannot both be ON"
assert not (ACTION_MASK and START_STOP), "ACTION_MASK and START_STOP cannot both be ON"

_IMU_COLS = list(range(_IMU_START, _IMU_START + N_IMU)) if USE_IMU else []
# Parking-spot inputs (USE_PARK): 4 columns added after everything else (after the IMU, if it is ON):
# parking spot ahead, left, sin and cos of the turn still needed there (add_park).
N_PARK = 4
PARK_COLS = list(range(_IMU_START + len(_IMU_COLS), _IMU_START + len(_IMU_COLS) + N_PARK))

TARGET_FROM_ROBOT = [21, 22, 20]    # target: ahead (m), left (m), height (m)
GAP_FROM_ROBOT = [31, 32, 30]       # end-effector to target: ahead (m), left (m), up (m)
JOINT_ANGLES = list(range(4, 11))   # 7 arm joint angles (rad)
JOINT_SPEEDS = list(range(11, 18))  # 7 arm joint speeds (rad/s)
TARGET_ON_MAP = [18, 19]            # target x, y on the map (world frame, m)
HEADING = [26, 27]                  # sin, cos of the robot's heading
GAP_EXTRA = [28, 29, 33]            # end-effector to target: x, y on the map (m), straight-line distance (m)

if INPUT_SIZE == 34:
    KEEP_COLS = list(range(_IMU_START)) + _IMU_COLS   # everything
    CONTROL_COLS = list(range(len(KEEP_COLS)))        # control_net reads everything
    # PHASE_COLS stays as set above (the column numbers are the same, because nothing is removed)
elif PARK_SETS:
    # INPUT_SIZE 19 or 22: two separate models, chosen by the drop check with the parking inputs.
    # Joint angle j is column 3 + j. Joint speed j is column 10 + j.
    DRIVE_COLS_ORIGINAL = ([3 + 3, 10 + 1]               # joint angle 3, joint speed 1
                           + PARK_COLS[:3]                # parking spot ahead, left, turn sin
                           + _IMU_COLS)
    ARM_COLS_ORIGINAL = ([30]                             # tip gap up
                         + [3 + j for j in (2, 4, 5, 6)]   # joint angles 2, 4, 5, 6
                         + [10 + j for j in (1, 2, 4, 5, 6, 7)])   # joint speeds 1, 2, 4, 5, 6, 7
    if INPUT_SIZE == 22:
        ARM_COLS_ORIGINAL = [31, 32, 33] + ARM_COLS_ORIGINAL   # + tip gap ahead, tip gap left, tip distance
    control = DRIVE_COLS_ORIGINAL + ARM_COLS_ORIGINAL
    phase = TARGET_FROM_ROBOT + PARK_COLS + _IMU_COLS      # phase_net: target ahead, left, height + parking
    KEEP_COLS = []
    for c in control + phase:   # every column once, in order
        if c not in KEEP_COLS:
            KEEP_COLS.append(c)
    CONTROL_COLS = [KEEP_COLS.index(c) for c in control]
    PHASE_COLS = [KEEP_COLS.index(c) for c in phase]
    N_DRIVE_IN = len(DRIVE_COLS_ORIGINAL)   # 5 (+5 with the IMU)
    N_ARM_IN = len(ARM_COLS_ORIGINAL)       # 19 option: 11. 22 option: 14.
elif TWO_MODELS:
    # Two separate models. control_net reads the driving model's columns first, then the arm model's.
    # A column can be read by both models (the joint numbers with INPUT_SIZE 24); it is stored only once.
    DRIVE_COLS_ORIGINAL = TARGET_FROM_ROBOT + TARGET_ON_MAP + HEADING + _IMU_COLS   # driving model
    ARM_COLS_ORIGINAL = GAP_FROM_ROBOT + JOINT_ANGLES                               # arm model
    if INPUT_SIZE == 24:
        DRIVE_COLS_ORIGINAL = DRIVE_COLS_ORIGINAL + JOINT_ANGLES + JOINT_SPEEDS
        ARM_COLS_ORIGINAL = ARM_COLS_ORIGINAL + JOINT_SPEEDS
    control = DRIVE_COLS_ORIGINAL + ARM_COLS_ORIGINAL
    phase = TARGET_FROM_ROBOT + (TARGET_ON_MAP + HEADING if PHASE_SIZE == 7 else []) + _IMU_COLS
    KEEP_COLS = []
    for c in control + phase:   # every column once, in order
        if c not in KEEP_COLS:
            KEEP_COLS.append(c)
    CONTROL_COLS = [KEEP_COLS.index(c) for c in control]
    PHASE_COLS = [KEEP_COLS.index(c) for c in phase]
    N_DRIVE_IN = len(DRIVE_COLS_ORIGINAL)   # 17 option: 7. 24 option: 21. (+5 with the IMU)
    N_ARM_IN = len(ARM_COLS_ORIGINAL)       # 17 option: 10. 24 option: 17.
else:
    control = TARGET_FROM_ROBOT + GAP_FROM_ROBOT
    if INPUT_SIZE != 6:
        control = control + JOINT_ANGLES
    if INPUT_SIZE in (20, 23):
        control = control + JOINT_SPEEDS
    if INPUT_SIZE == 23:
        control = control + GAP_EXTRA
    control = control + _IMU_COLS
    phase = TARGET_FROM_ROBOT + (TARGET_ON_MAP + HEADING if PHASE_SIZE == 7 else []) + _IMU_COLS
    # Store the control columns first, then any phase columns that control_net does not use.
    KEEP_COLS = control + [c for c in phase if c not in control]
    CONTROL_COLS = [KEEP_COLS.index(c) for c in control]
    PHASE_COLS = [KEEP_COLS.index(c) for c in phase]

# Parking-spot inputs (USE_PARK) for the other input sets: they replace the target's map x, y and the
# heading sin, cos in phase_net, and go to control_net (to the driving model when there are two models).
# INPUT_SIZE 19 and 22 already chose their parking inputs above, so they are skipped here.
if USE_PARK and not PARK_SETS:
    _control = [KEEP_COLS[i] for i in CONTROL_COLS]
    _phase = [KEEP_COLS[i] for i in PHASE_COLS]
    _phase = [c for c in _phase if c not in TARGET_ON_MAP + HEADING] + PARK_COLS
    if TWO_MODELS:
        _control = _control[:N_DRIVE_IN] + PARK_COLS + _control[N_DRIVE_IN:]   # park inputs go to the driving model
        N_DRIVE_IN = N_DRIVE_IN + N_PARK
    else:
        _control = _control + PARK_COLS
    KEEP_COLS = []
    for c in _control + _phase:   # every column once, in order
        if c not in KEEP_COLS:
            KEEP_COLS.append(c)
    CONTROL_COLS = [KEEP_COLS.index(c) for c in _control]
    PHASE_COLS = [KEEP_COLS.index(c) for c in _phase]

N_KEEP = len(KEEP_COLS)    # number of different input numbers the robot reads
N_IN = len(CONTROL_COLS)   # number of inputs control_net reads (it also gets the phase).
                           # With two models, a shared column is counted once per model.