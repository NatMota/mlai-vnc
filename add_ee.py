# =============================================================================
# add_ee.py
# This is for: working out where the end-effector was at every saved moment.
#              Only needed when USE_EE = True in features.py.
# How to run:  python3 add_ee.py data_dart
#              Run it ONCE. It is slow (15 to 40 minutes).
#              If stopped, run it again: it carries on where it stopped.
# Result:      a new folder, data_dart_ee, with one small file per run.
#              Your .npy files are not changed.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   d            one saved run (a table, one row per moment, 32 columns).
#   ee           end-effector position (x, y, z).
#   bearing      which way the robot faces (radians).
#   quat         the same direction written as a quaternion (x, y, z, w), which the robot model needs.
#   q_mes        all 17 joint angles: 4 wheels, 7 arm, 6 fingers.
#   q_all        base position + quat + q_mes, in the teacher's order.
#   q_pin        the same numbers, in the order the robot model (Pinocchio) needs.
#   Pinocchio    the robot model tool. It works out where the end-effector is from the joint angles.
#   median       the middle value when sorted.
#   90% below    9 out of 10 runs are below this number.
# -----------------------------------------------------------------------------

import os
import sys
import glob
import time
import numpy as np

os.environ["ROBOENV_HEADLESS"] = "1"
import pinocchio as pin
from simulation_and_control import pb, PinWrapper


# =============================================================================
# SETTINGS (change these, or type the folder after the command)
# =============================================================================

# Change data folder: which saved runs to work through.
DATA_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data_dart"
OUT_FOLDER = DATA_FOLDER + "_ee"
os.makedirs(OUT_FOLDER, exist_ok=True)


# =============================================================================
# STEP 1: Load the robot model (same as the teacher's code)
# =============================================================================

conf_file_name = "robotnik_arm.json"
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
ext_names = sim.getNameActiveJoints()
ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
model, data = dyn_model.pin_model, dyn_model.pin_data

FINGERS = np.full(6, 0.1)   # fingers are fixed at 0.1 in every run
WHEELS = np.zeros(4)        # wheel angles do not change where the end-effector is


# =============================================================================
# STEP 2: For one run, work out the end-effector x, y, z on every row
# =============================================================================

def ee_positions(d):
    out = np.empty((len(d), 3), dtype=np.float32)
    for i in range(len(d)):
        base_pos = d[i, 0:3]
        bearing = d[i, 3]
        # The floor is flat, so the base only turns around the vertical axis.
        quat = np.array([0.0, 0.0, np.sin(bearing / 2), np.cos(bearing / 2)])   # x, y, z, w
        q_mes = np.concatenate((WHEELS, d[i, 4:11], FINGERS))                   # all 17 joints
        q_all = np.concatenate((base_pos, quat, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(model, data, q_pin)
        pin.updateFramePlacement(model, data, ee_frame_id)
        out[i] = data.oMf[ee_frame_id].translation
    return out


# =============================================================================
# STEP 3: Go through every run, and save the results
# =============================================================================

files = sorted(glob.glob(f"{DATA_FOLDER}/*.npy"))
print(f"Working out the end-effector position for {len(files)} runs. This can take a while.")
start = time.time()
last_row_errors = []

for k, f in enumerate(files):   # k = run number, f = file name
    out_path = os.path.join(OUT_FOLDER, os.path.basename(f))
    d = np.load(f)
    if os.path.exists(out_path):
        ee = np.load(out_path)       # already done, skip the slow part
    else:
        ee = ee_positions(d)
        np.save(out_path, ee)
    # Remember how far the end-effector was from the dot on the last row (for the check below).
    last_row_errors.append(np.linalg.norm(ee[-1] - d[-1, 18:21]))

    # Print results: progress every 20 runs.
    if (k + 1) % 20 == 0:
        mins = (time.time() - start) / 60
        print(f"{k + 1} of {len(files)} runs done ({mins:.1f} minutes so far)")


# =============================================================================
# STEP 4: Print results: check the answers are right
# The expert stops when the end-effector is within 1 cm of the dot,
# so on the last row it should be about 1 cm away.
# =============================================================================

e = np.array(last_row_errors) * 100   # errors in cm
print("\nCHECK: end-effector to dot, on the last row of each run")
print(f"  median {np.median(e):.2f} cm   90% of runs below {np.percentile(e, 90):.2f} cm")
print("  About 1 cm or less means the positions are correct.")
