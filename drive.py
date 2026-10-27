# =============================================================================
# drive.py
# This is for: letting the trained model drive the robot to ONE test dot, with no expert.
# How to run:  python3 drive.py data_dart 0
#              (data folder, then which test dot: 0, 1, 2, ...)
# Watch it:    leave out ROBOENV_HEADLESS=1 to see the window.
# Result:      a check line every second, and a RESULT line at the end.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   x            the inputs at this moment (what the robot measures).
#   x_s          the same inputs, standardised (scaled like in training).
#   y_s          the model's 11 commands, still scaled.
#   y            the 11 commands in real units (wheel speeds and arm torques).
#   ee_pos       where the end-effector is (x, y, z).
#   p_reach      phase_net's answer, from 0 to 1. Above 0.5 means "time to reach".
#   phase        0 = driving, 1 = reaching.
#   q_mes        joint angles (17: 4 wheels, 7 arm, 6 fingers). [4:11] = the 7 arm joints.
#   qd_mes       joint speeds, in the same order.
#   clip         keep a number between a smallest and a biggest value.
# -----------------------------------------------------------------------------

import sys
import os
import json
import numpy as np

# No window if ROBOENV_HEADLESS=1. Stop after 30 seconds at most.
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "30000"))

import torch
import torch.nn as nn
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import StopCriteria, check_stop, configure_gripper_fingers, quaternion2bearing
from features import add_relative, add_ee, N_INPUTS, N_INPUTS_EE, PHASE_COLS, USE_EE, MODEL_NAME


# =============================================================================
# SETTINGS (change these)
# =============================================================================

# Change switch timing: phase_net must say "reach" for this many steps in a row (50 = 0.05 s).
SWITCH_STEPS = 50

# Change input: set USE_EE in features.py, not here. It must match train.py.
N_IN = N_INPUTS_EE if USE_EE else N_INPUTS

REACH_TOLERANCE = 0.01   # 1 cm, the same rule as the expert
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])   # fingers stay fixed


# =============================================================================
# STEP 1: Pick the test dot
# This is for: a fair test. Test runs were never used in training.
# =============================================================================

DATA_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data_dart"
TEST_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0

with open(f"{DATA_FOLDER}/split.json") as f:
    split = json.load(f)
test_file = split["test"][TEST_INDEX]
TARGET_EE_POS = np.load(test_file)[0, 18:21]   # columns 18 to 20 are the dot (x, y, z)


# =============================================================================
# STEP 2: Load the trained model (same shape as in train.py)
# =============================================================================

phase_net = nn.Sequential(
    nn.Linear(len(PHASE_COLS), 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
control_net = nn.Sequential(
    nn.Linear(N_IN + 1, 128), nn.ReLU(),
    nn.Linear(128, 128), nn.ReLU(),
    nn.Linear(128, 11),
)
saved = torch.load(f"{DATA_FOLDER}/{MODEL_NAME}_best.pt")
phase_net.load_state_dict(saved["phase_net"])
control_net.load_state_dict(saved["control_net"])
phase_net.eval()
control_net.eval()

# Load the standardising numbers from train.py.
scaler = np.load(f"{DATA_FOLDER}/scaler_{MODEL_NAME}.npz")
X_mean, X_std = scaler["X_mean"], scaler["X_std"]
Y_mean, Y_std = scaler["Y_mean"], scaler["Y_std"]

# Clipping limits: the smallest and biggest command the expert ever sent.
# This helps the model: it can never send a wild command and throw the robot away.
Y_min = np.full(11, np.inf)
Y_max = np.full(11, -np.inf)
for f in split["train"]:
    Y_file = np.load(f)[:, 21:]
    Y_min = np.minimum(Y_min, Y_file.min(axis=0))
    Y_max = np.maximum(Y_max, Y_file.max(axis=0))


def main():
    # =========================================================================
    # STEP 3: Set up the simulation (same as the teacher's code)
    # =========================================================================
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for j in wheel_joint_ids:
        client.changeDynamics(
            rid, j,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0]
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Draw the red dot.
    visual_shape_id = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
    client.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=TARGET_EE_POS.tolist())

    # Robot model, only used to measure where the end-effector is.
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")

    time_step = sim.GetTimeStep()
    current_time = 0
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
    cmd = MotorCommands()
    best_error = np.inf
    phase = 0.0          # 0 = driving, 1 = reaching
    reach_count = 0
    switch_time = None
    p_reach = 0.0

    # For checking only (never used to drive): where would the expert park?
    standoff = float(np.interp(TARGET_EE_POS[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]))
    target_xy = TARGET_EE_POS[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0])
    park_xy = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])

    # Print results: the test dot and the expert's parking spot.
    print(f"Model from {DATA_FOLDER}, test target {TEST_INDEX}: {np.round(TARGET_EE_POS, 3)}")
    print(f"Expert would park at {np.round(park_xy, 2)}, facing {np.degrees(target_bearing):.0f} degrees")

    # =========================================================================
    # STEP 4: The model drives (1000 times every second)
    # =========================================================================
    while True:
        # Measure the robot, the same way as when the data was collected.
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Where is the end-effector, and how far from the dot?
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(TARGET_EE_POS - ee_pos)
        best_error = min(best_error, ee_dist_error)

        # Build the inputs, in the same order as in training.
        x = np.concatenate((base_pos, [base_bearing], q_mes[4:11], qd_mes[4:11], TARGET_EE_POS))
        x = add_relative(x[None, :])          # add the 7 "look at the ball" inputs
        if USE_EE:
            x = add_ee(x, ee_pos[None, :])    # add the 6 end-effector-to-dot inputs
        x = x[0]
        x_s = torch.tensor((x - X_mean) / X_std, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            # Decide the phase. Once it switches to reaching, it never switches back.
            if phase == 0.0:
                p_reach = torch.sigmoid(phase_net(x_s[:, PHASE_COLS])).item()   # 0 to 1
                reach_count = reach_count + 1 if p_reach > 0.5 else 0
                if reach_count >= SWITCH_STEPS:
                    phase = 1.0
                    switch_time = current_time
                    print(f"Switched to reaching at t={current_time:.2f}s")

            # Ask control_net for the 11 commands.
            y_s = control_net(torch.cat([x_s, torch.tensor([[phase]])], dim=1)).numpy()[0]

        # Turn back into real units, then clip.
        y = np.clip(y_s * Y_std + Y_mean, Y_min, Y_max)
        wheel_cmds = y[:4]
        torque_arm = y[4:]
        if phase == 1.0:
            wheel_cmds = np.zeros(4)   # like the expert: base stays still while reaching

        # Send the commands to the robot (fingers stay fixed).
        cmd_all = np.concatenate((wheel_cmds, torque_arm, np.zeros(6)))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # Stop if reached, stuck, or out of time (same rules as the expert).
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, phase == 1.0)
        if stop_message:
            print(stop_message)
            break

        if not headless:
            import time
            time.sleep(0.001)

        current_time += time_step

        # Print results: one check line every second.
        if int(current_time / time_step) % 1000 == 0:
            dist_park = np.linalg.norm(base_pos[:2] - park_xy)
            face_err = np.degrees(np.arctan2(np.sin(target_bearing - base_bearing), np.cos(target_bearing - base_bearing)))
            print(f"t={current_time:.0f}s  base {np.round(base_pos[:2], 2)}  from parking spot {dist_park:.2f} m  "
                  f"facing error {face_err:.0f} deg  end-effector error {ee_dist_error:.2f} m  "
                  f"reach? {p_reach:.2f}  phase {int(phase)}")

    # Print results: the final RESULT line (test.py reads this).
    reached = ee_dist_error < REACH_TOLERANCE
    print(f"RESULT folder={DATA_FOLDER} target={TEST_INDEX} reached={reached} "
          f"final_error_cm={ee_dist_error * 100:.1f} best_error_cm={best_error * 100:.1f} time_s={current_time:.1f} "
          f"switch_s={switch_time}")


if __name__ == "__main__":
    main()
