# =============================================================================
# drive.py
# This is for: letting the trained model drive the robot to ONE test target, with no expert.
# How to run:  python3 drive.py data_dart 0
#              (data folder, then which test target: 0, 1, 2, ...)
#              python3 drive.py data_dart 0 val
#              (the same, but a validation target, for choosing settings fairly)
#              python3 drive.py data_dart2 0 test data_dart
#              (model from data_dart2, test target from data_dart)
# Watch it:    leave out ROBOENV_HEADLESS=1 to see the window.
# Result:      a check line every second, and a RESULT line at the end.
#              Saves a graph (drive_target<N>.png) and a recording (drive_target<N>_log.npz)
#              in the results folder. check_drive.py reads the recordings.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   x            the inputs at this moment (what the robot measures).
#   x_s          the same inputs, standardised (scaled like in training).
#   y_s          the model's 11 commands, still scaled.
#   y            the 11 commands in real units (wheel speeds and arm torques).
#   ee_pos       where the end-effector is (x, y, z).
#   p_reach      how sure the model is that it is time to stop and reach, from 0 to 1.
#                Above 0.5 means "time to reach". Comes from phase_net,
#                or from the stop command when START_STOP is ON.
#   phase        0 = driving, 1 = reaching.
#   q_mes        joint angles (17: 4 wheels, 7 arm, 6 fingers). [4:11] = the 7 arm joints.
#   qd_mes       joint speeds, in the same order.
#   clip         keep a number between a smallest and a biggest value.
# -----------------------------------------------------------------------------

import sys
import os
import json
import datetime
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
from features import (add_relative, add_ee, N_INPUTS, N_INPUTS_EE, PHASE_COLS, USE_EE, START_STOP,
                      USE_IMU, N_IMU, add_imu, KEEP_COLS, CONTROL_COLS, N_IN, MODEL_NAME, build_control_net,
                      DECIDE_EVERY, HOLD_COMMANDS, USE_PARK, add_park)


# =============================================================================
# SETTINGS
# The switches (USE_EE, ONLY_GOOD_RUNS, SPLIT_HEADS, START_STOP) are in features.py, not here.
# They must be the same as when train.py was run.
# =============================================================================

# SETTING: switch steps
#   phase_net must say "reach" this many steps in a row before the arm starts.
#   50 steps = 0.05 seconds. Bigger = waits longer to be sure.
SWITCH_STEPS = 50

# How often the robot decides: every DECIDE_EVERY steps with HOLD_COMMANDS ON, every step with it OFF.
DRIVE_EVERY = DECIDE_EVERY if HOLD_COMMANDS else 1
# The same time (0.05 s) counted in decisions, because the model may only decide every DRIVE_EVERY steps.
# For example DRIVE_EVERY = 33: 50 / 33 = 2 decisions in a row (0.066 s). Do not change.
SWITCH_DECISIONS = max(1, round(SWITCH_STEPS / DRIVE_EVERY))

# SETTING: reach tolerance
#   How close counts as "reached", in metres. 0.01 = 1 cm, the same as the expert.
REACH_TOLERANCE = 0.01

# Fixed by the teacher. Do not change.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])   # fingers stay fixed

# Number of inputs: N_IN comes from features.py (INPUT_SIZE, plus 5 if USE_IMU is ON).


# =============================================================================
# STEP 1: Pick the test target
# This is for: a fair test. Test runs were never used in training.
# =============================================================================

DATA_FOLDER = sys.argv[1] if len(sys.argv) > 1 else "data_dart"   # where the model is
TEST_INDEX = int(sys.argv[2]) if len(sys.argv) > 2 else 0
# Which group the target comes from: "test" (the final exam) or "val" (the practice exam).
GROUP = sys.argv[3] if len(sys.argv) > 3 else "test"
# Where the targets come from. Normally data_dart, so every model is tested on the same targets.
TARGETS_FOLDER = sys.argv[4] if len(sys.argv) > 4 else DATA_FOLDER
# Name for results: "_dart2" is added when the model was trained on data_dart2.
RUN_NAME = MODEL_NAME + ("" if DATA_FOLDER == "data_dart" else "_" + DATA_FOLDER.replace("data_", "").replace("/", "_"))

with open(f"{DATA_FOLDER}/split.json") as f:
    split = json.load(f)              # the model's own split (for the clipping limits)
with open(f"{TARGETS_FOLDER}/split.json") as f:
    targets_split = json.load(f)         # where the test and validation targets come from
test_file = targets_split[GROUP][TEST_INDEX]
TARGET_EE_POS = np.load(test_file)[0, 18:21]   # columns 18 to 20 are the target (x, y, z)


# =============================================================================
# STEP 2: Load the trained model (same shape as in train.py)
# =============================================================================

phase_net = nn.Sequential(
    nn.Linear(len(PHASE_COLS), 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
control_net = build_control_net(N_IN + 1)   # shape from the switches
print("Loading model:", f"{DATA_FOLDER}/{MODEL_NAME}_best.pt")
saved = torch.load(f"{DATA_FOLDER}/{MODEL_NAME}_best.pt")
phase_net.load_state_dict(saved["phase_net"])
control_net.load_state_dict(saved["control_net"])
# The seed this model was trained with (older models did not save it).
MODEL_SEED = saved.get("settings", {}).get("SEED", "unknown")
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
    Y_file = np.load(f)[:, 21:32]   # the 11 expert commands
    Y_min = np.minimum(Y_min, Y_file.min(axis=0))
    Y_max = np.maximum(Y_max, Y_file.max(axis=0))

# Graph colours and style (shared look for all graphs)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

def graph_style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


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

    # Draw the target.
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
    graph_time, graph_error = [], []   # for the graph: time (s) and end-effector error (cm)
    p_reach = 0.0
    # Recording of the run (every 10th step), for check_drive.py. Saved at the end as drive_target<N>_log.npz.
    log = {"time": [], "phase": [], "p_reach": [], "base_xy": [], "arm_angles": [], "ee_pos": [], "inputs_scaled": []}

    # For checking only (never used to drive): where would the expert park?
    standoff = float(np.interp(TARGET_EE_POS[2], [0.1, 0.5, 1.1, 1.25], [0.80, 0.65, 0.65, 0.35]))
    target_xy = TARGET_EE_POS[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0])
    park_xy = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])

    # Print results: the test target and the expert's parking spot.
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

        # Where is the end-effector, and how far from the target?
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(TARGET_EE_POS - ee_pos)
        best_error = min(best_error, ee_dist_error)
        if int(round(current_time / time_step)) % 10 == 0:   # keep every 10th step for the graph
            graph_time.append(current_time)
            graph_error.append(ee_dist_error * 100)

        # The model decides only every DRIVE_EVERY steps (DECIDE_EVERY and HOLD_COMMANDS in features.py).
        # In between, the last commands are held (sent again unchanged).
        decide_now = int(round(current_time / time_step)) % DRIVE_EVERY == 0
        if decide_now:
            # Build the inputs, in the same order as in training.
            x = np.concatenate((base_pos, [base_bearing], q_mes[4:11], qd_mes[4:11], TARGET_EE_POS))
            x = add_relative(x[None, :])          # add the 7 target-in-robot-frame inputs
            if USE_EE:
                x = add_ee(x, ee_pos[None, :])    # add the 6 end-effector-to-target inputs
            if USE_IMU:
                lin_vel, ang_vel = client.getBaseVelocity(rid)   # the IMU readings, from the simulation
                x = add_imu(x, [base_bearing], base_ori, lin_vel, ang_vel)
            if USE_PARK:
                x = add_park(x)                   # the 4 parking-spot inputs (as in training)
            x = x[0][KEEP_COLS]                   # keep only the inputs chosen by INPUT_SIZE (as in training)
            x_s = torch.tensor((x - X_mean) / X_std, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                # Decide the phase. Once it switches to reaching, it never switches back.
                if phase == 0.0:
                    if START_STOP:
                        # The stop command, from the base end (it does not see the phase).
                        out = control_net(torch.cat([x_s[:, CONTROL_COLS], torch.tensor([[phase]])], dim=1))
                        p_reach = torch.softmax(out[:, 11:13], dim=1)[0, 1].item()   # how sure it is about "stop", 0 to 1
                    else:
                        p_reach = torch.sigmoid(phase_net(x_s[:, PHASE_COLS])).item()   # 0 to 1
                    reach_count = reach_count + 1 if p_reach > 0.5 else 0
                    if reach_count >= SWITCH_DECISIONS:
                        phase = 1.0
                        switch_time = current_time
                        print(f"Switched to reaching at t={current_time:.2f}s")

                # Ask control_net for the 11 commands, with the phase just decided.
                y_s = control_net(torch.cat([x_s[:, CONTROL_COLS], torch.tensor([[phase]])], dim=1))[0, :11].numpy()

            # Turn back into real units, then clip.
            y = np.clip(y_s * Y_std + Y_mean, Y_min, Y_max)
            wheel_cmds = y[:4]
            torque_arm = y[4:]
            if phase == 1.0:
                wheel_cmds = np.zeros(4)   # like the expert: base stays still while reaching

        # Record this step (every 10th step only, to keep the file small).
        if int(round(current_time / time_step)) % 10 == 0:
            log["time"].append(current_time)
            log["phase"].append(phase)
            log["p_reach"].append(p_reach)
            log["base_xy"].append(base_pos[:2].copy())
            log["arm_angles"].append(q_mes[4:11].copy())
            log["ee_pos"].append(ee_pos.copy())
            log["inputs_scaled"].append(x_s[0].numpy().copy())

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
    print(f"RESULT folder={DATA_FOLDER} targets={TARGETS_FOLDER} target={TEST_INDEX} reached={reached} "
          f"final_error_cm={ee_dist_error * 100:.1f} best_error_cm={best_error * 100:.1f} time_s={current_time:.1f} "
          f"switch_s={switch_time} model={RUN_NAME} seed={MODEL_SEED} group={GROUP} decide_every={DRIVE_EVERY}")

    # The results folder for this run's graph and recording.
    # When test.py runs this file: the same folder as that test run.
    # When run on its own: a new folder, results/<model>_seed<seed>/drive_<group>_<date>_<time>/.
    test_stamp = os.environ.get("TEST_STAMP")   # given by test.py, so all its files share one folder
    if test_stamp:
        results_folder = f"results/{RUN_NAME}_seed{MODEL_SEED}/{GROUP}_{test_stamp}"
    else:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        results_folder = f"results/{RUN_NAME}_seed{MODEL_SEED}/drive_{GROUP}_{stamp}"
    os.makedirs(results_folder, exist_ok=True)

    # Save the recording, for check_drive.py (data mismatch, gravity drop, awkward pause).
    np.savez(f"{results_folder}/drive_target{TEST_INDEX}_log.npz",
             **{k: np.array(v) for k, v in log.items()},
             switch_time=np.nan if switch_time is None else switch_time,
             test_file=test_file, keep_cols=np.array(KEEP_COLS), X_mean=X_mean, X_std=X_std,
             use_imu=USE_IMU, use_park=USE_PARK)

    # =========================================================================
    # STEP 5: Graph: end-effector distance to the target over time
    # This is for: seeing when the robot switched to reaching, and how close it got.
    # When test.py runs this file: saved in the same folder as that test run.
    # When run on its own: saved in a new folder, results/<model>_seed<seed>/drive_<date>_<time>/.
    # Open it in VS Code to see it.
    # =========================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")   # draw to a file, no window
        import matplotlib.pyplot as plt
        import matplotlib.ticker

        fig, ax = plt.subplots(figsize=(7, 4), facecolor="#fcfcfb")
        graph_style(ax)
        ax.plot(graph_time, graph_error, color=BLUE, linewidth=2)
        ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
        ax.text(graph_time[-1], 1.0, "1 cm  ", color=INK_2, ha="right", va="bottom", fontsize=9)
        if switch_time is not None:
            ax.axvline(switch_time, color=ORANGE, linewidth=1.5, linestyle="--")
            ax.text(switch_time, max(graph_error), f"  switched to reaching ({switch_time:.1f} s)",
                    color=INK_2, va="top", fontsize=9)
        ax.set_yscale("log")   # log scale, so both 100 cm and 1 cm are easy to see
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))   # plain numbers, not 10^1
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Time (s)", color=INK)
        ax.set_ylabel("End-effector to target (cm)", color=INK)
        result = "reached" if reached else "not reached"
        ax.set_title(f"Test target {TEST_INDEX}: {result}, closest {best_error * 100:.1f} cm   "
                     f"({MODEL_NAME}, seed {MODEL_SEED})", color=INK, loc="left")
        fig.tight_layout()
        graph_file = f"{results_folder}/drive_target{TEST_INDEX}.png"
        fig.savefig(graph_file, dpi=150)
        print(f"Graph saved as {graph_file}")
    except ImportError:
        print("No graph: matplotlib is missing.")


if __name__ == "__main__":
    main()