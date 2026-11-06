# =============================================================================
# dagger.py
# This is for: DAgger. The MODEL drives to new random targets, and the EXPERT says
#              what it would do at every moment. The expert's answers are saved
#              as new training data, so the model can learn to fix its own mistakes.
# How to run:  python3 dagger.py 100     (100 new runs)
#              If stopped, run it again: it carries on, and keeps the runs already saved.
# Result:      a new folder, data_dagger_arm, with one .npy file per run
#              (the same 32 columns as data_dart), and data_dagger_arm_ee with the
#              end-effector positions.
#              Only the REACHING part of each run is saved (after the model stopped driving).
#              Why: the expert's base part has a memory (turning, driving, parking).
#              When the model drives its own way, that memory gets out of step,
#              so its wheel answers do not fit the moment (this spoiled the first tries).
#              The arm answers after the stop do not have this problem.
#              The expert's arm part also has a memory: its plan of where the arm should be.
#              It is reset to where the arm really is, every step, so it can never fall
#              behind and say "come back" (this pulled the arm away from the target in try 3).
#              Then set USE_DAGGER = True in features.py and run train.py.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   DAgger        the model drives, the expert gives the right answer, the model learns from it.
#   model         your trained model (phase_net + control_net). It drives the robot.
#   expert        the teacher's controller (robotnik_arm_control.py). It does NOT drive here.
#                 It only works out what it WOULD do, and that is saved.
#   expert_wheels the expert's 4 wheel commands at this moment (saved as the right answer).
#   expert_arm    the expert's 7 arm torques at this moment (saved as the right answer).
#   model_wheels  the model's 4 wheel commands (these are sent to the robot).
#   model_arm     the model's 7 arm torques (these are sent to the robot).
#   run number    each run gets a number. It also picks the random target, so runs can be repeated.
#   row           one moment of a run: what the robot measured + the expert's 11 answers.
# -----------------------------------------------------------------------------

import os
import sys
import glob
import json
import subprocess
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

# SETTING: number of new runs
#   Type it after the command, for example: python3 dagger.py 100
#   One run takes about 10 to 30 seconds.
N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "--one" else 100

# SETTING: data folder (the model is loaded from here)
#   Must be the same as DATA_FOLDER in train.py.
DATA_FOLDER = "data_dart"

# SETTING: DAgger folder (where the new runs are saved)
#   Must be the same as DAGGER_FOLDER in features.py.
OUT_FOLDER = "data_dagger_arm2"

# SETTING: max time for one run, in steps. 30000 steps = 30 seconds.
MAX_STEPS = 30000


# =============================================================================
# PART A: the loop. Runs PART B once per run, each in its own Python,
# the same way test.py runs drive.py. This keeps each run clean.
# =============================================================================

def run_all():
    os.makedirs(OUT_FOLDER, exist_ok=True)
    os.makedirs(OUT_FOLDER + "_ee", exist_ok=True)
    env = os.environ.copy()
    env["ROBOENV_HEADLESS"] = "1"   # no window, so it is faster
    for k in range(N_RUNS):
        out_file = os.path.join(OUT_FOLDER, f"dagger_{k:04d}.npy")
        skip_file = os.path.join(OUT_FOLDER + "_ee", f"dagger_{k:04d}.none")   # a run that never reached
        if os.path.exists(out_file) or os.path.exists(skip_file):
            print(f"Run {k + 1} of {N_RUNS}: already saved, skipped")
            continue
        out = subprocess.run([sys.executable, "dagger.py", "--one", str(k)],
                             env=env, capture_output=True, text=True)
        # Print results: the summary line of each run, or the error.
        lines = [line for line in out.stdout.splitlines() if line.startswith("DAGGER")]
        print(f"Run {k + 1} of {N_RUNS}: " + (lines[-1] if lines else "no result"))
        if out.returncode != 0:
            print(out.stderr[-1000:])
    n_saved = len(glob.glob(os.path.join(OUT_FOLDER, "*.npy")))
    print(f"\nDone. {n_saved} DAgger runs saved in {OUT_FOLDER}/")
    print("Next: set USE_DAGGER = True in features.py, then run train.py.")


# =============================================================================
# PART B: one run. The model drives, the expert answers.
# =============================================================================

def run_one(run_number):
    import torch
    import torch.nn as nn
    import pybullet as p
    import pinocchio as pin
    from simulation_and_control import pb, MotorCommands, PinWrapper
    from robotnik_arm_control import (
        ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
        arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
    )
    from features import add_relative, add_ee, N_INPUTS, N_INPUTS_EE, PHASE_COLS, USE_EE, MODEL_NAME, build_control_net

    # ---- Fixed numbers, the same as in the teacher's task file. Do not change. ----
    FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    BASE_NAV_TIMEOUT = 25.0
    ARM_REACH_LIMIT = 0.85
    STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
    STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
    REACH_TOLERANCE = 0.01
    SWITCH_STEPS = 50   # same as drive.py
    N_IN = N_INPUTS_EE if USE_EE else N_INPUTS

    # ---- STEP 1: a new random target. The run number picks it, so it can be repeated. ----
    # The target comes from the same box as the expert's runs, but it is NEW:
    # never a test or validation target.
    rng = np.random.default_rng(10000 + run_number)
    TARGET_EE_POS = np.array([rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5), rng.uniform(0.1, 1.25)])

    # ---- STEP 2: load the model (the same way as drive.py). ----
    phase_net = nn.Sequential(
        nn.Linear(len(PHASE_COLS), 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )
    control_net = build_control_net(N_IN + 1)
    saved = torch.load(f"{DATA_FOLDER}/{MODEL_NAME}_best.pt")
    phase_net.load_state_dict(saved["phase_net"])
    control_net.load_state_dict(saved["control_net"])
    phase_net.eval()
    control_net.eval()
    scaler = np.load(f"{DATA_FOLDER}/scaler_{MODEL_NAME}.npz")
    X_mean, X_std = scaler["X_mean"], scaler["X_std"]
    Y_mean, Y_std = scaler["Y_mean"], scaler["Y_std"]
    with open(f"{DATA_FOLDER}/split.json") as f:
        split = json.load(f)
    Y_min, Y_max = np.full(11, np.inf), np.full(11, -np.inf)
    for f in split["train"]:
        Y_file = np.load(f, mmap_mode="r")[:, 21:]
        Y_min = np.minimum(Y_min, Y_file.min(axis=0))
        Y_max = np.maximum(Y_max, Y_file.max(axis=0))

    # ---- STEP 3: set up the simulation (the same as the teacher's code). ----
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [j for j in range(client.getNumJoints(rid))
                       if "wheel" in client.getJointInfo(rid, j)[1].decode()]
    for j in wheel_joint_ids:
        client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                              anisotropicFriction=[1.0, 0.01, 1.0])
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    time_step = sim.GetTimeStep()

    # ---- STEP 4: set up the expert (the same as the teacher's code). ----
    # It is only asked for answers. It never drives the robot.
    standoff = float(np.interp(TARGET_EE_POS[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = TARGET_EE_POS[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_state = BaseNavState()
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    init_joint_angles = np.array(sim.GetMotorAngles(0))
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])

    # ---- STEP 5: the model drives, the expert answers, every step. ----
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=MAX_STEPS)
    cmd = MotorCommands()
    current_time = 0.0
    phase = 0.0          # the MODEL's phase: 0 = driving, 1 = reaching
    reach_count = 0
    rows, ee_rows = [], []
    best_error = np.inf

    while True:
        # Measure the robot.
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(TARGET_EE_POS - ee_pos)
        best_error = min(best_error, ee_dist_error)

        # 1st: ask the EXPERT what it would do here (saved, NOT sent to the robot).
        expert_wheels = base_controller(base_state, base_pos, base_bearing, base_goal_pos,
                                        target_bearing, current_time, base_params)
        # The expert reaches when the MODEL has switched to reaching.
        # (In the first try it did not know the model had stopped, and kept saying
        # "fold the arm, we are still driving". The model learned to pull its arm away.)
        reaching = phase == 1.0
        if phase == 1.0:
            expert_wheels = np.zeros(4)   # the model has parked, so the right wheel answer is "stay still"
        if reaching:
            q_des_arm = q_mes[4:11].copy()   # restart the expert's arm plan from where the arm really is
        expert_arm, q_des_arm, _ = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, TARGET_EE_POS, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=reaching,
        )

        # 2nd: save this moment, in the same 32 columns as data_dart.
        #      Only while reaching (see the top of this file for why).
        if phase == 1.0:
            rows.append(np.concatenate((base_pos, [base_bearing], q_mes[4:11], qd_mes[4:11], TARGET_EE_POS,
                                        expert_wheels, expert_arm)))
            ee_rows.append(ee_pos)

        # 3rd: ask the MODEL what to do (the same way as drive.py).
        x = np.concatenate((base_pos, [base_bearing], q_mes[4:11], qd_mes[4:11], TARGET_EE_POS))
        x = add_relative(x[None, :])
        if USE_EE:
            x = add_ee(x, ee_pos[None, :])
        x_s = torch.tensor((x[0] - X_mean) / X_std, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if phase == 0.0:
                p_reach = torch.sigmoid(phase_net(x_s[:, PHASE_COLS])).item()
                reach_count = reach_count + 1 if p_reach > 0.5 else 0
                if reach_count >= SWITCH_STEPS:
                    phase = 1.0
            y_s = control_net(torch.cat([x_s, torch.tensor([[phase]])], dim=1))[0, :11].numpy()
        y = np.clip(y_s * Y_std + Y_mean, Y_min, Y_max)
        model_wheels = np.zeros(4) if phase == 1.0 else y[:4]
        model_arm = y[4:]

        # 4th: send the MODEL's commands to the robot (fingers stay fixed).
        cmd_all = np.concatenate((model_wheels, model_arm, np.zeros(6)))
        cmd.SetControlCmd(cmd_all, ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6)
        sim.Step(cmd, "torque")

        # 5th: stop if reached, stuck, or out of time (the same rules as drive.py).
        if check_stop(stop_criteria, current_time, time_step, ee_dist_error, phase == 1.0):
            break
        current_time += time_step

    # ---- STEP 6: save the run. ----
    name = f"dagger_{run_number:04d}.npy"
    if not rows:
        # The model never switched to reaching, so there is nothing to save. Leave a note, so it is skipped next time.
        open(os.path.join(OUT_FOLDER + "_ee", name.replace(".npy", ".none")), "w").close()
        print(f"DAGGER run={run_number} rows=0 (never switched to reaching) best_error_cm={best_error * 100:.1f}")
        return
    np.save(os.path.join(OUT_FOLDER + "_ee", name), np.array(ee_rows, dtype=np.float32))
    np.save(os.path.join(OUT_FOLDER, name), np.array(rows))   # saved last, so a half run is never kept
    # Print results: one line for this run (dagger.py reads it).
    print(f"DAGGER run={run_number} rows={len(rows)} time_s={current_time:.1f} "
          f"final_error_cm={ee_dist_error * 100:.1f} best_error_cm={best_error * 100:.1f} "
          f"expert_parked={base_state.done}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--one":
        run_one(int(sys.argv[2]))
    else:
        run_all()
