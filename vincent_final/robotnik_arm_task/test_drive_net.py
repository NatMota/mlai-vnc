# =============================================================================
# test_drive_net.py
# This is for: testing the whole model on the robot. The driving, plan and torque networks give all the
#              wheel and arm commands. The switches below swap in the learned version's networks
#              or add the robustness tests. All switches OFF is the saved model.
# How to run:  ROBOENV_SEED=1 python3 test_drive_net.py 20 val     (20 validation targets, about 5 minutes)
#              The word after the number picks the targets: val, test, new, golden_val or golden_test.
# Result:      one RESULT line per target, then a summary with close and far targets apart.
#              Saved in results/three_nets<switches>_seed<seed>/<group>_<date>_<time>/summary.txt
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   close target  a target that starts within CLOSE_DISTANCE of the robot, on the floor.
#   park_cm       how far the base stopped from the expert's parking spot.
#   face_deg      how far the base's heading was from the expert's final heading when it stopped.
#   reached       True if the tip ended within 1 cm of the target.
# -----------------------------------------------------------------------------

import datetime
import json
import os
import subprocess
import sys
import numpy as np


# SETTING: seed of the networks to test
#   Type it before the command, for example: ROBOENV_SEED=1 python3 test_drive_net.py 120 val
SEED = int(os.environ.get("ROBOENV_SEED", "0"))

# SWITCH: arm plan network
#   ON  (True):  the arm plan network gives the joint speed plan during the reach. The expert's torque
#                formula follows that plan, so a miss now comes from the driving or the plan network.
#                Results are saved in results/drive_and_plan_seed<seed>/.
#   OFF (False): the expert's arm controller does the whole reach.
USE_PLAN_NET = True

# SWITCH: torque network
#   ON  (True):  the torque network gives all 7 arm torques at every step, while driving and reaching.
#                With USE_PLAN_NET ON too, no expert code controls the robot: this is the whole model.
#                Results are saved in results/three_nets_seed<seed>/.
#   OFF (False): the expert's torque formula is used.
USE_TORQUE_NET = True
assert USE_PLAN_NET or not USE_TORQUE_NET, "USE_TORQUE_NET needs USE_PLAN_NET = True"

# SWITCHES: networks instead of analytical calculations. Type them before the command, for example:
#   ROBOENV_SEED=1 ROBOENV_TIP_NET=1 python3 test_drive_net.py 20 val
#   Each one that is ON adds a word to the results folder's name. All OFF is the original model.
# ROBOENV_TIP_NET     ON: a network gives the tip position for the plan's input. OFF: the robot model's maths.
# ROBOENV_NO_ADDING   ON: arm network "option a" gives the angle gaps, so nothing adds up the plan. OFF: the adding-up.
# ROBOENV_PARK_NET    ON: a network gives the 4 parking inputs. OFF: the parking formula.
# ROBOENV_TARGET_NET  ON: a network gives the target seen from the robot, for the reach. Needs ROBOENV_TIP_NET. OFF: rotation maths.
USE_TIP_NET = os.environ.get("ROBOENV_TIP_NET", "0") == "1"
USE_NO_ADDING = os.environ.get("ROBOENV_NO_ADDING", "0") == "1"
USE_PARK_NET = os.environ.get("ROBOENV_PARK_NET", "0") == "1"
USE_TARGET_NET = os.environ.get("ROBOENV_TARGET_NET", "0") == "1"
assert USE_TIP_NET or not USE_TARGET_NET, "ROBOENV_TARGET_NET needs ROBOENV_TIP_NET=1"
assert USE_TORQUE_NET or not USE_NO_ADDING, "ROBOENV_NO_ADDING needs USE_TORQUE_NET = True"
# ROBOENV_DIRECT_ARM  ON: during the reach, arm network "option b" goes straight to the torques, with no plan and
#                     no adding-up. The torque network only holds the arm still while driving. OFF: plan and torque networks.
USE_DIRECT_ARM = os.environ.get("ROBOENV_DIRECT_ARM", "0") == "1"
assert not (USE_DIRECT_ARM and USE_NO_ADDING), "Choose ROBOENV_DIRECT_ARM or ROBOENV_NO_ADDING, not both"
# ROBOENV_GAP_NET     ON: a third arm network gives the angle gaps from train_gap_net.py, with the plan and torque
#                     networks kept. Nothing adds up the plan. OFF: the adding-up.
USE_GAP_NET = os.environ.get("ROBOENV_GAP_NET", "0") == "1"
# ROBOENV_GAP_ROBUST  ON: uses the angle gap network trained with errors on its speed inputs (ROBOENV_SPEED_NOISE in
#                     train_gap_net.py). OFF: the normal one.
USE_GAP_ROBUST = os.environ.get("ROBOENV_GAP_ROBUST", "0") == "1"
# ROBOENV_GAP_WINDOW  ON: the angle gap network also reads the last 100 ms of joint movement and plan (a stretch of time).
#                     OFF: it only reads the current moment.
USE_GAP_WINDOW = os.environ.get("ROBOENV_GAP_WINDOW", "0") == "1"
# ROBOENV_RUNS_DIVIDE 4 uses the driving, plan and torque networks trained on a quarter of the runs. 1 uses the normal networks.
RUNS_DIVIDE = int(os.environ.get("ROBOENV_RUNS_DIVIDE", "1"))
RUNS_TAG = f"_div{RUNS_DIVIDE}" if RUNS_DIVIDE > 1 else ""
assert not (USE_GAP_NET and (USE_DIRECT_ARM or USE_NO_ADDING)), "Choose only one way to replace the adding-up"
NO_MATHS_TAG = RUNS_TAG + "".join(word for word, on in (("_tipnet", USE_TIP_NET), ("_noadding", USE_NO_ADDING), ("_directarm", USE_DIRECT_ARM),
                                             ("_gapnet", USE_GAP_NET), ("_window", USE_GAP_WINDOW), ("_robust", USE_GAP_ROBUST),
                                             ("_parknet", USE_PARK_NET), ("_targetnet", USE_TARGET_NET)) if on)

# SETTING: sensor noise, for the robustness test. 0 means no noise. Each level multiplies the sizes below.
#   Type it before the command, for example: ROBOENV_NOISE=1 python3 test_drive_net.py 50 golden_val
#   The noise only changes what the networks read. Whether a target is reached still uses the true distance.
NOISE_LEVEL = float(os.environ.get("ROBOENV_NOISE", "0"))
NOISE_JOINT_ANGLE = 0.0005     # rad
NOISE_JOINT_SPEED = 0.01       # rad/s
NOISE_BASE_POSITION = 0.001    # m
NOISE_HEADING = 0.001          # rad
if NOISE_LEVEL:
    NO_MATHS_TAG += f"_noise{NOISE_LEVEL:g}"

# SWITCH: random start, to test new starting positions.
#   ON  (ROBOENV_RANDOM_START=1): the base starts at a random place within START_RADIUS of the expert's parking spot,
#                                 facing a random direction, the same as the random_start training data.
#   OFF (not set): the base starts at the map centre.
USE_RANDOM_START = os.environ.get("ROBOENV_RANDOM_START", "0") == "1"
START_RADIUS = 0.8             # m
if USE_RANDOM_START:
    NO_MATHS_TAG += "_randomstart"

# SETTING: physical pushes, for the robustness test. 0 means none. 1 is the same size as the training data's pushes:
#   standard deviations of 0.5 rad/s on the wheels while driving, 1.5 rad/s after parking, and 3.0 N m on the arm.
#   Unlike sensor errors, these really move the robot. For example: ROBOENV_PUSH_LEVEL=0.5 python3 test_drive_net.py 50 golden_val
PUSH_LEVEL = float(os.environ.get("ROBOENV_PUSH_LEVEL", "0"))
PUSH_WHEEL_DRIVING, PUSH_WHEEL_PARKED, PUSH_ARM = 0.5, 1.5, 3.0
if PUSH_LEVEL:
    NO_MATHS_TAG += f"_push{PUSH_LEVEL:g}"

# SETTING: constant sensor error, for the robustness test. 0 means none. It uses the same sizes as the noise above.
#   Each sensor gets one random error at the start of a run, and keeps it for the whole run, like a scale that always
#   reads too high. For example: ROBOENV_BIAS=1 python3 test_drive_net.py 50 golden_val
BIAS_LEVEL = float(os.environ.get("ROBOENV_BIAS", "0"))
if BIAS_LEVEL:
    NO_MATHS_TAG += f"_bias{BIAS_LEVEL:g}" + (f"_{os.environ['ROBOENV_BIAS_ONLY']}" if os.environ.get("ROBOENV_BIAS_ONLY") else "")

# SETTING: number of targets, and which group ("val" for choosing settings, "test" is the final exam)
N_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
GROUP = sys.argv[2] if len(sys.argv) > 2 else "val"

# SETTING: new targets (GROUP = "new")
#   Targets that no model has been trained, tuned or tested on. They are spread evenly over the same
#   box as the task: x and y from -1.5 to 1.5 m, z from 0.1 to 1.25 m. The fixed seed gives every
#   model the same targets. For example: python3 test_drive_net.py 200 new
NEW_TARGETS_SEED = 2026
NEW_TARGETS_MAX = 1000

# SETTING: the file with the "val", "test", "golden_val" and "golden_test" target positions.
#   make_targets_file.py makes it, so these tests do not need the big data folders.
#   For example: python3 test_drive_net.py 100 golden_test
TARGETS_FILE = "report_targets.json"

# SETTING: how many decisions in a row must say "stop". 2 decisions = about 0.066 s.
STOP_DECISIONS = 2

# SETTING: close target distance (metres), the same as CLOSE_DISTANCE in features.py
CLOSE_DISTANCE = 0.5

# Fixed by the teacher and the expert's code. Do not change.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
ARM_REACH_LIMIT = 0.85
REACH_TOLERANCE = 0.01
MAX_STEPS = 30000


def run_one(target_index):
    """Drives to one target and prints its RESULT line. Each target runs in its own process."""
    os.environ["ROBOENV_HEADLESS"] = "1"
    import torch
    import pybullet as p
    import pinocchio as pin
    from simulation_and_control import pb, MotorCommands, PinWrapper
    from robotnik_arm_control import (ArmControllerParams, StopCriteria, arm_controller, check_stop,
                                      configure_gripper_fingers, quaternion2bearing)
    from features import STANDOFF_Z_POINTS, STANDOFF_POINTS
    from drive_net import DECIDE_EVERY, MODEL_FOLDER, MODEL_TAG, USE_IMU, DriveNet, make_inputs

    if GROUP == "new":
        target = np.random.default_rng(NEW_TARGETS_SEED).uniform(
            [-1.5, -1.5, 0.1], [1.5, 1.5, 1.25], size=(NEW_TARGETS_MAX, 3))[target_index]
    else:
        with open(TARGETS_FILE) as f:
            target = np.array(json.load(f)[GROUP][target_index], dtype=np.float64)

    saved = torch.load(f"{MODEL_FOLDER}/drive_net_seed{SEED}{MODEL_TAG}{RUNS_TAG}.pt", weights_only=False)
    net = DriveNet()
    net.load_state_dict(saved["net"])
    net.eval()

    if USE_PLAN_NET:
        import arm_plan_net
        saved_plan = torch.load(f"{arm_plan_net.MODEL_FOLDER}/arm_plan_seed{SEED}{RUNS_TAG}.pt", weights_only=False)
        plan_net = arm_plan_net.ArmPlanNet()
        plan_net.load_state_dict(saved_plan["net"])
        plan_net.eval()
    qd_plan = np.zeros(7)                    # the plan, held between decisions

    import input_nets
    tip_weights = dict(np.load("input_net_model/tip_net.npz")) if USE_TIP_NET else None

    def load_input_net(kind, n_out):
        saved_net = torch.load(f"{input_nets.MODEL_FOLDER}/{kind}_net.pt", weights_only=False)
        one_net = input_nets.InputNet(n_out)
        one_net.load_state_dict(saved_net["net"])
        one_net.eval()
        return one_net, saved_net

    def run_input_net(one_net, saved_net, x21):
        x = torch.from_numpy((input_nets.make_inputs(x21) - saved_net["x_mean"]) / saved_net["x_std"])
        with torch.no_grad():
            return one_net(x)[0].numpy() * saved_net["y_std"] + saved_net["y_mean"]

    park_net = load_input_net("park", 4) if USE_PARK_NET else None
    target_net = load_input_net("target", 3) if USE_TARGET_NET else None

    if USE_NO_ADDING:
        # Option a: 9 inputs, 14 outputs: 7 angle gaps and 7 planned speeds. Replaces the plan network.
        saved_plan = torch.load(f"arm_option_model/option_a_seed{SEED}.pt", weights_only=False)
        plan_net = input_nets.nn.Sequential(input_nets.nn.Linear(9, 128), input_nets.nn.ReLU(), input_nets.nn.Linear(128, 128),
                                            input_nets.nn.ReLU(), input_nets.nn.Linear(128, 14))
        plan_net.load_state_dict({k.replace("layers.", ""): v for k, v in saved_plan["net"].items()})
        plan_net.eval()
    angle_gap = None                         # option a's angle gaps, held between decisions

    if USE_DIRECT_ARM:
        # Option b: 16 inputs (tip gap, joint angles 1 to 6, 7 joint speeds), 7 torques out.
        saved_direct = torch.load(f"arm_option_model/option_b_seed{SEED}.pt", weights_only=False)
        direct_net = input_nets.nn.Sequential(input_nets.nn.Linear(16, 128), input_nets.nn.ReLU(), input_nets.nn.Linear(128, 128),
                                              input_nets.nn.ReLU(), input_nets.nn.Linear(128, 7))
        direct_net.load_state_dict({k.replace("layers.", ""): v for k, v in saved_direct["net"].items()})
        direct_net.eval()
    tip_gap_and_angles = None                # the 9 plan inputs: the tip gap is held between decisions

    if USE_GAP_NET:
        import gap_net
        saved_gap = torch.load(f"{gap_net.MODEL_FOLDER}/gap{'_window' if USE_GAP_WINDOW else ''}{'_robust' if USE_GAP_ROBUST else ''}_seed{SEED}.pt", weights_only=False)
        angle_gap_net = gap_net.GapNet(len(saved_gap["X_mean"]))
        angle_gap_net.load_state_dict(saved_gap["net"])
        angle_gap_net.eval()

    if USE_TORQUE_NET:
        import arm_torque_net
        torch.set_num_threads(1)             # one small network, 1000 times per second
        saved_torque = torch.load(f"{arm_torque_net.MODEL_FOLDER}/arm_torque_seed{SEED}{RUNS_TAG}.pt", weights_only=False)
        torque_net = arm_torque_net.ArmTorqueNet()
        torque_net.load_state_dict(saved_torque["net"])
        torque_net.eval()

    # Set up the simulation, the same way as when the data was collected.
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    for j in range(client.getNumJoints(rid)):
        if "wheel" in client.getJointInfo(rid, j)[1].decode():
            client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                  anisotropicFriction=[1.0, 0.01, 1.0])
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
    configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    ext_names_arr = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")
    lower_limits, upper_limits = sim.GetBotJointsLimit()
    arm_lower, arm_upper = np.array(lower_limits[4:11]), np.array(upper_limits[4:11])
    q_init_arm = np.array(sim.GetMotorAngles(0))[4:11].copy()
    q_des_arm = q_init_arm.copy()
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)

    # Where the expert would park. Only used to measure the result, never to drive.
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    expert_bearing = np.arctan2(target[1], target[0]) if np.linalg.norm(target[:2]) > 1e-4 else 0.0
    park_xy = target[:2] - standoff * np.array([np.cos(expert_bearing), np.sin(expert_bearing)])
    if USE_RANDOM_START:
        start_rng = np.random.default_rng(target_index)      # each target always gets the same start
        radius = START_RADIUS * np.sqrt(start_rng.uniform(0.0, 1.0))
        angle = start_rng.uniform(-np.pi, np.pi)
        start_xy = park_xy + radius * np.array([np.cos(angle), np.sin(angle)])
        client.resetBasePositionAndOrientation(rid, [float(start_xy[0]), float(start_xy[1]), float(sim.GetBasePosition()[2])],
                                               p.getQuaternionFromEuler([0.0, 0.0, float(start_rng.uniform(-np.pi, np.pi))]))
    start_distance = np.linalg.norm(target[:2] - (start_xy if USE_RANDOM_START else np.array(sim.GetBasePosition())[:2]))
    push_rng = np.random.default_rng(target_index + 100000)  # each target always gets the same pushes

    time_step = sim.GetTimeStep()
    current_time = 0.0
    step = 0
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=MAX_STEPS)
    cmd = MotorCommands()
    parked = False
    stop_count = 0
    wheels = np.zeros(2)                     # right wheels, left wheels
    switch_time, park_cm, face_deg = None, None, None
    best_error = np.inf
    noise_rng = np.random.default_rng(target_index)   # each target always gets the same noise
    angle_history, plan_history = [], []              # one entry per step, for the window version
    # The constant errors for this run. They are all zero when BIAS_LEVEL is 0.
    bias_base = noise_rng.normal(0, NOISE_BASE_POSITION * BIAS_LEVEL, 3)
    bias_heading = noise_rng.normal(0, NOISE_HEADING * BIAS_LEVEL)
    bias_angle = noise_rng.normal(0, NOISE_JOINT_ANGLE * BIAS_LEVEL, 7)
    bias_speed = noise_rng.normal(0, NOISE_JOINT_SPEED * BIAS_LEVEL, 7)
    # SETTING: which sensor gets the constant error: base, heading, angle or speed. Not set means all of them.
    #   For example: ROBOENV_BIAS=1 ROBOENV_BIAS_ONLY=speed python3 test_drive_net.py 50 golden_val
    bias_only = os.environ.get("ROBOENV_BIAS_ONLY", "")
    if bias_only:
        bias_base = bias_base if bias_only == "base" else np.zeros(3)
        bias_heading = bias_heading if bias_only == "heading" else 0.0
        bias_angle = bias_angle if bias_only == "angle" else np.zeros(7)
        bias_speed = bias_speed if bias_only == "speed" else np.zeros(7)

    while True:
        # Measure the robot.
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))
        q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = np.linalg.norm(target - ee_pos)
        best_error = min(best_error, ee_error)

        # The robustness test: from here on, the networks read noisy measurements.
        if NOISE_LEVEL or BIAS_LEVEL:
            base_pos = base_pos + bias_base + noise_rng.normal(0, NOISE_BASE_POSITION * NOISE_LEVEL, 3)
            base_bearing = base_bearing + bias_heading + noise_rng.normal(0, NOISE_HEADING * NOISE_LEVEL)
            q_mes, qd_mes = q_mes.copy(), qd_mes.copy()
            q_mes[4:11] += bias_angle + noise_rng.normal(0, NOISE_JOINT_ANGLE * NOISE_LEVEL, 7)
            qd_mes[4:11] += bias_speed + noise_rng.normal(0, NOISE_JOINT_SPEED * NOISE_LEVEL, 7)
            q_pin = dyn_model.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q_mes)), "pos")
            pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
            pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
            ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

        if USE_GAP_WINDOW:
            angle_history.append(q_mes[4:11].copy())
            plan_history.append(qd_plan.copy())

        # The driving network decides 30 times per second. The wheels hold their commands in between.
        if not parked and step % DECIDE_EVERY == 0:
            x21 = np.concatenate((base_pos, [base_bearing], q_mes[4:11], qd_mes[4:11], target))[None, :]
            if USE_IMU:
                lin_vel, ang_vel = client.getBaseVelocity(rid)   # the IMU readings, from the simulation
                raw = make_inputs(x21, base_ori[None, :], np.array(lin_vel)[None, :], np.array(ang_vel)[None, :])
            else:
                raw = make_inputs(x21)
            if USE_PARK_NET:
                raw[0, :4] = run_input_net(*park_net, x21)
            x = torch.from_numpy((raw - saved["X_mean"]) / saved["X_std"])
            with torch.no_grad():
                out = net(x)[0].numpy()
            wheels = np.clip(out[:2] * saved["Y_std"] + saved["Y_mean"], saved["Y_min"], saved["Y_max"])
            stop_count = stop_count + 1 if out[2] > 0 else 0
            if stop_count >= STOP_DECISIONS:
                parked = True
                wheels = np.zeros(2)
                switch_time = current_time
                park_cm = np.linalg.norm(base_pos[:2] - park_xy) * 100
                face_deg = abs(np.degrees(np.arctan2(np.sin(expert_bearing - base_bearing),
                                                     np.cos(expert_bearing - base_bearing))))

        if USE_PLAN_NET and parked:
            # The plan network decides 30 times per second. The plan is held in between.
            if step % arm_plan_net.DECIDE_EVERY == 0:
                x21 = np.concatenate((base_pos, [base_bearing], q_mes[4:11], qd_mes[4:11], target))[None, :]
                x = arm_plan_net.make_inputs(x21, ee_pos[None, :])
                if USE_TIP_NET:
                    # The tip gap: target minus tip, both seen from the robot. Height is measured from the floor.
                    tip = input_nets.tip_from_net(tip_weights, q_mes[4:11])
                    if USE_TARGET_NET:
                        seen_target = run_input_net(*target_net, x21)
                    else:
                        c, s = np.cos(base_bearing), np.sin(base_bearing)
                        d = target[:2] - base_pos[:2]
                        seen_target = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], target[2]])
                    x[0, :3] = seen_target - tip - np.array([0, 0, base_pos[2]])
                tip_gap_and_angles = x[0].copy()
                if not USE_DIRECT_ARM:
                    x = torch.from_numpy((x - saved_plan["X_mean"]) / saved_plan["X_std"])
                    with torch.no_grad():
                        out = plan_net(x)[0].numpy()
                    out = np.clip(out * saved_plan["Y_std"] + saved_plan["Y_mean"], saved_plan["Y_min"], saved_plan["Y_max"])
                    if USE_NO_ADDING:
                        angle_gap, qd_plan = out[:7], out[7:]
                    else:
                        qd_plan = out
            q_arm, qd_arm = q_mes[4:11], qd_mes[4:11]
            if USE_DIRECT_ARM:
                # The tip gap is held between decisions. The joint angles and speeds are new at every step.
                x = np.r_[tip_gap_and_angles[:3], q_arm[:6], qd_arm].astype(np.float32)
                x = torch.from_numpy((x - saved_direct["X_mean"]) / saved_direct["X_std"])[None, :]
                with torch.no_grad():
                    out = direct_net(x)[0].numpy()
                torque_arm = np.clip(out * saved_direct["Y_std"] + saved_direct["Y_mean"], saved_direct["Y_min"], saved_direct["Y_max"])
            elif USE_GAP_NET:
                # The third arm network gives the angle gap at every step, so nothing adds up the plan.
                x = gap_net.make_inputs(tip_gap_and_angles[:3][None, :], q_arm[None, :], qd_arm[None, :], qd_plan[None, :])
                if USE_GAP_WINDOW:
                    past = [max(len(angle_history) - 1 - j * gap_net.WINDOW_MS, 0) for j in range(1, gap_net.WINDOW_STEPS + 1)]
                    q_past = np.array([angle_history[i] for i in past])[None, :]
                    plan_past = np.array([plan_history[i] for i in past])[None, :]
                    x = np.c_[x, gap_net.window_inputs(q_arm[None, :], q_past, plan_past)]
                x = torch.from_numpy((x - saved_gap["X_mean"]) / saved_gap["X_std"])
                with torch.no_grad():
                    out = angle_gap_net(x)[0].numpy()
                q_des_arm = q_arm + np.clip(out * saved_gap["Y_std"] + saved_gap["Y_mean"], saved_gap["Y_min"], saved_gap["Y_max"])
                torque_arm = np.zeros(7)                 # replaced by the torque network below
            elif USE_NO_ADDING:
                # The network's angle gap is held between decisions, so nothing adds up the plan.
                q_des_arm = q_arm + angle_gap
                torque_arm = np.zeros(7)                 # replaced by the torque network below
            else:
                # The expert's own steps, from arm_controller: add up the plan, keep it near the arm and
                # inside the joint limits, then the joint PD law with gravity compensation.
                q_des_arm = np.clip(q_des_arm + qd_plan * time_step,
                                    q_arm - arm_params.max_q_lead, q_arm + arm_params.max_q_lead)
                q_des_arm = np.where(arm_lower < arm_upper,
                                     np.clip(q_des_arm, arm_lower + arm_params.limit_margin,
                                             arm_upper - arm_params.limit_margin),
                                     q_des_arm)
                tau_g = pin.computeGeneralizedGravity(dyn_model.pin_model, dyn_model.pin_data, q_pin)[6:13]
                torque_arm = arm_params.kp * (q_des_arm - q_arm) + arm_params.kd * (qd_plan - qd_arm) + tau_g
                torque_arm = np.clip(torque_arm, -arm_params.max_torque, arm_params.max_torque)
        elif USE_NO_ADDING or USE_DIRECT_ARM or USE_GAP_NET:
            # While driving: hold the starting pose. The torque network below gives the torques.
            q_des_arm = q_init_arm.copy()
            torque_arm = np.zeros(7)
        else:
            # The expert's arm controller: holds the arm while driving, reaches once parked.
            torque_arm, q_des_arm, _ = arm_controller(
                dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, target, ee_pos, ee_frame_id,
                shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params, reach_active=parked)

        # The torque network replaces the torques worked out above. While driving, the plan is
        # "hold the starting pose" (q_des_arm = q_init_arm, qd_plan = 0), as in the training data.
        if USE_TORQUE_NET and not (USE_DIRECT_ARM and parked):
            x = arm_torque_net.make_inputs(q_des_arm[None, :], qd_plan[None, :],
                                           q_mes[4:11][None, :], qd_mes[4:11][None, :])
            x = torch.from_numpy((x - saved_torque["X_mean"]) / saved_torque["X_std"])
            with torch.no_grad():
                out = torque_net(x)[0].numpy()
            torque_arm = np.clip(out * saved_torque["Y_std"] + saved_torque["Y_mean"],
                                 saved_torque["Y_min"], saved_torque["Y_max"])

        wheel_cmds = np.array([wheels[0], wheels[1], wheels[1], wheels[0]])
        if PUSH_LEVEL:
            wheel_size = (PUSH_WHEEL_PARKED if parked else PUSH_WHEEL_DRIVING) * PUSH_LEVEL
            wheel_cmds = wheel_cmds + push_rng.normal(0.0, wheel_size, 4)
            torque_arm = torque_arm + push_rng.normal(0.0, PUSH_ARM * PUSH_LEVEL, 7)
        cmd.SetControlCmd(np.concatenate((wheel_cmds, torque_arm, np.zeros(6))),
                          ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6)
        sim.Step(cmd, "torque")

        if check_stop(stop_criteria, current_time, time_step, ee_error, parked):
            break
        current_time += time_step
        step += 1

    # Print results: the RESULT line that the summary below reads.
    print(f"RESULT target={target_index} start_m={start_distance:.2f} reached={ee_error < REACH_TOLERANCE} "
          f"final_cm={ee_error * 100:.1f} best_cm={best_error * 100:.1f} parked={parked} switch_s={switch_time} "
          f"park_cm={park_cm if park_cm is None else round(park_cm, 1)} "
          f"face_deg={face_deg if face_deg is None else round(face_deg, 1)}")


def summary_lines(results):
    """Turns the RESULT lines into a summary, with close and far targets apart."""
    def get(line, key):
        return line.split(key + "=")[1].split()[0]

    def score(lines, name):
        reached = sum(get(l, "reached") == "True" for l in lines)
        parked = [l for l in lines if get(l, "parked") == "True"]
        text = f"{name}: reached {reached} of {len(lines)}, stopped {len(parked)} of {len(lines)}"
        if parked:
            text += (f", average park error {np.mean([float(get(l, 'park_cm')) for l in parked]):.1f} cm"
                     f", average heading error {np.mean([float(get(l, 'face_deg')) for l in parked]):.1f} deg")
        return text

    close = [l for l in results if float(get(l, "start_m")) < CLOSE_DISTANCE]
    far = [l for l in results if float(get(l, "start_m")) >= CLOSE_DISTANCE]
    return [score(results, "All targets"), score(close, "Close targets"), score(far, "Far targets")]


if __name__ == "__main__":
    # One target only (test_drive_net.py runs itself once per target, with a new simulation each time).
    if len(sys.argv) > 3:
        run_one(int(sys.argv[3]))
        sys.exit(0)

    results = []
    for i in range(N_TARGETS):
        out = subprocess.run([sys.executable, sys.argv[0], str(N_TARGETS), GROUP, str(i)],
                             capture_output=True, text=True)
        lines = [l for l in out.stdout.splitlines() if l.startswith("RESULT")]
        # Print results: the RESULT line, or the error so it can be fixed.
        print(lines[0] if lines else f"Target {i} crashed:\n{out.stderr[-1000:]}", flush=True)
        results += lines

    summary = results + summary_lines(results)
    print("\n".join(summary[-3:]))

    # Save the summary in a new folder.
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    name = "three_nets" if USE_TORQUE_NET else ("drive_and_plan" if USE_PLAN_NET else "drive_net")
    from drive_net import MODEL_TAG
    folder = f"results/{name}{MODEL_TAG}{NO_MATHS_TAG}_seed{SEED}/{GROUP}_{stamp}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/summary.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"Results saved in {folder}/")
