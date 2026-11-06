# =============================================================================
# robotnik_arm_task_dart.py
# This is for: one expert run for data collection. It is a copy of the teacher's robotnik_arm_task.py, with
#              DART pushes, targets spread evenly by distance, and random starts near the parking spot added.
# How to run:  collect.py runs it, so you do not normally run it yourself.
#              collect.py chooses the settings with ROBOENV_PUSH, ROBOENV_TARGETS and ROBOENV_START.
# Result:      one run saved as a .npy file in a data_v2 folder (see OUT_FOLDER below).
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   DART    adding random pushes to the commands sent to the robot, while saving the expert's clean commands,
#           so the data shows the expert recovering when the robot goes off course.
#   push    one random change added to a wheel or arm command.
#   target  the position the arm tip must reach.
# -----------------------------------------------------------------------------

import numpy as np
import time
import os

# Headless mode and step cap for CI / direct runs
headless = os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes")
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

import matplotlib
if headless:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)


# ==============================================================================
# TARGET SPECIFICATION
# Fixed 3D Cartesian coordinates for the Kinova end-effector in the world frame:
# [X (forward/meters), Y (lateral/meters), Z (height/meters)]
# TARGET_EE_POS = np.array([0.0, 1.1, 0.5])
# Note: this fixed target is replaced by the random target below

# Make a random number generator
rng = np.random.default_rng()

# SETTING: how the targets are spread (chosen by collect.py)
#   "square" = a random point in the 3 m x 3 m square around the robot's start (as in all the
#              older data). Close targets are rare: only about 2% land within 25 cm of the robot.
#   "even"   = a random direction and a random distance (0 to 2.12 m), kept only if it is inside
#              the same square. Near and far targets are about equally common up to 1.5 m, and the
#              square's corners are still covered (like the test targets). check_rare.py showed the
#              model failed on close targets because they were rare, so this balances the data.
TARGETS = os.environ.get("ROBOENV_TARGETS", "square")

# SETTING: where the robot starts (chosen by collect.py)
#   "centre" = at the map centre, facing +x (as in all the older data, and in every test).
#   "random" = somewhere near the expert's parking spot, facing a random direction.
#              Watching the model on validation target 13 showed it can end up facing the target
#              but too close, a situation the expert never shows (it always starts at the centre).
#              Starting the expert in many such awkward places records how it recovers
#              (start-state randomisation, the same idea as the DART pushes).
START = os.environ.get("ROBOENV_START", "centre")
START_RADIUS = 0.8   # metres: the random start is within this distance of the parking spot


def pick_target_xy():
    """Target x, y (world frame, m) for one run, spread as set by TARGETS."""
    if TARGETS == "even":
        while True:
            distance = rng.uniform(0.0, 1.5 * np.sqrt(2))   # up to the square's corners (2.12 m)
            direction = rng.uniform(-np.pi, np.pi)
            x, y = distance * np.cos(direction), distance * np.sin(direction)
            if abs(x) <= 1.5 and abs(y) <= 1.5:              # keep it inside the square
                return x, y
    return rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)


# Pick a new random target (shown as a red sphere) for each run
_x, _y = pick_target_xy()
TARGET_EE_POS = np.array([
    _x,                       # x: forwards/backwards (metres)
    _y,                       # y: left/right (metres)
    rng.uniform(0.1, 1.25),   # z: height (metres), matches the expert's standoff table
])

# Fixed gripper finger configuration (angles in radians, held fixed during simulation):
# [finger_1, finger_tip_1, finger_2, finger_tip_2, finger_3, finger_tip_3]
# Set to None to use initial configuration from config/URDF, or an array of 6 angles.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Max time (s) allowed for the base to reach its standoff pose before the arm starts reaching.
# Targets behind the robot need a large turn, so this must be generous.
BASE_NAV_TIMEOUT = 25.0
# Max EE distance from the shoulder the arm is asked to reach (m). Targets further away are projected
# onto this sphere, so the arm settles at the closest pose instead of fully stretching (singularity).
ARM_REACH_LIMIT = 0.85
# Base-to-target horizontal standoff (m) as a function of target height: larger for low targets,
# default for mid heights, smaller for high targets (far above the shoulder, which sits ~0.49 m high).
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
# Stop once the EE is this close to the target (m); max iterations set via ROBOENV_MAX_STEPS
# SETTING: stopping distance. Chosen by collect.py: "finish" uses 0.005, so the expert keeps going to 5 mm.
#   The expert's control code does not change, only when the run ends.
REACH_TOLERANCE = float(os.environ.get("ROBOENV_REACH_TOLERANCE", "0.01"))

# SETTING: push size (DART)
#   Chosen by collect.py (python3 collect.py big 800  or  python3 collect.py medium 900).
#   "big"    = the same as data_dart:  wheels 1.5, arm 3.0, wheels pushed ALL the time
#              (so the base also slides after parking).
#   "medium" = the same as data_dart2: wheels 0.5, arm 1.0, wheels pushed only while driving.
#   "mixed"  = the good parts of both in one: wheels 0.5 while driving (the expert gets lost less),
#              wheels 1.5 after parking (the base slides, like data_dart), arm 3.0 (lots of recovery).
#   WHEEL_PUSH = wheel push while driving, PARKED_WHEEL_PUSH = wheel push after parking (0 = none).
PUSH = os.environ.get("ROBOENV_PUSH", "medium")
if PUSH == "big":
    WHEEL_PUSH, PARKED_WHEEL_PUSH, ARM_PUSH = 1.5, 1.5, 3.0
elif PUSH == "mixed":
    WHEEL_PUSH, PARKED_WHEEL_PUSH, ARM_PUSH = 0.5, 1.5, 3.0
else:
    WHEEL_PUSH, PARKED_WHEEL_PUSH, ARM_PUSH = 0.5, 0.0, 1.0

# SETTING: data folder
#   Where the runs are saved. Version 2 of the data, with two folders inside:
#   data_v2/big_push and data_v2/medium_push. data_dart and data_dart2 stay as they are.
OUT_FOLDER = os.path.join("data_v2", {"big": "big_push", "mixed": "mixed_push"}.get(PUSH, "medium_push"))
if TARGETS == "even":
    OUT_FOLDER = os.path.join("data_v2", "balanced")   # one balanced dataset (mixed pushes, even targets)
if START == "random":
    OUT_FOLDER = os.path.join("data_v2", "random_start")   # balanced targets, mixed pushes, random starts
if REACH_TOLERANCE != 0.01:
    OUT_FOLDER = os.path.join("data_v2", "finish_5mm")   # balanced targets, mixed pushes, the expert keeps going to 5 mm

# A note inside data_v2, so anyone can see what each folder holds.
DATA_V2_NOTE = """data_v2: version 2 of the training data. Made by robotnik_arm_task_dart.py and collect.py.

In every run, the expert drives to a random target. Random pushes are added to its commands
(this is called DART), but the expert's clean commands are saved. The pushes show the expert
fixing mistakes, so the model can learn that too.

big_push/
    Big pushes: wheels 1.5 rad/s, arm 3.0 N m.
    The wheels are pushed all the time, so the base also slides after parking.
    The same pushes as the older data_dart folder.
    Made with: python3 collect.py big 800

medium_push/
    Medium pushes: wheels 0.5 rad/s, arm 1.0 N m.
    The wheels are only pushed while driving, so the base stays still after parking.
    The same pushes as the older data_dart2 folder.
    Made with: python3 collect.py medium 900

mixed_push/
    The good parts of both: wheels 0.5 rad/s while driving, 1.5 rad/s after parking, arm 3.0 N m.
    The expert gets lost less, but the base still slides after parking.
    Made with: python3 collect.py mixed 1000 good 3

random_start/
    Like balanced/ (mixed pushes, even targets), but the robot starts at a random place within
    0.8 m of the expert's parking spot, facing a random direction. It records how the expert
    recovers from awkward places, like facing the target while too close.
    Meant to be used together with balanced/ for training.
    Made with: python3 collect.py randomstart 500 good 4

balanced/
    ONE balanced dataset, meant to be used on its own for training.
    Mixed pushes (as mixed_push), and the targets are spread evenly by distance:
    a random direction and a random distance, kept inside the same 3 m x 3 m square.
    So close targets (the ones the model failed on) are as common as far ones.
    Made with: python3 collect.py balanced 1500 good 4

Only GOOD runs are saved (the expert ended within 2 cm of the target), in float32.

Each .npy file is one run, with one row per moment (1000 rows per second) and 63 columns:
    0-2    base position x, y, z (m)
    3      base bearing (rad)
    4-10   arm joint angles, q (rad)
    11-17  arm joint speeds, qd (rad/s)
    18-20  target position x, y, z (m)
    21-24  expert wheel commands (rad/s)
    25-31  expert arm torques (N m)
    32-35  base_ori: real tilt of the base (quaternion x, y, z, w)   NEW in version 2
    36-38  ee_pos: real end-effector position x, y, z (m)          NEW in version 2
    39-45  q_des: the expert's arm plan, joint angles (rad)       NEW in version 2
    46-52  qd_des: the expert's speed plan, joint speeds (rad/s)  NEW in version 2
    53-55  base linear velocity x, y, z (world, m/s)            NEW in version 2 (IMU)
    56-58  base angular velocity x, y, z (world, rad/s)         NEW in version 2 (IMU)
    59-62  wheel speeds (rad/s)                                 NEW in version 2 (wheel encoders)
"""
# ==============================================================================


def main():
    # --------------------------------------------------------------------------
    # Configuration and Simulation Setup
    # --------------------------------------------------------------------------
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    # Configure realistic ground contact and skid-steer friction dynamics
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

    # --------------------------------------------------------------------------
    # Random start (only when START = "random"): move the robot to a random place near the
    # expert's parking spot, facing a random direction, before anything is measured.
    # The parking spot is worked out the same way as the expert does below.
    # --------------------------------------------------------------------------
    if START == "random":
        start_standoff = float(np.interp(TARGET_EE_POS[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        t_xy = TARGET_EE_POS[:2]
        t_bearing = np.arctan2(t_xy[1], t_xy[0]) if np.linalg.norm(t_xy) > 1e-4 else 0.0
        park_xy = t_xy - start_standoff * np.array([np.cos(t_bearing), np.sin(t_bearing)])
        r = START_RADIUS * np.sqrt(rng.uniform(0.0, 1.0))   # even spread inside the circle
        a = rng.uniform(-np.pi, np.pi)
        start_xy = park_xy + r * np.array([np.cos(a), np.sin(a)])
        start_yaw = rng.uniform(-np.pi, np.pi)               # facing a random direction
        start_z = sim.GetBasePosition()[2]                   # keep the normal height
        client.resetBasePositionAndOrientation(
            rid, [float(start_xy[0]), float(start_xy[1]), float(start_z)],
            p.getQuaternionFromEuler([0.0, 0.0, float(start_yaw)]))
        print(f"Random start: base at {np.round(start_xy, 2)}, facing {np.degrees(start_yaw):.0f} degrees "
              f"(parking spot {np.round(park_xy, 2)})")

    # --------------------------------------------------------------------------
    # Gripper Fingers: Fix gripper finger configuration
    # --------------------------------------------------------------------------
    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    # Visualize the target reach reference in the PyBullet 3D environment as a target
    target_pos_list = TARGET_EE_POS.tolist()
    visual_shape_id = client.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.03,
        rgbaColor=[1.0, 0.0, 0.0, 0.85]
    )
    client.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=target_pos_list
    )
    client.addUserDebugPoints(
        pointPositions=[target_pos_list],
        pointColorsRGB=[[1.0, 0.0, 0.0]],
        pointSize=10
    )

    # Active joints from the simulation
    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    source_names = ["pybullet"]

    # Pinocchio dynamic & kinematic model
    dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, source_names, False, 0, cur_dir)

    controlled_frame_name = "robot_j2s7s300_end_effector"
    ee_frame_id = dyn_model.pin_model.getFrameId(controlled_frame_name)
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())

    # Forward kinematics for initial end-effector position
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    print("==================================================================")
    print(f"Target Cartesian EE Position : {TARGET_EE_POS}")
    print(f"Initial Cartesian EE Position: {np.round(init_ee_pos, 4)}")
    print(f"Initial Cartesian Error (m)  : {np.linalg.norm(TARGET_EE_POS - init_ee_pos):.4f}")
    print("==================================================================")

    # Joint limits
    lower_limits, upper_limits = sim.GetBotJointsLimit()

    time_step = sim.GetTimeStep()
    current_time = 0

    # --------------------------------------------------------------------------
    # Coordinated Motion Planning: Base Navigation & Arm Reach
    # --------------------------------------------------------------------------
    # Standoff distance between robot base center and target, depending on target height.
    standoff = float(np.interp(TARGET_EE_POS[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = TARGET_EE_POS[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0

    # Mobile base goal location facing the target
    base_goal_pos = target_xy - standoff * np.array([np.cos(target_bearing), np.sin(target_bearing)])
    base_goal_bearing = target_bearing

    # Base navigation controller (align -> drive -> orient -> done), see BaseControllerParams
    base_state = BaseNavState()
    base_navigation_done = base_state.done
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)

    # Arm controller (differential IK + joint PD), see ArmControllerParams
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    # Stop conditions (step cap, target reached, unreachable-target stall), see StopCriteria
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)

    # Command interface
    cmd = MotorCommands()

    # Data logging for plotting
    time_all = []
    ee_pos_all = []
    ee_error_all = []
    base_pos_all = []
    q_mes_all = []
    qd_mes_all = []
    q_d_all = []
    qd_d_all = []
    data_rows = [] # Makes a list. A row of numbers is added to it at every moment of the run.

    # --------------------------------------------------------------------------
    # Main Simulation Loop
    # --------------------------------------------------------------------------
    while True:
        # Measure current states
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        # Full kinematic state vector in Pinocchio format
        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")

        # Compute end-effector forward kinematics
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_error = TARGET_EE_POS - ee_pos
        ee_dist_error = np.linalg.norm(ee_error)

        # ----------------------------------------------------------------------
        # 1. Mobile Base Control (Wheel velocities)
        # ----------------------------------------------------------------------
        wheel_cmds = base_controller(
            base_state, base_pos, base_bearing, base_goal_pos, base_goal_bearing, current_time, base_params
        )
        base_navigation_done = base_state.done

        # ----------------------------------------------------------------------
        # 2. Arm Control (reaching via differential IK + gravity-compensated joint PD)
        # ----------------------------------------------------------------------
        # While the base is travelling the arm holds its nominal configuration.
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model, q_pin, q_mes[4:11], qd_mes[4:11], q_des_arm, q_init_arm, TARGET_EE_POS, ee_pos,
            ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params,
            reach_active=base_navigation_done,
        )

        # ----------------------------------------------------------------------
        # 3. Gripper / Finger Control (Fingers locked at fixed configuration)
        # ----------------------------------------------------------------------
        torque_fingers = np.zeros(6)

        # ----------------------------------------------------------------------
        # Send Multi-Interface Control Commands
        # ----------------------------------------------------------------------
        # 4 wheels (velocity) + 7 arm joints (torque) + 6 fingers (torque)
        # cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))

        # DART: add a random push to the commands sent to the robot.
        # While driving: WHEEL_PUSH. After parking: PARKED_WHEEL_PUSH (0 for "medium", so the base stays still).
        wheel_push_now = PARKED_WHEEL_PUSH if base_navigation_done else WHEEL_PUSH
        push_wheel = wheel_cmds + rng.normal(0.0, wheel_push_now, 4) if wheel_push_now > 0 else wheel_cmds
        push_arm = torque_arm + rng.normal(0.0, ARM_PUSH, 7)
        cmd_all = np.concatenate((push_wheel, push_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        # Makes one row of numbers for this moment and adds it to the list.
        # Columns 0 to 31 are the same as before, so the old code still works:
        #   0-2 base position, 3 bearing, 4-10 arm angles, 11-17 arm speeds, 18-20 target,
        #   21-24 wheel commands, 25-31 arm torques.
        # Columns 32 to 52 are new (from the teammate's idea):
        #   32-35 base_ori: the real tilt of the base (quaternion x, y, z, w)
        #   36-38 ee_pos:   the real end-effector position (x, y, z)
        #   39-45 q_des:    the expert's arm plan (the joint angles it wants)
        #   46-52 qd_des:   the expert's speed plan (the joint speeds it wants)
        # Columns 53 to 62 are the IMU and wheel sensors (Nat's idea):
        #   53-55 base linear velocity (x, y, z, world, m/s)    like an IMU
        #   56-58 base angular velocity (x, y, z, world, rad/s) like an IMU (z = turning speed)
        #   59-62 wheel speeds (rad/s)                          like wheel encoders
        base_lin_vel, base_ang_vel = client.getBaseVelocity(rid)
        row = np.concatenate((base_pos, [base_bearing], q_mes[4:11], qd_mes[4:11], TARGET_EE_POS, wheel_cmds, torque_arm,
                              base_ori, ee_pos, q_des_arm, qd_des_arm,
                              base_lin_vel, base_ang_vel, qd_mes[0:4]))
        data_rows.append(row)

        # Visualization in Pinocchio GUI if active
        if dyn_model.visualizer:
            for index in range(len(sim.bot)):
                q = sim.GetMotorAngles(index)
                dyn_model.DisplayModel(q)

        # Exit logic with 'q' key in PyBullet GUI window
        keys = sim.GetPyBulletClient().getKeyboardEvents()
        qKey = ord('q')
        if qKey in keys and keys[qKey] and sim.GetPyBulletClient().KEY_WAS_TRIGGERED:
            break

        # Stop criteria: maximum steps, target reached within fine tolerance, or unreachable target
        stop_message = check_stop(stop_criteria, current_time, time_step, ee_dist_error, base_navigation_done)
        if stop_message:
            print(stop_message)
            break

        # Data logging for plotting
        time_all.append(current_time)
        ee_pos_all.append(ee_pos)
        ee_error_all.append(ee_error)
        base_pos_all.append(base_pos)
        q_mes_all.append(q_mes)
        qd_mes_all.append(qd_mes)
        current_desired_q = np.concatenate((q_mes[:4], q_des_arm, target_finger_angles))
        current_desired_qd = np.concatenate((wheel_cmds, qd_des_arm, np.zeros(6)))
        q_d_all.append(current_desired_q)
        qd_d_all.append(current_desired_qd)

        if not headless:
            time.sleep(0.001)

        current_time += time_step
        if int(current_time / time_step) % 500 == 0:
            print(f"t={current_time:.2f}s | Base pos: {np.round(base_pos[:2], 3)} | EE pos: {np.round(ee_pos, 3)} | EE error: {ee_dist_error:.4f}m | yaw {base_bearing:.2f} phase {base_state.phase} nav_done {base_navigation_done}")
    
    # The run is over. Saves the list of rows to a file.
    # np.save("run_002.npy", np.array(data_rows))

    # Save each run into OUT_FOLDER, with the time in the name
    out_dir = os.path.join(cur_dir, OUT_FOLDER)
    os.makedirs(out_dir, exist_ok=True)
    note_file = os.path.join(cur_dir, "data_v2", "README.txt")
    if not os.path.exists(note_file):
        with open(note_file, "w") as f:
            f.write(DATA_V2_NOTE)
    # Save the run, but only if it is a GOOD run (the expert ended within 2 cm of the target).
    # Bad runs are not saved, to save disk space: training skips them anyway.
    # float32 = half the size of the normal numbers, and training uses float32 anyway.
    final_error_cm = np.linalg.norm(TARGET_EE_POS - ee_pos) * 100
    if final_error_cm < 2.0:
        np.save(os.path.join(out_dir, f"dart_{time.time_ns()}_{os.getpid()}.npy"), np.array(data_rows, dtype=np.float32))
        print(f"SAVED good run (final error {final_error_cm:.1f} cm)")
    else:
        print(f"NOT SAVED bad run (final error {final_error_cm:.1f} cm)")

    # Final summary report
    print("==================================================================")
    print(f"Final Base Position          : {np.round(base_pos[:2], 4)}")
    print(f"Final End-Effector Position  : {np.round(ee_pos, 4)}")
    print(f"Target End-Effector Position : {TARGET_EE_POS}")
    print(f"Final Cartesian Error (m)    : {np.linalg.norm(TARGET_EE_POS - ee_pos):.4f}")
    print(f"Fixed Finger Angles          : {np.round(target_finger_angles, 4)}")
    print(f"Final Finger Angles          : {np.round(q_mes[11:17], 4)}")
    print(f"Max Finger Deviation (rad)   : {np.max(np.abs(q_mes[11:17] - target_finger_angles)):.6f}")
    print("==================================================================")

    if headless:
        print(f"Headless run finished at t={current_time:.2f}s")
        return

    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)
    time_all = np.array(time_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=TARGET_EE_POS[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in q_d_all], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()