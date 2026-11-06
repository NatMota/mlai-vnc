# =============================================================================
# dagger_labels.py
# This is for: one DAgger round for the one-network arm. The arm's own recorded runs show where it really goes.
#              Here the expert says what torque it would have used at each of those moments.
#              The expert is only asked for answers. It never moves the robot.
# How to run:  python3 dagger_labels.py OUT.npy RECORDING1.npy RECORDING2.npy ...
# Result:      saves OUT.npy, one row per recorded moment: true tip gap (3), joint angles 1 to 6, joint speeds (7),
#              expert torques (7), then the target number. train_arm_no_maths.py can add these to its training data.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   target     the red ball the gripper has to reach.
#   tip gap    target minus tip, in the robot's frame (m).
#   recording  frames saved by my_arm.py with MY_ARM_RECORD=1, on training targets only.
#   DAgger     the model drives, the expert gives the right answer, the model learns from it.
# -----------------------------------------------------------------------------

import os
import sys
import numpy as np
os.environ["ROBOENV_HEADLESS"] = "1"
import pinocchio as pin
from simulation_and_control import pb, PinWrapper
from robotnik_arm_control import ArmControllerParams, arm_controller, configure_gripper_fingers

# Fixed by the teacher and the expert's code. Do not change.
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
ARM_REACH_LIMIT = 0.85

conf_file_name = "robotnik_arm.json"
cur_dir = os.path.dirname(os.path.abspath(__file__))
sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=False)
configure_gripper_fingers(sim, sim.pybullet_client, sim.bot[0].bot_pybullet, FIXED_FINGER_ANGLES)
ext_names_arr = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
dyn_model = PinWrapper(conf_file_name, "pybullet", ext_names_arr, ["pybullet"], False, 0, cur_dir)
ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")
lower_limits, upper_limits = sim.GetBotJointsLimit()
arm_lower, arm_upper = np.array(lower_limits[4:11]), np.array(upper_limits[4:11])
time_step = sim.GetTimeStep()
arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
q_hold = np.array(sim.GetMotorAngles(0))[4:11].copy()

rows = []
tip_differences = []
for path in sys.argv[2:]:
    frames = np.load(path)
    for frame in frames:
        target_id, tip, true_target, angle, speed = frame[0], frame[3:6], frame[6:9], frame[9:16], frame[16:23]
        # The robot's frame: the base at the origin, facing forward. The target is already in this frame.
        q_mes = np.r_[np.zeros(4), angle, FIXED_FINGER_ANGLES]
        qd_mes = np.r_[np.zeros(4), speed, np.zeros(6)]
        q_pin = dyn_model.ReoderJoints2PinVec(np.r_[0, 0, 0, 0, 0, 0, 1, q_mes], "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        tip_differences.append(np.linalg.norm(ee_pos - tip))
        # The expert's plan starts from where the arm really is, so it can never fall behind and say "come back".
        torque, _, _ = arm_controller(dyn_model, q_pin, angle, speed, angle.copy(), q_hold, true_target, ee_pos,
                                      ee_frame_id, shoulder_frame_id, arm_lower, arm_upper, time_step, arm_params, reach_active=True)
        rows.append(np.r_[true_target - tip, angle[:6], speed, torque, target_id])

rows = np.array(rows, dtype=np.float32)
np.save(sys.argv[1], rows)
# Print results: how many rows, and whether this model's tip agrees with the recording.
print(f"rows {len(rows)}, tip difference from the recording: median {1000 * np.median(tip_differences):.2f} mm, largest {1000 * np.max(tip_differences):.2f} mm")
print(f"expert torque: largest size per joint {np.round(np.abs(rows[:, 16:23]).max(axis=0), 1)}")
