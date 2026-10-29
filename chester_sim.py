"""One self-contained rollout engine for the expert, learned policies and DAgger collection.

Simulation setup, measurements and stop rules copy drive.py exactly, so success numbers are
comparable with test.py. One process runs one episode; chester_parallel_eval.py runs many.

Policy protocol:
    policy.reset(robot, target, obs0)
    policy.act(obs) -> (wheel_cmds[4], torque_arm[7], reach_active: bool)
A labeler (optional) is reset the same way and its label_row(obs, reach_active, policy)
is appended to each recorded row.
"""
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pinocchio as pin
from simulation_and_control import pb, MotorCommands, PinWrapper
from robotnik_arm_control import (
    ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria,
    arm_controller, base_controller, check_stop, configure_gripper_fingers, quaternion2bearing,
)

REACH_TOLERANCE = 0.01
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
# Expert settings, identical to robotnik_arm_task.py / robotnik_arm_task_dart.py
BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
ROOT = Path(__file__).resolve().parent


class Robot:
    """The simulated robot, set up exactly like drive.py and the expert scripts."""

    def __init__(self, target, gui=False):
        conf = "robotnik_arm.json"
        self.sim = pb.SimInterface(conf, conf_file_path_ext=str(ROOT), use_gui=gui)
        sim = self.sim
        sim.SetFloorFriction(1.0)
        client = sim.pybullet_client
        rid = sim.bot[0].bot_pybullet
        for j in range(client.getNumJoints(rid)):
            if "wheel" in client.getJointInfo(rid, j)[1].decode():
                client.changeDynamics(rid, j, contactStiffness=30000, contactDamping=1000,
                                      anisotropicFriction=[1.0, 0.01, 1.0])
        sim.bot[0].servo_motor_model.set_motor_gains(400, 30)
        configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)
        shape = client.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.0, 0.0, 0.85])
        client.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, basePosition=list(target))
        names = np.expand_dims(np.array(sim.getNameActiveJoints()), axis=0)
        self.dyn = PinWrapper(conf, "pybullet", names, ["pybullet"], False, 0, str(ROOT))
        self.ee_id = self.dyn.pin_model.getFrameId("robot_j2s7s300_end_effector")
        self.shoulder_id = self.dyn.pin_model.getFrameId("robot_j2s7s300_link_2")
        lower, upper = sim.GetBotJointsLimit()
        self.arm_lower = np.array(lower[4:11])
        self.arm_upper = np.array(upper[4:11])
        self.dt = sim.GetTimeStep()
        self.cmd = MotorCommands()

    def observe(self):
        sim = self.sim
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q = np.array(sim.GetMotorAngles(0))
        qd = np.array(sim.GetMotorVelocities(0))
        q_pin = self.dyn.ReoderJoints2PinVec(np.concatenate((base_pos, base_ori, q)), "pos")
        pin.forwardKinematics(self.dyn.pin_model, self.dyn.pin_data, q_pin)
        pin.updateFramePlacements(self.dyn.pin_model, self.dyn.pin_data)
        ee = self.dyn.pin_data.oMf[self.ee_id].translation.copy()
        return {"base_pos": base_pos, "bearing": bearing, "q": q[4:11], "qd": qd[4:11],
                "ee": ee, "q_pin": q_pin}

    def step(self, wheel_cmds, torque_arm):
        cmd_all = np.concatenate((wheel_cmds, torque_arm, np.zeros(6)))
        self.cmd.SetControlCmd(cmd_all, ["velocity"] * 4 + ["torque"] * 13)
        self.sim.Step(self.cmd, "torque")


def raw_state(obs, target):
    """The 21 saved inputs, in the column order of the expert data files."""
    return np.concatenate((obs["base_pos"], [obs["bearing"]], obs["q"], obs["qd"], target))


class Expert:
    """The expert controller as a policy, optionally with DART pushes on the executed command."""

    def __init__(self, wheel_push=0.0, arm_push=0.0, seed=None):
        self.wheel_push = wheel_push
        self.arm_push = arm_push
        self.rng = np.random.default_rng(seed)

    def reset(self, robot, target, obs0):
        self.robot = robot
        self.target = np.asarray(target, dtype=float)
        standoff = float(np.interp(self.target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
        xy = self.target[:2]
        self.goal_bearing = float(np.arctan2(xy[1], xy[0])) if np.linalg.norm(xy) > 1e-4 else 0.0
        self.goal_pos = xy - standoff * np.array([np.cos(self.goal_bearing), np.sin(self.goal_bearing)])
        self.state = BaseNavState()
        self.base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
        self.q_hold = obs0["q"].copy()
        self.q_des = self.q_hold.copy()
        self.qd_des = np.zeros(7)
        self.time = 0.0

    def clean(self, obs):
        """Noise-free expert command; advances the expert's base state machine and arm reference."""
        wheels = base_controller(self.state, obs["base_pos"], obs["bearing"], self.goal_pos,
                                 self.goal_bearing, self.time, self.base_params)
        torque, self.q_des, self.qd_des = arm_controller(
            self.robot.dyn, obs["q_pin"], obs["q"], obs["qd"], self.q_des, self.q_hold, self.target,
            obs["ee"], self.robot.ee_id, self.robot.shoulder_id, self.robot.arm_lower,
            self.robot.arm_upper, self.robot.dt, self.arm_params, reach_active=self.state.done)
        self.time += self.robot.dt
        return wheels, torque, self.state.done

    def act(self, obs):
        wheels, torque, active = self.clean(obs)
        self.last_clean = (wheels.copy(), torque.copy())
        if not active and self.wheel_push:
            wheels = wheels + self.rng.normal(0.0, self.wheel_push, 4)
        if self.arm_push:
            torque = torque + self.rng.normal(0.0, self.arm_push, 7)
        return wheels, torque, active


def run_episode(target, policy, max_steps=30000, gui=False, record=None, labeler=None):
    """Run one episode with drive.py's stop rules.

    record (a list) collects one row per step: 21 raw state, 3 ee, 1 reach flag,
    4 executed wheel commands, 7 executed torques, then labeler.label_row(...) if given.
    """
    target = np.asarray(target, dtype=float)
    robot = Robot(target, gui=gui)
    obs = robot.observe()
    policy.reset(robot, target, obs)
    if labeler is not None:
        labeler.reset(robot, target, obs)
    stop = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
    t, best, switch_time, reason = 0.0, np.inf, None, ""
    started = time.perf_counter()
    while True:
        obs = robot.observe()
        err = float(np.linalg.norm(target - obs["ee"]))
        best = min(best, err)
        wheels, torque, active = policy.act(obs)
        if active and switch_time is None:
            switch_time = t
        if record is not None:
            row = [raw_state(obs, target), obs["ee"], [float(active)], wheels, torque]
            if labeler is not None:
                row.append(labeler.label_row(obs, active, policy))
            record.append(np.concatenate(row))
        robot.step(wheels, torque)
        message = check_stop(stop, t, robot.dt, err, active)
        if message:
            reason = message
            break
        if gui:
            time.sleep(0.001)
        t += robot.dt
    # Same final measure as drive.py: the error seen on the last loop iteration.
    return {"reached": err < REACH_TOLERANCE, "final_error_cm": err * 100, "best_error_cm": best * 100,
            "time_s": t, "switch_s": switch_time, "stop": reason,
            "final_base": obs["base_pos"][:2].tolist(), "final_bearing": float(obs["bearing"]),
            "wall_s": time.perf_counter() - started}
