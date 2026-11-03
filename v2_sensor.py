"""Two-rate arm on Nat's sensor-only driving (v2, iteration 13), as a sim.run_episode policy.

Every 1 ms physics step:
  at each 30 Hz decision (step round(k * 1000 / 30), as Nat's sensor_parking):
    render Nat's corner fisheye from the true pose -> red pixels -> dot in the robot frame -> tracker
    wheel encoders -> Nat's driving network -> 4 wheel commands and the PARK latch
  arm encoders (Kinova Jaco 2 resolution, speed from the change over 1 ms)
  before PARK: the hold controller of Nat's simulator and the expert (kp 100, kd 30, gravity, +-30 Nm)
  after PARK: the two-rate arm (vel_net about 30 Hz, torque_net 1 kHz), from sensor or true inputs

With sensor inputs the arm sees a robot-frame state: base at the origin with bearing 0, encoder joint angles and
speeds, the tracked dot as the goal, and the end-effector from forward kinematics of the encoder angles. The arm
checkpoint ignores the world-frame inputs (all_world ablation), so only robot-frame quantities reach it. The
simulator's true state is used only to render the camera image and to score the run, as in Nat's camera_loop.py.
"""
import os
from pathlib import Path
import time

import numpy as np
import pinocchio as pin
import pybullet as p

ROOT = Path(__file__).resolve().parent
# wrist_camera.py reads the robot file path when it is imported. Always this repo's copy (identical to Nat's).
os.environ["CAMERA_URDF"] = str(ROOT / "models/summit_xl_arm_description/summit_xl_complete.urdf")

from fused_policy import HIGH_LEVEL_STEPS
from memory_policy import MemoryPolicy, features
from sim import ARM_REACH_LIMIT, FIXED_FINGER_ANGLES, Expert
from nat_v2 import sensors
from nat_v2.wrist_camera import DOT_RADIUS, INSIDE, Scene, pixel_to_ray
from robotnik_arm_control import ArmControllerParams

# ---- Copied word for word from Nat's camera_loop.py (nat_v2/SOURCE.md); tests check they still match. ----
MOUNT = np.array([0.29, -0.13, 0.36])        # corner camera, base frame (x forward, y left, z up)
WHEEL_RADIUS, TRACK = 0.11, 0.66            # skid-steer effective track width (expert's calibrated value)
DT = 1 / 30


def camera_to_robot(p_cam):
    """Step 8: camera axes (x right, y down, z forward) -> robot axes (x forward, y left, z up), fixed mount."""
    return np.array([MOUNT[0] + p_cam[2], MOUNT[1] - p_cam[0], MOUNT[2] - p_cam[1]])


def see_dot(rgb, depth):
    """Steps 2-7. Returns the dot centre in the camera frame, or None if not visible."""
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    red = INSIDE & (r > 100) & (r > 1.7 * g) & (r > 1.7 * b)
    if not red.any():
        return None
    rows, cols = red.nonzero()
    return pixel_to_ray(rows.mean(), cols.mean()) * (np.median(depth[red]) + DOT_RADIUS)


class Tracker:
    """Keeps the dot in the robot frame. Seen: replace. Hidden: move it opposite to the robot's own motion."""
    def __init__(self):
        self.dot = None
    def update(self, seen, wheel_speed):
        if seen is not None:
            self.dot = seen[:2].copy()
        elif self.dot is not None:
            right, left = wheel_speed[[0, 3]].mean(), wheel_speed[[1, 2]].mean()
            v = WHEEL_RADIUS * (right + left) / 2
            w = WHEEL_RADIUS * (right - left) / TRACK
            x, y = self.dot[0] - v * DT, self.dot[1]
            c, s = np.cos(-w * DT), np.sin(-w * DT)
            self.dot = np.array([c * x - s * y, s * x + c * y])
        return self.dot


def policy(w, x):
    for layer in ('shared.0', 'shared.2'):
        x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    head = float((x @ w['park.weight'].T + w['park.bias'])[0])
    if 'ttp_clip' in w:      # time-to-park head: report a 0/1 'probability' from the fixed threshold, keep the seconds too
        policy.ttp = head * float(w['ttp_clip'])
        return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), float(policy.ttp <= float(w['ttp_thresh']))
    return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), float(1 / (1 + np.exp(-head)))
# ---- End of the copy. ----


def decision_step(k):
    """Physics step of Nat's k-th 30 Hz decision (33 or 34 ms apart, 30 per second)."""
    return round(k * 1000 / 30)


class HeightTracker(Tracker):
    """Nat's tracker, also keeping the dot height (robot frame), which stays at its last seen value while hidden."""

    def __init__(self):
        super().__init__()
        self.height = None

    def update(self, seen, wheel_speed):
        if seen is not None:
            self.height = float(seen[2])
        xy = super().update(seen, wheel_speed)
        return None if xy is None else np.r_[xy, self.height]


HOLD = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)       # the expert's gains (kp 100, kd 30, 30 Nm)


def hold_torque(q, qd, hold, gravity):
    """The arm hold controller of the expert and of Nat's simulator before PARK."""
    return np.clip(HOLD.kp * (hold - q) - HOLD.kd * qd + gravity, -HOLD.max_torque, HOLD.max_torque)


class NatDriver:
    """Nat's driving network: 12 wheel encoder inputs, dot distance, sin and cos of its direction, visible flag."""

    def __init__(self, weights_path):
        self.name = Path(weights_path).stem
        self.w = dict(np.load(weights_path))
        if self.w["shared.0.weight"].shape[1] != 16:
            raise ValueError(f"{weights_path}: expected the 16-input camera + flag network")
        if int(self.w.get("info_encoders", 0)) != 1:
            raise ValueError(f"{weights_path}: expected a network trained on wheel encoder inputs")

    def decide(self, wheel_x, dot_xy, visible):
        """(4 wheel commands, PARK now?) for one decision, built exactly as in camera_loop.py."""
        if dot_xy is None:
            dot_in = [0.0, 0.0, 0.0]            # never seen yet: no information
        else:
            a = np.arctan2(dot_xy[1], dot_xy[0])
            dot_in = [float(np.hypot(*dot_xy)), np.sin(a), np.cos(a)]
        cmd, prob = policy(self.w, np.r_[wheel_x, dot_in, float(visible)].astype(np.float32))
        return cmd, prob >= 0.5


class V2Policy(MemoryPolicy):
    """Nat's driving (or the expert's) until PARK, then the two-rate arm. See the module docstring."""

    @classmethod
    def build(cls, arm_checkpoint, driver, arm_inputs="sensor", use_arm=True):
        """driver: a NatDriver, or "expert". arm_inputs: "sensor" or "true". use_arm=False holds the arm all run."""
        if arm_inputs not in ("sensor", "true"):
            raise ValueError("arm_inputs must be 'sensor' or 'true'")
        policy = cls.load(arm_checkpoint, vel_every=HIGH_LEVEL_STEPS)
        policy.driver, policy.arm_inputs, policy.use_arm = driver, arm_inputs, use_arm
        policy.scene = None
        return policy

    def reset(self, robot, target, obs0):
        super().reset(robot, target, obs0)
        if self.scene is None:                   # one render-only PyBullet client per process
            self.scene = Scene()
        self.names = list(robot.sim.getNameActiveJoints())
        self.pin_data = robot.dyn.pin_model.createData()
        self.expert = None
        if self.driver == "expert":
            self.expert = Expert()
            self.expert.reset(robot, target, obs0)
        joints = np.array(robot.sim.GetMotorAngles(0))
        self.wheel_enc = sensors.WheelEncoders()
        self.wheel_enc.reset(joints[:4])
        self.arm_enc = sensors.ArmEncoders(dt=robot.dt)
        self.arm_enc.reset(obs0["q"])
        self.tracker = HeightTracker()
        self.hold = obs0["q"].copy()
        self.step, self.decisions, self.next_decision = 0, 0, 0
        self.parked, self.wheels, self.dot = False, np.zeros(4), None
        self.report = {"parked": False, "park_s": None, "first_seen_s": None, "decisions": 0, "visible_decisions": 0,
                       "seen_after_park": 0, "camera_xy_error_at_park_cm": None, "camera_height_error_at_park_cm": None,
                       "camera_xy_error_last_cm": None, "camera_height_error_last_cm": None,
                       "dot_never_seen": False, "render_s": 0.0}

    def act(self, obs):
        t = self.step * self.robot.dt
        if self.step == self.next_decision:
            self._decide(obs, t)
            self.decisions += 1
            self.next_decision = decision_step(self.decisions)
        if self.expert is not None and not self.parked:
            wheels, _, done = self.expert.clean(obs)
            self.wheels = wheels
            if done:
                self._park(obs, t)
        reading = self.arm_enc.read(obs["q"])
        if self.parked and self.use_arm and (self.arm_inputs == "true" or self.dot is not None):
            torque = self._reach(obs, reading)
        else:
            gravity = pin.computeGeneralizedGravity(self.robot.dyn.pin_model, self.pin_data, obs["q_pin"])[6:13]
            torque = hold_torque(obs["q"], obs["qd"], self.hold, gravity)
        self.step += 1
        return (np.zeros(4) if self.parked else self.wheels), torque, self.parked

    def _decide(self, obs, t):
        """One 30 Hz decision: camera, wheel encoders, tracker, and Nat's driving network until PARK."""
        started = time.perf_counter()
        sim = self.robot.sim
        base, quat = np.array(sim.GetBasePosition()), np.array(sim.GetBaseOrientation())
        joints = np.array(sim.GetMotorAngles(0))
        # Render the simulated camera from the true robot state (this is the sensor, not an input).
        self.scene.pose(np.r_[base, quat], self.names, joints, self.target)
        B = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        rgb, depth, _ = self.scene.fisheye(base + B @ MOUNT, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
        self.report["render_s"] += time.perf_counter() - started
        seen = see_dot(rgb, depth)
        wheel = self.wheel_enc.read(joints[:4])
        self.dot = self.tracker.update(camera_to_robot(seen) if seen is not None else None, wheel["wheel_speed"])
        self.report["decisions"] += 1
        if self.dot is not None:            # log only: the goal estimate the arm uses, against the truth
            xy, height = self._camera_error()
            self.report.update(camera_xy_error_last_cm=xy, camera_height_error_last_cm=height)
        if seen is not None:
            self.report["visible_decisions"] += 1
            self.report["seen_after_park"] += int(self.parked)
            if self.report["first_seen_s"] is None:
                self.report["first_seen_s"] = t
        if isinstance(self.driver, NatDriver) and not self.parked:
            wheel_x = sensors.wheel_features(wheel["wheel_angle"], wheel["wheel_speed"])[0]
            self.wheels, park_now = self.driver.decide(wheel_x, None if self.dot is None else self.dot[:2],
                                                       seen is not None)
            if park_now:
                self._park(obs, t)

    def _park(self, obs, t):
        """PARK latches: the wheels stop and the arm starts reaching (it never switches back)."""
        self.parked, self.phase = True, 1.0
        self.report.update(parked=True, park_s=t, dot_never_seen=self.dot is None)
        if self.dot is not None:
            xy, height = self._camera_error()
            self.report.update(camera_xy_error_at_park_cm=xy, camera_height_error_at_park_cm=height)

    def _camera_error(self):
        """Tracked dot against the true dot in the robot frame, in cm (horizontal, height). For the log only."""
        sim = self.robot.sim
        B = np.array(p.getMatrixFromQuaternion(sim.GetBaseOrientation())).reshape(3, 3)
        true_dot = B.T @ (self.target - np.array(sim.GetBasePosition()))
        return (float(np.linalg.norm(self.dot[:2] - true_dot[:2]) * 100),
                float(abs(self.dot[2] - true_dot[2]) * 100))

    def _reach(self, obs, reading):
        if self.arm_inputs == "true":
            state = np.concatenate((obs["base_pos"], [obs["bearing"]], obs["q"], obs["qd"], self.target))
            return self._arm(features(state, obs["ee"]), obs["q"], obs["qd"])
        q, qd = reading["arm_angle"], reading["arm_speed"]
        state = np.concatenate((np.zeros(4), q, qd, self.dot))      # base at the origin, bearing 0
        return self._arm(features(state, self.ee_robot_frame(q)), q, qd)

    def ee_robot_frame(self, q_arm):
        """End-effector in the robot frame, from forward kinematics of the arm angles (base at the origin)."""
        joints = np.r_[np.zeros(4), q_arm, FIXED_FINGER_ANGLES]       # wheels and fingers do not move the hand
        q_pin = self.robot.dyn.ReoderJoints2PinVec(np.r_[np.zeros(3), [0.0, 0.0, 0.0, 1.0], joints], "pos")
        pin.forwardKinematics(self.robot.dyn.pin_model, self.pin_data, q_pin)
        pin.updateFramePlacements(self.robot.dyn.pin_model, self.pin_data)
        return self.pin_data.oMf[self.robot.ee_id].translation.copy()
