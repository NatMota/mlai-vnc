"""The exported two-rate arm after Nat's golden driving network (iteration 43), scored as in Nat's full_run.py.

Every 1 ms physics step (sim.Robot):
  at each 30 Hz decision (step round(k * 1000 / 30)):
    render Nat's corner fisheye from the true pose -> red pixels -> dot in the robot frame -> two trackers
    (driving: x, y; arm: x, y, height) -> wheel encoders -> Nat's 11-input driving network -> wheels and PARK
  arm encoders (Kinova resolution, speed over 1 ms)
  before PARK: the hold controller; after PARK: The exported two-rate arm (no world-frame inputs) from encoders, the
  arm tracker's goal and forward kinematics of the encoder angles
  STOP (sensor only, fixed before any test run): the estimated distance |FK(encoder angles) - tracked goal| stays at
  or below STOP_DISTANCE_M for STOP_HOLD_STEPS steps.

Scoring follows full_run.py: success = parked, STOP issued and the end-effector within SUCCESS_MM of the target at
the STOP step; the run ends RUN_AFTER_STOP_STEPS after STOP, or at the expert's stop time + LIMIT_AFTER_EXPERT_S.
The simulator's true state is used only to render the camera and to score.
"""
import os
from pathlib import Path
import time

import numpy as np
import pinocchio as pin
import pybullet as p

ROOT = Path(__file__).resolve().parent
GOLDEN_DIR = ROOT / "nat_golden"
# wrist_camera.py reads the robot file path when it is imported. Always this repo's copy (identical to Nat's).
os.environ["CAMERA_URDF"] = str(ROOT / "models/summit_xl_arm_description/summit_xl_complete.urdf")

from nat_golden import sensors
from nat_golden.sensors import DRIVING_NO_SPEED_COLS
from nat_golden.wrist_camera import DOT_RADIUS, INSIDE, Scene, pixel_to_ray
from v2_sensor import V2Policy, decision_step, hold_torque

STOP_DISTANCE_M = 0.005          # estimated hand-to-goal distance for STOP (half the 10 mm success limit)
STOP_HOLD_STEPS = 100            # 100 ms at 1 kHz: three camera updates
RUN_AFTER_STOP_STEPS = 500       # full_run.py: the run continues 0.5 s after STOP
SUCCESS_MM = 10.0
LIMIT_AFTER_EXPERT_S = 4.0       # full_run.py: expert park + 3 s + (expert stop - park) + 1 s = expert stop + 4 s

# ---- Copied word for word from Nat's camera_loop.py (nat_golden/SOURCE.md); tests check they still match. ----
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
    """Keeps the dot in the robot frame. Seen: replace. Hidden: move it opposite to the robot's own motion.
    keep_height=False (driving): the dot is (x, y). keep_height=True (arm): (x, y, height); while hidden the height stays
    as last seen, because the robot drives on flat ground."""
    def __init__(self, keep_height=False):
        self.dot = None; self.keep_height = keep_height
    def update(self, seen, wheel_speed):
        if seen is not None:
            self.dot = seen[:3 if self.keep_height else 2].copy()
        elif self.dot is not None:
            right, left = wheel_speed[[0, 3]].mean(), wheel_speed[[1, 2]].mean()
            v = WHEEL_RADIUS * (right + left) / 2
            w = WHEEL_RADIUS * (right - left) / TRACK
            x, y = self.dot[0] - v * DT, self.dot[1]
            c, s = np.cos(-w * DT), np.sin(-w * DT)
            self.dot = np.r_[c * x - s * y, s * x + c * y, self.dot[2:]]
        return self.dot


def driving_inputs(w, wheel, dot, visible, height=None):
    """Legacy 15/16 inputs, or 17 with tracked robot-frame height appended in metres."""
    n = w['shared.0.weight'].shape[1]
    if n not in (11, 15, 16, 17):
        raise ValueError(f'Unsupported driving input count: {n}')
    camera = [0., 0., 0.] if dot is None else [float(np.hypot(*dot[:2])), np.sin(np.arctan2(dot[1], dot[0])), np.cos(np.arctan2(dot[1], dot[0]))]
    x = np.r_[wheel, camera, [float(visible)] if n in (11, 16, 17) else []]
    if n in (11, 17):
        if not bool(w.get('info_camera_height', False)):
            raise ValueError('Height-enabled driving checkpoint must declare camera height')
        x = np.r_[x, 0. if height is None else height]
    if n == 11:
        if bool(w.get('info_wheel_velocity', True)):
            raise ValueError('11-input checkpoint must declare omitted wheel velocities')
        x = x[DRIVING_NO_SPEED_COLS]
    return x.astype(np.float32)


def policy(w, x):
    policy.ttp = None
    policy.park_distance_m = None
    if 'export_input_mean' in w:
        x = (x - w['export_input_mean']) / w['export_input_scale']
    for layer in ('shared.0', 'shared.2'):
        x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    head = float((x @ w['park.weight'].T + w['park.bias'])[0])
    if 'park_distance_scale_m' in w:
        policy.park_distance_m = head * float(w['park_distance_scale_m'])
        return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), float(policy.park_distance_m <= float(w['park_distance_threshold_m']))
    if 'ttp_clip' in w:      # time-to-park head: report a 0/1 'probability' from the fixed threshold, keep the seconds too
        policy.ttp = head * float(w['ttp_clip'])
        return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), float(policy.ttp <= float(w['ttp_thresh']))
    return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), float(1 / (1 + np.exp(-head)))
# ---- End of the copy. ----


def time_limit_s(case):
    return float(case["stop_s"]) + LIMIT_AFTER_EXPERT_S


def expert_park_xy(case):
    """The expert's base position at its first parked frame, or None when its reference is not valid."""
    if not case.get("reference_valid", True):
        return None
    return next((np.asarray(f["base"][:2]) for f in case["frames"] if f["parked"]), None)


class SensorStop:
    """STOP once the estimated distance stays at or below the threshold for hold_steps steps in a row; latched."""

    def __init__(self, threshold_m=STOP_DISTANCE_M, hold_steps=STOP_HOLD_STEPS):
        self.threshold_m, self.hold_steps = threshold_m, hold_steps
        self.count, self.stopped = 0, False

    def update(self, distance_m):
        if not self.stopped:
            self.count = self.count + 1 if distance_m <= self.threshold_m else 0
            self.stopped = self.count >= self.hold_steps
        return self.stopped


class GoldenPolicy(V2Policy):
    """Nat's iteration 43 driving until PARK, then The two-rate arm with a sensor-only STOP. See the module docstring."""

    @classmethod
    def build_golden(cls, arm_checkpoint, driving_path, use_arm=True):
        golden = cls.build(arm_checkpoint, driver=None, arm_inputs="sensor", use_arm=use_arm)
        with np.load(driving_path, allow_pickle=False) as weights:
            golden.drive_w = dict(weights)
        return golden

    def reset(self, robot, target, obs0):
        if self.scene is None:                   # Nat's current camera code, one render-only client per process
            self.scene = Scene()
        super().reset(robot, target, obs0)
        joints = np.array(robot.sim.GetMotorAngles(0))
        self.wheel_enc = sensors.WheelEncoders()
        self.wheel_enc.reset(joints[:4])
        self.arm_enc = sensors.ArmEncoders(dt=robot.dt)
        self.arm_enc.reset(obs0["q"])
        self.drive_tracker, self.arm_tracker = Tracker(), Tracker(keep_height=True)
        self.stop, self.stop_step = SensorStop(), None
        self.report.update(stop_s=None, estimate_mm_at_stop=None)

    def act(self, obs):
        t = self.step * self.robot.dt
        if self.step == self.next_decision:
            self._decide(obs, t)
            self.decisions += 1
            self.next_decision = decision_step(self.decisions)
        reading = self.arm_enc.read(obs["q"])
        if self.parked and self.use_arm and self.dot is not None:
            torque = self._reach(obs, reading)
            estimate = float(np.linalg.norm(self.ee_robot_frame(reading["arm_angle"]) - self.dot))
            if self.stop_step is None and self.stop.update(estimate):
                self.stop_step = self.step
                self.report.update(stop_s=t, estimate_mm_at_stop=1000 * estimate)
        else:
            gravity = pin.computeGeneralizedGravity(self.robot.dyn.pin_model, self.pin_data, obs["q_pin"])[6:13]
            torque = hold_torque(obs["q"], obs["qd"], self.hold, gravity)
        self.step += 1
        return (np.zeros(4) if self.parked else self.wheels), torque, self.parked

    def _decide(self, obs, t):
        """One 30 Hz decision as in full_run.py: camera, wheel encoders, both trackers, then driving until PARK."""
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
        wheel_x, wheel_speed = sensors.wheel_features(wheel["wheel_angle"], wheel["wheel_speed"])[0], wheel["wheel_speed"]
        point = camera_to_robot(seen) if seen is not None else None
        drive_dot = self.drive_tracker.update(point, wheel_speed)
        self.dot = self.arm_tracker.update(point, wheel_speed)
        self.report["decisions"] += 1
        if self.dot is not None:            # log only: the goal estimate the arm uses, against the truth
            xy, height = self._camera_error()
            self.report.update(camera_xy_error_last_cm=xy, camera_height_error_last_cm=height)
        if seen is not None:
            self.report["visible_decisions"] += 1
            self.report["seen_after_park"] += int(self.parked)
            if self.report["first_seen_s"] is None:
                self.report["first_seen_s"] = t
        if not self.parked:
            x = driving_inputs(self.drive_w, wheel_x, drive_dot, seen is not None,
                               self.dot[2] if self.dot is not None else None)
            self.wheels, prob = policy(self.drive_w, x.astype(np.float32))
            if prob >= 0.5:
                self._park(obs, t)


def score(parked, stop_step, at_stop_m):
    """Status and success as in full_run.py."""
    if not parked:
        return "no_park_timeout", False
    if stop_step is None:
        return "no_stop_timeout", False
    success = 1000 * at_stop_m <= SUCCESS_MM
    return ("success" if success else "stop_outside_10mm"), success


def run_golden_episode(case, golden_policy, parity=False):
    """One full run on one golden case. parity=True (arm held) ends 0.5 s after PARK, to check the driving only."""
    from sim import Robot
    target = np.asarray(case["target"], dtype=float)
    if parity:
        golden_policy.use_arm = False
    robot = Robot(target)
    golden_policy.reset(robot, target, robot.observe())
    limit_steps = round(time_limit_s(case) * 1000)
    step, closest, at_stop, park_xy, park_step = 0, np.inf, None, None, None
    started = time.perf_counter()
    while True:
        obs = robot.observe()
        if not np.isfinite(np.r_[obs["base_pos"], obs["q"], obs["qd"], obs["ee"]]).all():
            raise FloatingPointError("Nonfinite simulator state")
        error = float(np.linalg.norm(target - obs["ee"]))
        wheels, torque, parked = golden_policy.act(obs)
        if parked:
            closest = min(closest, error)
            if park_xy is None:
                park_xy, park_step = obs["base_pos"][:2].copy(), step
        if golden_policy.stop_step == step:
            at_stop = error
        robot.step(wheels, torque)
        step += 1
        if golden_policy.stop_step is not None:
            done = step >= golden_policy.stop_step + RUN_AFTER_STOP_STEPS
        elif parity and park_step is not None:
            done = step >= park_step + RUN_AFTER_STOP_STEPS
        else:
            done = step >= limit_steps
        if done:
            break
    final_obs = robot.observe()
    if not np.isfinite(np.r_[final_obs["base_pos"], final_obs["q"], final_obs["qd"], final_obs["ee"]]).all():
        raise FloatingPointError("Nonfinite terminal simulator state")
    end = float(np.linalg.norm(target - final_obs["ee"]))
    status, success = score(park_xy is not None, golden_policy.stop_step, at_stop)
    ref_xy = expert_park_xy(case)
    return {**golden_policy.report, "parked": park_xy is not None, "status": status, "success": success,
            "park_s": None if park_step is None else park_step / 1000,
            "stop_s": None if golden_policy.stop_step is None else golden_policy.stop_step / 1000,
            "at_stop_mm": None if at_stop is None else 1000 * at_stop,
            "closest_mm": None if not np.isfinite(closest) else 1000 * closest, "end_mm": 1000 * end,
            "park_error_cm": None if park_xy is None or ref_xy is None else float(np.linalg.norm(park_xy - ref_xy) * 100),
            "limit_s": time_limit_s(case), "expert_park_s": case["park_s"], "expert_stop_s": case["stop_s"],
            "final_s": step / 1000, "wall_s": time.perf_counter() - started}
