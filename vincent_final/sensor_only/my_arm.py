# =============================================================================
# my_arm.py
# This is for: testing Vincent's two arm designs with Nat's newest driving, on the six validation targets
#              that Nat's driving_height_no_speed11 experiment uses. Same sensors and same scoring as full_run.py.
#                both = two arm networks: plan (30 Hz) and torque (1 kHz). Uses the fixed maths and the 0.15 rad limit.
#                one  = one arm network straight to the torques (1 kHz). No plan and no fixed maths.
#                his  = Nat's reaching network, to check this computer gives the same result as his.
# How to run:  python my_arm.py both|one|his TAG [--jobs 3] [--targets 673 506 ...]
#              python my_arm.py one TAG --data ~/COMP0245_Golden_300      (all 100 validation targets)
# Result:      results/my_arm_TAG.json, and one line per target on the screen.
#              With --data, each finished target is also saved in results/my_arm_TAG/, and running the same
#              command again carries on from there. The test targets are never opened.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   target     the red ball the gripper has to reach.
#   tip        the end of the gripper, worked out from the arm encoders and the robot model, in the robot's frame.
#   tip gap    target from the camera tracker, minus the tip (m).
#   plan       the joint speeds the plan network wants (rad/s).
#   STOP       the arm has finished. Here it is a fixed rule: the estimated tip gap is 1 cm or less.
# -----------------------------------------------------------------------------

import argparse
import json
import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import sensors
import pybullet as p

# SETTING: the expert's limits on the plan (rad), the same as test_full.py.
Q_LEAD, LIMIT_MARGIN = 0.15, 0.05

# SETTING: STOP when the estimated tip gap is this small (m).
STOP_GAP = 0.01

# SETTING: the ball fit. The fewest red pixels it will use, and how many times it improves its answer.
FIT_MIN_PIXELS = 8
FIT_STEPS = 8

# SETTING: the stop network. STOP when it predicts the tip is this close, for this many camera frames in a row,
#   but not in the first moments after PARK.
LEARNED_STOP_M = float(os.environ.get('MY_ARM_STOP_M', '0.009'))
LEARNED_STOP_FRAMES = int(os.environ.get('MY_ARM_STOP_FRAMES', '5'))
LEARNED_SKIP_MS = 300

# SETTING: the settled stop rule. STOP when the tip has barely moved over the last 10 camera frames (1/3 s)
#   and the estimated tip gap is small enough.
SETTLED_FRAMES = 10
SETTLED_MOVE_M = 0.002
SETTLED_GAP_M = 0.015

# SETTING: Nat's newest driving and his reaching network.
EXPERIMENT =HERE / 'results/driving_height_no_speed11'

# SETTING: camera fixes. Only use a camera reading once the base has stopped and the hand is far from the target.
SETTLE_MS = 300         # milliseconds to wait after PARK
HAND_CLEAR_M = 0.10     # the tip must be at least this far from the camera's target position (m)
BALL_RADIUS = 0.03      # the target's radius (m), the same as DOT_RADIUS in wrist_camera.py
CAMERA_MOUNT = np.array([0.29, -0.13, 0.36])    # the corner camera in the robot's frame (m), the same as MOUNT in camera_loop.py

# The camera code adds the full ball radius to the median distance of the red pixels. The median pixel is already
# part way round the ball, at about 0.71 of the radius from the centre, so the code overstates the distance.
OVERSTATED_M = (1 - np.sqrt(0.5)) * BALL_RADIUS


def fit_ball(rgb, depth):
    """The ball's centre from one camera picture: every red pixel gives a point on the ball's surface (its ray times
    its depth), and the known radius leaves one best position for the centre. It works when part of the ball is hidden.
    Returns the centre in the robot's frame, or None when too few red pixels can be seen."""
    from camera_loop import camera_to_robot
    from wrist_camera import INSIDE, RAYS, pixel_to_ray
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    red = INSIDE & (r > 100) & (r > 1.7 * g) & (r > 1.7 * b) & np.isfinite(depth)
    if red.sum() < FIT_MIN_PIXELS:
        return None
    points = RAYS[red] * depth[red][:, None]
    rows, cols = red.nonzero()
    centre = pixel_to_ray(rows.mean(), cols.mean()) * (np.median(depth[red]) + 0.7 * BALL_RADIUS)
    for _ in range(FIT_STEPS):
        to_centre = centre - points
        distance = np.linalg.norm(to_centre, axis=1)
        centre = centre + np.linalg.lstsq(to_centre / distance[:, None], BALL_RADIUS - distance, rcond=None)[0]
    return camera_to_robot(centre)


def red_pixel_summary(rgb, depth):
    """A short description of the red pixels in the robot's frame: average point (3), spread (3), median and smallest
    distance, and pixel count. This is what the ball network reads. Returns None when no red pixel can be seen."""
    from wrist_camera import INSIDE, RAYS
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    red = INSIDE & (r > 100) & (r > 1.7 * g) & (r > 1.7 * b) & np.isfinite(depth)
    if not red.any():
        return None
    p_cam = RAYS[red] * depth[red][:, None]
    points = np.c_[CAMERA_MOUNT[0] + p_cam[:, 2], CAMERA_MOUNT[1] - p_cam[:, 0], CAMERA_MOUNT[2] - p_cam[:, 1]]
    return np.r_[points.mean(axis=0), points.std(axis=0), np.median(depth[red]), depth[red].min(), red.sum()].tolist()


def ball_centre(camera_position):
    """Moves the camera's target position back along the camera's line of sight by the overstated distance."""
    line_of_sight = camera_position - CAMERA_MOUNT
    return camera_position - OVERSTATED_M * line_of_sight / np.linalg.norm(line_of_sight)


def forward(w, x):
    """One of my networks (three linear layers, ReLU): raw inputs to real units, clipped to the training range."""
    x = (np.asarray(x, np.float32) - w['X_mean']) / w['X_std']
    for layer in ('layers.0', 'layers.2'):
        x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    return np.clip((x @ w['layers.4.weight'].T + w['layers.4.bias']) * w['Y_std'] + w['Y_mean'], w['Y_min'], w['Y_max'])


class MyArm:
    """Gives the arm torques from the arm encoders and the camera tracker. Call step() every 1 ms after PARK."""

    def __init__(self, design, first_angles):
        models = HERE / 'models'
        # SETTING: which one-network model to use. Type it before the command: MY_ARM_NET=option_b_finish_seed1
        self.direct = dict(np.load(models / f"{os.environ.get('MY_ARM_NET', 'option_b_seed1')}.npz")) if design == 'one' else None
        self.plan = dict(np.load(models / 'vincent_plan_seed1.npz')) if design == 'both' else None
        self.torque = dict(np.load(models / 'vincent_torque_seed1.npz')) if design == 'both' else None
        # SWITCH: third arm network. ON (MY_ARM_GAP_NET=1, with design 'both') gets the angle gaps from a network, so nothing
        #   adds up the plan. OFF (not set) uses the fixed adding-up.
        self.gap_net = dict(np.load(models / 'vincent_gap_seed1.npz')) if os.environ.get('MY_ARM_GAP_NET') == '1' else None
        self.hand = sensors.HandPose()
        info = [p.getJointInfo(self.hand.robot, j, physicsClientId=self.hand.cid) for j in self.hand.ids[4:11]]
        self.lower, self.upper = np.array([i[8] for i in info]), np.array([i[9] for i in info])
        self.q_des, self.qd_plan = np.array(first_angles, float), np.zeros(7)
        self.gap, self.started = None, False
        # SWITCH: camera correction. ON (MY_ARM_CAMERA_OFFSET=1) removes the offset measured by camera_offset.py on the
        #   training runs, for the arm only. OFF (not set) uses the camera tracker's position unchanged.
        use_offset = os.environ.get('MY_ARM_CAMERA_OFFSET') == '1'
        self.camera_offset = np.array(json.loads((models / 'camera_offset.json').read_text())) if use_offset else np.zeros(3)
        # SWITCH: camera fixes. ON (MY_ARM_CAMERA_FIX=1) uses ball_centre() and keeps an averaged target position.
        #   OFF (not set) uses the camera tracker's position unchanged. No true positions are used either way.
        self.camera_fix = os.environ.get('MY_ARM_CAMERA_FIX') == '1'
        self.good_readings, self.ms_since_park = [], 0
        # SWITCH: stop rule. ON (MY_ARM_SETTLED_STOP=1) waits for the arm to settle. OFF (not set) stops at a 1 cm estimated gap.
        self.settled_stop = os.environ.get('MY_ARM_SETTLED_STOP') == '1'
        self.recent_tips = deque(maxlen=SETTLED_FRAMES + 1)
        # SWITCH: ball fit. ON (MY_ARM_BALL_FIT=1) uses fit_ball() for the arm's target, keeping the last good answer when too
        #   little of the ball can be seen. OFF (not set) uses the camera tracker's position.
        self.ball_fit = os.environ.get('MY_ARM_BALL_FIT') == '1'
        self.fitted_target = None
        # SWITCH: recording for training. ON (MY_ARM_RECORD=1) saves every camera frame and never says stop.
        self.record = os.environ.get('MY_ARM_RECORD') == '1'
        self.frames, self.driving_frames = [], []
        # SWITCH: learned ball and stop. ON (MY_ARM_LEARNED=1) uses the two networks from train_ball_stop.py instead of the
        #   ball fit and the fixed stop rule. OFF (not set) does not use them.
        self.learned = dict(np.load(models / 'ball_stop_net.npz')) if os.environ.get('MY_ARM_LEARNED') == '1' else None
        # SWITCH: networks with memory. ON (MY_ARM_MEMORY_NETS=1) uses the networks from recordings made with memory:
        #   the ball network reads each good view for the memory, and the stop network says stop.
        self.memory_nets = os.environ.get('MY_ARM_MEMORY_NETS') == '1'
        # SWITCH: tip network. ON (MY_ARM_TIP_NET=1) uses train_tip_net.py's network for the tip. OFF (not set) uses the robot model.
        self.tip_net = dict(np.load(models / 'tip_net.npz')) if os.environ.get('MY_ARM_TIP_NET') == '1' else None
        if self.memory_nets:
            self.learned = dict(np.load(models / os.environ.get('MY_ARM_NET_FILE', 'ball_stop_net_final.npz')))
        self.learned_target, self.last_speed, self.frames_below = None, np.zeros(7), 0
        # SWITCH: memory. ON (MY_ARM_MEMORY=1) keeps the target from the last good view, from the start of the run, and
        #   ignores views with too few red pixels. OFF (not set) does not use it.
        from camera_loop import Tracker
        self.memory = Tracker(keep_height=True) if os.environ.get('MY_ARM_MEMORY') == '1' else None
        # SWITCH: wheel network. ON (MY_ARM_WHEEL_NET=1) moves the memory using train_wheel_net.py's network.
        #   OFF (not set) moves it with the fixed wheel formula.
        self.wheel_net = dict(np.load(models / 'wheel_net.npz')) if os.environ.get('MY_ARM_WHEEL_NET') == '1' else None

    def ball_network(self, rgb, depth):
        """The ball network's target position from one view, or None when too few red pixels can be seen."""
        s = red_pixel_summary(rgb, depth)
        if s is None or s[8] < FIT_MIN_PIXELS:
            return None
        return np.array(s[:3]) + self.small_net('ball', np.r_[s[:3], np.array(s[3:6]) * 30, s[6:8], np.log(s[8])])

    def tip_network(self, angle):
        """The tip's position in the robot's frame, from the network in train_tip_net.py instead of the robot model."""
        w = self.tip_net
        out = (np.r_[angle, np.sin(angle), np.cos(angle)] - w['x_mean']) / w['x_std']
        for layer in ('0', '2', '4'):
            out = np.maximum(out @ w[f'{layer}.weight'].T + w[f'{layer}.bias'], 0)
        return (out @ w['6.weight'].T + w['6.bias']) * w['y_std'] + w['y_mean']

    def small_net(self, name, x):
        """One of the two networks in ball_stop_net.npz."""
        w = self.learned
        out = (np.asarray(x) - w[f'{name}_x_mean']) / w[f'{name}_x_std']
        for layer in ('0', '2'):
            out = np.maximum(out @ w[f'{name}.{layer}.weight'].T + w[f'{name}.{layer}.bias'], 0)
        return (out @ w[f'{name}.4.weight'].T + w[f'{name}.4.bias']) * w[f'{name}_y_std']

    def record_frame(self, rgb, depth, reading, true_target, true_distance, tracker=None):
        """Saves what the robot could sense, and the true answers, for training the ball and stop networks.
        The true answers are only saved here. They never reach the arm."""
        summary = red_pixel_summary(rgb, depth)
        fitted = fit_ball(rgb, depth)
        self.frames.append(dict(ms=self.ms_since_park, tracker=None if tracker is None else tracker.tolist(),
                                memory=None if self.memory is None or self.memory.dot is None else self.memory.dot.tolist(), summary=summary, fitted=None if fitted is None else fitted.tolist(),
                                angle=reading['arm_angle'].tolist(), speed=reading['arm_speed'].tolist(),
                                true_target=list(true_target), true_distance=float(true_distance)))

    def record_driving_frame(self, rgb, depth, true_target):
        """Saves one camera view from before PARK, with the true answer, for training the ball network."""
        self.driving_frames.append(dict(summary=red_pixel_summary(rgb, depth), true_target=list(true_target)))

    def remember(self, rgb, depth, wheel_speed):
        """Call for every camera picture from the start of the run, while driving too. A good view (enough red pixels)
        replaces the memory. Otherwise the wheel encoders move the memory as the robot moves."""
        if self.memory is not None:
            reading = self.ball_network(rgb, depth) if self.memory_nets else fit_ball(rgb, depth)
            if self.wheel_net is None or reading is not None or self.memory.dot is None:
                self.memory.update(reading, wheel_speed)
            elif np.any(wheel_speed):
                # The wheel network says how the base moved, so the remembered target moves the opposite way.
                out = (np.asarray(wheel_speed) - self.wheel_net['x_mean']) / self.wheel_net['x_std']
                for layer in ('0', '2'):
                    out = np.maximum(out @ self.wheel_net[f'{layer}.weight'].T + self.wheel_net[f'{layer}.bias'], 0)
                move = (out @ self.wheel_net['4.weight'].T + self.wheel_net['4.bias']) * self.wheel_net['y_std']
                x, y = self.memory.dot[0] - move[0], self.memory.dot[1] - move[1]
                c, s = np.cos(-move[2]), np.sin(-move[2])
                self.memory.dot = np.r_[c * x - s * y, s * x + c * y, self.memory.dot[2:]]

    def read_camera(self, rgb, depth):
        """Call once for each new camera picture after PARK."""
        if self.learned is not None and not self.memory_nets:
            reading = self.ball_network(rgb, depth)
            self.learned_target = reading if reading is not None else self.learned_target
        if self.ball_fit:
            fitted = fit_ball(rgb, depth)
            self.fitted_target = fitted if fitted is not None else self.fitted_target

    def should_stop(self):
        """True when the arm has finished. Checked once per camera frame."""
        size = np.linalg.norm(self.gap)
        if self.record:
            return False
        if self.learned is not None:
            moved = np.linalg.norm(self.recent_tips[-1] - self.recent_tips[0])
            if self.learned['stop.0.weight'].shape[1] == 6:     # the stop network that reads no joint speeds
                moved_short = np.linalg.norm(self.recent_tips[-1] - self.recent_tips[max(0, len(self.recent_tips) - 4)])
                distance = self.small_net('stop', np.r_[self.gap, size, moved * 100, moved_short * 100])[0]
            else:
                distance = self.small_net('stop', np.r_[self.gap, size, np.abs(self.last_speed), moved * 100])[0]
            self.frames_below = self.frames_below + 1 if distance <= LEARNED_STOP_M else 0
            return self.ms_since_park > LEARNED_SKIP_MS and self.frames_below >= LEARNED_STOP_FRAMES
        if not self.settled_stop:
            return size <= STOP_GAP
        if len(self.recent_tips) <= SETTLED_FRAMES:
            return False
        moved = max(np.linalg.norm(tip - self.recent_tips[-1]) for tip in self.recent_tips)
        return moved <= SETTLED_MOVE_M and size <= SETTLED_GAP_M

    def target_for_arm(self, target_estimate, camera_reading, tip):
        """Fix 1: the ball's true centre. Fix 2: average the camera readings from before the hand gets close, then keep them."""
        if camera_reading is not None and self.ms_since_park >= SETTLE_MS and np.linalg.norm(camera_reading - tip) >= HAND_CLEAR_M:
            self.good_readings.append(camera_reading)
        return ball_centre(np.mean(self.good_readings, axis=0) if self.good_readings else target_estimate)

    def step(self, reading, target_estimate, new_camera_frame, camera_reading=None):
        """Returns torques (None until the target has been seen) and the size of the estimated tip gap (m)."""
        angle, speed = reading['arm_angle'], reading['arm_speed']
        self.ms_since_park += 1
        # Every 1/30 s: a new tip gap and, for the two-network arm, a new plan.
        if new_camera_frame or not self.started:
            self.started = True
            if target_estimate is not None:
                q = np.zeros(17)
                q[4:11] = angle
                tip = self.tip_network(angle) if self.tip_net is not None else self.hand([0, 0, 0], [0, 0, 0, 1], q)[0]
                self.recent_tips.append(tip)
                target = self.target_for_arm(target_estimate, camera_reading, tip) if self.camera_fix else target_estimate
                target = self.fitted_target if self.fitted_target is not None else target
                target = self.learned_target if self.learned_target is not None else target
                target = self.memory.dot if self.memory is not None and self.memory.dot is not None else target
                self.last_speed = speed
                self.gap = target - self.camera_offset - tip
                if self.plan is not None:
                    self.qd_plan = forward(self.plan, np.r_[self.gap, angle[:6]])
        if self.gap is None:
            return None, None
        if self.direct is not None:
            tau = forward(self.direct, np.r_[self.gap, angle[:6], speed])
        elif self.gap_net is not None:
            # Three arm networks: the plan is held between camera frames, the gap and torque networks run every 1 ms.
            angle_gap = forward(self.gap_net, np.r_[self.gap, angle[:6], speed, self.qd_plan])
            tau = forward(self.torque, np.r_[angle_gap, self.qd_plan - speed, angle[:6]])
        else:
            # The fixed maths: add up the plan, with the expert's limits.
            self.q_des = np.clip(self.q_des + self.qd_plan * 1e-3, angle - Q_LEAD, angle + Q_LEAD)
            limited = np.clip(self.q_des, self.lower + LIMIT_MARGIN, self.upper - LIMIT_MARGIN)
            self.q_des = np.where(self.lower < self.upper, limited, self.q_des)
            tau = forward(self.torque, np.r_[self.q_des - angle, self.qd_plan - speed, angle[:6]])
        return tau, float(np.linalg.norm(self.gap))

    def close(self):
        p.disconnect(self.hand.cid)


def case_from_record(ghost):
    """Target, deadline and expert reference from the record in Nat's experiment, for one target."""
    frames = ghost['frames']
    at_park = next(f for f in frames if f['parked'])
    manifest = dict(target_position_world_m=ghost['target'], steps=round(ghost['stop_s'] * 1000) + 1,
                    final_pre_step_ee_error_m=ghost['stop_mm'] / 1000)
    reference = dict(park_time_s=ghost['park_s'], park_base_xy=at_park['base'][:2], final_base_xy=frames[-1]['base'][:2],
                     park_base_quaternion=at_park['quaternion'])
    return dict(manifest=manifest, reference=reference, split='val')


def case_from_training_run(root, dot):
    """Target, deadline and expert reference for one training target, from the golden data."""
    with np.load(root / 'expert/train' / f'run_{dot}.npz', allow_pickle=False) as run:
        manifest = json.loads(str(run['manifest']))
        park = int(manifest['parking_step'])
        reference = dict(park_time_s=park / 1000, park_base_xy=run['base_position'][park, :2].tolist(),
                         final_base_xy=run['base_position'][-1, :2].tolist(),
                         park_base_quaternion=run['base_quaternion'][park].tolist())
    return dict(manifest=manifest, reference=reference, split='train')


def attempt(job):
    from full_run import one
    design, dot, case, driving, reaching = job
    row = one(driving, reaching, dot, False, case=case, batch_driving=True, my_arm_design=None if design == 'his' else design)
    row.update(dot=dot, design=design)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('design', choices=['both', 'one', 'his'])
    ap.add_argument('tag')
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--targets', type=int, nargs='*')
    ap.add_argument('--data', type=Path, help='the golden data folder: use its 100 validation targets')
    # SETTING: which golden targets to use. val is for development. test is only for the final score of a frozen design.
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--training-targets', type=int, help='use this many golden training targets instead, for recording')
    ap.add_argument('--training-skip', type=int, default=0, help='skip this many training targets first')
    a = ap.parse_args()
    # SETTING: which driving network. Type it before the command: MY_ARM_DRIVING=path/to/driving_seed1.npz
    driving = dict(np.load(os.environ.get('MY_ARM_DRIVING', EXPERIMENT / 'models/driving_seed7.npz'), allow_pickle=False))
    reaching = dict(np.load(EXPERIMENT / 'models/reaching_seed7.npz', allow_pickle=False))
    if a.training_targets:
        targets = json.loads((a.data / 'split.json').read_text())['train'][a.training_skip:a.training_skip + a.training_targets]
        cases = {dot: case_from_training_run(a.data, dot) for dot in targets}
    elif a.data:
        from validate_golden import case_for
        allowed = json.loads((a.data / 'split.json').read_text())[a.split]
        targets = a.targets or allowed
        assert set(targets) <= set(allowed), f'Only {a.split} targets are allowed'
        cases = {dot: case_for(a.data, dot, a.split)[0] for dot in targets}
    else:
        ghosts = json.loads((EXPERIMENT / 'results/expert_validation.json').read_text())
        targets = a.targets or [int(k) for k in ghosts]
        cases = {dot: case_from_record(ghosts[str(dot)]) for dot in targets}
    # Finished targets are kept, so an interrupted run can carry on.
    saved = HERE / f'results/my_arm_{a.tag}'
    saved.mkdir(exist_ok=True)
    done = {dot: json.loads((saved / f'{dot}.json').read_text()) for dot in targets if (saved / f'{dot}.json').exists()}
    jobs = [(a.design, dot, cases[dot], driving, reaching) for dot in targets if dot not in done]
    with ProcessPoolExecutor(a.jobs) as pool:
        for row in pool.map(attempt, jobs):
            (saved / f"{row['dot']}.json").write_text(json.dumps(row))
            done[row['dot']] = row
            print('finished', len(done), 'of', len(targets), flush=True)
    rows = [done[dot] for dot in targets]
    (HERE / f'results/my_arm_{a.tag}.json').write_text(json.dumps(rows))
    # Print results: one line per target.
    for r in rows:
        print(r['dot'], 'park error cm', round(100 * r['park_error_vs_expert_m'], 1), 'park', r['park_s'], 'stop', r['stop_s'],
              'closest', None if r['min_mm'] is None else round(r['min_mm'], 1),
              'at stop', None if r['at_network_stop_mm'] is None else round(r['at_network_stop_mm'], 1),
              'end', round(r['end_mm'], 1), r['status'], flush=True)
