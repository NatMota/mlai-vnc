"""Closed loop with a real sensor pipeline: fast native physics (sensor_parking) + Python camera, tracker and policy.

Per 30 Hz decision:
  native state -> render corner fisheye (RGB-D) -> red pixels -> ray + depth -> point in ROBOT frame (fixed mount,
  no world pose) -> tracker (keeps / dead-reckons the dot with wheel encoders) -> policy -> wheel commands + PARK.
The simulator's true pose is used only to render the image and, separately, to log the true dot for comparison.

Usage (any Python with PyBullet):  python camera_loop.py WEIGHTS.npz BEARING_DEG OUT.json
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pybullet as p

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from wrist_camera import Scene, INSIDE, RAYS, pixel_to_ray, DOT_RADIUS

NATIVE = Path(os.environ.get('NATIVE_ROOT', HERE.parents[1] / 'collection/native'))
PIXI = next(x for x in [Path.home() / '.local/bin/pixi', Path.home() / '.pixi/bin/pixi'] if x.exists())
MOUNT = np.array([0.29, -0.13, 0.36])        # corner camera, base frame (x forward, y left, z up)
WHEEL_RADIUS, TRACK = 0.11, 0.66            # skid-steer effective track width (expert's calibrated value)
DT = 1 / 30
FULL_FRAMES = os.environ.get("FULL_FRAMES") == "1"      # also log the full robot state (for the 3D player)


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


def wheel_features(q, qd, q0):
    t = q[:4] - q0
    r, l = t[[0, 3]].mean(), t[[1, 2]].mean(); vr, vl = qd[[0, 3]].mean(), qd[[1, 2]].mean()
    return np.r_[t, qd[:4], (r + l) / 2, r - l, (vr + vl) / 2, vr - vl]


def driving_inputs(w, wheel, dot, visible, height=None):
    """Legacy 15/16 inputs, or 17 with tracked robot-frame height appended in metres."""
    n = w['shared.0.weight'].shape[1]
    if n not in (15, 16, 17):
        raise ValueError(f'Unsupported driving input count: {n}')
    camera = [0., 0., 0.] if dot is None else [float(np.hypot(*dot[:2])), np.sin(np.arctan2(dot[1], dot[0])), np.cos(np.arctan2(dot[1], dot[0]))]
    x = np.r_[wheel, camera, [float(visible)] if n >= 16 else []]
    if n == 17:
        if not bool(w.get('info_camera_height', False)):
            raise ValueError('17-input driving checkpoint must declare camera height')
        x = np.r_[x, 0. if height is None else height]
    return x.astype(np.float32)


def policy(w, x):
    for layer in ('shared.0', 'shared.2'):
        x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    head = float((x @ w['park.weight'].T + w['park.bias'])[0])
    if 'ttp_clip' in w:      # time-to-park head: report a 0/1 'probability' from the fixed threshold, keep the seconds too
        policy.ttp = head * float(w['ttp_clip'])
        return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), float(policy.ttp <= float(w['ttp_thresh']))
    return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), float(1 / (1 + np.exp(-head)))


def run(weights_path, bearing, max_s, use_visible_flag=False):
    w = dict(np.load(weights_path))
    use_encoders = int(w.get('info_encoders', 0)) == 1      # wheel inputs from simulated encoders (sensors.py)
    if use_encoders:
        import sensors
        robot_sensors = sensors.Sensors()
    b = np.deg2rad(bearing); target = [1.1 * np.cos(b), 1.1 * np.sin(b), 0.5]
    env = dict(os.environ, EXPERT_SKIP_VISUALS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    sim = subprocess.Popen([str(PIXI), 'run', 'build-clang19/sensor_parking', str(NATIVE / 'assets')], cwd=NATIVE,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    def call(msg=None):
        if msg is not None:
            sim.stdin.write(json.dumps(msg) + '\n'); sim.stdin.flush()
        while True:
            line = sim.stdout.readline()
            if not line:
                raise RuntimeError('sensor_parking exited')
            if '{"' in line:                       # engine warnings can share a line with the JSON reply
                out = json.loads(line[line.index('{"'):])
                if out.get('type') == 'error':
                    raise RuntimeError(out['error'])
                return out
    names = call()['joint_names']
    state = call(dict(op='reset', target=target, max_duration_s=max_s))
    height_input = bool(w.get('info_camera_height', False))
    scene, tracker, log = Scene(), Tracker(keep_height=height_input), []
    q0 = np.array(state['frame']['joints'][:4]); parked = False
    if use_encoders:
        robot_sensors.reset(state['frame'])
    try:
        timing = dict(physics_s=0.0, camera_s=0.0, other_s=0.0)
        while not state['done']:
            t0 = time.perf_counter()
            f = state['frame']; pose = np.r_[f['base'], f['quaternion']]
            # Render the simulated camera from the true robot state (this is the sensor, not an input).
            scene.pose(pose, names, f['joints'], target)
            B = np.array(p.getMatrixFromQuaternion(f['quaternion'])).reshape(3, 3)
            eye = np.array(f['base']) + B @ MOUNT
            rgb, depth, _ = scene.fisheye(eye, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
            t1 = time.perf_counter(); timing['camera_s'] += t1 - t0
            seen = see_dot(rgb, depth)
            if use_encoders:
                reading = robot_sensors.read(f)
                wheel_x, wheel_speed = sensors.wheel_features(reading['wheel_angle'], reading['wheel_speed'])[0], reading['wheel_speed']
            else:
                wheel_x, wheel_speed = wheel_features(np.array(f['joints']), np.array(f['wheel_velocity']), q0), np.array(f['wheel_velocity'])
            dot = tracker.update(camera_to_robot(seen) if seen is not None else None, wheel_speed)
            if dot is None:
                dot_in = [0.0, 0.0, 0.0]            # never seen yet: no information
            else:
                a = np.arctan2(dot[1], dot[0]); dot_in = [float(np.hypot(*dot[:2])), np.sin(a), np.cos(a)]
            x = driving_inputs(w, wheel_x, dot, seen is not None, dot[2] if height_input and dot is not None else None)
            cmd, prob = policy(w, x.astype(np.float32))
            park = parked or prob >= 0.5; parked = park
            # True dot relative to the robot, for logging only.
            dx, dy = target[0] - f['base'][0], target[1] - f['base'][1]
            ta = np.arctan2(dy, dx) - f['yaw']
            log.append(dict(t=f['time_s'], visible=seen is not None, tracked=dot is not None,
                            est_dist=dot_in[0], est_dir_deg=float(np.rad2deg(np.arctan2(dot_in[1], dot_in[2]))) if dot is not None else None,
                            true_dist=float(np.hypot(dx, dy)), true_dir_deg=float(np.rad2deg(np.arctan2(np.sin(ta), np.cos(ta)))),
                            park_prob=prob, ttp_s=getattr(policy, 'ttp', None), base=f['base'][:2], yaw=f['yaw']))
            if FULL_FRAMES:      # full robot state for the 3D player
                log[-1].update(base3=f['base'], quaternion=f['quaternion'], joints=f['joints'],
                               wheel_velocity=f['wheel_velocity'], command=cmd.tolist(), parked=bool(park))
            t2 = time.perf_counter(); timing['other_s'] += t2 - t1
            state = call(dict(op='step', wheel=cmd.tolist(), park=bool(park)))
            timing['physics_s'] += time.perf_counter() - t2
        final = state['frame']
    finally:
        sim.kill()
    return dict(bearing=bearing, parked=parked, decisions=len(log), timing=timing, final_base=final['base'][:2], final_yaw=final['yaw'], log=log)


if __name__ == '__main__':
    weights, bearing, out = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])
    expert = json.loads((HERE / 'expert_reference.json').read_text())['bearings'][str(int(bearing))]
    flag = np.load(weights)['shared.0.weight'].shape[1] == 16      # 16 inputs = trained with the visible flag
    result = run(weights, bearing, expert['park_time_s'] + 3.0, use_visible_flag=flag)
    result['expert_park_base'] = expert['park_base_xy']; result['expert_final_base'] = expert['final_base_xy']
    result['expert_final_yaw'] = expert['final_yaw']
    out.write_text(json.dumps(result))
    print('done', out)
