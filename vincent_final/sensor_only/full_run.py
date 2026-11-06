"""The full run: the driving network drives and parks, the arm network reaches and stops. No expert in the loop.

Inputs come only from the robot's own sensors: the corner camera and tracker (rendered every 1/30 s from the true
state, as in camera_loop.py and test_arm.py --camera), the wheel encoders and the arm encoders. The simulator's true
state is used to render the image and to log the truth, never as a network input.

Per run: the simulator (sensor_parking, arm mode, no expert_drive) steps 1 ms at a time. Every 1/30 s the camera is
rendered, both trackers are updated (driving: distance and bearing; arm: x, y, height) and the driving network
decides the wheel speeds and PARK; after its PARK the arm network runs every 1 ms with the held camera estimate and
says stop. The run ends 0.5 s after the stop, or at the time limit.

usage: python full_run.py DRIVING_MODEL ARM_MODEL [--bearings B ...] [--jobs N] [--tag T]
  DRIVING_MODEL from driving_v2/models (16 inputs, encoders), ARM_MODEL from arm/models (17 inputs, camera dot).
  -> results/full_<DRIVING>_<ARM><TAG>.json
"""
import argparse, json, os, subprocess, sys, time
from functools import partial
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import sensors
from train_arm import DATA, SPLIT, dot_in_robot, select_inputs, policy as arm_policy
from test_arm import NATIVE, PIXI                                   # the built simulator, as the arm tests use it
from camera_loop import MOUNT, Scene, Tracker, see_dot, camera_to_robot, policy as drive_policy, driving_inputs

DRIVING = HERE.parent / 'driving_v2'
EXPERT = json.loads((DRIVING / 'expert_reference.json').read_text())['bearings']


def one(wd, wa, bearing, tag_frames=True, case=None, camera_dir=None, batch_driving=False, my_arm_design=None):
    """my_arm_design: 'both' or 'one' uses the arm in my_arm.py instead of the arm network wa."""
    import pybullet as p
    if camera_dir is not None:
        from render_golden import png
        camera_dir = Path(camera_dir)
        camera_dir.mkdir(parents=True, exist_ok=True)
    if case is None:
        m = json.loads((DATA / f'run_{bearing // 12 - 1}/manifest.json').read_text())
        ref, split = EXPERT[str(int(bearing))], SPLIT[bearing]
    else:
        m, ref, split = case['manifest'], case['reference'], case['split']
    target, expert_stop = m['target_position_world_m'], (m['steps'] - 1) / 1000
    limit = ref['park_time_s'] + 3.0 + (expert_stop - ref['park_time_s']) + 1.0      # the expert's time plus 3 s to park and 1 s to reach
    sim = subprocess.Popen([str(PIXI), 'run', 'build-clang19/sensor_parking', str(NATIVE / 'assets')], cwd=NATIVE,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                           env=dict(os.environ, EXPERT_SKIP_VISUALS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    def call(msg=None):
        if msg is not None:
            sim.stdin.write(json.dumps(msg) + '\n'); sim.stdin.flush()
        while True:
            line = sim.stdout.readline()
            if not line:
                raise RuntimeError('sensor_parking exited')
            if '{"' in line:
                out = json.loads(line[line.index('{"'):])
                if out.get('type') == 'error':
                    raise RuntimeError(out['error'])
                return out
    try:
        names = call()['joint_names']
        state = call(dict(op='reset', target=target, max_duration_s=limit, arm=True))
        f = state['frame']
        scene = Scene(); drive_tracker, arm_tracker = Tracker(), Tracker(keep_height=True)
        wheels = sensors.Sensors(); wheels.reset(f)
        enc = sensors.ArmEncoders(1e-3); enc.reset(np.array(f['joints'][sensors.ARM]))
        n_in = wa['shared.0.weight'].shape[1]
        use_time = bool(wa['info_time']) if 'info_time' in wa else n_in == 19
        cmd = np.zeros(4); parked = False; park_step = None; stop_t = None; stop_mm = None; est = None; seen = None
        next_frame = 0; frames = []; log = []; closest = np.inf; tau = np.zeros(7); ttp = None; never_seen = 0
        if my_arm_design is not None:
            from my_arm import MyArm
            mine = MyArm(my_arm_design, enc.last.copy())
        while not state['done']:
            f = state['frame']; t = f['time_s']; step = f['step']
            if not np.isfinite(np.r_[f['base'], f['quaternion'], f['joints'], state['ee_distance_m']]).all():
                raise FloatingPointError('Nonfinite simulator state')
            new_camera_frame = step >= next_frame
            if new_camera_frame:                                     # a camera frame and a driving decision, every 1/30 s
                B = np.array(p.getMatrixFromQuaternion(f['quaternion'])).reshape(3, 3)
                scene.pose(np.r_[f['base'], f['quaternion']], names, f['joints'], target)
                rgb, depth, _ = scene.fisheye(np.array(f['base']) + B @ MOUNT, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
                camera_src = None
                if camera_dir is not None:
                    image_path = camera_dir / f'{step:06d}.png'
                    png(image_path, rgb[::2, ::2])  # 128 px replay preview; detector still sees the original 256 px image.
                    camera_src = f'{camera_dir.name}/{image_path.name}'
                seen = see_dot(rgb, depth)
                reading = wheels.read(f)
                wheel_x, wheel_speed = sensors.wheel_features(reading['wheel_angle'], reading['wheel_speed'])[0], reading['wheel_speed']
                point = camera_to_robot(seen) if seen is not None else None
                dot = drive_tracker.update(point, wheel_speed); est = arm_tracker.update(point, wheel_speed)
                if my_arm_design is not None:
                    mine.remember(rgb, depth, wheel_speed)
                    if mine.record and not parked:
                        mine.record_driving_frame(rgb, depth, dot_in_robot(target, np.array(f['base']), f['yaw']).tolist())
                driving_x, driving_ttp = None, None
                if not parked:
                    if dot is None:
                        dot_in = [0.0, 0.0, 0.0]
                    else:
                        a = np.arctan2(dot[1], dot[0]); dot_in = [float(np.hypot(*dot)), np.sin(a), np.cos(a)]
                    x = driving_inputs(wd, wheel_x, dot, seen is not None, est[2] if est is not None else None)
                    cmd, prob = drive_policy(wd, x.astype(np.float32))
                    driving_x, driving_ttp = x.tolist(), getattr(drive_policy, 'ttp', None)
                    if prob >= 0.5:
                        parked = True; park_step = step; cmd = np.zeros(4)
                dx, dy = target[0] - f['base'][0], target[1] - f['base'][1]
                frames.append(dict(time_s=t, base=f['base'], quaternion=f['quaternion'], joints=f['joints'], ee=f['ee'],
                                   parked=parked, stopped=stop_t is not None, ttp_s=ttp, tau=np.round(tau, 3).tolist(),
                                   ee_mm=1000 * state['ee_distance_m'], visible=seen is not None, true_dist=float(np.hypot(dx, dy)),
                                   est_dist=dot_in[0] if not parked else None, wheel=np.round(cmd, 3).tolist(), camera_src=camera_src,
                                   driving_input=driving_x, driving_ttp_s=driving_ttp))
                next_frame = round(len(frames) * 1000 / 30)
            req = dict(op='step', wheel=cmd.tolist(), park=parked, ms=1)
            if parked and my_arm_design is not None:
                r = enc.read(np.array(f['joints'][sensors.ARM]))
                never_seen += est is None
                if new_camera_frame:
                    mine.read_camera(rgb, depth)
                    if mine.record:
                        mine.record_frame(rgb, depth, r, dot_in_robot(target, np.array(f['base']), f['yaw']).tolist(), state['ee_distance_m'], est)
                tau_mine, ttp = mine.step(r, est, new_camera_frame, point if new_camera_frame else None)
                if tau_mine is not None:                             # until the target is first seen, the hold controller keeps the arm
                    tau = tau_mine
                    if stop_t is None and new_camera_frame and mine.should_stop():
                        stop_t = t
                        stop_mm = 1000 * state['ee_distance_m']
                    req.update(arm=np.clip(tau, -30, 30).tolist(), stop=stop_t is not None)
                closest = min(closest, 1000 * state['ee_distance_m'])
            elif parked:
                r = enc.read(np.array(f['joints'][sensors.ARM]))
                dot_a = est if est is not None else np.zeros(3); never_seen += est is None
                x = np.r_[(r['arm_angle'], r['arm_speed'], dot_a, t - park_step / 1000, t)].astype(np.float32)
                x = select_inputs(x, False, True, use_time)
                tau, ttp = arm_policy(wa, x)
                if stop_t is None and ttp <= float(wa['ttp_thresh']):
                    stop_t = t
                    stop_mm = 1000 * state['ee_distance_m']
                req.update(arm=np.clip(tau, -30, 30).tolist(), stop=stop_t is not None)
                closest = min(closest, 1000 * state['ee_distance_m'])
                if step % 10 == 0:
                    true_dot = dot_in_robot(target, np.array(f['base']), f['yaw'])
                    log.append(dict(t=t, ee_mm=1000 * state['ee_distance_m'], ttp_s=ttp, tau=np.round(tau, 3).tolist(),
                                    est_err_mm=None if est is None else float(1000 * np.linalg.norm(est - true_dot))))
            else:
                enc.read(np.array(f['joints'][sensors.ARM]))            # keep the encoder's speed estimate current
                if batch_driving:
                    # The native hold controller still runs every 1 ms. Stop one tick before
                    # the next camera decision so the arm encoder has the exact previous sample at PARK.
                    req['ms'] = max(1, next_frame - step - 1)
            state = call(req)
        f = state['frame']
        park_frame = next((fr for fr in frames if fr['parked']), None)
        park_xy = frames[-1]['base'][:2] if park_frame is None else park_frame['base'][:2]
        heading_error = None
        if park_frame is not None and 'park_base_quaternion' in ref:
            yaw = p.getEulerFromQuaternion(park_frame['quaternion'])[2]
            expert_yaw = p.getEulerFromQuaternion(ref['park_base_quaternion'])[2]
            heading_error = float(np.arctan2(np.sin(yaw-expert_yaw), np.cos(yaw-expert_yaw)))
        expected_end = (stop_t + .5) if stop_t is not None else limit
        early = f['time_s'] < expected_end - .002
        success = bool(parked and stop_mm is not None and stop_mm <= 10 and not early)
        reason = ('early_native_termination' if early else 'success' if success else
                  'no_park_timeout' if not parked else 'no_stop_timeout' if stop_t is None else 'stop_outside_10mm')
        recorded = mine.frames if my_arm_design is not None else []
        return dict(recorded_frames=recorded, recorded_driving=mine.driving_frames if my_arm_design is not None else [], bearing=bearing, split=split, parked=parked, success=success, status=reason,
                    park_heading_error_rad=heading_error, timed_out=stop_t is None and not early,
                    final_time_s=f['time_s'], final_step=f['step'],
                    post_stop_drift_mm=None if stop_mm is None else 1000 * state['ee_distance_m'] - stop_mm,
                    post_park_base_drift_m=None if park_frame is None else float(np.linalg.norm(np.array(f['base'][:2])-park_xy)), park_s=None if park_step is None else park_step / 1000,
                    expert_park_s=ref['park_time_s'], park_error_vs_expert_m=float(np.hypot(*(np.array(park_xy) - np.array(ref['park_base_xy'])))),
                    final_base_error_vs_expert_m=float(np.hypot(*(np.array(f['base'][:2]) - np.array(ref['final_base_xy'])))),
                    stop_s=stop_t, expert_stop_s=expert_stop, expert_stop_mm=1000 * m['final_pre_step_ee_error_m'],
                    min_mm=float(closest) if np.isfinite(closest) else None, end_mm=1000 * state['ee_distance_m'],
                    at_expert_stop_mm=next((fr['ee_mm'] for fr in frames if fr['time_s'] >= expert_stop), 1000 * state['ee_distance_m']),
                    at_network_stop_mm=stop_mm,
                    visible_at_park=None if park_step is None else next(fr['visible'] for fr in frames if fr['parked']),
                    steps_without_estimate=never_seen, limit_s=limit, log=log, frames=frames if tag_frames else [])
    finally:
        sim.kill()
        sim.wait()
        if 'mine' in locals(): mine.close()
        if 'scene' in locals(): p.disconnect()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('driving'); ap.add_argument('arm')
    ap.add_argument('--bearings', type=int, nargs='*'); ap.add_argument('--jobs', type=int, default=6); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    wd = dict(np.load(DRIVING / f'models/{a.driving}.npz')); wa = dict(np.load(HERE / f'models/{a.arm}.npz'))
    bearings = a.bearings or [b for b in sorted(SPLIT) if SPLIT[b] != 'holdout']
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(partial(one, wd, wa), bearings))
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    path = out / f'full_{a.driving}_{a.arm}{a.tag}.json'; path.write_text(json.dumps(rows))
    for r in rows:
        print(f"{r['bearing']:4d} {r['split']:6s} parked {str(r['parked']):5s} at {r['park_s']} s (expert {r['expert_park_s']:.2f}), "
              f"park error {100 * r['park_error_vs_expert_m']:.0f} cm | closest {r['min_mm']} mm, stop {r['stop_s']} s, at stop {r['at_network_stop_mm']} mm")
    print(f'{len(rows)} runs in {time.time() - t0:.0f} s -> {path}')
