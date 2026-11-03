"""Test an arm network (train_arm.py) in the fast simulator.

The expert drives and parks itself (expert_drive), so only the arm network is tested. From PARK the network gets
its inputs at its own rate (info_hz: every 30 Hz decision, or every 1 ms) and each set of 7 torques is held until
the next one. The full robot state is logged every 1/30 s for the 3D player. It says stop when the
predicted time to stop is 0.25 s or less (latched); the run ends 0.5 s after stop, or 1 s after the expert's stop.

--camera: the dot comes from the corner camera and tracker (as in driving, keeping the height) instead of the
simulator. The image is taken every 1/30 s; the tracker moves the dot with the wheel encoder speeds while the robot
drives; the network uses the latest estimate. Results go to results/MODEL_camera.json, with the estimate error logged.

usage: python test_arm.py MODEL [--bearings B ...] [--camera] [--jobs N]  ->  results/MODEL[_camera].json
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import sensors
from train_arm import DATA, SPLIT, dot_in_robot, policy, select_inputs

NATIVE = Path(os.environ.get('NATIVE_ROOT', Path.home() / 'github/comp0245-native-expert'))
PIXI = next(x for x in [Path.home() / '.local/bin/pixi', Path.home() / '.pixi/bin/pixi'] if x.exists())


def one(w, bearing, camera=False):
    m = json.loads((DATA / f'run_{bearing // 12 - 1}/manifest.json').read_text())
    target, expert_stop = m['target_position_world_m'], m['steps'] - 1
    sim = subprocess.Popen([str(PIXI), 'run', 'build-clang19/sensor_parking', str(NATIVE / 'assets')], cwd=NATIVE,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                           env=dict(os.environ, EXPERT_SKIP_VISUALS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    def call(req):
        sim.stdin.write(json.dumps(req) + '\n'); sim.stdin.flush()
        while True:
            line = sim.stdout.readline()
            if not line:
                raise RuntimeError('sensor_parking exited')
            if '{"' in line:
                out = json.loads(line[line.index('{"'):])
                if out['type'] == 'error':
                    raise RuntimeError(out['error'])
                if out['type'] == 'state':
                    return out
    try:
        state = call(dict(op='reset', target=target, max_duration_s=expert_stop / 1000 + 1.0, arm=True, expert_drive=True))
        hz = int(w.get('info_hz', 30))
        if camera:                                 # the driving test's camera, dot finder and tracker
            import pybullet as p
            from camera_loop import Scene, Tracker, see_dot, camera_to_robot, MOUNT
            names = json.loads((HERE.parent / 'driving_v2/closed_loop_native/result_wheels_live_dot.json').read_text())['joint_names']
            scene, tracker = Scene(), Tracker(keep_height=True)
            wheels = sensors.WheelEncoders(); wheels.reset(np.array(state['frame']['joints'][:4]))
            seen = None; est = None; never_seen = 0
        enc = sensors.ArmEncoders(1 / hz); enc.reset(np.array(state['frame']['joints'][sensors.ARM]))
        n_in = w['shared.0.weight'].shape[1]
        use_imu = bool(w['info_imu']) if 'info_imu' in w else False
        use_enc = bool(w['info_encoders']) if 'info_encoders' in w else True
        use_time = bool(w['info_time']) if 'info_time' in w else n_in == 19
        if use_imu:
            hand, imu = sensors.HandPose(), sensors.EndEffectorIMU(seed=1000 + bearing)   # noise seeded differently from training
        imu_on = False
        stop_t = None; log = []; frames = []; closest = np.inf; next_frame = 0; tau = np.zeros(7); ttp = None
        while not state['done']:
            f = state['frame']; t = f['time_s']
            r = enc.read(np.array(f['joints'][sensors.ARM]))
            req = dict(op='step', wheel=[0] * 4)
            if hz == 1000:
                req['ms'] = 1
            true_dot = dot_in_robot(target, np.array(f['base']), f['yaw'])
            if camera and f['step'] >= next_frame:     # every 1/30 s: new camera image, tracker update
                B = np.array(p.getMatrixFromQuaternion(f['quaternion'])).reshape(3, 3)
                scene.pose(np.r_[f['base'], f['quaternion']], names, f['joints'], target)
                rgb, depth, _ = scene.fisheye(np.array(f['base']) + B @ MOUNT, np.c_[-B[:, 1], -B[:, 2], B[:, 0]])
                seen = see_dot(rgb, depth)
                est = tracker.update(camera_to_robot(seen) if seen is not None else None, wheels.read(np.array(f['joints'][:4]))['wheel_speed'])
            if state['parked']:
                if camera:
                    dot = est if est is not None else np.zeros(3)
                    never_seen += est is None
                else:
                    dot = true_dot
                extra = []
                if use_imu:
                    pose = hand(f['base'], f['quaternion'], f['joints'])
                    if not imu_on:
                        imu.reset(*pose); imu_on = True
                    m_ = imu.read(*pose); extra = [m_['imu_acc'], m_['imu_gyro']]
                x = np.r_[(r['arm_angle'], r['arm_speed'], dot, *extra, t - state['park_step'] / 1000, t)].astype(np.float32)
                x = select_inputs(x, use_imu, use_enc, use_time)
                assert len(x) == n_in
                tau, ttp = policy(w, x)
                if stop_t is None and ttp <= float(w['ttp_thresh']):
                    stop_t = t
                req.update(arm=np.clip(tau, -30, 30).tolist(), stop=stop_t is not None)
                closest = min(closest, 1000 * state['ee_distance_m'])
                if hz == 30 or f['step'] >= next_frame:
                    log.append(dict(t=t, ee_mm=1000 * state['ee_distance_m'], ttp_s=ttp, tau=np.round(tau, 3).tolist()))
                    if camera:
                        log[-1].update(visible=seen is not None, est=None if est is None else np.round(est, 4).tolist(),
                                       true=np.round(true_dot, 4).tolist(),
                                       est_err_mm=None if est is None else float(1000 * np.linalg.norm(est - true_dot)))
            if f['step'] >= next_frame:            # every 1/30 s: full state for the 3D player
                frames.append(dict(time_s=t, base=f['base'], quaternion=f['quaternion'], joints=f['joints'], ee=f['ee'],
                                   parked=state['parked'], stopped=stop_t is not None, ttp_s=ttp, tau=np.round(tau, 3).tolist(),
                                   ee_mm=1000 * state['ee_distance_m']))
                next_frame = round(len(frames) * 1000 / 30)
            state = call(req)
        d = [l['ee_mm'] for l in log]
        at_expert_stop = next((l['ee_mm'] for l in log if l['t'] >= expert_stop / 1000), 1000 * state['ee_distance_m'])
        return dict(bearing=bearing, split=SPLIT[bearing], expert_termination=m['termination'],
                    expert_stop_mm=1000 * m['final_pre_step_ee_error_m'], expert_stop_s=expert_stop / 1000,
                    stop_s=stop_t, at_network_stop_mm=next((l['ee_mm'] for l in log if stop_t is not None and l['t'] >= stop_t), None),
                    min_mm=float(closest) if d else None, at_expert_stop_mm=at_expert_stop,
                    end_mm=1000 * state['ee_distance_m'], log=log, frames=frames,
                    **(dict(camera=True, visible_at_park=log[0]['visible'] if log else None,
                            est_err_at_park_mm=log[0]['est_err_mm'] if log else None,
                            height_err_at_park_mm=(1000 * (log[0]['est'][2] - log[0]['true'][2]) if log and log[0]['est'] else None),
                            steps_without_estimate=never_seen) if camera else {}))
    finally:
        sim.kill()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--bearings', type=int, nargs='*')
    ap.add_argument('--camera', action='store_true'); ap.add_argument('--jobs', type=int, default=6)
    a = ap.parse_args()
    w = dict(np.load(HERE / f'models/{a.model}.npz'))
    bearings = a.bearings or [b for b in sorted(SPLIT) if SPLIT[b] != 'holdout']
    from concurrent.futures import ProcessPoolExecutor      # processes: the camera renders in Python
    from functools import partial
    with ProcessPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(partial(one, w, camera=a.camera), bearings))
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    (out / f"{a.model}{'_camera' if a.camera else ''}.json").write_text(json.dumps(rows))
    for r in rows:
        print(r['bearing'], r['split'], 'expert', round(r['expert_stop_mm'], 1), 'mm | stop', r['stop_s'], 'expert stop', r['expert_stop_s'],
              '| min', round(r['min_mm'], 1), 'at expert stop', round(r['at_expert_stop_mm'], 1), 'end', round(r['end_mm'], 1))
