"""Test an arm network (train_arm.py) in the fast simulator.

The expert drives and parks itself (expert_drive), so only the arm network is tested. From PARK the network gets
its inputs at every 30 Hz decision and its 7 torques are held until the next decision. It says stop when the
predicted time to stop is 0.25 s or less (latched); the run ends 0.5 s after stop, or 1 s after the expert's stop.

usage: python test_arm.py MODEL [--bearings B ...]  ->  results/MODEL.json
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import sensors
from train_arm import DATA, SPLIT, dot_in_robot, policy

NATIVE = Path(os.environ.get('NATIVE_ROOT', Path.home() / 'github/comp0245-native-expert'))
PIXI = next(x for x in [Path.home() / '.local/bin/pixi', Path.home() / '.pixi/bin/pixi'] if x.exists())


def one(w, bearing):
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
        enc = sensors.ArmEncoders(); enc.reset(np.array(state['frame']['joints'][sensors.ARM]))
        stop_t = None; log = []
        while not state['done']:
            f = state['frame']; t = f['time_s']
            r = enc.read(np.array(f['joints'][sensors.ARM]))
            req = dict(op='step', wheel=[0] * 4)
            if state['parked']:
                x = np.r_[r['arm_angle'], r['arm_speed'],
                          dot_in_robot(target, np.array(f['base']), f['yaw']), t - state['park_step'] / 1000, t].astype(np.float32)
                tau, ttp = policy(w, x)
                if stop_t is None and ttp <= float(w['ttp_thresh']):
                    stop_t = t
                req.update(arm=np.clip(tau, -30, 30).tolist(), stop=stop_t is not None)
                log.append(dict(t=t, ee_mm=1000 * state['ee_distance_m'], ttp_s=ttp, tau=np.round(tau, 3).tolist()))
            state = call(req)
        d = [l['ee_mm'] for l in log]
        at_expert_stop = next((l['ee_mm'] for l in log if l['t'] >= expert_stop / 1000), 1000 * state['ee_distance_m'])
        return dict(bearing=bearing, split=SPLIT[bearing], expert_termination=m['termination'],
                    expert_stop_mm=1000 * m['final_pre_step_ee_error_m'], expert_stop_s=expert_stop / 1000,
                    stop_s=stop_t, min_mm=min(d) if d else None, at_expert_stop_mm=at_expert_stop,
                    end_mm=1000 * state['ee_distance_m'], log=log)
    finally:
        sim.kill()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('--bearings', type=int, nargs='*')
    a = ap.parse_args()
    w = dict(np.load(HERE / f'models/{a.model}.npz'))
    bearings = a.bearings or [b for b in sorted(SPLIT) if SPLIT[b] != 'holdout']
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(lambda b: one(w, b), bearings))
    out = HERE / 'results'; out.mkdir(exist_ok=True)
    (out / f'{a.model}.json').write_text(json.dumps(rows))
    for r in rows:
        print(r['bearing'], r['split'], 'expert', round(r['expert_stop_mm'], 1), 'mm | stop', r['stop_s'], 'expert stop', r['expert_stop_s'],
              '| min', round(r['min_mm'], 1), 'at expert stop', round(r['at_expert_stop_mm'], 1), 'end', round(r['end_mm'], 1))
