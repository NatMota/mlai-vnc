"""Arm smoke tests 2 and 3: send the expert's own arm torques through the stepping server's arm input.

The expert drives and parks itself (expert_drive), then the arm gets only the torques sent by this script:
  test 2 (--rate 1000): the expert's torque for every 1 ms step, exactly as it was applied.
  test 3 (--rate 30):   the expert's torque averaged over each 30 Hz decision window and held for that window.
  test 3b (--expert-hz N): the expert's own arm controller, closed loop, with its torque sampled at N Hz and held.
For each bearing it reports the gripper-to-dot distance at the expert's stop time and 0.5 s after it, against
the expert's own distance at its stop.

usage: python smoke_arm.py --rate 1000|30 [--bearings 12 24 ...] OUT.json
"""
import argparse, json, os, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data/expert_full'
NATIVE = Path(os.environ.get('NATIVE_ROOT', Path.home() / 'github/comp0245-native-expert'))
PIXI = next(x for x in [Path.home() / '.local/bin/pixi', Path.home() / '.pixi/bin/pixi'] if x.exists())


def table(run, name):
    cols = json.loads((run / f'{name}.schema.json').read_text())['columns']
    return dict(zip(cols, np.load(run / f'{name}.npy').T))


def expert_run(bearing):
    run = DATA / f'run_{bearing // 12 - 1}'
    m = json.loads((run / 'manifest.json').read_text())
    arm = table(run, 'arm_controller')
    tau = np.c_[[arm[f'tau_command_Nm_arm{j}'] for j in range(1, 8)]].T
    return m, tau


def window_end(step):
    """First 30 Hz decision boundary after this step (the server runs to round(k * 1000 / 30))."""
    k = int(step * 30 / 1000)
    while round(k * 1000 / 30) <= step:
        k += 1
    return round(k * 1000 / 30)


def one(bearing, rate, expert_hz=0):
    m, tau = expert_run(bearing)
    stop = m['steps'] - 1                       # the expert's last executed step
    sim = subprocess.Popen([str(PIXI), 'run', 'build-clang19/sensor_parking', str(NATIVE / 'assets')], cwd=NATIVE,
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                           env=dict(os.environ, EXPERT_SKIP_VISUALS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    def call(req):
        sim.stdin.write(json.dumps(req) + '\n'); sim.stdin.flush()
        while True:
            line = sim.stdout.readline()
            if not line:
                raise RuntimeError('sensor_parking exited')
            if '{"' not in line:                    # engine warnings
                continue
            out = json.loads(line[line.index('{"'):])
            if out['type'] != 'ready':
                if out['type'] == 'error':
                    raise RuntimeError(out['error'])
                return out
    try:
        state = call(dict(op='reset', target=m['target_position_world_m'], max_duration_s=stop / 1000 + 1.0,
                          arm=True, expert_drive=True, expert_arm_hz=expert_hz))
        at_stop = park = None
        while not state['done']:
            step = state['frame']['step']
            if state['parked'] and park is None:
                park = step
            if expert_hz:
                req = dict(op='step', wheel=[0] * 4, stop=window_end(step) > stop)
            elif rate == 1000:
                req = dict(op='step', wheel=[0] * 4, ms=1, arm=tau[min(step, len(tau) - 1)].tolist(), stop=step >= stop)
            else:
                end = window_end(step)
                req = dict(op='step', wheel=[0] * 4, arm=tau[step:min(end, len(tau))].mean(0).tolist() if step < len(tau) else tau[-1].tolist(),
                           stop=end > stop)
            state = call(req)
            if at_stop is None and state['frame']['step'] >= stop:
                at_stop = state['ee_distance_m']
        return dict(bearing=bearing, expert_termination=m['termination'], expert_park_step=m['parking_step'], server_park_step=park,
                    expert_stop_mm=1000 * m['final_pre_step_ee_error_m'], at_stop_mm=1000 * at_stop,
                    end_mm=1000 * state['ee_distance_m'], end_step=state['frame']['step'])
    finally:
        sim.kill()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rate', type=int, choices=[1000, 30], default=30)
    ap.add_argument('--expert-hz', type=int, default=0, help='test 3b: the expert arm controller sampled at this rate')
    ap.add_argument('--bearings', type=int, nargs='*', default=list(range(12, 361, 12))); ap.add_argument('out')
    a = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(6) as pool:
        rows = list(pool.map(lambda b: one(b, a.rate, a.expert_hz), a.bearings))
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(r)
