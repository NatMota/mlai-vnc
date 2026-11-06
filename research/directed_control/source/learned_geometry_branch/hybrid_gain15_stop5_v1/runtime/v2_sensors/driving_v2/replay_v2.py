"""Read native closed-loop rollouts of the rung-1 models, check them, and build the 3D ghost-expert player.

Rollouts come from collection/native/build-clang19/policy_replay (the fast collection engine).
Nothing here runs physics or training.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'driving_30'          # reuse the old player: template, meshes, URDF geometry
sys.path.insert(0, str(OLD))
import replay as old_replay

RUNS = HERE / 'closed_loop_native'
MODELS = {'wheels': 'wheels', 'wheels+start_dot': 'wheels_start_dot', 'wheels+live_dot': 'wheels_live_dot',
          'wheels+live_dot+skid': 'wheels_live_dot_skid'}
NATIVE_INPUTS = {'wheels': 'wheels', 'wheels+start_dot': 'start_dot', 'wheels+live_dot': 'live_dot',
                 'wheels+live_dot+skid': 'live_dot'}


def _angle_err(a, b):
    return float(np.rad2deg(abs((a - b + np.pi) % (2 * np.pi) - np.pi)))


def _features(f, initial, bearing_deg, target, inputs):
    q = np.array(f['joints'][:4]) - initial; v = np.array(f['wheel_velocity'])
    r, l = q[[0, 3]].mean(), q[[1, 2]].mean(); vr, vl = v[[0, 3]].mean(), v[[1, 2]].mean()
    x = np.r_[q, v, (r + l) / 2, r - l, (vr + vl) / 2, vr - vl]
    if 'start_dot' in inputs:
        b = np.deg2rad(bearing_deg); x = np.r_[x, np.sin(b), np.cos(b)]
    if 'live_dot' in inputs:
        dx, dy = target[0] - f['base'][0], target[1] - f['base'][1]
        a = np.arctan2(dy, dx) - f['yaw']; x = np.r_[x, np.hypot(dx, dy), np.sin(a), np.cos(a)]
    return x.astype(np.float32)


def _numpy_policy(w, x):
    for layer in ('shared.0', 'shared.2'):
        x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    prob = float(1 / (1 + np.exp(-(x @ w['park.weight'].T + w['park.bias'])[0])))
    return np.clip(x @ w['wheels.weight'].T + w['wheels.bias'], -3, 3), prob


def load(model):
    """Metrics per bearing, plus the max NumPy-vs-native prediction difference (should be ~1e-6)."""
    payload = json.loads((RUNS / f'result_{MODELS[model]}.json').read_text())
    w = np.load(HERE / f'models/rung1_{MODELS[model]}.npz')
    rows, max_diff = [], 0.0
    for run in payload['runs']:
        m, e = run['model'], run['expert']
        initial = np.array(m['frames'][0]['joints'][:4])
        for f in m['frames']:
            if not f['decision'] or (m['park'] and f['step'] > m['park']['step']):
                continue
            cmd, prob = _numpy_policy(w, _features(f, initial, run['bearing_deg'], run['target'], model))
            max_diff = max(max_diff, abs(prob - f['park_probability']))
            if not f['parked']:
                max_diff = max(max_diff, float(np.max(np.abs(cmd - f['command']))))
        p, ep, mf, ef = m['park'], e['park'], m['final'], e['final']
        row = dict(model=model, split=run['split'], bearing_deg=run['bearing_deg'], parked=p is not None,
                   model_park_s=p['time_s'] if p else np.nan, expert_park_s=ep['time_s'],
                   park_xy_error_m=float(np.linalg.norm(np.array(p['base'][:2]) - ep['base'][:2])) if p else np.nan,
                   park_yaw_error_deg=_angle_err(p['yaw'], ep['yaw']) if p else np.nan,
                   final_xy_error_m=float(np.linalg.norm(np.array(mf['base'][:2]) - ef['base'][:2])),
                   final_yaw_error_deg=_angle_err(mf['yaw'], ef['yaw']))
        run['metrics'] = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
        rows.append(row)
    return payload, pd.DataFrame(rows), max_diff


def build_viewer(model):
    payload, metrics, _ = load(model)
    payload['geometry'] = old_replay.geometry(old_replay.native_root(), payload['joint_names'])
    page = ((OLD / 'replay_template.html').read_text()
            .replace('__PLOTLY__', (OLD / 'replay_assets/plotly.min.js').read_text())
            .replace('__DATA__', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')))
    out = RUNS / f'viewer_{MODELS[model]}.html'
    out.write_text(page)
    return out


def run_native(model, cases):
    """Run one model through the fast native engine, unless its result already exists."""
    import os, subprocess
    out = RUNS / f'result_{MODELS[model]}.json'
    if out.exists():
        return out
    w = np.load(HERE / f'models/rung1_{MODELS[model]}.npz')
    request = RUNS / f'request_{MODELS[model]}.json'
    request.write_text(json.dumps(dict(inputs=NATIVE_INPUTS[model], cases=cases,
                                       weights={k: w[k].tolist() for k in w.files})))
    native = old_replay.native_root()
    pixi = next(p for p in [Path.home() / '.local/bin/pixi', Path.home() / '.pixi/bin/pixi'] if p.exists())
    env = dict(os.environ, EXPERT_SKIP_VISUALS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    with (RUNS / f'log_{MODELS[model]}.txt').open('w') as log:
        subprocess.run([str(pixi), 'run', 'build-clang19/policy_replay', str(native / 'assets'), str(request), str(out)],
                       cwd=native, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return out


def build_camera_viewer(name, out_name):
    """3D player for a camera-loop model (camera_loop_frames/NAME, logged with FULL_FRAMES=1) against the expert.
    The expert frames are the expert half of closed_loop_native/result_wheels_live_dot.json."""
    from v2_results import REF
    expert = json.loads((RUNS / 'result_wheels_live_dot.json').read_text())
    runs = []
    for e in sorted(expert['runs'], key=lambda r: r['bearing_deg']):
        b = int(e['bearing_deg']); ref = REF[str(b)]
        r = json.loads((HERE / f'camera_loop_frames/{name}/run_{b}.json').read_text())
        frames, prev = [], None
        for l in r['log']:
            base = l['base3']
            speed = 0.0 if prev is None else float(np.hypot(base[0] - prev[0], base[1] - prev[1]) * 30)
            frames.append(dict(time_s=l['t'], base=base, quaternion=l['quaternion'], joints=l['joints'], ee=base,
                               command=[0.0] * 4 if l['parked'] else l['command'], parked=l['parked'],
                               park_probability=l['park_prob'], ttp_s=l['ttp_s'], speed_m_s=speed))
            prev = base
        park = next((f for f in frames if f['parked']), None)
        metrics = dict(model_park_s=park['time_s'] if park else None, expert_park_s=ref['park_time_s'],
                       park_xy_error_m=float(np.hypot(*np.subtract(park['base'][:2], ref['park_base_xy']))) if park else None,
                       park_yaw_error_deg=_angle_err(next(l['yaw'] for l in r['log'] if l['parked']), ref['park_yaw']) if park else None,
                       final_xy_error_m=float(np.hypot(*np.subtract(r['final_base'], ref['final_base_xy']))))
        runs.append(dict(bearing_deg=b, split=ref['split'], target=e['target'], metrics=metrics,
                         model=dict(frames=frames, final=dict(time_s=frames[-1]['time_s'])), expert=e['expert']))
    payload = dict(joint_names=expert['joint_names'], runs=runs,
                   recording_description='Model: camera + wheel encoders, tested in the simulator. Expert: recorded run for the same bearing.')
    payload['geometry'] = old_replay.geometry(old_replay.native_root(), payload['joint_names'])
    page = ((OLD / 'replay_template.html').read_text()
            .replace('Park probability ≥ 0.5 latches zero wheel commands',
                     'PARK when the predicted time to park is 0.25 s or less; it latches zero wheel commands')
            .replace('Park probability ${m.park_probability.toFixed(3)}', 'Predicted time to park ${m.ttp_s.toFixed(2)} s')
            .replace('__PLOTLY__', (OLD / 'replay_assets/plotly.min.js').read_text())
            .replace('__DATA__', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')))
    out = RUNS / f'viewer_{out_name}.html'
    out.write_text(page)
    return out
