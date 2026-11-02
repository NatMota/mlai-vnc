"""Log every v2 iteration to MLflow with one fixed schema.

Step 1, in the training environment (needs torch):   python mlflow_runs.py collect
   Re-runs the notebook's training cells to get the loss curves, reads the saved test results, writes mlflow_runs.json.
Step 2, in any environment with mlflow:              MLFLOW_TRACKING_URI=... python mlflow_runs.py log
   Creates one run per iteration in the experiment 'v2-sensors'.
Step 3:                                              MLFLOW_TRACKING_URI=... python mlflow_runs.py describe [NOTEBOOK.html]
   Writes each run's description (linked to its notebook section) and attaches the notebook.
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / 'mlflow_runs.json'
EXPERIMENT = 'v2-sensors'
WHEEL_INPUTS = ['wheel_angle_fr', 'wheel_angle_fl', 'wheel_angle_bl', 'wheel_angle_br', 'wheel_speed_fr', 'wheel_speed_fl',
                'wheel_speed_bl', 'wheel_speed_br', 'mean_angle', 'right_minus_left_angle', 'mean_speed', 'right_minus_left_speed']
ITERATIONS = [   # iteration, name, model file, dot input, PARK output, threshold (s), description
    (1, 'wheels', 'rung1_wheels', 'none', 'park_yes_no', None, 'Wheel encoders only.'),
    (2, 'wheels+start_dot', 'rung1_wheels_start_dot', 'true_direction_at_start', 'park_yes_no', None, 'Adds the true dot direction at the start (simulator truth).'),
    (3, 'wheels+live_dot', 'rung1_wheels_live_dot', 'true_position_every_step', 'park_yes_no', None, 'Adds the true dot position every step (simulator truth). Ceiling for the camera models.'),
    (4, 'wheels+live_dot+skid', 'rung1_wheels_live_dot_skid', 'true_position_every_step', 'park_yes_no', None, 'Iteration 3 with a skid-steer output: 2 learned speeds copied to 4 wheels.'),
    (5, 'camera+flag', 'camera_flag', 'camera', 'park_yes_no', None, 'Dot from the corner fisheye camera and tracker, plus a visible flag.'),
    (6, 'camera+flag+time_to_park', 'camera_flag_ttp', 'camera', 'time_to_park', 0.25, 'Iteration 5 with PARK replaced by time to park. Model to build from.'),
    (7, 'camera+time_to_park', 'camera_ttp_noflag', 'camera', 'time_to_park', 0.35, 'Iteration 6 without the visible flag.'),
]
CAMERA_RESULTS = {'camera+flag': 'camera_flag', 'camera+flag+time_to_park': 'camera_flag_ttp', 'camera+time_to_park': 'camera_ttp_noflag_0.35s'}


def digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def collect():
    import nbformat, io, contextlib
    nb = nbformat.read(HERE / 'driving_v2_sensors.ipynb', as_version=4)
    g = {}
    cells = [i for i, c in enumerate(nb.cells) if c.cell_type == 'code']
    want = [cells[0], 3, 7, 27] + [i for i, c in enumerate(nb.cells) if c.source.startswith("CAM = Path('camera_features')")]
    with contextlib.redirect_stdout(io.StringIO()):
        for i in want:
            exec(nb.cells[i].source, g)
    hist, cam_hist = g['history'], g['camera_history']
    import replay_v2
    ref = json.loads((HERE / 'expert_reference.json').read_text())['bearings']
    seen = {int(b): json.loads((HERE / f'camera_loop/camera_flag_ttp/run_{b}.json').read_text())['log'][0]['visible'] for b in ref}
    data = HERE.parent / 'driving_30/data/expert_30hz'
    datasets = {'expert_30hz': dict(digest=digest(list((data / 'sampled').glob('*.npz'))), source='v2_sensors/driving_30/data/expert_30hz',
                                    rows_train=int(sum(len(e['x']) for e in g['train'])), runs_train=len(g['train']), runs_test=len(g['test'])),
                'camera_inputs': dict(digest=digest(list((HERE / 'camera_features').glob('*.npy'))), source='v2_sensors/driving_v2/camera_features')}
    runs = []
    for it, name, file, dot, park, thr, desc in ITERATIONS:
        w = np.load(HERE / f'models/{file}.npz')
        n_in = int(w['shared.0.weight'].shape[1])
        n_params = int(sum(w[k].size for k in w.files if not k.startswith('ttp_')))
        if name.endswith('skid'):
            n_params -= 2 * 17            # trained with 2 wheel outputs; the saved file folds them into 4
        inputs = WHEEL_INPUTS + {'none': [], 'true_direction_at_start': ['dot_sin_start', 'dot_cos_start'],
                                 'true_position_every_step': ['dot_distance', 'dot_sin', 'dot_cos'],
                                 'camera': ['dot_distance', 'dot_sin', 'dot_cos'] + (['dot_visible'] if n_in == 16 else [])}[dot]
        assert len(inputs) == n_in, (name, len(inputs), n_in)
        curves = {}
        if name.startswith('camera'):
            h = cam_hist[cam_hist.model == name]
            for k in ('train_wheel_mse', 'test_wheel_mse', 'train_park_loss', 'test_park_loss'):
                curves[k.replace('park_loss', 'time_to_park_mse' if park == 'time_to_park' else 'park_bce')] = h[k].tolist()
        else:
            h = hist[hist.model == name]
            curves = {'train_wheel_mse': h.train_wheel_mse.tolist(), 'test_wheel_mse': h.test_wheel_mse.tolist(), 'train_park_bce': h.train_park_bce.tolist()}
        rows = []
        if name.startswith('camera'):
            for b, x in ref.items():
                r = json.loads((HERE / f'camera_loop/{CAMERA_RESULTS[name]}/run_{b}.json').read_text()); L = r['log']
                pt = next((l['t'] for l in L if l['park_prob'] >= 0.5), None)
                rows.append(dict(bearing=int(b), split=x['split'], parked=bool(r['parked']),
                                 final_error_cm=100 * float(np.hypot(*np.subtract(r['final_base'], x['final_base_xy']))),
                                 heading_error_deg=float(np.rad2deg(abs((r['final_yaw'] - x['final_yaw'] + np.pi) % (2 * np.pi) - np.pi))),
                                 park_time_delta_s=None if pt is None else pt - x['park_time_s'],
                                 cam_dist_err_cm=[100 * abs(l['est_dist'] - l['true_dist']) for l in L if l['visible']],
                                 cam_dir_err_deg=[abs((l['est_dir_deg'] - l['true_dir_deg'] + 180) % 360 - 180) for l in L if l['visible']]))
        else:
            _, m, _ = replay_v2.load(name)
            for _, x in m.iterrows():
                rows.append(dict(bearing=int(x.bearing_deg), split=x.split, parked=bool(x.parked), final_error_cm=100 * x.final_xy_error_m,
                                 heading_error_deg=x.final_yaw_error_deg,
                                 park_time_delta_s=None if not x.parked else x.model_park_s - x.expert_park_s))
        P = lambda rs: int(sum(r['parked'] for r in rs))
        metrics = {'parked': P(rows), 'parked_rate': P(rows) / len(rows),
                   'parked_dot_in_view_at_start': P([r for r in rows if seen[r['bearing']]]),
                   'parked_dot_out_of_view_at_start': P([r for r in rows if not seen[r['bearing']]]),
                   'parked_test_split': P([r for r in rows if r['split'] == 'test']),
                   'median_final_error_cm': float(np.median([r['final_error_cm'] for r in rows])),
                   'mean_final_error_cm': float(np.mean([r['final_error_cm'] for r in rows])),
                   'median_heading_error_deg': float(np.median([r['heading_error_deg'] for r in rows]))}
        dt = [abs(r['park_time_delta_s']) for r in rows if r['park_time_delta_s'] is not None]
        if dt:
            metrics['median_abs_park_time_error_s'] = float(np.median(dt))
        if name.startswith('camera'):
            metrics['camera_distance_error_cm'] = float(np.mean([v for r in rows for v in r['cam_dist_err_cm']]))
            metrics['camera_direction_error_deg'] = float(np.mean([v for r in rows for v in r['cam_dir_err_deg']]))
            for r in rows:
                r.pop('cam_dist_err_cm'); r.pop('cam_dir_err_deg')
        for k, v in curves.items():
            metrics[f'final_{k}'] = v[-1]
        params = dict(version='v2', iteration=it, model=name, inputs=n_in, input_names=','.join(inputs), dot_input=dot,
                      visible_flag=n_in == 16 and dot == 'camera', park_output=park, park_threshold_s=thr if thr is not None else 'n/a',
                      wheel_output='skid_2_to_4' if name.endswith('skid') else '4_free', hidden='2x16_relu', parameters=n_params,
                      optimizer='adam', learning_rate=0.01, epochs=300, batch='full', seed=7,
                      park_loss='mse_time_to_park_clip3s_weight1' if park == 'time_to_park' else 'bce_pos_weight_x0.1',
                      trajectory_weighting='equal', input_scaling='none', control_hz=30,
                      camera='corner_fisheye_180_rgbd_mount_0.29_-0.13_0.36' if dot == 'camera' else 'none',
                      renderer='pybullet_tiny_cpu' if dot == 'camera' else 'n/a', tracker='wheel_odometry' if dot == 'camera' else 'n/a',
                      test_bearings=len(rows), test_engine='sensor_parking' if dot == 'camera' else 'policy_replay')
        runs.append(dict(iteration=it, name=name, description=desc, params=params, metrics=metrics, curves=curves, per_bearing=rows,
                         model_file=f'models/{file}.npz', datasets=['expert_30hz', 'camera_inputs'] if dot == 'camera' else ['expert_30hz']))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    OUT.write_text(json.dumps(dict(commit=commit, datasets=datasets, runs=runs), indent=1, default=float))
    print('wrote', OUT, len(runs), 'runs')


def log():
    import mlflow, pandas as pd
    from mlflow.entities import Metric
    payload = json.loads(OUT.read_text())
    mlflow.set_experiment(EXPERIMENT)
    client = mlflow.MlflowClient()
    for r in payload['runs']:
        tags = {'mlflow.user': 'nat', 'mlflow.source.type': 'LOCAL', 'mlflow.source.name': 'v2_sensors/driving_v2/driving_v2_sensors.ipynb',
                'mlflow.source.git.repoURL': 'https://github.com/NatMota/mlai-vnc', 'mlflow.source.git.branch': 'v2-sensor',
                'mlflow.source.git.commit': payload['commit'], 'mlflow.note.content': r['description'],
                'version': 'v2', 'iteration': str(r['iteration']), 'model': r['name'], 'author': 'nat'}
        with mlflow.start_run(run_name=f"v2-iter{r['iteration']:02d}-{r['name'].replace('+', '-')}", tags=tags) as run:
            mlflow.log_params(r['params'])
            for d in r['datasets']:
                info = payload['datasets'][d]
                ds = mlflow.data.from_pandas(pd.DataFrame([info]), source=info['source'], name=d, digest=info['digest'])
                mlflow.log_input(ds, context='training' if d == 'expert_30hz' else 'inputs')
            mlflow.log_metrics(r['metrics'])
            t = int(run.info.start_time)
            batch = [Metric(k, float(v), t, step) for k, vals in r['curves'].items() for step, v in enumerate(vals)]
            for i in range(0, len(batch), 1000):
                client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
            mlflow.log_artifact(str(HERE / r['model_file']), 'model')
            mlflow.log_table(pd.DataFrame(r['per_bearing']), 'tests/per_bearing.json')
        print('logged', run.info.run_name, run.info.run_id)


NOTEBOOK_MD = 'https://github.com/NatMota/mlai-vnc/blob/ef677675aa9a37623b1affacbf7bcb341785ec55/v2_sensors/driving_v2/driving_v2_sensors.md'
INTRO, LOOP, SKID, SKID_TEST = '#driving-v2--rung-1-does-the-network-work-when-it-can-see-the-dot', '#closed-loop-the-network-drives-the-robot', '#change-1-skid-steer-output', '#closed-loop-skid-output-vs-four-free-outputs'
CAMERA, CURVES, TESTS, PLOTS = '#camera-models-the-dot-comes-from-the-corner-camera', '#training-curves', '#testing', '#test-metrics-plotted'
WHEELS = 'the angle each wheel has turned since the start, each wheel speed, and the average and right minus left of both'
DESCRIPTIONS = {
    'wheels': dict(change='Starting point. Wheel encoders only, nothing about the dot.',
        inputs=f'12: {WHEELS}.',
        lesson='At the start every bearing gives the same inputs but a different expert command, so the network cannot know which way to turn. This is the floor any dot input has to beat.',
        links=[('What the iterations are', INTRO), ('Test results', LOOP)]),
    'wheels+start_dot': dict(change='Adds the true direction of the dot at the start, taken from the simulator and fixed for the whole run.',
        inputs=f'14: {WHEELS}, plus sin and cos of the starting dot direction.',
        lesson='The starting direction helps most when the dot starts in front (9 of 11 parked) and little when it starts behind (2 of 14). The input never updates after the start, and the robot often stops in the wrong place.',
        links=[('What the iterations are', INTRO), ('Test results', LOOP)]),
    'wheels+live_dot': dict(change='Gives the true dot position relative to the robot at every step, taken from the simulator.',
        inputs=f'15: {WHEELS}, plus the dot distance and sin and cos of its direction.',
        lesson='This is a perfect sensor, so it is the ceiling for the camera models. It uses simulator truth and does not count as a v2 result.',
        links=[('What the iterations are', INTRO), ('Test results', LOOP)]),
    'wheels+live_dot+skid': dict(change='Iteration 3 with a skid-steer output. The network predicts one speed for the right side and one for the left, and they are copied to the front and back wheels, because the expert always gives front and back the same speed.',
        inputs=f'15, as in iteration 3.',
        lesson='Its final position error is lower than iteration 3, but PARK rarely fires (its PARK probability mostly stays near 0.01), so it parks less often. Not carried forward.',
        links=[('Why skid steering', SKID), ('Test results', SKID_TEST)]),
    'camera+flag': dict(change='First model with no simulator truth. The dot comes from the corner fisheye camera: red pixels give a ray and a depth, the fixed mount turns that into a position relative to the robot, and a tracker keeps it, moving it with the wheel encoders while the dot is hidden.',
        inputs=f'16: {WHEELS}, plus the tracked dot distance, sin and cos of its direction (all 0 until the dot is first seen), and a visible flag (1 when the camera sees the dot in this frame).',
        lesson='When the camera sees the dot it is accurate to about 1 cm, so the camera is not the weak point. Most failures are PARK not firing: the robot stops 3 to 36 cm from the expert parking spot, but the PARK probability stays below 0.35.',
        links=[('How the camera inputs are made', CAMERA), ('Training curves', CURVES), ('Test results', TESTS), ('Plots', PLOTS)]),
    'camera+flag+time_to_park': dict(change='Iteration 5 with the PARK output replaced by time to park. The network predicts how many seconds are left until the expert would park, and PARK fires when that drops to 0.25 s or less.',
        inputs='16, as in iteration 5.',
        lesson='When it parks, it parks close to the expert in time and place. When it fails, the robot stalls 20 to 40 cm short and the predicted time to park stays above 0.25 s. This is the model to build from.',
        links=[('How the camera inputs are made', CAMERA), ('Training curves', CURVES), ('Test results', TESTS), ('Plots', PLOTS)]),
    'camera+time_to_park': dict(change='Iteration 6 without the visible flag. PARK fires at 0.35 s or less.',
        inputs=f'15: {WHEELS}, plus the tracked dot distance and sin and cos of its direction.',
        lesson='Without the flag the robot stalls about 50 cm short in most runs, and it never parks when the dot starts out of view. The flag is needed.',
        links=[('How the camera inputs are made', CAMERA), ('Training curves', CURVES), ('Test results', TESTS), ('Plots', PLOTS)]),
}


def description(r):
    d, m, pr = DESCRIPTIONS[r['name']], r['metrics'], r['params']
    park = (f"time to park in seconds (clipped at 3 s), PARK when it is {pr['park_threshold_s']} s or less" if pr['park_output'] == 'time_to_park'
            else 'PARK probability, PARK when it is 0.5 or more')
    loss = ('wheel MSE plus time to park MSE (weight 1)' if pr['park_output'] == 'time_to_park'
            else 'wheel MSE plus 0.1 times class-weighted PARK cross entropy')
    lines = [f"**Iteration {r['iteration']}: `{r['name']}`**", '',
             f"**What changed.** {d['change']}", '',
             f"**Inputs.** {d['inputs']}", '',
             f"**Outputs.** Four wheel speed commands (rad/s, clipped to ±3) and {park}.", '',
             f"**Training.** 2×16 ReLU network, {pr['parameters']} parameters. Loss: {loss}, each training run counted equally. Adam, lr 0.01, 300 full-batch epochs, seed 7, 21 training runs.", '',
             f"**Test.** The model drives the robot in the simulator from 25 bearings (21 training, 4 test).", '',
             f"- Parked {m['parked']} of 25: {m['parked_dot_in_view_at_start']} of 11 with the dot in view at the start, {m['parked_dot_out_of_view_at_start']} of 14 with it out of view, {m['parked_test_split']} of 4 test runs.",
             f"- Final distance from the expert: median {m['median_final_error_cm']:.0f} cm. Heading: median {m['median_heading_error_deg']:.0f}° off."]
    if 'median_abs_park_time_error_s' in m:
        lines.append(f"- When it parks, it parks a median {m['median_abs_park_time_error_s']:.1f} s from the expert's park time.")
    if 'camera_distance_error_cm' in m:
        lines.append(f"- Camera, when it sees the dot: {m['camera_distance_error_cm']:.1f} cm and {m['camera_direction_error_deg']:.1f}° mean error.")
    lines += ['', f"**What it tells us.** {d['lesson']}", '',
              '**Notebook.** ' + ', '.join(f"[{t}]({NOTEBOOK_MD}{a})" for t, a in d['links']) +
              '. The full notebook is attached under Artifacts (`notebook/`).']
    return '\n'.join(lines)


def describe():
    import mlflow
    payload = json.loads(OUT.read_text())
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    html = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    for r in payload['runs']:
        name = f"v2-iter{r['iteration']:02d}-{r['name'].replace('+', '-')}"
        run = client.search_runs([exp.experiment_id], filter_string=f"attributes.run_name = '{name}'")[0]
        client.set_tag(run.info.run_id, 'mlflow.note.content', description(r))
        client.log_artifact(run.info.run_id, str(HERE / 'driving_v2_sensors.ipynb'), 'notebook')
        if html:
            client.log_artifact(run.info.run_id, str(html), 'notebook')
        print('described', name)


# Iterations trained with train_v2.py: loss curves come from models/NAME_history.json, tests from camera_loop/FOLDER.
# iteration, name, model file and result folder, notebook anchors, what changed, lesson
NEW_ITERATIONS = {
    8: dict(name='camera+flag+time_to_park+scaled', file='camera_flag_ttp_scaled', seed=7, scale=True, tail=0, encoders=False,
            anchors=[('Why scale', '#why-the-inputs-are-on-very-different-scales'), ('What changed', '#what-changed'),
                     ('Test results', '#testing-25-bearings-compared-with-iteration-6')],
            change='Iteration 6 with the six angle inputs (4 wheel angles, mean angle, right minus left angle) standardised with the training-set mean and standard deviation. The angles reach about 14 rad while the dot inputs stay between -1 and 1. The scaling is folded into the first layer, so the saved network takes raw inputs.',
            lesson='Every bearing where the dot starts in view now parks (11 of 11, up from 7), but the search for a dot behind the robot got worse (3 of 14, down from 9). One seed cannot separate the change from chance; iteration 9 measures that.'),
    9: dict(name='camera+flag+time_to_park+scaled+seed1', file='camera_flag_ttp_scaled_seed1', seed=1, scale=True, tail=0, encoders=False,
            anchors=[('What the seed changes', '#what-the-seed-changes'), ('What the seeds show', '#what-the-seeds-show')],
            change='Iteration 8 trained with seed 1 instead of 7. Same data, inputs, outputs and training; only the starting weights differ.',
            lesson='Parked counts across seeds 7, 1, 2 and 3 are 14, 0, 4 and 21 of 25 with nothing else changed, so single-seed differences mean little. Test losses are almost identical. Seed 1 ends within 20 cm of the expert in 19 of 25 runs but never parks, because its predicted time to park never drops below 0.35 s. The fixed 0.25 s threshold decides most of the outcome.'),
    10: dict(name='camera+flag+time_to_park+scaled+seed2', file='camera_flag_ttp_scaled_seed2', seed=2, scale=True, tail=0, encoders=False,
            anchors=[('What the seed changes', '#what-the-seed-changes'), ('What the seeds show', '#what-the-seeds-show')],
            change='Iteration 8 trained with seed 2 instead of 7. Same data, inputs, outputs and training; only the starting weights differ.',
            lesson='Parked counts across seeds 7, 1, 2 and 3 are 14, 0, 4 and 21 of 25 with nothing else changed, so single-seed differences mean little. Test losses are almost identical. Seed 1 ends within 20 cm of the expert in 19 of 25 runs but never parks, because its predicted time to park never drops below 0.35 s. The fixed 0.25 s threshold decides most of the outcome.'),
    11: dict(name='camera+flag+time_to_park+scaled+seed3', file='camera_flag_ttp_scaled_seed3', seed=3, scale=True, tail=0, encoders=False,
            anchors=[('What the seed changes', '#what-the-seed-changes'), ('What the seeds show', '#what-the-seeds-show')],
            change='Iteration 8 trained with seed 3 instead of 7. Same data, inputs, outputs and training; only the starting weights differ.',
            lesson='Parked counts across seeds 7, 1, 2 and 3 are 14, 0, 4 and 21 of 25 with nothing else changed, so single-seed differences mean little. Test losses are almost identical. Seed 1 ends within 20 cm of the expert in 19 of 25 runs but never parks, because its predicted time to park never drops below 0.35 s. The fixed 0.25 s threshold decides most of the outcome.'),
    12: dict(iteration=12, name='camera+flag+time_to_park+scaled+parked_rows', file='camera_flag_ttp_scaled_tail100', seed=7, scale=True, tail=100, encoders=False,
            anchors=[('Why add parked rows', '#iteration-12-100-rows-after-the-expert-parks'), ('Results over four seeds', '#iteration-12-results-over-four-seeds')],
            change='Iteration 8 (seed 7) plus 100 rows after the expert parks in every training run: wheels stopped, time to park 0. The expert was replayed for 3.4 s past PARK in the fast simulator; before PARK the replay matches the saved data to 3e-8.',
            lesson='Averaged over seeds 7, 1, 2 and 3 it parked 5.75 of 25 against 9.75 for iterations 8 to 11, and the seeds still range from 0 to 12. Runs ending within 20 cm of the expert hardly changed (13.25 against 13). The parked rows did not make PARK reliable. The variation between seeds is the bigger problem.'),
    '12-seed1': dict(iteration=12, name='camera+flag+time_to_park+scaled+parked_rows+seed1', file='camera_flag_ttp_scaled_tail100_seed1', seed=1, scale=True, tail=100, encoders=False,
            anchors=[('Why add parked rows', '#iteration-12-100-rows-after-the-expert-parks'), ('Results over four seeds', '#iteration-12-results-over-four-seeds')],
            change='Iteration 8 (seed 1) plus 100 rows after the expert parks in every training run: wheels stopped, time to park 0. The expert was replayed for 3.4 s past PARK in the fast simulator; before PARK the replay matches the saved data to 3e-8.',
            lesson='Averaged over seeds 7, 1, 2 and 3 it parked 5.75 of 25 against 9.75 for iterations 8 to 11, and the seeds still range from 0 to 12. Runs ending within 20 cm of the expert hardly changed (13.25 against 13). The parked rows did not make PARK reliable. The variation between seeds is the bigger problem.'),
    '12-seed2': dict(iteration=12, name='camera+flag+time_to_park+scaled+parked_rows+seed2', file='camera_flag_ttp_scaled_tail100_seed2', seed=2, scale=True, tail=100, encoders=False,
            anchors=[('Why add parked rows', '#iteration-12-100-rows-after-the-expert-parks'), ('Results over four seeds', '#iteration-12-results-over-four-seeds')],
            change='Iteration 8 (seed 2) plus 100 rows after the expert parks in every training run: wheels stopped, time to park 0. The expert was replayed for 3.4 s past PARK in the fast simulator; before PARK the replay matches the saved data to 3e-8.',
            lesson='Averaged over seeds 7, 1, 2 and 3 it parked 5.75 of 25 against 9.75 for iterations 8 to 11, and the seeds still range from 0 to 12. Runs ending within 20 cm of the expert hardly changed (13.25 against 13). The parked rows did not make PARK reliable. The variation between seeds is the bigger problem.'),
    '12-seed3': dict(iteration=12, name='camera+flag+time_to_park+scaled+parked_rows+seed3', file='camera_flag_ttp_scaled_tail100_seed3', seed=3, scale=True, tail=100, encoders=False,
            anchors=[('Why add parked rows', '#iteration-12-100-rows-after-the-expert-parks'), ('Results over four seeds', '#iteration-12-results-over-four-seeds')],
            change='Iteration 8 (seed 3) plus 100 rows after the expert parks in every training run: wheels stopped, time to park 0. The expert was replayed for 3.4 s past PARK in the fast simulator; before PARK the replay matches the saved data to 3e-8.',
            lesson='Averaged over seeds 7, 1, 2 and 3 it parked 5.75 of 25 against 9.75 for iterations 8 to 11, and the seeds still range from 0 to 12. Runs ending within 20 cm of the expert hardly changed (13.25 against 13). The parked rows did not make PARK reliable. The variation between seeds is the bigger problem.'),
}


def add(iteration):
    """Collect, log and describe one train_v2.py iteration. Needs mlflow, numpy and pandas (no torch)."""
    import mlflow, pandas as pd
    from mlflow.entities import Metric
    import v2_results
    spec = NEW_ITERATIONS[iteration]; name = spec['name']; iteration = spec.get('iteration', iteration)
    h = v2_results.history(spec['file'])
    df = v2_results.tests(spec['file'])
    seen = df.seen_at_start
    m = {'parked': int(df.parked.sum()), 'parked_rate': float(df.parked.mean()),
         'parked_dot_in_view_at_start': int(df[seen].parked.sum()), 'parked_dot_out_of_view_at_start': int(df[~seen].parked.sum()),
         'parked_test_split': int(df[df.split == 'test'].parked.sum()),
         'median_final_error_cm': float(df.final_error_cm.median()), 'mean_final_error_cm': float(df.final_error_cm.mean()),
         'median_heading_error_deg': float(df.heading_error_deg.median()),
         'camera_distance_error_cm': float(df.camera_distance_error_cm.mean())}
    dt = df.park_time_vs_expert_s.dropna().abs()
    if len(dt):
        m['median_abs_park_time_error_s'] = float(dt.median())
    for k in ('train_wheel_mse', 'test_wheel_mse', 'train_time_to_park_mse', 'test_time_to_park_mse'):
        m['final_' + k] = float(h[k].iloc[-1])
    inputs = WHEEL_INPUTS + ['dot_distance', 'dot_sin', 'dot_cos', 'dot_visible']
    params = dict(version='v2', iteration=iteration, model=name, inputs=16, input_names=','.join(inputs), dot_input='camera',
                  visible_flag=True, park_output='time_to_park', park_threshold_s=0.25, wheel_output='4_free', hidden='2x16_relu',
                  parameters=629, optimizer='adam', learning_rate=0.01, epochs=300, batch='full', seed=spec['seed'],
                  park_loss='mse_time_to_park_clip3s_weight1', trajectory_weighting='equal',
                  input_scaling='angles_standardised' if spec['scale'] else 'none', park_tail_rows=spec['tail'],
                  wheel_sensor='encoders_4096_counts_per_rev' if spec['encoders'] else 'simulator_joint_values', control_hz=30,
                  camera='corner_fisheye_180_rgbd_mount_0.29_-0.13_0.36', renderer='pybullet_tiny_cpu', tracker='wheel_odometry',
                  test_bearings=len(df), test_engine='sensor_parking')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    link = NOTEBOOK_MD.replace(NOTEBOOK_MD.split('/blob/')[1].split('/')[0], commit)
    mlflow.set_experiment(EXPERIMENT); client = mlflow.MlflowClient()
    desc = '\n'.join([f"**Iteration {iteration}: `{name}`**", '', f"**What changed.** {spec['change']}", '',
        f"**Inputs.** 16: {WHEELS}{' (read from simulated wheel encoders)' if spec['encoders'] else ''}, plus the tracked dot distance, sin and cos of its direction (0 until first seen) and a visible flag.", '',
        '**Outputs.** Four wheel speed commands (rad/s, clipped to ±3) and time to park in seconds (clipped at 3 s), PARK when it is 0.25 s or less.', '',
        f"**Training.** 2×16 ReLU network, 629 parameters. Loss: wheel MSE plus time to park MSE (weight 1), each training run counted equally. Adam, lr 0.01, 300 full-batch epochs, seed {spec['seed']}, 21 training runs"
        + (f", plus {spec['tail']} rows after the expert parks in each run" if spec['tail'] else '') + '.', '',
        '**Test.** The model drives the robot in the simulator from 25 bearings (21 training, 4 test).', '',
        f"- Parked {m['parked']} of 25: {m['parked_dot_in_view_at_start']} of 11 with the dot in view at the start, {m['parked_dot_out_of_view_at_start']} of 14 with it out of view, {m['parked_test_split']} of 4 test runs.",
        f"- Final distance from the expert: median {m['median_final_error_cm']:.0f} cm. Heading: median {m['median_heading_error_deg']:.0f}° off."]
        + ([f"- When it parks, it parks a median {m['median_abs_park_time_error_s']:.1f} s from the expert's park time."] if 'median_abs_park_time_error_s' in m else [])
        + [f"- Camera, when it sees the dot: {m['camera_distance_error_cm']:.1f} cm mean distance error.", '',
           f"**What it tells us.** {spec['lesson']}", '',
           '**Notebook.** ' + ', '.join(f"[{t}]({link}{a})" for t, a in spec['anchors']) + '. The full notebook is attached under Artifacts (`notebook/`).'])
    tags = {'mlflow.user': 'nat', 'mlflow.source.type': 'LOCAL', 'mlflow.source.name': 'v2_sensors/driving_v2/driving_v2_sensors.ipynb',
            'mlflow.source.git.repoURL': 'https://github.com/NatMota/mlai-vnc', 'mlflow.source.git.branch': 'v2-sensor',
            'mlflow.source.git.commit': commit, 'mlflow.note.content': desc,
            'version': 'v2', 'iteration': str(iteration), 'model': name, 'author': 'nat'}
    with mlflow.start_run(run_name=f"v2-iter{iteration:02d}-{name.replace('+', '-')}", tags=tags) as run:
        mlflow.log_params(params)
        data = HERE.parent / 'driving_30/data/expert_30hz'
        for dname, src, dig, ctx in (('expert_30hz', 'v2_sensors/driving_30/data/expert_30hz', digest(list((data / 'sampled').glob('*.npz'))), 'training'),
                                     ('camera_inputs', 'v2_sensors/driving_v2/camera_features', digest(list((HERE / 'camera_features').glob('*.npy'))), 'inputs')):
            mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame([dict(source=src)]), source=src, name=dname, digest=dig), context=ctx)
        if spec['tail']:
            src = 'v2_sensors/driving_v2/park_tail'
            mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame([dict(source=src)]), source=src, name='park_tail',
                                                     digest=digest(list((HERE / 'park_tail').glob('tail_*.npz')))), context='training')
        mlflow.log_metrics(m)
        t = int(run.info.start_time)
        batch = [Metric(k, float(v), t, int(e)) for k in ('train_wheel_mse', 'test_wheel_mse', 'train_time_to_park_mse', 'test_time_to_park_mse')
                 for e, v in zip(h.epoch, h[k])]
        for i in range(0, len(batch), 1000):
            client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
        mlflow.log_artifact(str(HERE / f"models/{spec['file']}.npz"), 'model')
        mlflow.log_table(df, 'tests/per_bearing.json')
        mlflow.log_artifact(str(HERE / 'driving_v2_sensors.ipynb'), 'notebook')
        if len(sys.argv) > 3:
            mlflow.log_artifact(sys.argv[3], 'notebook')
    print('logged', run.info.run_name, run.info.run_id, 'commit', commit[:7])


if __name__ == '__main__':
    if sys.argv[1] == 'add':
        key = sys.argv[2]                 # python mlflow_runs.py add ITERATION [NOTEBOOK.html]
        add(int(key) if key.isdigit() else key)
    else:
        {'collect': collect, 'log': log, 'describe': describe}[sys.argv[1]]()
