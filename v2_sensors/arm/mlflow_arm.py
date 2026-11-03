"""Log the arm networks (train_arm.py + test_arm.py) to MLflow, in the same experiment and with the same metadata
as the driving iterations (driving_v2/mlflow_runs.py).

usage: MLFLOW_TRACKING_URI=... python mlflow_arm.py ITERATION [NOTEBOOK.html]     (ITERATION: e.g. 17, 17-seed1, 18)
       MLFLOW_TRACKING_URI=... python mlflow_arm.py cv18 [NOTEBOOK.html]          (the 5-fold epoch choice for iteration 18)
Needs mlflow, numpy and pandas (no torch). Run it after the notebook section is committed and pushed: the
description links to the Markdown copy of the notebook at the current commit.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
from mlflow_runs import EXPERIMENT, digest

NOTEBOOK = HERE.parent / 'driving_v2/driving_v2_sensors.ipynb'
NOTEBOOK_MD = 'https://github.com/NatMota/mlai-vnc/blob/{commit}/v2_sensors/driving_v2/driving_v2_sensors.md'
INPUTS = ([f'arm_angle_{j}' for j in range(1, 8)] + [f'arm_speed_{j}' for j in range(1, 8)]
          + ['dot_x_robot', 'dot_y_robot', 'dot_z_robot', 'time_since_park_s', 'time_since_start_s'])
PLAN30 = [('Plan and expected results', '#iterations-14-to-16-plan-and-expected-results'),
          ('loss curves', '#iterations-14-to-16-loss-curves'),
          ('results', '#arm-30-hz-torque-networks-iterations-14-to-16')]
ARM_ITERATIONS = {
    14: dict(name='arm-30hz-one-bearing', file='arm30_sample_b12', hz=30, target='sample', bearings='12',
             anchors=PLAN30 + [('3D replay', '#iteration-14-3d-replay-arm30_sample_b12')],
             change='A 30 Hz torque network trained on bearing 12 only, to check whether a network that fits one reach almost exactly can repeat it in the simulator.',
             lesson='It fits its training rows almost exactly (torque loss 0.0001) and still swings the arm away 0.53 s after PARK. The failure comes from holding torques for 33 ms, not from poor learning.'),
    15: dict(name='arm-30hz-sampled', file='arm30_sample', hz=30, target='sample', bearings='21 training bearings',
             anchors=PLAN30 + [('3D replay', '#iteration-15-3d-replay-arm30_sample')],
             change='The arm network outputs the 7 torques once per 30 Hz decision, trained on the expert torque at each decision step.',
             lesson='The arm swings away about 0.5 s after PARK on every bearing, as the expert does when its own torques are held for 33 ms (smoke test 3b). Test loss is lowest at epoch 70 and rises after: it overfits.'),
    16: dict(name='arm-30hz-averaged', file='arm30_mean', hz=30, target='mean', bearings='21 training bearings',
             anchors=PLAN30 + [('3D replay', '#iteration-16-3d-replay-arm30_mean')],
             change='As iteration 15, trained on the expert torque averaged over the 33 ms until the next decision.',
             lesson='Much easier to learn (test loss 0.06 against 1.46) but the arm swings away in the same way. Averaging does not make held torques work.'),
    17: dict(name='arm-1khz', file='arm1k', hz=1000, target='sample', bearings='21 training bearings',
             anchors=[('Plan and expected results', '#iteration-17-plan-and-expected-results'), ('loss curves', '#iteration-17-loss-curves'),
                      ('results', '#arm-1-khz-torque-network-iteration-17'), ('3D replay', '#iteration-17-3d-replay-arm1k')],
             change='As iteration 15, but the network runs every 1 ms, the rate at which the expert outputs its torques.',
             lesson='The arm stays stable and the gripper gets to a median 18 mm from the dot, then drifts to a median 124 mm by the expert stop. Test loss is lowest at epoch 120 and rises after: it overfits the 21 training reaches.'),
}
R18 = [('Plan and expected results', '#iteration-18-plan-and-expected-results'), ('choosing the number of epochs', '#iteration-18-choosing-the-number-of-epochs'),
       ('results over four seeds', '#iteration-18-results-over-four-seeds'), ('loss curves', '#iteration-18-loss-curves'),
       ('reading the results', '#iteration-18-reading-the-results')]
for s_ in (1, 2, 3):
    ARM_ITERATIONS[f'17-seed{s_}'] = dict(iteration=17, name=f'arm-1khz-seed{s_}', file=f'arm1k_seed{s_}', hz=1000, target='sample', seed=s_,
        bearings='21 training bearings', anchors=R18 + [('3D replay', f'#iteration-17-seed-{s_}-arm1k_seed{s_}')],
        change=f'Iteration 17 with seed {s_}, to compare iterations 17 and 18 over the same four seeds.',
        lesson='Over seeds 7, 1, 2 and 3, iteration 17 gets the gripper to a median 23 mm from the dot and ends a median 197 mm away at the expert stop. The seed changes the result a lot (75 to 488 mm at the expert stop).')
for s_ in (7, 1, 2, 3):
    k = '18' if s_ == 7 else f'18-seed{s_}'
    f_ = 'arm1k_e77' if s_ == 7 else f'arm1k_e77_seed{s_}'
    ARM_ITERATIONS[18 if s_ == 7 else k] = dict(iteration=18, name='arm-1khz-77-epochs' + ('' if s_ == 7 else f'-seed{s_}'), file=f_, hz=1000, target='sample',
        seed=s_, epochs=77, bearings='21 training bearings', anchors=R18 + [('3D replay', f'#iteration-18-seed-{s_}-{f_}')],
        change='Iteration 17 trained for 77 epochs instead of 300. 77 is where the held-out torque loss, averaged over 5 folds of training bearings, is lowest.',
        lesson='Test torque loss falls (0.38 against 0.64 over four seeds) but the reach in the simulator does not improve (median closest 24 against 23 mm, 190 against 197 mm at the expert stop), and the network says stop far less often (7 against 18 of 25). Overfitting is not the main reason the arm drifts.')


def add(iteration):
    import mlflow, pandas as pd
    from mlflow.entities import Metric
    spec = ARM_ITERATIONS[iteration]; iteration = spec.get('iteration', iteration); seed = spec.get('seed', 7); epochs = spec.get('epochs', 300)
    h = pd.DataFrame(json.loads((HERE / f"models/{spec['file']}_history.json").read_text()))
    rows = json.loads((HERE / f"results/{spec['file']}.json").read_text())
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('log', 'frames')} for r in rows])
    df['stop_vs_expert_s'] = df.stop_s - df.expert_stop_s
    blown = [any(f['parked'] and f['ee_mm'] > 1500 for f in r['frames']) for r in rows]
    one = spec['file'].endswith('_b12')
    m = {'bearings_tested': len(df), 'median_closest_mm': float(df.min_mm.median()), 'best_closest_mm': float(df.min_mm.min()),
         'closest_within_1cm': int((df.min_mm <= 10).sum()), 'median_at_expert_stop_mm': float(df.at_expert_stop_mm.median()),
         'within_1cm_at_expert_stop': int((df.at_expert_stop_mm <= 10).sum()), 'stopped': int(df.stop_s.notna().sum()),
         'gripper_past_1_5m': int(sum(blown)), 'final_train_torque_mse': float(h.train_torque_mse.iloc[-1]),
         'final_train_time_to_stop_mse': float(h.train_time_to_stop_mse.iloc[-1])}
    if df.stop_vs_expert_s.notna().any():
        m['median_stop_vs_expert_s'] = float(df.stop_vs_expert_s.median())
    if not one:
        m.update(final_test_torque_mse=float(h.test_torque_mse.iloc[-1]), final_test_time_to_stop_mse=float(h.test_time_to_stop_mse.iloc[-1]),
                 best_test_torque_mse=float(h.test_torque_mse.min()), best_test_torque_epoch=int(h.epoch[h.test_torque_mse.idxmin()]))
    w = np.load(HERE / f"models/{spec['file']}.npz")
    n_params = int(sum(w[k].size for k in w.files if k.endswith(('.weight', '.bias'))))
    params = dict(version='v2', iteration=iteration, model=spec['name'], part='arm', inputs=len(INPUTS), input_names=','.join(INPUTS),
                  outputs='7_arm_torques_Nm+time_to_stop', stop_threshold_s=0.25, control_hz=spec['hz'], torque_target=spec['target'],
                  hidden='2x64_relu', parameters=n_params, optimizer='adam', learning_rate=0.01, epochs=epochs, batch='full', seed=seed,
                  loss='torque_mse_standardised+time_to_stop_mse_clip3s_weight1', trajectory_weighting='equal',
                  input_scaling='all_standardised', output_scaling='torques_standardised', train_bearings=spec['bearings'],
                  arm_sensor='encoders_kinova_ka75_3686400_ka58_2534400_counts_per_turn', dot_input='simulator_robot_frame',
                  driving='expert_drive', torque_clip_Nm=30, test_bearings=len(df), test_engine='sensor_parking_arm_mode')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    link = NOTEBOOK_MD.format(commit=commit)
    rate = '30 Hz (each torque held 33 ms)' if spec['hz'] == 30 else '1 kHz (a new torque every 1 ms)'
    desc = '\n'.join([
        f"**Iteration {iteration}: `{spec['file']}` (arm)**", '', f"**What changed.** {spec['change']}", '',
        '**Inputs.** 19: arm encoder angles and speeds (7 each, Kinova Jaco 2 actuator resolutions), the dot in the robot frame (x, y, z, from the simulator for this test), time since PARK and time since the start.', '',
        f"**Outputs.** The 7 arm torques (N m, clipped to ±30) at {rate}, and time to stop in seconds (clipped at 3 s); stop when it is 0.25 s or less.", '',
        f"**Training.** 2×64 ReLU network, {n_params:,} parameters. Loss: torque MSE in standardised units plus time-to-stop MSE (weight 1), each training run counted equally. Adam, lr 0.01, {epochs} full-batch epochs, seed {seed}, trained on {spec['bearings']}. Data: the expert's full runs recorded every 1 ms, from PARK to its stop.", '',
        '**Test.** In the simulator on 25 bearings. The expert drives and parks, then only the arm network moves the arm.', '',
        f"- Closest the gripper gets to the dot: median {m['median_closest_mm']:.0f} mm, best {m['best_closest_mm']:.0f} mm; within 1 cm on {m['closest_within_1cm']} of {len(df)}.",
        f"- At the expert's stop time: median {m['median_at_expert_stop_mm']:.0f} mm from the dot. The gripper went more than 1.5 m from the dot on {m['gripper_past_1_5m']} of {len(df)}.",
        f"- The network said stop on {m['stopped']} of {len(df)}" + (f", a median {abs(m['median_stop_vs_expert_s']):.1f} s {'before' if m['median_stop_vs_expert_s'] < 0 else 'after'} the expert." if 'median_stop_vs_expert_s' in m else '.'), '',
        f"**What it tells us.** {spec['lesson']}", '',
        '**Notebook.** ' + ', '.join(f"[{t}]({link}{a})" for t, a in spec['anchors']) + '. The full notebook and the 3D player are attached under Artifacts.'])
    tags = {'mlflow.user': 'nat', 'mlflow.source.type': 'LOCAL', 'mlflow.source.name': 'v2_sensors/driving_v2/driving_v2_sensors.ipynb',
            'mlflow.source.git.repoURL': 'https://github.com/NatMota/mlai-vnc', 'mlflow.source.git.branch': 'v2-sensor',
            'mlflow.source.git.commit': commit, 'mlflow.note.content': desc,
            'version': 'v2', 'iteration': str(iteration), 'model': spec['name'], 'part': 'arm', 'author': 'nat'}
    mlflow.set_experiment(EXPERIMENT); client = mlflow.MlflowClient()
    with mlflow.start_run(run_name=f"v2-iter{iteration:02d}-{spec['name']}", tags=tags) as run:
        mlflow.log_params(params)
        data = HERE / 'data/expert_full'
        src = 'v2_sensors/arm/data/expert_full (collected with expert_compatible --targets ../driving_30/data/expert_30hz/targets.json --binary)'
        files = sorted(data.glob('run_*/manifest.json')) + sorted(data.glob('run_*/arm_controller.npy'))
        mlflow.log_input(mlflow.data.from_pandas(pd.DataFrame([dict(source=src)]), source=src, name='expert_full_1khz', digest=digest(files)), context='training')
        mlflow.log_metrics(m)
        t = int(run.info.start_time)
        keys = ['train_torque_mse', 'train_time_to_stop_mse'] + ([] if one else ['test_torque_mse', 'test_time_to_stop_mse'])
        batch = [Metric(k, float(v), t, int(e)) for k in keys for e, v in zip(h.epoch, h[k])]
        for i in range(0, len(batch), 1000):
            client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
        mlflow.log_artifact(str(HERE / f"models/{spec['file']}.npz"), 'model')
        mlflow.log_artifact(str(HERE / f"models/{spec['file']}_history.json"), 'model')
        mlflow.log_table(df, 'tests/per_bearing.json')
        mlflow.log_artifact(str(HERE / f"results/viewer_{spec['file']}.html"), 'player')
        mlflow.log_artifact(str(NOTEBOOK), 'notebook')
        if len(sys.argv) > 2:
            mlflow.log_artifact(sys.argv[2], 'notebook')
    print('logged', run.info.run_name, run.info.run_id, 'commit', commit[:7])


def cv18():
    """One run for the 5-fold validation that chose 77 epochs for iteration 18 (no simulator test)."""
    import mlflow, pandas as pd
    from mlflow.entities import Metric
    H = [pd.DataFrame(json.loads((HERE / f'models/arm1k_cv_fold{f}_history.json').read_text())) for f in range(5)]
    mean = np.mean([h.val_torque_mse for h in H], 0); chosen = int(np.argmin(mean))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(); link = NOTEBOOK_MD.format(commit=commit)
    desc = '\n'.join(['**Iteration 18: choosing the number of epochs (5-fold validation)**', '',
        'For each of the 5 folds of the 21 training bearings, the iteration 17 network (1 kHz arm torques) was trained on the other 4 folds for 300 epochs and its torque loss on the held-out fold was recorded every epoch. The test bearings were not used.', '',
        f'- The held-out loss averaged over the 5 folds is lowest at epoch {chosen} ({mean[chosen]:.3f}); at epoch 300 it is {mean[-1]:.3f}.',
        '- ' + '; '.join(f'fold {f}: best epoch {int(h.epoch[h.val_torque_mse.idxmin()])}' for f, h in enumerate(H)) + '. The mean is pulled down mostly by fold 4, whose held-out loss rises from 0.57 to 13.8.', '',
        '**Notebook.** ' + ', '.join(f'[{t}]({link}{a})' for t, a in R18[:2]) + '.'])
    tags = {'mlflow.user': 'nat', 'mlflow.source.type': 'LOCAL', 'mlflow.source.name': 'v2_sensors/driving_v2/driving_v2_sensors.ipynb',
            'mlflow.source.git.repoURL': 'https://github.com/NatMota/mlai-vnc', 'mlflow.source.git.branch': 'v2-sensor',
            'mlflow.source.git.commit': commit, 'mlflow.note.content': desc, 'version': 'v2', 'iteration': '18', 'model': 'arm-1khz-epoch-choice', 'part': 'arm', 'author': 'nat'}
    mlflow.set_experiment(EXPERIMENT); client = mlflow.MlflowClient()
    with mlflow.start_run(run_name='v2-iter18-arm-1khz-epoch-choice', tags=tags) as run:
        mlflow.log_params(dict(version='v2', iteration=18, model='arm-1khz-epoch-choice', part='arm', method='5_fold_validation_on_training_bearings',
                               folds=5, epochs_per_fold=300, control_hz=1000, seed=7, chosen_epochs=chosen))
        mlflow.log_metrics(dict(chosen_epochs=chosen, mean_val_torque_mse_at_chosen=float(mean[chosen]), mean_val_torque_mse_at_300=float(mean[-1])))
        t = int(run.info.start_time)
        batch = [Metric('mean_val_torque_mse', float(v), t, e) for e, v in enumerate(mean)]
        batch += [Metric(f'fold{f}_val_torque_mse', float(v), t, int(e)) for f, h in enumerate(H) for e, v in zip(h.epoch, h.val_torque_mse)]
        for i in range(0, len(batch), 1000):
            client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
        for f in range(5):
            mlflow.log_artifact(str(HERE / f'models/arm1k_cv_fold{f}_history.json'), 'folds')
        mlflow.log_artifact(str(HERE / 'results/cv_epochs_arm1k.json'), 'folds')
        mlflow.log_artifact(str(NOTEBOOK), 'notebook')
        if len(sys.argv) > 2:
            mlflow.log_artifact(sys.argv[2], 'notebook')
    print('logged', run.info.run_name, run.info.run_id, 'commit', commit[:7])


if __name__ == '__main__':
    key = sys.argv[1]
    cv18() if key == 'cv18' else add(int(key) if key.isdigit() else key)
