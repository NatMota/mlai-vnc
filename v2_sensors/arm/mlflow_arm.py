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


ARM_ITERATIONS[19] = dict(iteration=19, name='arm-1khz-stop-on-distance', file='arm1k_e77_dist', hz=1000, target='sample', seed=7, epochs=77,
    bearings='21 training bearings',
    anchors=[('Plan and expected results', '#iteration-19-plan-and-expected-results'), ('results', '#iteration-19-results'),
             ('predicted against actual distance', '#iteration-19-predicted-against-actual-distance'), ('loss curves', '#iteration-19-loss-curves'),
             ('3D replay', '#iteration-19-3d-replay-arm1k_e77_dist'), ('reading the results', '#iteration-19-reading-the-results')],
    change='Iteration 18, seed 7, with the stop head trained on the distance from the gripper to the dot (from the expert recordings) instead of the time to stop. Stop when the predicted distance is 1 cm or less.',
    lesson='Stopping on distance works far better than on time: the network stops on all 25 bearings, a median 35 mm from the dot against 257 mm, and on 18 of 25 within 10 mm of the closest point the gripper reaches. The reach itself is now the limit: the gripper never gets within 1 cm (median closest 29 mm) and drifts after stop. Near the dot the predicted distance is a median 14 mm too low.')


ARM_ITERATIONS[20] = dict(iteration=20, name='arm-1khz-camera-dot', file='arm1k_e77_dist', results='arm1k_e77_dist_camera', hz=1000, target='sample',
    seed=7, epochs=77, bearings='21 training bearings', dot_input='camera_tracker',
    anchors=[('Plan and expected results', '#iteration-20-plan-and-expected-results'), ('results', '#iteration-20-results'),
             ("the camera's estimate during the reach", '#iteration-20-the-cameras-estimate-during-the-reach'), ('loss curves', '#iteration-20-loss-curves'),
             ('3D replay', '#iteration-20-3d-replay-arm1k_e77_dist-dot-from-the-camera'), ('reading the results', '#iteration-20-reading-the-results')],
    change="Iteration 19's network, not retrained, with the dot from the corner camera and tracker (keeping the height) instead of the simulator. The expert still drives.",
    lesson='The camera is not good enough for the arm as it is used now: the gripper gets a median 105 mm from the dot (29 mm with the simulator dot) and the network stops a median 214 mm away. The estimate error itself is moderate (27 mm at PARK, 11 mm during the reach), but the dot input jumps when a hidden dot comes into view after PARK, which the network never saw in training, and the arm swings away 0.5 s after PARK. Next: one fixed dot position after a fresh view.')


_A21 = [('Plan and expected results', '#iteration-21-plan-and-expected-results'), ('results', '#iteration-21-results'),
        ('gripper height after PARK', '#iteration-21-gripper-height-after-park'), ('loss curves', '#iteration-21-loss-curves'),
        ('the dot input at PARK', '#iteration-21-the-dot-input-at-park-training-against-the-camera'), ('reading the results', '#iteration-21-reading-the-results')]
ARM_ITERATIONS[21] = dict(iteration=21, name='arm-1khz-no-time-inputs', file='arm1k_e77_dist_notime', hz=1000, target='sample', seed=7, epochs=77,
    bearings='21 training bearings', n_inputs=17, anchors=_A21 + [('3D replay', '#iteration-21-3d-replay-with-the-simulators-dot-arm1k_e77_dist_notime')],
    change='Iteration 19 retrained without the two time inputs (time since PARK, time since the start): 17 inputs. Tested with the simulator dot.',
    lesson='The clock was not why the arm falls: without it the arm falls more (median 111 mm after its closest point, against 5 mm). The time inputs were what told the network to hold late in the run. Iteration 19 stays the best arm network with the simulator dot.')
ARM_ITERATIONS['21-camera'] = dict(iteration=21, name='arm-1khz-no-time-inputs-camera-dot', file='arm1k_e77_dist_notime', results='arm1k_e77_dist_notime_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=_A21 + [('3D replay', '#iteration-21-3d-replay-with-the-camera-arm1k_e77_dist_notime-dot-from-the-camera')],
    change='The iteration 21 network (no time inputs) tested with the dot from the corner camera and tracker.',
    lesson='The arm still falls (median 386 mm). The cause is the dot height input: over the 21 training runs it spreads by only 0.2 mm, so the camera height error of 11 mm at PARK is about 60 standard deviations, far outside training. Next: leave the height out, or train with the camera error or other dot heights.')


_A22 = [('Plan and expected results', '#iteration-22-plan-and-expected-results'), ('results', '#iteration-22-results'),
        ('gripper height after PARK', '#iteration-22-gripper-height-after-park'), ('loss curves', '#iteration-22-loss-curves'),
        ('reading the results', '#iteration-22-reading-the-results')]
ARM_ITERATIONS[22] = dict(iteration=22, name='arm-1khz-no-scaling-camera-dot', file='arm1k_e77_dist_notime_raw', results='arm1k_e77_dist_notime_raw_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=_A22 + [('3D replay', '#iteration-22-3d-replay-arm1k_e77_dist_notime_raw-dot-from-the-camera')],
    change='Iteration 21 (17 inputs, no time) retrained with no scaling: raw inputs (rad, rad/s, m) and raw torques (N m) instead of mean/std on all of them. Tested with the dot from the corner camera and tracker.',
    lesson='Without scaling the network fits the torques poorly (2.17 N m² on the training runs against 0.37 for iteration 21 and 3.68 for the mean torque). With the camera it says stop on all 25 bearings but a median 0.64 s after PARK and 239 mm from the dot; the median fall of 195 mm (386 mm in iteration 21) is lower only because the run ends sooner. The gripper drops from PARK and never holds.')


ARM_ITERATIONS[23] = dict(iteration=23, name='arm-1khz-physical-input-scaling-camera-dot', file='arm1k_e77_dist_notime_phys', results='arm1k_e77_dist_notime_phys_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=[('Plan and expected results', '#iteration-23-plan-and-expected-results'), ('results', '#iteration-23-results'),
             ('gripper height after PARK', '#iteration-23-gripper-height-after-park'), ('torque error per joint', '#iteration-23-torque-error-per-joint'),
             ('loss curves', '#iteration-23-loss-curves'), ('3D replay', '#iteration-23-3d-replay-arm1k_e77_dist_notime_phys-dot-from-the-camera'),
             ('reading the results', '#iteration-23-reading-the-results')],
    change='Iteration 21 (17 inputs, no time) with each input divided by a fixed physical size instead of standardised: angles by π, speeds by the speed limits (0.63 rad/s joints 1 to 4, 0.84 rad/s joints 5 to 7), the dot by 1 m. Torques standardised as before. Tested with the dot from the corner camera and tracker.',
    lesson='The input scaling was the main reason the camera runs failed. With inputs divided by physical sizes the gripper gets a median 57 mm from the dot (189 mm in iteration 21), within 50 mm on 9 of 25, with no fall and 0.2% of torques at the limit. The stop is now the limit: the network says stop on all 25 bearings but a median 0.81 s after PARK and 104 mm from the dot, while the arm is still moving towards it. Training is not finished at 77 epochs (losses still falling).')

def _stop_anchors(n, model):
    return [('Plan and expected results', f'#iteration-{n}-plan-and-expected-results'), ('results', f'#iteration-{n}-results'),
            ('gripper height and the stop prediction', f'#iteration-{n}-gripper-height-and-the-stop-prediction'),
            ('loss curves', f'#iteration-{n}-loss-curves'), ('3D replay', f'#iteration-{n}-3d-replay-{model}-dot-from-the-camera'),
            ('reading the results', f'#iteration-{n}-reading-the-results')]
ARM_ITERATIONS[24] = dict(iteration=24, name='arm-1khz-stop-layer-16-camera-dot', file='arm1k_e77_phys_stop16', results='arm1k_e77_phys_stop16_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=_stop_anchors(24, 'arm1k_e77_phys_stop16'),
    change='Iteration 23 with a 16-unit ReLU layer of its own for the stop head, after the second shared layer. Tested with the dot from the corner camera and tracker.',
    lesson='A layer of its own does not fix the stop at 77 epochs: the median stop is 0.78 s after PARK (0.81 s in iteration 23), and on 12 of 25 bearings the network says stop within 2 ms of PARK, where the head predicts a median 40 mm against about 225 mm. The reach is unchanged; the median closest is 115 mm because the runs stop sooner.')

ARM_ITERATIONS[26] = dict(iteration=26, name='arm-1khz-stop-layer-16-300-epochs-camera-dot', file='arm1k_e300_phys_stop16', results='arm1k_e300_phys_stop16_camera',
    hz=1000, target='sample', seed=7, epochs=300, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=_stop_anchors(26, 'arm1k_e300_phys_stop16'),
    change='Iteration 25 (the same network as iteration 24: stop head with its own 16-unit ReLU layer, stop loss weight 1) trained for 300 epochs instead of 77. Tested with the dot from the corner camera and tracker.',
    lesson='Training longer fixes the stop at PARK (the head predicts a median 218 mm there, against 40 mm in iteration 24) and lowers the losses (test torque 0.40, stop 0.0004), but makes the arm worse in the camera test: the gripper comes to about 125 mm from the dot by 0.8 s, then rises and drops hard, with 17% of torques at the limit; no bearing gets within 50 mm. 10 of 25 still stop at about 0.4 s. A better fit to the expert data does not carry over once the network drives the arm off the expert paths. Iteration 23 stays the best reach with the camera.')

ARM_ITERATIONS[27] = dict(iteration=27, name='arm-1khz-separate-stop-network-camera-dot', file='arm1k_e77_phys_stopnet', results='arm1k_e77_phys_stopnet_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=_stop_anchors(27, 'arm1k_e77_phys_stopnet')[:2] + [("the stop network's epochs", "#iteration-27-the-stop-networks-epochs")] + _stop_anchors(27, 'arm1k_e77_phys_stopnet')[2:],
    change="Iteration 23's torque network unchanged, with the stop from a separate feedforward network (17 -> 32 ReLU -> 16 ReLU -> 1, same inputs and target), trained on its own with its epochs chosen by 5-fold validation. Tested with the dot from the corner camera and tracker.",
    lesson="Separating the stop from the torques works: no bearing stops at PARK (11 in iteration 23), the median stop moves from 0.81 s to 1.19 s after PARK, and the arm moves exactly as in iteration 23 (median closest 54 mm, 57 mm at the expert stop time). The stop still comes a median 88 mm from the dot: the stop network reads the poses where iteration 23's arm stalls short of the dot as close to it. The limit is now the reach.")
ARM_ITERATIONS[28] = dict(iteration=28, name='arm-1khz-imu-and-encoders-camera-dot', file='arm1k_e77_phys_imu', results='arm1k_e77_phys_imu_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=23, dot_input='camera_tracker',
    anchors=_stop_anchors(28, 'arm1k_e77_phys_imu')[:2] + [('torque error per joint', '#iteration-28-torque-error-per-joint')] + _stop_anchors(28, 'arm1k_e77_phys_imu')[2:],
    change='Iteration 23 with an IMU on the end effector added to the arm encoders: 3 accelerations and 3 rotation rates in the hand frame every 1 ms, Bosch BMI088 noise, accelerations divided by 9.81 m/s^2 and rotation rates by 1 rad/s. 23 inputs. Tested with the dot from the corner camera and tracker.',
    lesson='The IMU improves the fit to the expert data (test torque loss 0.455 against 0.59) but not the arm in testing: 13 of 25 stop at PARK with the shared stop head, the median closest is 116 mm and 4.6% of torques are at the limit. The accelerometer reads the effect of the torque just before, so on expert data the network can read its target from it, while in testing its own errors come back as inputs.')
ARM_ITERATIONS[29] = dict(iteration=29, name='arm-1khz-imu-no-encoders-camera-dot', file='arm1k_e77_phys_imu_noenc', results='arm1k_e77_phys_imu_noenc_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=9, dot_input='camera_tracker',
    anchors=_stop_anchors(29, 'arm1k_e77_phys_imu_noenc')[:2] + [('torque error per joint', '#iteration-29-torque-error-per-joint')] + _stop_anchors(29, 'arm1k_e77_phys_imu_noenc')[2:],
    change='Iteration 23 with an IMU on the end effector instead of the arm encoders: the dot position and the 6 IMU readings, 9 inputs. Tested with the dot from the corner camera and tracker.',
    lesson='Without the arm encoders the fit to the expert data is still good (test torque loss 0.517), for the same reason as iteration 28, but in testing the arm is out of control: it falls a median 350 mm after its closest point, 23% of torques are at the limit and the gripper passes 1.5 m from the dot on 2 bearings. The median closest of 61 mm comes from falling through the dot. The IMU alone cannot replace the encoders.')

_A30 = [('Plan and expected results', '#iteration-30-plan-and-expected-results'), ('results', '#iteration-30-results'),
        ('by dot height', '#iteration-30-by-dot-height'), ('the expert data under each effect', '#iteration-30-the-expert-data-under-each-effect'),
        ('reading the results', '#iteration-30-reading-the-results')]
ARM_ITERATIONS[30] = dict(iteration=30, name='arm-1khz-iteration-23-on-new-dots-camera', file='arm1k_e77_dist_notime_phys', results='arm1k_e77_dist_notime_phys_camera_dots',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    expert_file='results/dot_cases_test_baseline.json',
    anchors=_A30 + [('3D replay', '#iteration-30-3d-replay-iteration-23-on-the-new-dots')],
    change="Iteration 23's network, unchanged, tested on the 100 new test dots (1.0 to 2.0 m away, 0.1 to 1.25 m high, all bearings), with the dot from the camera and the expert driving. All simulator effects off.",
    lesson="On new dots the median closest is 205 mm (57 mm on the 25 original bearings), 15 of 100 within 50 mm. It is 76 mm on dots 0.4 to 0.6 m high and 265 mm at other heights: the closest point falls in a V against the dot's height, lowest near 0.6 m, and does not depend on the dot's distance. The network reaches to about one height whatever the camera says. The shared stop head says stop in the first 0.5 s on 66 of 100 dots.")
ARM_ITERATIONS['30-stopnet'] = dict(iteration=30, name='arm-1khz-iteration-27-on-new-dots-camera', file='arm1k_e77_phys_stopnet', results='arm1k_e77_phys_stopnet_camera_dots',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    expert_file='results/dot_cases_test_baseline.json',
    anchors=_A30 + [('3D replay', '#iteration-30-3d-replay-iteration-27-on-the-new-dots')],
    change="Iteration 27's networks (iteration 23's torques and the separate stop network), unchanged, tested on the 100 new test dots, with the dot from the camera and the expert driving. All simulator effects off.",
    lesson='On new dots the median closest is 181 mm (54 mm on the 25 original bearings), 21 of 100 within 50 mm. It is 61 mm on dots 0.4 to 0.6 m high and 241 mm at other heights, the same V against height as iteration 23. The separate stop network never says stop in the first 0.5 s. The expert data under each effect: damping, elasticity and inertia leave the expert unchanged (993 to 997 of 1000 dots reached); noise (198), self-collision (73) and the 3 ms delay (0) break it.')

ARM_ITERATIONS[31] = dict(iteration=31, name='arm-1khz-trained-on-camera-dot', file='arm1k_e77_phys_camdot', results='arm1k_e77_phys_camdot_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=[('Plan and expected results', '#iterations-31-and-32-plan-and-expected-results')] + _stop_anchors(31, 'arm1k_e77_phys_camdot')[1:],
    change="Iteration 23 retrained with the dot input from the corner camera and tracker along the expert's recorded runs (camera_dot.py: the test's own render, dot finder and height-keeping tracker every 1/30 s, held between frames) instead of the simulator's exact dot. Everything else unchanged. Tested with the dot from the corner camera and tracker.",
    lesson="Training on the camera's dot does not help the arm: the fit is unchanged (test torque loss 0.583 against 0.59) but in testing the arm is worse on every one of the 25 bearings, by a median 8 mm: median closest 70 mm against 57 mm, 1 bearing within 50 mm against 9, 12 of 25 stop at PARK with the shared head. The camera's error is the same in training and testing (30 mm against 26 mm at PARK, bearing by bearing), so the match was real and still cost more than it gained. For the arm, the exact dot in training is the better choice.")
ARM_ITERATIONS[32] = dict(iteration=32, name='arm-1khz-separate-stop-network-trained-on-camera-dot', file='arm1k_e77_phys_camdot_stopnet', results='arm1k_e77_phys_camdot_stopnet_camera',
    hz=1000, target='sample', seed=7, epochs=77, bearings='21 training bearings', n_inputs=17, dot_input='camera_tracker',
    anchors=[('Plan and expected results', '#iterations-31-and-32-plan-and-expected-results')] + _stop_anchors(32, 'arm1k_e77_phys_camdot_stopnet')[1:][:2] + [("the stop network's epochs", "#iteration-32-the-stop-networks-epochs")] + [('Plan and expected results', '#iterations-31-and-32-plan-and-expected-results')] + _stop_anchors(32, 'arm1k_e77_phys_camdot_stopnet')[1:][2:],
    change="Iteration 27 on top of iteration 31: the separate stop network (17 -> 32 ReLU -> 16 ReLU -> 1) trained on the same camera dot as the torque network, epochs chosen by 5-fold validation. Tested with the dot from the corner camera and tracker.",
    lesson="The separate stop network trained on the camera's dot still never stops at PARK, but stops earlier (median 0.73 s after PARK against 1.19 s) and further from the dot (115 mm against 88 mm), on every bearing. Median closest 68 mm against 54 mm in iteration 27, no bearing within 50 mm. Its stop network fits the expert's data a little less well (test stop loss 0.00020 against 0.00014, 593 epochs chosen). Iteration 27, both networks trained on the exact dot, stays the best arm with the camera; the camera's 3 cm bias is better left to networks that never saw it than trained in.")


def add(iteration):
    import mlflow, pandas as pd
    from mlflow.entities import Metric
    spec = ARM_ITERATIONS[iteration]; iteration = spec.get('iteration', iteration); seed = spec.get('seed', 7); epochs = spec.get('epochs', 300)
    h = pd.DataFrame(json.loads((HERE / f"models/{spec['file']}_history.json").read_text()))
    res = spec.get('results', spec['file'])
    rows = json.loads((HERE / f"results/{res}.json").read_text())
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('log', 'frames')} for r in rows])
    df['stop_vs_expert_s'] = df.stop_s - df.expert_stop_s
    blown = [any(f['parked'] and f['ee_mm'] > 1500 for f in r['frames']) for r in rows]
    one = spec['file'].endswith('_b12')
    stop = 'time_to_stop' if 'train_time_to_stop_mse' in h else 'distance'      # what the stop head learns
    m = {'bearings_tested': len(df), 'median_closest_mm': float(df.min_mm.median()), 'best_closest_mm': float(df.min_mm.min()),
         'closest_within_1cm': int((df.min_mm <= 10).sum()), 'median_at_expert_stop_mm': float(df.at_expert_stop_mm.median()),
         'within_1cm_at_expert_stop': int((df.at_expert_stop_mm <= 10).sum()), 'stopped': int(df.stop_s.notna().sum()),
         'gripper_past_1_5m': int(sum(blown)), 'final_train_torque_mse': float(h.train_torque_mse.dropna().iloc[-1]),
         f'final_train_{stop}_mse': float(h[f'train_{stop}_mse'].dropna().iloc[-1])}
    if 'at_network_stop_mm' in df and df.at_network_stop_mm.notna().any():
        m['median_at_network_stop_mm'] = float(df.at_network_stop_mm.median())
    if df.stop_vs_expert_s.notna().any():
        m['median_stop_vs_expert_s'] = float(df.stop_vs_expert_s.median())
    if not one:
        m.update(final_test_torque_mse=float(h.test_torque_mse.dropna().iloc[-1]), **{f'final_test_{stop}_mse': float(h[f'test_{stop}_mse'].dropna().iloc[-1])},
                 best_test_torque_mse=float(h.test_torque_mse.min()), best_test_torque_epoch=int(h.epoch[h.test_torque_mse.idxmin()]))
    w = np.load(HERE / f"models/{spec['file']}.npz")
    n_params = int(sum(w[k].size for k in w.files if k.endswith(('.weight', '.bias'))))
    time_on = bool(w['info_time']) if 'info_time' in w.files else w['shared.0.weight'].shape[1] == 19
    imu_on = bool(w['info_imu']) if 'info_imu' in w.files else False; enc_on = bool(w['info_encoders']) if 'info_encoders' in w.files else True
    full = INPUTS[:14] + INPUTS[14:17] + (['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_gyro_x', 'imu_gyro_y', 'imu_gyro_z'] if imu_on else []) + INPUTS[17:]
    names = [n for n in full if (enc_on or not n.startswith('arm_')) and (time_on or not n.startswith('time_'))]
    stopnet = 'stopnet.0.weight' in w.files
    stopnet_desc = (f", with the stop from a separate network ({w['info_stopnet']}, {int(w['info_stopnet_epochs'])} epochs chosen by 5-fold validation)"
                    if stopnet else '')
    scaling = str(w['info_scaling']) if 'info_scaling' in w.files else 'standardised'; raw = scaling == 'none'
    sh = int(w['info_stop_hidden']) if 'info_stop_hidden' in w.files else 0; sw = float(w['info_stop_weight']) if 'info_stop_weight' in w.files else 1.0
    params = dict(version='v2', iteration=iteration, model=spec['name'], part='arm', inputs=len(names), input_names=','.join(names),
                  outputs=f'7_arm_torques_Nm+{stop}', stop_target=stop, stop_threshold=float(w['ttp_thresh']), control_hz=spec['hz'], torque_target=spec['target'],
                  hidden='2x64_relu' + (f'+stop_branch_{sh}_relu' if sh else '') + ('+separate_stop_net_32_16_relu' if stopnet else ''), imu=int(imu_on), arm_encoders=int(enc_on), stop_hidden=sh, stop_loss_weight=sw, parameters=n_params, optimizer='adam', learning_rate=0.01, epochs=epochs, batch='full', seed=seed,
                  loss=f"torque_mse_{'Nm2' if raw else 'standardised'}+{stop}_mse_scaled_weight{sw:g}", trajectory_weighting='equal',
                  input_scaling={'none': 'none', 'physical_inputs': 'physical_angles_pi_speeds_limit_dot_1m'}.get(scaling, 'all_standardised'), output_scaling='none' if raw else 'torques_standardised', train_bearings=spec['bearings'],
                  arm_sensor='encoders_kinova_ka75_3686400_ka58_2534400_counts_per_turn', dot_input=spec.get('dot_input', 'simulator_robot_frame'),
                  driving='expert_drive', torque_clip_Nm=30, test_bearings=len(df), test_engine='sensor_parking_arm_mode')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    link = NOTEBOOK_MD.format(commit=commit)
    rate = '30 Hz (each torque held 33 ms)' if spec['hz'] == 30 else '1 kHz (a new torque every 1 ms)'
    desc = '\n'.join([
        f"**Iteration {iteration}: `{spec['file']}` (arm)**", '', f"**What changed.** {spec['change']}", '',
        f"**Inputs.** {len(names)}: " + ('arm encoder angles and speeds (7 each, Kinova Jaco 2 actuator resolutions), ' if enc_on else 'no arm encoders, ') + "the dot in the robot frame (x, y, z, " + ('from the corner camera and tracker, image every 1/30 s' if spec.get('dot_input') == 'camera_tracker' else 'from the simulator for this test') + ')' + (', an IMU on the end effector (3 accelerations and 3 rotation rates in the hand frame, every 1 ms, Bosch BMI088 noise)' if imu_on else '') + (', time since PARK and time since the start' if any(n.startswith('time_') for n in names) else '') + '.', '',
        f"**Outputs.** The 7 arm torques (N m, clipped to ±30) at {rate}, and " + ("time to stop in seconds (clipped at 3 s); stop when it is 0.25 s or less." if stop == 'time_to_stop' else "the distance from the gripper to the dot (clipped at 0.25 m); stop when it is 1 cm or less."), '',
        f"**Training.** 2×64 ReLU network{f', with a {sh}-unit ReLU layer of its own for the stop head' if sh else ''}{stopnet_desc}, {n_params:,} parameters. Loss: torque MSE in {'N m² (no scaling of inputs or torques)' if raw else 'standardised units'}{' (inputs divided by physical sizes: angles by π, speeds by the speed limits, the dot by 1 m)' if scaling == 'physical_inputs' else ''} plus {stop.replace('_', '-')} MSE (weight {sw:g}), each training run counted equally. Adam, lr 0.01, {epochs} full-batch epochs, seed {seed}, trained on {spec['bearings']}. Data: the expert's full runs recorded every 1 ms, from PARK to its stop.", '',
        ('**Test.** In the simulator on the 100 new test dots (1.0 to 2.0 m away, 0.1 to 1.25 m high, all bearings; seed 0, collection/native/effects/targets_1000.json), all simulator effects off. The expert drives and parks, then only the arm network moves the arm.' if spec.get('expert_file') else '**Test.** In the simulator on 25 bearings. The expert drives and parks, then only the arm network moves the arm.'), '',
        f"- Closest the gripper gets to the dot: median {m['median_closest_mm']:.0f} mm, best {m['best_closest_mm']:.0f} mm; within 1 cm on {m['closest_within_1cm']} of {len(df)}.",
        f"- At the expert's stop time: median {m['median_at_expert_stop_mm']:.0f} mm from the dot. The gripper went more than 1.5 m from the dot on {m['gripper_past_1_5m']} of {len(df)}.",
        (f"- When the network said stop, the gripper was a median {m['median_at_network_stop_mm']:.0f} mm from the dot." if 'median_at_network_stop_mm' in m else ''),
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
        keys = ['train_torque_mse', f'train_{stop}_mse'] + ([] if one else ['test_torque_mse', f'test_{stop}_mse'])
        batch = [Metric(k, float(v), t, int(e)) for k in keys for e, v in zip(h.epoch, h[k]) if v == v]     # skip epochs a part was not trained (NaN)
        for i in range(0, len(batch), 1000):
            client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
        mlflow.log_artifact(str(HERE / f"models/{spec['file']}.npz"), 'model')
        mlflow.log_artifact(str(HERE / f"models/{spec['file']}_history.json"), 'model')
        mlflow.log_table(df, 'tests/per_bearing.json')
        mlflow.log_artifact(str(HERE / f"results/viewer_{res}.html"), 'player')
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


def cv25():
    """Iteration 25: the stop loss weight chosen by 5-fold validation on the training bearings (no new simulator test)."""
    import mlflow, pandas as pd
    from mlflow.entities import Metric
    choice = json.loads((HERE / 'results/stop_weight_choice_25.json').read_text()); W = [1, 3, 10, 30, 100]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(); link = NOTEBOOK_MD.format(commit=commit)
    anchors = [('Plan and expected results', '#iteration-25-plan-and-expected-results'), ('results', '#iteration-25-results'),
               ('validation loss curves', '#iteration-25-validation-loss-curves'), ('reading the results', '#iteration-25-reading-the-results')]
    rows = [f"- weight {w}: validation torque loss {choice['weights'][str(w)]['val_torque']:.3f} ({choice['weights'][str(w)]['val_torque'] / choice['weights']['1']['val_torque']:.2f} of weight 1), "
            f"validation stop loss {choice['weights'][str(w)]['val_stop']:.4f}, held-out runs stopping in the first 0.5 s {choice['weights'][str(w)]['early_stops']} of 21" for w in W]
    desc = '\n'.join(['**Iteration 25: choosing the stop loss weight (5-fold validation)**', '',
        'Iteration 24 (stop head with its own 16-unit ReLU layer, inputs divided by physical sizes, 77 epochs, seed 7) trained with stop loss weights 1, 3, 10, 30 and 100, '
        'each 5 times with one fold of the training bearings held out. The test bearings were not used.', '',
        f"Rule, set before training: {choice['rule']}.", ''] + rows + ['',
        f"Chosen: weight {choice['chosen']}. Every larger weight raises the torque loss by more than 10%, so the network is the same as iteration 24's and is not tested again. "
        'Larger weights make the head stop at PARK on fewer held-out runs (10 of 21 at weight 1, 4 of 21 at 30), at a large cost in torque accuracy.', '',
        '**Notebook.** ' + ', '.join(f'[{t}]({link}{a})' for t, a in anchors) + '.'])
    tags = {'mlflow.user': 'nat', 'mlflow.source.type': 'LOCAL', 'mlflow.source.name': 'v2_sensors/driving_v2/driving_v2_sensors.ipynb',
            'mlflow.source.git.repoURL': 'https://github.com/NatMota/mlai-vnc', 'mlflow.source.git.branch': 'v2-sensor',
            'mlflow.source.git.commit': commit, 'mlflow.note.content': desc, 'version': 'v2', 'iteration': '25', 'model': 'arm-1khz-stop-weight-choice', 'part': 'arm', 'author': 'nat'}
    mlflow.set_experiment(EXPERIMENT); client = mlflow.MlflowClient()
    with mlflow.start_run(run_name='v2-iter25-arm-1khz-stop-weight-choice', tags=tags) as run:
        mlflow.log_params(dict(version='v2', iteration=25, model='arm-1khz-stop-weight-choice', part='arm', method='5_fold_validation_on_training_bearings',
                               folds=5, epochs=77, control_hz=1000, seed=7, stop_hidden=16, input_scaling='physical_angles_pi_speeds_limit_dot_1m',
                               candidate_weights='1,3,10,30,100', rule=choice['rule'], chosen_stop_loss_weight=choice['chosen']))
        mlflow.log_metrics({f'w{w}_{k}': float(choice['weights'][str(w)][k]) for w in W for k in ('val_torque', 'val_stop', 'early_stops')})
        t = int(run.info.start_time); batch = []
        for w in W:
            H = [pd.DataFrame(json.loads((HERE / f'models/arm25_w{w}_fold{f}_history.json').read_text())) for f in range(5)]
            for k in ('val_torque_mse', 'val_distance_mse'):
                batch += [Metric(f'w{w}_mean_{k}', float(v), t, e) for e, v in enumerate(np.mean([h[k] for h in H], 0))]
        for i in range(0, len(batch), 1000):
            client.log_batch(run.info.run_id, metrics=batch[i:i + 1000])
        for w in W:
            for f in range(5):
                mlflow.log_artifact(str(HERE / f'models/arm25_w{w}_fold{f}_history.json'), 'folds')
        mlflow.log_artifact(str(HERE / 'results/stop_weight_choice_25.json'), 'folds')
        mlflow.log_artifact(str(NOTEBOOK), 'notebook')
        if len(sys.argv) > 2:
            mlflow.log_artifact(sys.argv[2], 'notebook')
    print('logged', run.info.run_name, run.info.run_id, 'commit', commit[:7])


if __name__ == '__main__':
    key = sys.argv[1]
    cv18() if key == 'cv18' else cv25() if key == 'cv25' else add(int(key) if key.isdigit() else key)
