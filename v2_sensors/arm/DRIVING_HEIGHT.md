This development experiment adds camera-tracked robot-frame target height in metres as input 17. Inputs 1–16, both hidden layers, wheel-speed and time-to-PARK losses, thresholds, 300 training epochs and seed 7 are retained. The reaching checkpoint is copied unchanged from the frozen golden baseline. There is no fixed search rule.

The 100 training and 100 validation recordings already contain causal tracked height in the `arm` camera array. Before the first sighting, height is zero; when temporarily hidden, its last estimate is retained. Live evaluation uses the same height-keeping tracker. No simulator target height becomes a policy input.

Run from the repository root. Choose a new output directory. Use the existing CUDA training environment for `prepare` and `train`, and a NumPy/PyBullet environment for `evaluate`.

```bash
python v2_sensors/arm/driving_height.py prepare --data /path/to/golden --baseline /path/to/frozen/models --out /path/to/new/experiment
python v2_sensors/arm/driving_height.py train --data /path/to/golden --baseline /path/to/frozen/models --out /path/to/new/experiment --device cuda
CAMERA_RENDERER=egl NATIVE_ROOT=/path/to/native CAMERA_URDF=/path/to/native/assets/summit_xl_arm_description/summit_xl_complete.urdf python v2_sensors/arm/driving_height.py evaluate --data /path/to/golden --baseline /path/to/frozen/models --out /path/to/new/experiment --jobs 3
```

`prepare` chooses one initially visible validation target per low/middle/high height and near/far distance stratum, using proximity to each stratum's centre. This freezes six IDs before training without inspecting model outcomes: 673, 506, 830, 720, 16, 333. Evaluation runs both baseline and height models from the normal start through reaching. Full rendering and 30 Hz driving/camera plus 1 kHz torque/physics are preserved. Recorded numerical trajectories include driving inputs and time-to-PARK predictions; no camera videos are generated.

Checks: `python v2_sensors/arm/check_driving_height.py` and `python v2_sensors/arm/check_train_golden.py`. Training verifies all dataset hashes and NumPy/PyTorch export agreement. Reaching and original feature scaling are checked unchanged. Baseline target 506 exactly reproduces its earlier validation PARK/STOP times, parking error, closest reach and distance at STOP.

Saved experiment: `results/driving_height17/`, including configuration, NumPy checkpoints, histories, per-case trajectories and summaries. The resumable PyTorch `.pt` checkpoint is retained on Mint and excluded from Git by the existing repository policy. The golden notebook appends the analysis; earlier cells and the original historical notebook are preserved. This validation screen creates no official test iteration or MLflow run.

On these six cases, median parking error fell from 36.0 cm to 23.3 cm, improving four targets. Median closest reach changed from 438.7 mm to 428.8 mm, but median distance at STOP worsened from 459.1 mm to 552.9 mm. Both variants issued PARK and STOP on all six, with zero successes within 10 mm. Training including loading took 14.1 seconds; the twelve simulations took 43.7 seconds. These timings exclude setup, checks and reporting.

This is partial improvement in parking, not a stable end-to-end result. The largest gains were on middle-height targets; the two far low/high targets worsened. One seed and six initially visible validation targets do not establish generalisation. Test results were not used to choose these cases or this checkpoint, and no reserved test trajectory was opened during this experiment.

## Follow-up: remove wheel-speed inputs while retaining height

Use the previous height model directory as `--baseline`, a new `--out`, and add `--without-wheel-velocity` to the `prepare` command. The same `train` and `evaluate` commands then run the comparison. The 11 retained inputs are four accumulated wheel angles, mean angle and right-minus-left angle, camera range, bearing sine/cosine, visibility and height. Encoder speeds still feed the tracker. Hidden layers, losses, seed, training schedule and reaching weights stay unchanged.

Results are in `results/driving_height_no_speed11/`. On the same six validation cases, all six parking errors improved; the median fell from 23.3 cm to 4.4 cm. Median closest gripper distance fell from 428.8 mm to 142.8 mm, and median distance at STOP fell from 552.9 mm to 242.1 mm. Both models issued PARK and STOP in all six cases; neither succeeded within 10 mm. Two new-model runs approached within 31 mm but did not stop there. One later stopped more than a metre away, so reaching and STOP remain unstable.

Training including loading took 11.9 seconds, and the 12 full simulations took 45.9 seconds. The baseline reproduced all six previous height-model PARK/STOP times, parking errors and reaching metrics exactly. Retained feature scaling and reaching weights were verified unchanged. This supports omitting the speed inputs on this small validation screen; it does not establish the causal mechanism or generalisation across seeds and targets. Only the new notebook analysis cells were executed, preserving earlier outputs and avoiding unnecessary replay rebuilding.
