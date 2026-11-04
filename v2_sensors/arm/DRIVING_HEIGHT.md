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
