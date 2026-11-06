# COMP0245 assignment: behaviour cloning of a mobile robot arm

This folder trains machine learning models to copy the expert controller. The models drive the base and move the
arm tip to within 1 cm of a target in the PyBullet simulator, with all commands coming from the networks.

There are three versions:

```
Saved model       driving, plan and torque networks, with analytical calculations for some inputs
Learned version   the saved model with those calculations replaced by parking, target, tip and angle gap networks
Sensor-only       a harder extension with no true positions, using a wrist camera and encoders
```

`ITERATIONS.md` lists every iteration in order, with its result and the switches that step into it.

The submission has two folders:

```
robotnik_arm_task/   the saved model and the learned version: code, saved seed 1 models and report_targets.json
sensor_only/         our files for the sensor-only extension, which run inside the team's repository
```


## Setup

Copy the `models` and `configs` folders from RoboEnv into `robotnik_arm_task`.
Run every command from `robotnik_arm_task`, inside the RoboEnv environment. On our machine, each command starts with:

```
pixi run --manifest-path /home/vincent/RoboEnv/pixi.toml -e ml
```

The commands below leave this part out, to keep them short.

The IMU inputs are OFF by default in `drive_net.py`, as in our reported models. `ROBOENV_USE_IMU=1` turns them ON.


## Quick check, with no data

The seed 1 models are included in this submission, so you can skip steps 1 and 2 and test straight away.
This runs one new target on the saved model:

```
ROBOENV_SEED=1 python3 test_drive_net.py 1 new
```

We ran this and the learned version's command from step 3, each on one target, in a clean copy holding only the code,
the seed 1 models, and the `models` and `configs` folders. Both reached their target.
In the same kind of copy, with `report_targets.json` added, we ran the saved model on two targets from each of `val`, `test`,
`golden_val` and `golden_test`, with no data folders.


## What each part needs

```
All robot tests                      the code, saved models and report_targets.json
train_drive_net.py                   data_v2/balanced, data_v2/random_start, data_dart and data_dart_ee
train_arm_plan.py, train_arm_torque.py, train_tip_net.py, train_gap_net.py    data_v2/balanced
train_input_nets.py                  no data
make_targets_file.py                 data_dart (2.9 GB) and the golden dataset (836 MB)
```

The data folders are too large to submit. They are in one zip file (7.2 GB, UCL login needed):
https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabcy7_ucl_ac_uk/IQDmVRMPgJ3jQLptnikBDr7pAdfgebgWc3X80FUjRUIn2Qc?e=00NfCi


## Step 1: collect the data

```
python3 collect.py balanced 1500 good 4
python3 collect.py randomstart 500 good 4
```

This saves one file per run in `data_v2/balanced` and `data_v2/random_start` (about 4.9 GB, and it takes 20 minutes or more).
The random targets in `collect.py` have no fixed seed, so new data will not match our runs exactly.

`data_dart` is an older dataset of 800 runs with big pushes. It cannot be remade exactly, so use our copy.
`train_drive_net.py` uses its validation runs to pick the best epoch, and also needs their tip positions.
This makes them in `data_dart_ee` (15 to 40 minutes):

```
python3 add_ee.py data_dart
```

`report_targets.json` holds the positions of the 440 `val`, `test`, `golden_val` and `golden_test` targets.
`make_targets_file.py` made it from `data_dart` and the team's golden dataset, so you do not need to run it again.
That script expects the golden dataset at `~/COMP0245_Golden_300`. You can change this with `GOLDEN_DATA` in the script.


## Step 2: train the models

The saved model, for seed 1 (use `ROBOENV_SEED=0` for seed 0):

```
ROBOENV_SEED=1 python3 train_drive_net.py
ROBOENV_SEED=1 python3 train_arm_plan.py
ROBOENV_SEED=1 python3 train_arm_torque.py
```

The extra networks for the learned version:

```
python3 train_input_nets.py park
python3 train_input_nets.py target
python3 train_tip_net.py
ROBOENV_SEED=1 python3 train_gap_net.py
```

Each model is saved in its own folder: `drive_net_model`, `arm_plan_model`, `arm_torque_model`, `input_net_model` and `arm_gap_model`.
Our `park` and `target` networks used the default seed 0, and the tip network uses seed 1, set in its file.


## Step 3: test on the robot

The saved model on the 200 new targets:

```
ROBOENV_SEED=1 python3 test_drive_net.py 200 new
```

The learned version on the same targets:

```
ROBOENV_SEED=1 ROBOENV_PARK_NET=1 ROBOENV_TIP_NET=1 ROBOENV_TARGET_NET=1 ROBOENV_GAP_NET=1 python3 test_drive_net.py 200 new
```

The word after the number picks the targets: `val`, `test`, `new`, `golden_val` or `golden_test`.
Each run prints one `RESULT` line per target and a summary, and saves them in
`results/three_nets<switches>_seed<seed>/<group>_<date>_<time>/summary.txt`.

These switches give the other experiments in the report. Type them before the command:

```
ROBOENV_NOISE=1 or 5                  random sensor errors
ROBOENV_BIAS=1 or 5                   constant sensor errors
ROBOENV_BIAS_ONLY=base                constant error on one sensor: base, heading, angle or speed
ROBOENV_RANDOM_START=1                start near the parking spot, facing a random direction
ROBOENV_PUSH_LEVEL=0.5 or 1           random physical pushes
ROBOENV_RUNS_DIVIDE=4 or 16           networks trained on less data (train them with the same switch first)
ROBOENV_GAP_WINDOW=1                  angle gap network that reads the last 100 ms (train with the same switch first)
ROBOENV_GAP_ROBUST=1                  angle gap network trained with speed errors (train with ROBOENV_SPEED_NOISE=0.05)
ROBOENV_NO_ADDING=1                   one network for the plan and angle gaps, instead of ROBOENV_GAP_NET (train_arm_no_maths.py a)
ROBOENV_DIRECT_ARM=1                  one network straight to torques, instead of ROBOENV_GAP_NET (train_arm_no_maths.py b)
```

`train_arm_no_maths.py` also saves a copy to `~/mlai-vnc-golden/v2_sensors/arm/models`.
If you do not have that folder, change `TEST_MODELS` at the top of the file first.

To count parameters, file sizes and computation time. The times change a little between runs:

```
ROBOENV_SEED=1 python3 count_nets.py
```


## Sensor-only extension

This code runs in the `v2_sensors/arm` folder of the team's repository, https://github.com/NatMota/mlai-vnc,
on the `v2-sensor` branch at commit 36d1597 (27e96dd before the history was rewritten on 6 November). Copy everything in our `sensor_only` folder, with its `models`
and `results` folders, into `v2_sensors/arm`. The seed 7 driving network is already in that repository.
It also needs the golden dataset and the native simulator, built from the `collection/native` folder of that repository:

```
pixi run cmake --build build-clang19 --target sensor_parking policy_replay
```

Then, from `v2_sensors/arm`, replacing `NATIVE` with the full path to your `collection/native` folder:

```
export CAMERA_RENDERER=tiny OMP_NUM_THREADS=1 NATIVE_ROOT=NATIVE
MY_ARM_MEMORY=1 MY_ARM_MEMORY_NETS=1 MY_ARM_STOP_FRAMES=15 MY_ARM_TIP_NET=1 MY_ARM_WHEEL_NET=1 \
  pixi run --manifest-path NATIVE/pixi.toml \
  python my_arm.py one TAG --data ~/COMP0245_Golden_300 --split test --jobs 4
```

Results are saved as `results/my_arm_TAG.json`. Leave out `--split test` to use the validation targets.
We have only run these sensor-only commands on our own machine.

The stop network can be retrained from the recorded runs we include. This remakes `models/ball_stop_net_final.npz` exactly:

```
python train_stop_net.py results/my_arm_record_train50b_nets_frames.npy
```

`MY_ARM_STOP_SPEED=0` gives the version without joint speeds from the report, which was not used.


## External code and what we changed

External code, not written by us:

```
simulation_and_control, from RoboEnv     the simulator interface
PyBullet, Pinocchio                      physics and the robot model
PyTorch, NumPy, Matplotlib               networks, maths and figures
models/, configs/                        the robot files, from RoboEnv
```

From the teacher's template:

```
robotnik_arm_control.py      the expert controller. Not changed.
robotnik_arm_task.py         changed: random targets and saving each run's data
robotnik_arm_task_dart.py    our copy of the template, with DART pushes, even target spread and random starts added
```

Written by us, in this folder:

```
collect.py, add_ee.py, make_targets_file.py               data collection and the targets file
features.py, train.py, drive.py, test.py                  the earlier single model
drive_net.py, arm_plan_net.py, arm_torque_net.py          the three networks
train_drive_net.py, train_arm_plan.py, train_arm_torque.py    training for them
input_nets.py, gap_net.py                                 the learned version's networks
train_input_nets.py, train_tip_net.py, train_gap_net.py   training for them
train_arm_no_maths.py, dagger.py, dagger_labels.py        experiments that are reported as not working
test_drive_net.py                                         all robot tests
count_nets.py                                             efficiency numbers
check_rare.py                                             how evenly the targets are spread by distance
```

In the sensor-only extension, the driving network, camera, sensors and golden dataset are the team's code in `v2_sensors`.
`my_arm.py`, `train_ball_stop.py`, `train_stop_net.py`, `train_tip_net.py` and `train_wheel_net.py` were added,
and `full_run.py` was changed so it can run them.
