# How to use this folder

This folder trains a machine learning model to copy the expert controller.
The model drives the robot base and moves the arm, so the end-effector reaches the red dot.
It does this on its own, with no expert helping.


## Before you start

Every command is run from this folder, with the ml environment switched on.
Start each command with:

```
pixi run --manifest-path /home/xxxxxx/RoboEnv/pixi.toml -e ml
```

For example:

```
pixi run --manifest-path /home/xxxxxx/RoboEnv/pixi.toml -e ml python3 test.py
```


## The files

Teacher's files (do not change):

```
README.md                 the lecture's instructions
robotnik_arm_control.py   the expert controller
```

Collecting data:

```
robotnik_arm_task.py        the expert drives to a random dot and saves the run (data_normal)
robotnik_arm_task_dart.py   the same, with random pushes (data_dart)
collect.py                  runs one of the two above many times
```

The model:

```
features.py   the SWITCHES, and the extra inputs. You do not run this file.
add_ee.py     works out where the end-effector was in every saved run. Run once.
train.py      trains the model
drive.py      lets the model drive to ONE test dot
test.py       lets the model drive to many test dots, and prints the scores
```

Folders:

```
data_normal     runs without pushes
data_dart       runs with pushes, plus the trained models and split.json
data_dart_ee    end-effector positions, made by add_ee.py
```


## Steps, in order

Step 1. Collect data (only if you need more):

```
python3 collect.py dart 300
```

Step 2. Work out the end-effector positions (once, after collecting):

```
python3 add_ee.py data_dart
```

Check the last line. The median should be about 1 cm.

Step 3. Train:

```
python3 train.py
```

Step 4. Test on 20 dots it has never seen:

```
python3 test.py
```

Step 5. Watch one test dot, with the window open:

```
python3 drive.py data_dart 0
```

The last number picks the test dot: 0, 1, 2, and so on.


## The switches

The switches are at the top of features.py. True = ON, False = OFF.
After changing a switch, run train.py again, then test.py.

```
USE_EE           ON:  the model also sees the gap from the end-effector to the dot.
                 OFF: it does not.

ONLY_GOOD_RUNS   ON:  learn only from runs where the expert reached the dot.
                 OFF: learn from all runs, including the failed ones.
```

Each switch setting saves its own model, so nothing is overwritten:

```
USE_EE  ONLY_GOOD_RUNS   saved model file
OFF     OFF              data_dart/phase_rel_best.pt
OFF     ON               data_dart/phase_rel_good_best.pt
ON      OFF              data_dart/ee_best.pt
ON      ON               data_dart/ee_good_best.pt
```


## Results so far (same 20 test dots)

```
Model                                     Within 1 cm   Average closest
Two phases + "look at the ball" inputs     4 of 20        2.9 cm
+ only good runs                          10 of 20        2.0 cm
+ end-effector inputs                     12 of 20        1.3 cm
```


## Input and output format

The model is two networks. Both are in the checkpoint file (<name>_best.pt).

control_net gets 35 numbers and gives 11 commands.

```
Input   What it is                                          Unit
1-3     base position x, y, z                               m
4       base bearing (which way it faces)                   rad
5-11    arm joint angles (q_mes[4:11])                      rad
12-18   arm joint speeds (qd_mes[4:11])                     rad/s
19-21   dot position x, y, z                                m
22      dot in front of the robot (negative = behind)       m
23      dot to the left of the robot (negative = right)     m
24      dot distance on the floor                           m
25-26   sin and cos of the turn needed to face the dot      -
27-28   sin and cos of the bearing                          -
29-31   dot minus end-effector, x, y, z                     m
32-33   that gap in front of / to the left of the robot     m
34      that gap as a straight-line distance                m
35      phase: 0 = driving, 1 = reaching                    -
```

Inputs 22 to 34 are made by features.py. Inputs 29 to 34 are only used when USE_EE is ON.

```
Output  What it is                                                       Unit
1-4     wheel speeds: front right, front left, back left, back right     rad/s
5-11    arm joint torques, joints 1 to 7                                 N m
```

phase_net gets inputs 1-4 and 19-28 (14 numbers) and gives 1 number. Above 0 means "reach".

Before the networks, inputs 1-34 are standardised with X_mean and X_std from scaler_<name>.npz.
The outputs come out standardised. drive.py turns them back with Y_mean and Y_std,
then limits each command to the smallest and biggest value the expert used in the training runs.
The robot switches to reaching when phase_net says "reach" for 50 steps in a row, and never switches back.
While reaching, the wheel commands are set to 0.


## What the checkpoint file holds

```
phase_net, control_net   the trained weights
optimiser, scheduler     to carry on training (saved before that epoch's scheduler step)
epoch, val_mse           which epoch was best, and its validation MSE
settings                 seed, data folder, learning rate, batch size, epochs, switches
```