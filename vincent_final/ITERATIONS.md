# Vincent's iterations, in order

Each iteration changed one thing (or says what else changed) and was tested on the robot. The score is the number of targets
reached: within 1 cm for true position (T), and within 10 mm at STOP for sensor-only (S). "val" and "test" are the 120
`data_dart` targets, "golden" the team's 100 golden targets, "new" 200 fresh random targets, "golden50" the first 50 golden
validation targets. Seed 1 unless stated.

The code was not saved as one commit per iteration. Instead, one version of the code holds every iteration as a switch,
so you step into an iteration by typing its switches, and step out by leaving them off. Older models that a later iteration
replaced are in `COMP0245_extras.zip`. Every result file named here is in that zip too.


## How to step in

True position, from `robotnik_arm_task`:

```
Earlier single model (T1 to T13)   set the switches in features.py (see the name table in REPRODUCE_EXTRA.md),
                                   then python3 train.py and python3 test.py 20 (test targets) or python3 test.py 20 val
Three networks (T14 onwards)       ROBOENV_SEED=1 <switches> python3 test_drive_net.py <N> <val|test|golden_test|golden_val|new>
```

Sensor-only, from `v2_sensors/arm` (see the main README for the full command):

```
<switches> python my_arm.py one <tag> --data ~/COMP0245_Golden_300 [--targets <list>] --jobs 4
<37> = the 37 targets in sensor_only/target_lists/val37_targets.txt, <10> = the 10 in quick_targets.txt
```


## T. True position

### Earlier single model: one decision network and one control network for all 11 commands

```
#    Date   What changed                                         Score                              Kept?
T1   27 Oct tip (end-effector) inputs, one control network       9/20 test (seed 0)                 base
T2   27 Oct separate wheel and arm heads (Nat's idea)            12/20 test, 11/20 seed 0           yes
T3   27 Oct start and stop decisions                             0/20, 5/20 test                    no
T4   28 Oct park label starts 200 ms early                       13/20 val (8/20 without), 13/20 test yes
T5   28 Oct DAgger, four attempts                                1, 1, 0, 3 of 20 test              no
T6   28 Oct add data_dart2 (medium pushes) to data_dart          17/20 test, 16/20 seed 0, 106/120  yes
T7   1 Nov  fewer inputs: 20, 23, 17, 6                          10, 16, 0, 0 of 20 test            23 kept
            (the 6 input model also had an action mask and every 10th driving row)
T8   2 Nov  decide 30 or 100 times per second, not every step   0 to 7/20, 11/20 val               no
T9   2 Nov  close targets repeated 5 times                       16/20 val, 15/20 seed 1            no
T10  2 Nov  balanced data only (data_v2/balanced)                17/20 val                          yes
T11  2 Nov  add random-start data                                13/20 val                          no
T12  2 Nov  parking inputs (parking spot and turn still needed)  18/20 val, 100/120 test            final single model
T13  2 Nov  drop one more input (22)                             0/20 val                           no
```

To step into T12, use the default `features.py`. For the others, change the switches named in each model's folder name.

### Three networks: driving, plan and torque

```
#    Date   What changed                                Score                                   Switches or files
T14  2 Nov  driving network, expert arm                 20/20 val (seed 0)                      USE_PLAN_NET = False, USE_TORQUE_NET = False
T15  2 Nov  plus plan network, expert torque            19/20 val                               USE_TORQUE_NET = False
T16  2 Nov  plus torque network (30 plan epochs)        113/120 val (seed 0)                    arm_plan_seed0_30epochs.pt
T17  2 Nov  80 plan epochs                              117/120 seed 0, 112/120 seed 1          arm_plan_seed<s>_80epochs.pt
T18  2 Nov  near-target weight 5 in the plan loss       117/120 both seeds                      drive_net_seed<s>_balanced.pt
T19  2 Nov  random-start runs added to driving          116 and 117/120 val, 118 and 117/120 test   default: the saved model
T20  3 Nov  2 IMU inputs for driving                    120/120 seed 0, 114/120 seed 1 val      ROBOENV_USE_IMU=1 (not kept)
            new targets: 198/200 (IMU, seed 0) against 195/200 (saved model, seed 1)
```

T14 and T15 are constants at the top of `test_drive_net.py`. For T16 to T18, copy the named backup model over the normal
file name (for example `arm_plan_seed0.pt`), run the test, then copy the normal model back.

### Learned version: replacing the analytical calculations

```
#    Date   What changed                                   Score                                   Switches
T21  5 Nov  tip network                                    18/20 val                               ROBOENV_TIP_NET=1
T22  5 Nov  parking network                                20/20 val                               ROBOENV_PARK_NET=1
T23  5 Nov  tip, parking and target networks               17/20, 112/120 val, 95 and 97 golden    + ROBOENV_TARGET_NET=1
T24  5 Nov  one network for the plan and angle gaps        90/120 val, 83 and 66 golden            + ROBOENV_NO_ADDING=1 (not kept)
T25  5 Nov  one network straight to torques                14/20 val                               + ROBOENV_DIRECT_ARM=1 (not kept)
T26  5 Nov  third arm network for the angle gap            17/20, 106/120 val, 94 and 94 golden,   + ROBOENV_GAP_NET=1
            (the learned version)                          185/200 new
```

### Tests on the saved model and the learned version (no new design)

```
#    Test                                    Saved model    Learned version   Switches (added to each version's)
T27  golden, both seeds                      100 and 100    94 and 94         python3 test_drive_net.py 100 golden_test
T28  golden50, no error                      49             43                python3 test_drive_net.py 50 golden_val
T29  random sensor errors, level 1 and 5     50, 50         43, 44            ROBOENV_NOISE=1 or 5
T30  constant sensor errors, level 1 and 5   49, 32         16, 1             ROBOENV_BIAS=1 or 5
T31  constant error on one sensor (level 1)  -              base 43, heading 43, angle 44, speed 15   ROBOENV_BIAS=1 ROBOENV_BIAS_ONLY=<sensor>
T32  angle gap trained with speed errors     -              45, then 16 and 1 constant, 47 random L5  ROBOENV_GAP_ROBUST=1
T33  angle gap reads the last 100 ms         -              46, then 16 constant L1                    ROBOENV_GAP_WINDOW=1
T34  a quarter and a sixteenth of the data   110 and 69/120 val   -           ROBOENV_RUNS_DIVIDE=4 or 16
T35  random starts near the parking spot     48             46                ROBOENV_RANDOM_START=1
T36  physical pushes, half and full size     50, 47         45, 44            ROBOENV_PUSH_LEVEL=0.5 or 1
```

T32 to T34 need their own trained models first (see the README).


## S. Sensor-only

All use Nat's seed 7 driving network and Vincent's arm network option b (`option_b_seed1`), unless stated.

```
#    Date   What changed                                       Score              Switches
S1   4 Nov  arm network with the camera's target               4/100 val          (none)
S2   4 Nov  camera fixes                                       1/37               MY_ARM_CAMERA_FIX=1 <37>
S3   5 Nov  camera offset and a settled stop rule              19/37              MY_ARM_CAMERA_OFFSET=1 MY_ARM_SETTLED_STOP=1 <37>
S4   5 Nov  ball fit instead of the offset                     17/37              MY_ARM_BALL_FIT=1 MY_ARM_SETTLED_STOP=1 <37>
            arm trained to finish closer (option_b_finish)     2/10               + MY_ARM_NET=option_b_finish_seed1 <10> (not kept)
S5   5 Nov  learned ball and stop networks                     5/10, 16/37        MY_ARM_LEARNED=1 <10> or <37>
S6   5 Nov  target memory, settled stop rule                   21/37              MY_ARM_MEMORY=1 MY_ARM_SETTLED_STOP=1 <37>
S7   5 Nov  memory with ball and stop networks trained on      6/10, 18/37        MY_ARM_MEMORY=1 MY_ARM_MEMORY_NETS=1
            recordings made with memory                                           MY_ARM_NET_FILE=ball_stop_net_memory.npz <37>
S8   5 Nov  stop network retrained on the robot's own runs,    74/100 val         MY_ARM_MEMORY=1 MY_ARM_MEMORY_NETS=1 MY_ARM_STOP_FRAMES=15
            stop after 15 frames in a row
S9   5 Nov  tip network                                        23/37, 74/100      (S8) + MY_ARM_TIP_NET=1
S10  5 Nov  DAgger round 1, DAgger near the target             3/37, 2/10         (S9) + MY_ARM_NET=option_b_dagger_seed1 or option_b_daggernear_seed1 (not kept)
S11  5 Nov  stop at 12 mm instead of 9 mm                      23/37              (S9) + MY_ARM_STOP_M=0.012 (not kept)
S12  5 Nov  arm seed 0                                         69/100             (S9) + MY_ARM_NET=option_b_seed0 (seed check)
S13  5 Nov  stop network without joint speeds, 10 frames       22/37              (S9 but MY_ARM_STOP_FRAMES=10) + MY_ARM_NET_FILE=ball_stop_net_nospeed.npz (not kept)
S14  5 Nov  wheel network moves the memory                     76/100 val         (S9) + MY_ARM_WHEEL_NET=1   frozen design
S15  5 Nov  driving seeds 1 and 2, seed 7 retrained            23 and 27/100, 24/37   (S14) + MY_ARM_DRIVING=results/driving_nospeed_seed<n>/models/driving_seed<n>.npz
S16  5 Nov  frozen design, run once on the golden test         82/100 test        (S14) --split test
S17  5 Nov  after the test: three arm networks                 24/37 (25 for S14) (S14) + MY_ARM_GAP_NET=1, design both (not kept)
```

S7 names its stop network file because the default later changed to `ball_stop_net_final.npz`, made in S8.
S2 to S13 used models and recordings that are in `COMP0245_extras.zip` (`sensor_only/models`, `sensor_only/results`).
The S10 round 1 model was not kept; `REPRODUCE_EXTRA.md` gives its labels and training command.
