# V2: two-rate arm on Nat's sensor-only driving

## Setup

Nat's iteration 13 driving network (wheel encoders and a corner camera only) drives until PARK. Then the
two-rate arm reaches: `vel_net` at about 30 Hz and `torque_net` at 1 kHz, the final arm from
`docs/fusion_results.md`. Nat found that an arm network sending torques at 30 Hz fails on every bearing,
because each torque is held for 33 ms.

| Part | Source |
| --- | --- |
| Wheel commands and PARK | Nat's iteration 13 network, 30 Hz (`nat_v2/models/iter13*.npz`) |
| Wheel inputs | Nat's wheel encoders: 4096 counts per turn, speed from the change over 1/30 s |
| Goal | Nat's corner fisheye camera and tracker, with the goal height added (robot frame) |
| Arm joint angles and speeds | Nat's arm encoders: Kinova Jaco 2 resolution, speed from the change over 1 ms |
| End-effector | forward kinematics of the encoder angles, base at the origin |
| Arm before PARK | the expert's hold controller, as in Nat's simulator |

The simulator's true state is used only to render the camera image and to score the run. Nat's code is copied
unchanged into `nat_v2/` (see `nat_v2/SOURCE.md`). The test uses his 25 bearings: 21 train and 4 test, each
with the goal 1.1 m away at 0.5 m height. His 5 holdout bearings were not used. Success means the end-effector
is within 1 cm of the goal after at least 8 s.

## Results

| Run | Driving | Arm inputs | Parked | Reached 1 cm (train / test / all) | Within 2 cm | Median final error |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| A | Nat seed 7 | arm held | 25/25 | - | - | - |
| B | expert | sensors | 25/25 | 0/21, 0/4, **0/25** | 0/25 | 2.86 cm |
| C | Nat seed 7 | sensors | 25/25 | 8/21, 2/4, **10/25** | 11/25 | 2.19 cm |
| D | Nat seed 7 | true state | 25/25 | 20/21, 3/4, **23/25** | 23/25 | 0.57 cm |
| E1 | Nat seed 1 | sensors | 25/25 | 0/21, 0/4, **0/25** | 4/25 | 4.09 cm |
| E2 | Nat seed 2 | sensors | 25/25 | 11/21, 1/4, **12/25** | 16/25 | 1.46 cm |
| E3 | Nat seed 3 | sensors | 19/25 | 5/21, 0/4, **5/25** | 10/25 | 2.80 cm |

Full v2 (C) reaches 10 of 25 bearings. Over the four driving seeds (C, E1-E3) it reaches 27 of 100.

All 175 episodes ran without a process error. Two full runs gave identical results, episode for episode (the first
batch is kept in `outputs/v2_first`).

## Findings

1. **Nat's driving runs the same in this simulator.**
   - In run A it parked on 25 of 25 bearings; Nat reports 24 of 25.
   - Compared with Nat's own saved runs for iteration 13 seed 7 (`v2_sensors/driving_v2/camera_loop/camera_flag_ttp_scaled_tail100_encoders/run_*.json`
     on his branch, not copied here), the final base positions differ by a median of 0.5 cm, at most 8.1 cm.
   - The bearing he does not count (12 degrees) parks here at 6.17 s, which equals his time limit for that
     bearing (expert park time 3.17 s + 3 s, `nat_v2/expert_reference.json`).
   - Against the expert's own final base positions, the network parks a median of 7.3 cm away, and 25 cm on
     bearing 12.
2. **The two-rate arm works after sensor-only driving.** With the true goal and joint state (D), it reaches 23
   of 25 bearings, with a median final error of 0.57 cm. Both misses (84 and 108 degrees) swing far away. These
   are the two bearings where Nat's network parks with the largest heading errors, 46 and 25 degrees from the
   expert's final heading, against a median of 9 degrees.
3. **The camera goal estimate limits sensor-only reaching.** The arm ends where the tracker places the goal.
   - In C, the median 3D camera error at the end is 1.05 cm on the bearings that succeed and 2.60 cm on those
     that miss.
   - Across C the camera error and the final error correlate at 0.81.
   - In C, all 13 bearings with a camera error above 1.5 cm missed, and 10 of the 12 at or below it reached.
4. **The camera is worst at the expert's parking spot.** In B the expert parks a median of 0.26 cm (at most
   1.2 cm) from its own reference, but the camera error at the end is 2.4-3.9 cm horizontally on every bearing, so no reach succeeds. The
   camera's view of the goal depends on where the robot stops.
5. **Driving seeds matter as much as the arm.** Seeds 1, 2 and 3 park a median of 17-21 cm from the expert,
   against 7.5 cm for seed 7.
   - Seed 1 has a small camera error (median 1.3 cm in 3D) but reaches nothing, so its parking poses are the
     problem there.
   - Seed 3 parks on only 19 of 25 bearings.

## What this means

- Sending torques at 1 kHz from a 30 Hz velocity plan solves Nat's 30 Hz arm problem: 23 of 25 with exact
  inputs.
- To reach within 1 cm on sensors alone, the goal estimate needs to be better than about 1 cm. Possible steps:
  - average several camera sightings after PARK;
  - add a camera on the wrist for the final approach;
  - train the arm with goal noise.
- The arm was trained on goals across the whole training box, not on Nat's 25 bearings. It was not retrained
  for this test.

## Limits

- One seed of the arm; four seeds of the driving.
- The camera is simulated: PyBullet's CPU renderer, no lens noise, and the same red-pixel detector as Nat's.
- Encoders are ideal apart from counting in whole ticks: no noise, no offset, no delay.
- The rows in `outputs/v2` were made before the last change to the provenance code in `v2_eval.py`.
  The sensor, policy and simulator code was the same. The rerun of run A afterwards gave identical results.

## Reproduce

```
python -B -m unittest tests.test_v2_sensor
python -B v2_eval.py A B C D E1 E2 E3 --workers 12 --fresh
```

Each episode renders the camera 30 times a second, so one episode takes about 1-7 minutes. The 175 episodes
took about 28 minutes with 12 workers. Results go to `outputs/v2/RUN.jsonl` and `outputs/v2/RUN.summary.json`.
