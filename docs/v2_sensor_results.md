# V2: two-rate arm on Nat's sensor-only driving

## Setup

Nat's iteration 13 driving network (wheel encoders and a corner camera only) drives until PARK. Then the
two-rate arm reaches: `vel_net` at about 30 Hz and `torque_net` at 1 kHz. Nat found that an arm network sending
torques at 30 Hz fails on every bearing, because each torque is held for 33 ms.

| Part | Source |
| --- | --- |
| Wheel commands and PARK | Nat's iteration 13 network, 30 Hz (`nat_v2/models/iter13*.npz`) |
| Wheel inputs | Nat's wheel encoders: 4096 counts per turn, speed from the change over 1/30 s |
| Goal | Nat's corner fisheye camera and tracker, with the goal height added (robot frame) |
| Arm joint angles and speeds | Nat's arm encoders: Kinova Jaco 2 resolution, speed from the change over 1 ms |
| End-effector | forward kinematics of the encoder angles, base at the origin |
| Arm before PARK | the expert's hold controller, as in Nat's simulator |

**The arm has no world-frame inputs.** `v2_sensor.export_arm` takes `vel_net` and `torque_net` from the final arm
(`outputs/ablation/all_world/policy.pt`, MLflow run 540bdcf9) and deletes the 11 world-frame inputs from their
first layers, giving 23 inputs for `vel_net` and 38 for `torque_net`. These inputs were set to zero in training
and in use, so deleting them changes no output: the tests check this to 1e-9. The exported file
(`outputs/v2_arm/arm_no_world_inputs.pt`) has no base or switch network.

Nat's code is copied unchanged into `nat_v2/` (see `nat_v2/SOURCE.md`). The test uses his 25 bearings: 21 train
and 4 test, each with the goal 1.1 m away at 0.5 m height. His 5 holdout bearings were not used. Success means
the end-effector is within 1 cm of the goal after at least 8 s.

## V2 results: sensors only

In these runs every network input comes from the robot's own sensors. The simulator's true state is used only to
render the camera image and to score the run.

| Run | Driving | Parked | Reached 1 cm (train / test / all) | Within 2 cm | Median final error |
| --- | --- | ---: | ---: | ---: | ---: |
| C | Nat seed 7 | 25/25 | 8/21, 2/4, **10/25** | 11/25 | 2.19 cm |
| E1 | Nat seed 1 | 25/25 | 0/21, 0/4, **0/25** | 4/25 | 3.86 cm |
| E2 | Nat seed 2 | 25/25 | 10/21, 1/4, **11/25** | 16/25 | 1.46 cm |
| E3 | Nat seed 3 | 19/25 | 5/21, 0/4, **5/25** | 10/25 | 2.84 cm |

Full v2 with Nat's seed 7 driving reaches 10 of 25 bearings. Over the four driving seeds it reaches 26 of 100.

**Caveats that go with these numbers:**
- The arm was trained on expert data. The velocity network learned from the expert's internal joint reference
  (`q_des`), and the torque network was trained with that reference as an input. At run time the reference comes
  from the arm's own velocity network, not from the expert.
- The integrator that turns velocity into the reference, and its 0.15 rad clip, copy the expert's design.
- The arm was trained on goals across the whole training box, not on Nat's 25 bearings, and was not retrained
  for this test.

## Upper bounds and diagnostics: not v2 results

These runs use simulator truth that the robot cannot measure, so they are not v2 results. They only show where
the v2 misses come from.

| Run | What uses simulator truth | Parked | Reached 1 cm (train / test / all) | Within 2 cm | Median final error |
| --- | --- | ---: | ---: | ---: | ---: |
| A | arm held all run (checks Nat's driving in this simulator) | 25/25 | - | - | - |
| B | expert driving; arm on sensors | 25/25 | 0/21, 0/4, 0/25 | 0/25 | 2.96 cm |
| D | arm uses the true goal and joint state | 25/25 | 20/21, 3/4, 23/25 | 23/25 | 0.57 cm |
| F | arm uses the true goal; Kinova encoders | 25/25 | 20/21, 3/4, 23/25 | 23/25 | 0.57 cm |
| G | arm uses the true goal; encoders with 4096 counts per turn | 25/25 | 13/21, 2/4, 15/25 | 18/25 | 1.00 cm |

All 225 episodes ran without a process error. Every row is stamped with the hash of the code and files that
made it.

## Findings

1. **Nat's driving runs the same in this simulator (A).**
   - It parked on 25 of 25 bearings; Nat reports 24 of 25.
   - Compared with Nat's own saved runs for iteration 13 seed 7 (`v2_sensors/driving_v2/camera_loop/camera_flag_ttp_scaled_tail100_encoders/run_*.json`
     on his branch), the final base positions differ by a median of 0.5 cm, at most 8.1 cm.
   - The bearing he does not count (12 degrees) parks here at 6.17 s, which equals his time limit for that
     bearing (expert park time 3.17 s + 3 s).
2. **The camera goal estimate limits v2 reaching.** The arm ends where the tracker places the goal.
   - In C, all 12 bearings with a 3D camera error above 1.5 cm at the end missed.
   - Of the 13 bearings at or below it, 10 reached.
   - The median camera error is 1.05 cm where the arm succeeded and 2.38 cm where it missed.
   - With the true goal instead (F), the same encoders reach 23 of 25, so the Kinova encoders are not the
     limit; 4096 counts per turn (G) drops this to 15 of 25.
3. **The camera is worst at the expert's parking spot (B).** The expert parks a median of 0.26 cm (at most
   1.2 cm) from its own reference, but the camera's horizontal error at the end is 2.6-4.0 cm on every bearing,
   so no reach succeeds.
4. **Parking pose matters.**
   - Seeds 1, 2 and 3 park a median of 17-21 cm from the expert, against 7.8 cm for seed 7.
   - Seed 1 has a small camera error (median 1.3 cm in 3D) and still reaches nothing.
   - In D, the two misses (84 and 108 degrees) are the bearings where Nat's network parks with the largest heading
     errors: 36 and 30 degrees from the expert's final heading (run A), against a median of 9 degrees.

## What this means

- To reach within 1 cm on sensors alone, the goal estimate needs to be better than about 1 cm. Possible steps:
  - average several camera sightings after PARK;
  - add a camera on the wrist for the final approach;
  - train the arm with goal noise.
- The two-rate split (velocity at 30 Hz, torque at 1 kHz) keeps the arm stable at the camera's rate. Nat's 30 Hz
  torque networks swing away.

## Limits

- One seed of the arm; four seeds of the driving.
- The camera is simulated: PyBullet's CPU renderer, no lens noise, and the same red-pixel detector as Nat's.
- Encoders are ideal apart from counting in whole ticks: no noise, no offset, no delay.
- **Comparison with the earlier masked arm.** The exported arm gives the same network outputs as the earlier
  masked arm (`outputs/v2_masked`) to rounding, but rounding differences grow over a run.
  - Success counts are the same except E2 (12 before, 11 now) and G within 2 cm (17 before, 18 now).
  - Single trajectories can differ a lot when the arm swings away.

## Reproduce

```
python -B -m unittest tests.test_v2_sensor
python -B v2_eval.py A B C D E1 E2 E3 F G --workers 12 --fresh
```

`v2_eval.py` exports the arm first when the export is missing or out of date. Each episode renders the camera
30 times a second, so one episode takes about 1-3.5 minutes. The 225 episodes took about 30 minutes with 12
workers. Results go to `outputs/v2/RUN.jsonl` and `outputs/v2/RUN.summary.json`.
