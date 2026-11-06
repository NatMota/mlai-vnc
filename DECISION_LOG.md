# Decision log

We moved the tables of our COMP0245 report (Group 02) into this file to fit the 6-page paper. The paper covers behaviour cloning of a simulated mobile manipulator from onboard sensors, and it cites these tables as Appendix A to Appendix H. Each row names its saved result evidence; notebook links open the relevant analysis cell. Live notebooks require access to our Tailscale network. The adjacent saved views can be read on GitHub without Tailscale. A row that rests only on our working notes says "development record".

[Home](README.md) · [Glossary](docs/GLOSSARY.md) · [Notebook access](docs/NOTEBOOKS.md) · [Data and reproduction instructions](docs/REPRODUCE.md)

Source paths start in `v2_sensors/` unless they start with `collection/`. Distances are in millimetres unless a unit is given.

## Terms

- PARK: the signal that stops the wheels and starts the reach.
- STOP: the signal that ends the reach.
- Iteration: one trained and tested configuration.
- Bearing: the direction of a target from the robot's start, in degrees from straight ahead. Up to iteration 33 each bearing is one target, 1.1 m away and 0.5 m high.
- Full run: both networks (driving and arm) in control with no expert, from reset to STOP plus 0.5 s.
- DART-inspired: fixed-noise augmentation, an approximation of DART. Fixed noise is added to the expert's commands while collecting data, and each row keeps the expert's clean command as its label.
- DAgger: we run the networks, label every state they reach with the expert's command, add those rows to the training data and retrain. We use three rounds.
- Closest: the smallest gripper-to-target distance in a run, counted up to 0.5 s after STOP. At STOP: that distance at the STOP step. Park error: the distance from the expert's parking position to where the robot parked. Values are medians over targets unless a count is given.

## Appendix A: Design choices

| Choice | Why | Evidence | Source | Notebook section |
|---|---|---|---|---|
| Corner camera | The forearm blocks a camera near the arm's centre line at PARK | Detected at PARK: hand 0, nose 12, corner 19 of 30 | `driving_v2/camera/wrist_fisheye_check.npz`, `prow_fisheye_check.npz`, `corner_fisheye_check.npz` (last frame of each bearing) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=fd8ef63e) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-48) |
| No wheel speed inputs | They let the network repeat its last command and ignore range | Park error 18.7 to 5.6 cm, 100 targets | `arm/results/time_cumulative_test/scores.json` (iterations 37 and 38) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=tc-38-title) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-204) |
| No arm IMU | Better fit and worse control, the same pattern as wheel speeds | Torque loss 23% lower, closest 116 against 57 mm, 25 bearings | `arm/models/arm1k_e77_phys_imu_history.json` (0.455 against 0.590), `arm/results/arm1k_e77_phys_imu_camera.json`, `arm/results/arm1k_e77_dist_notime_phys_camera.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=d8e7d676) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-242) |
| Target height in driving | The expert's standoff depends on height | Seed 7 park error 27.3 to 18.7 cm | `arm/results/golden_11110/scores.json` (seed 7), `arm/results/time_cumulative_test/scores.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=tc-37-title) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-196) |
| Time to PARK output | A distance target parked further from the expert in a direct test | Park error 4.4 to 8.4 cm, six targets | `arm/results/driving_height_no_speed11/results/height_no_speed_dot*.json` (4.4), `arm/results/park_distance_four/results/no_speed_dot*.json` (8.4) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pd-plan) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-114) |
| Separate STOP network | A shared head fired at the PARK transition | Stops within 0.5 s of PARK: 11 of 25 to 0 | `arm/results/arm1k_e77_dist_notime_phys_camera.json`, `arm/results/arm1k_e77_phys_stopnet_camera.json`, with `driving_v2/expert_reference.json` for the PARK times | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=3716de1e) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-241) |
| Single instant inputs | 50 and 100 ms of history made STOP worse | At STOP 132 to 218 mm, six targets | `arm/results/arm_cumulative/results/wider_dot*.json` and `history_dot*.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=ab-history-title) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-104) |
| 128 units in the arm | 64 units stopped further from the target | At STOP 194 to 85 mm, 100 targets | `arm/results/time_cumulative_test/scores.json` (iterations 39 and 40) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=tc-40-title) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-220) |
| Standardised arm inputs | Physical scales gave a worse reach once the network held the arm | At STOP 273 to 169 mm, 100 targets | `arm/results/overnight_20261105/baseline/scores.csv`, `basics_standard/scores.csv` (iterations 41 and 43) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pending-full-basics_standard-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-251) |

## Appendix B: What was tried, by iteration

Parked counts are of 25 bearings up to iteration 33, and of 100 test targets from iteration 34. Iteration 4 (a skid-steer wheel output, parked 9 of 25) is not in the table. Some older files in the repository number the recovery experiments 52 to 58. This table uses 51 to 57.

| It. | Change | What we wanted to learn | Result | Source | Notebook section |
|---|---|---|---|---|---|
| 1 to 3 | Wheels only, then true target at start, then at every step | The ceiling with perfect target input | True target: 23 of 25 parked | `driving_v2/closed_loop_native/result_wheels.json`, `result_wheels_start_dot.json`, `result_wheels_live_dot.json` (12, 11, 23 of 25) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=fd71951c) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-0) |
| 5 | Target from corner camera, with visible flag | Can a camera replace the true target? | 17 of 25 | `driving_v2/camera_loop/camera_flag/` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=89dc70b4) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-55) |
| 6, 7 | Time to PARK output, then no visible flag | Is the flag needed? | 16, then 6 of 25 | `driving_v2/camera_loop/camera_flag_ttp/`, `camera_ttp_noflag_0.35s/` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=89dc70b4) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-55) |
| 8 to 11 | Scaled wheel angles, four seeds | How much is seed alone? | 14, 0, 4, 21 of 25 | `driving_v2/camera_loop/camera_flag_ttp_scaled/`, `..._seed1/`, `..._seed2/`, `..._seed3/` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=6a36a6b3) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-80) |
| 12, 13 | 100 rows after PARK, then encoder readings | Can PARK become reliable? | Mean 5.75, then 18 of 25 | `driving_v2/camera_loop/camera_flag_ttp_scaled_tail100*/` (9, 12, 2, 0 of 25), `..._tail100_encoders*/` (24, 22, 11, 15 of 25). Note: both are means over four seeds. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=5a43ae03) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-95) |
| 14 to 17 | Arm torque at 30 Hz, then 1 kHz | Is the camera rate enough? | 30 Hz swings away after 0.5 s; 1 kHz stable, 18 mm closest | `arm/results/arm30_sample.json`, `arm30_mean.json`, `arm1k.json` (closest 189, 221, 18 mm). The 0.5 s: development record. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=7cb0a93b) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-116) |
| 18, 19 | 77 epochs, then STOP from a distance head | Stop overfitting, stop on every run | Loss 0.64 to 0.38; STOP on 25 of 25, 35 mm | `arm/models/arm1k_history.json`, `arm1k_seed1_history.json` to `arm1k_seed3_history.json` (300 epochs) and `arm1k_e77_history.json`, `arm1k_e77_seed1_history.json` to `arm1k_e77_seed3_history.json` (77 epochs): mean test torque loss over four seeds 0.64, then 0.38. `arm/results/arm1k_e77_dist.json` (35 mm, 25 stops). | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=f025dc66) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-136) |
| 20 | Target from camera in the arm | What does the sensor cost? | Closest 29 to 105 mm | `arm/results/arm1k_e77_dist.json`, `arm1k_e77_dist_camera.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=fc408727) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-169) |
| 21 to 23 | No time inputs, no scaling, physical scales | Does the arm stop falling through the target? | Closest 189, 186, 57 mm | `arm/results/arm1k_e77_dist_notime_camera.json`, `..._notime_raw_camera.json`, `..._notime_phys_camera.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=25dea735) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-179) |
| 24 to 27 | STOP head weights and rows, then separate network | Stop firing at PARK | 11 of 25 to 0 with a separate network | `arm/results/arm1k_e77_phys_stop16_camera.json`, `arm1k_e300_phys_stop16_camera.json`, `stop_weight_choice_25.json`, `arm1k_e77_phys_stopnet_camera.json`. Stops within 0.5 s of PARK, counted with `driving_v2/expert_reference.json`: 11 (iteration 23), 12 (24), 0 (27). | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=a37fe9ef) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-215) |
| 28, 29 | Arm IMU, then IMU without arm encoders | Does an extra sensor help? | Loss down 23%, closest 116 against 57 mm | `arm/models/arm1k_e77_phys_imu_history.json`, `arm/results/arm1k_e77_phys_imu_camera.json`, `arm1k_e77_phys_imu_noenc_camera.json` (iteration 29 closest 61 mm) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=3716de1e) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-241) |
| 30 | 100 new targets | Does it generalise? | Closest 181 to 205 mm, height decides | `arm/results/arm1k_e77_dist_notime_phys_camera_dots.json` (205), `arm1k_e77_phys_stopnet_camera_dots.json` (181). "Height decides": development record. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=66e9cd55) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-277) |
| 31 to 33 | Train on camera target; both networks together | Does it help to train on the sensor's error? | Fit unchanged, closest 70 mm; together 101 mm | `arm/results/arm1k_e77_phys_camdot_camera.json` (70), `arm/models/arm1k_e77_phys_camdot_history.json` (test loss 0.583 against 0.590), `arm/results/full_camera_flag_ttp_scaled_tail100_encoders_arm1k_e77_phys_stopnet.json` (101). Note: iteration 33 used 8 bearings. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=49b24250) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-289) |
| 34 to 36 | Golden data, three seeds | Seed spread on spread targets | PARK 47, 98, 96; park error 22 to 29 cm | `arm/results/golden_11110/scores.json` (seeds 7, 1, 2: PARK 47, 98, 96; park error 27.3, 22.1, 28.6 cm) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=golden-dataset-008) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-8) |
| 37 | Target height in driving | Is height the missing input? | PARK 87, 18.7 cm | `arm/results/time_cumulative_test/scores.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=tc-37-title) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-196) |
| 38 | Remove six speed inputs | Does repeating the last command cause it? | PARK 100, 5.6 cm | `arm/results/time_cumulative_test/scores.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=tc-38-title) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-204) |
| 39, 40 | 1,200 epochs, then 128 units | Is the arm underfitting? | At STOP 194, then 85 mm | `arm/results/time_cumulative_test/scores.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=tc-39-title) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-212) |
| 41, 43 | Learned hold, then standardised | Can a network hold the arm? | Closest 184, then 75 mm | `arm/results/overnight_20261105/baseline/scores.csv`, `basics_standard/scores.csv` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=full-41-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-233) |
| 42, 44 to 48 | Effects added in turn to 41 | Which effects does it tolerate? | Closest 556 to 860 mm | `arm/results/overnight_20261105/` `encoder_noise`, `damping`, `elasticity`, `inertia`, `delay`, `random_start_fixed` (each `scores.csv`: 574, 556, 662, 641, 764, 860) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=full-42-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-242) |
| 49, 50 | Retrain on random starts | Does more varied data help? | PARK 85, 29.5 cm | `arm/results/overnight_20261105/random_start_retrained/scores.csv` (iteration 49: PARK 85, park error 29.5 cm; iteration 50 is `random_start_retention`) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pending-full-random_start_retrained-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-305) |
| 52, 54, 56 | Fixed noise, DAgger, both | Does recovery data help? | At STOP 61, 37, 34 mm | `arm/results/overnight_20261105/dart/`, `dagger/`, `dart_dagger/` (each `scores.csv`) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pending-full-dart-heading) · [saved view](v2_sensors/review/golden_dataset_51_57.md#cell-332) |
| 51, 53, 55, 57 | Same models, all effects | Does it help under disturbance? | PARK 48, 65, 64, 87 | `arm/results/overnight_20261105/standard_stack/`, `dart_stack/`, `dagger_stack/`, `dart_dagger_stack/` (each `scores.csv`) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pending-full-standard_stack-heading) · [saved view](v2_sensors/review/golden_dataset_51_57.md#cell-323) |

## Appendix C: Implementation parameters

| Item | Value | Source | Notebook section |
|---|---|---|---|
| Targets | Range 1 to 2 m, any bearing, height 0.10 to 1.25 m; expert reached 997 of 1,000; split 800, 100, 100; 100 training targets by Latin hypercube selection | `collection/native/effects/targets_1000.json`, `split_1000.json`, `dots_spread.json`; `collection/native/tools/select_dots.py`; `arm/results/effects_collection_summary.json` (baseline: 997 reached, 3 stalled) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=golden-dataset-002) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-2) |
| Encoders | Quantised readings: each angle is a whole number of counts, 4,096 per wheel turn and the Kinova datasheet resolution on the arm; speeds are backward differences | `driving_v2/sensors.py` (4,096 per wheel turn; 3,686,400 per turn on arm joints 1 to 4 and 2,534,400 on joints 5 to 7) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=c41d529d) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-101) |
| Camera | 180° fisheye RGB-D, 256×256, 30 Hz; at (0.29, −0.13, 0.36) m on the front-right chassis corner | `driving_v2/camera/corner_fisheye_check.npz` (mount and 256×256 frames), `arm/results/time_cumulative_test/config.json` (`camera_hz` 30, `driving_hz` 30, `arm_hz` 1000) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=golden-dataset-004) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-4) |
| Driving MLP | 11 inputs; ReLU 16, 16; 549 parameters; 30 Hz | `arm/results/overnight_20261105/baseline/models/driving_seed7.npz` (549 counted, PARK output included) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=full-41-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-233) |
| Torque MLP | 18 inputs; ReLU 128, 128; torques clipped at ±30 N m; 19,847 parameters plus 129 for the distance output used in training, 19,976 in all; 1 kHz | `arm/results/overnight_20261105/baseline/models/torque_seed7.npz` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=full-41-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-233) |
| STOP MLP | 18 inputs; ReLU 32, 16; STOP at 10 mm or less, after PARK; 1153 parameters; 1 kHz | `arm/results/overnight_20261105/baseline/models/stop_seed7.npz` (1,153 counted) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=full-41-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-233) |
| Training | Adam, learning rate 0.01, full batches; up to 300, 1,200 and 600 epochs; lowest validation loss; seed 7 | `arm/results/overnight_20261105/baseline/models/config.json` (1,200 and 600 epochs, rate 0.01, seed 7), `driving_seed7_summary.json` (selected epoch 299) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=full-41-heading) · [saved view](v2_sensors/review/golden_dataset_41_50.md#cell-233) |
| DART-inspired | 0.025 rad/s on wheels, 0.5 N m on arm, held 33 ms; 199 usable runs from 200; 2,657,685 STOP rows against 884,912 | `arm/results/overnight_20261105/dart/models/stop_seed7_summary.json` (2,657,685 rows), `baseline/models/stop_seed7_summary.json` (884,912 rows), `arm/collect_recovery.py` (defaults 0.025 and 0.5, held 33 steps). The 199 usable runs: development record. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pending-full-dart-heading) · [saved view](v2_sensors/review/golden_dataset_51_57.md#cell-332) |
| DAgger | Three rounds on 100 training targets; six validation targets chose the round | `arm/results/overnight_20261105/dagger/` (`scores.csv`, `models/`). The round choice: development record. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pending-full-dagger-heading) · [saved view](v2_sensors/review/golden_dataset_51_57.md#cell-350) |
| Effects | Encoder angle noise 0.0001 rad before counting; τ += M(q)(−c·q̇ − k·q) with c = 1 s⁻¹, k = 1 s⁻²; motor inertia −J·q̈ with J = 0.001 kg m²; 3 ms delay; random starts ±0.15 m and ±30° | `collection/native/effects/noise.json`, `damping.json`, `elasticity.json`, `inertia.json`, `delay.json`; `arm/collect_recovery.py` (random start range) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=full-episode-plan) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-229) |
| Full run | Reset to STOP plus 0.5 s; deadline 4 s after the expert's finish | `arm/results/time_cumulative_test/config.json` (`timeout`: expert completion + 4 s; observe 0.5 s after STOP) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=tc-plan) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-194) |

Camera mount comparison, target detected at PARK on 30 bearings: 0 from the hand, 12 from the chassis nose, 19 from the corner (`driving_v2/camera/wrist_fisheye_check.npz`, `prow_fisheye_check.npz`, `corner_fisheye_check.npz`).

## Appendix D: Loss choices

The three losses, in plain text. Driving: the wheel-command squared error divided by 4, plus (predicted output minus min(t, 3) / 3) squared, where t is the time in seconds to the expert's PARK. Arm: the standardised-torque squared error divided by 7, plus (auxiliary output minus min(d, 0.25) / 0.25) squared, where d is the gripper distance to the target in metres. STOP: (output minus min(d, 0.25) / 0.25) squared. Each is averaged within a run and then across runs, so long runs do not count for more (`arm/train_golden.py`, function `objective`).

| Choice | Tried | Result | Kept | Source | Notebook section |
|---|---|---|---|---|---|
| PARK output | One-row PARK class, binary cross-entropy | Parked 17 of 25, with PARK probability under 0.35 when the robot stopped 3 to 36 cm short; 16 of 25 with time, and those park close to the expert in time and place | Time to PARK | `driving_v2/camera_loop/camera_flag/run_*.json` (17 parked; highest PARK probability 0.34 on the 8 that did not; 2.6 to 36.3 cm short), `driving_v2/camera_loop/camera_flag_ttp/run_*.json` (16 parked); code `driving_v2/train_camera.py` (cross-entropy, PARK row weighted up), `driving_v2/train_camera_ttp.py` (time target). "Close to the expert": development record. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=89dc70b4) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-55) |
| PARK target | Distance to the expert's park position | Park error 4.4 cm with time, 8.4 with distance, six targets | Time to PARK | `arm/PARK_DISTANCE.md`, `arm/results/park_distance_four/results/`, `arm/results/driving_height_no_speed11/results/height_no_speed_dot*.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pd-plan) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-114) |
| PARK weight | 10 on that distance | Park error 8.4 to 11.1 cm, worse on 6 of 6 | Weight 1 | `arm/PARK_WEIGHT10.md`, `arm/results/park_weight10/results/summary.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pw-plan) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-166) |
| PARK clip | 0.5 m on that distance | First PARK within 5 cm: 38 to 60 of 100 runs, then 37 at 1,000 epochs | Time, clipped at 3 s | `arm/PARK_CLIP05.md`, `arm/results/park_clip05_e300/offline_audit.json` (38, 62, 60 and 37 of 100 for the four models), `arm/results/park_clip05_e1000/` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=pc-plan) · [saved view](v2_sensors/review/golden_dataset_34_40.md#cell-179) |
| Torque scale | Raw N m | Training error 2.17 N m², against 3.68 for each joint's mean and 0.37 standardised | Standardised | `arm/models/arm1k_e77_dist_notime_raw_history.json` (2.17). The 3.68 and 0.37: development record. | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=24ce2b48) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-193) |
| STOP target | Time to STOP | At STOP 257 mm on the 15 of 25 bearings that stopped, against 35 mm on all 25 with distance | Distance | `arm/results/arm1k_e77.json` (median of the first log row at or after STOP), `arm/results/arm1k_e77_dist.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=41e01273) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-159) |
| STOP weight | 3, 10, 30, 100 | Validation torque loss up 11% to 109% | Weight 1 | `arm/results/stop_weight_choice_25.json` (up 11%, 30%, 60%, 109%) | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=40c617b4) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-225) |
| STOP head | Shared with torque layers | Stops within 0.5 s of PARK: 11 of 25, then 0 when separate | Separate network | `arm/results/arm1k_e77_dist_notime_phys_camera.json`, `arm1k_e77_phys_stopnet_camera.json`, with `driving_v2/expert_reference.json` | [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb#cell-id=3716de1e) · [saved view](v2_sensors/review/driving_v2_sensors.md#cell-241) |

## Appendix E: Normal against fast simulator

Related setup analysis: [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c012) · [saved view](v2_sensors/review/report_statistics.md#cell-5); exact audits and benchmark files are listed below.

The normal simulator is the course Python expert in PyBullet. The fast one is a C++ port of the same expert (`expert_compatible`) that calls the same Bullet build, with physics and control at 1 kHz and no Python in the loop. We read every number and flag below from the files named, in `collection/native/`.

Where to look:

- `collection/native/BENCHMARKS.md` (Apple M1) and `collection/native/MINT-BENCHMARKS.md` (Linux desktop). Raw records are in `collection/native/results/`.
- The valid audit is `results/FINAL.json` (`verified_runs` 10, `verified_samples` 80010, `all_exact` true) and `results/verified_summary.json` (`all_passed` true, `all_exact` true). Details are in the file table below.
- The reference captures from the Python simulator are not in Git (`reference/run_*/` is ignored). Make them with `tools/capture_reference.py`.

Commands, run from `collection/native/`. We took the flags from each tool's own argument parser and from `src/main.cpp`.

```sh
pixi run build-compatible
pixi run collect --targets targets.json --repeat 10 --binary --diagnostics --out data/NAME
python tools/capture_reference.py --run run_001 --target 0.0 1.1 0.5
python tools/audit_suite.py ORIGINAL NATIVE --tag NAME
python tools/compare.py REF CAND --report FILE
```

- `collect` with one worker and `--repeat 10` runs ten episodes. Target i of `targets.json` goes to `data/NAME/run_i`, for i from 0 to 9. The output folder must be new. `--diagnostics` adds the 23×23 mass matrix at every tick.
- `capture_reference.py` makes one unchanged Python run, headless, in `reference/RUN/`. RUN must start with `run_` and must not exist yet. Without `--target` the course script keeps its own target. It needs the course Python environment, and it finds the course script through the environment variable `EXPERT_REFERENCE_SOURCE` (the course lab repository's `robotnik_arm_task.py`). Capturing adds overhead, so its wall time is no benchmark.
- `audit_suite.py` runs `compare.py` on ten pairs. ORIGINAL holds the captures named below, NATIVE holds `run_0` to `run_9`. It writes `results/NAME_0.json` to `results/NAME_9.json` and `results/NAME_summary.json`, and exits with 0 only when all ten pass and all ten are exact.
- `compare.py` checks one pair. `--prefix` compares only the rows the candidate has, for driving-only runs.
- Timing: `tools/reference_timing.py` times the Python expert, and `tools/benchmark.py` times the native one (`--workers 1 2 4 8 --repeat 2`, optional `--fork-workers`, `--diagnostics` and `--compact`).

The ten audited targets, in the order `audit_suite.py` pairs them:

| Native folder | Python capture folder | Target x, y, z (m), as in `targets.json` |
|---|---|---|
| `run_0` | `run_ahead_001` | 1.1, 0.0, 0.5 |
| `run_1` | `run_001` | 0.0, 1.1, 0.5 |
| `run_2` | `run_003` | −0.8, 0.6, 0.65 |
| `run_3` | `run_004` | 0.8, 0.8, 0.5 |
| `run_4` | `run_005` | 0.6, −0.9, 0.5 |
| `run_5` | `run_006` | −1.0, 0.0, 0.65 |
| `run_6` | `run_007` | −0.6, −0.8, 0.7 |
| `run_7` | `run_008` | 0.4, 0.9, 0.9 |
| `run_8` | `run_009` | 1.0, 0.3, 0.3 |
| `run_9` | `run_010` | −0.4, 0.9, 1.05 |

Tolerances in `tools/compare.py`:

| Check | Rule |
|---|---|
| Seven tables: `target`, `joint_state`, `base_state`, `kinematics`, `base_controller`, `arm_controller`, `commands` | Same row count and column order; string and boolean labels equal; NaN in the same places; numbers within numpy `allclose` with rtol 1e-7 and atol 1e-8 |
| Simulation clocks: `step`, `time_s`, `post_step_time_s` in the `timing` table | Exactly equal |
| Tensors: `mass_matrix`, `arm_jacobian`, `q_pin` | Same shape; same `allclose` bounds. A tensor the native run did not record is listed as not recorded and does not fail the run. |
| Exact equality | Reported separately for every table and tensor (`exact_recorded_values`). `audit_suite.py` needs it as well as the tolerance pass. |
| Wall-clock columns | Not compared |

Timing. Every episode is the straight-ahead target (1.1, 0, 0.5) with 8,001 samples (8 s at 1 kHz). We measured on 29 October 2026, with other work running on both machines: desktop use and background photo processing on the M1, a load average of roughly 8 to 11 on the Mint desktop.

| Number | Machine and workers | Included | Left out | File |
|---|---|---|---|---|
| Python 5.46 to 5.77 s (the report says 5.5 to 5.8) | Apple M1 (8 cores, 16 GB, macOS 14.6.1), one process, inside `main()` | Simulation and control, headless, no profiler | Imports (0.66 to 1.35 s), process start-up, dataset saving (none done) | `results/python_reference_timing.json` (`main_s`). With imports 6.16 to 6.81 s, with process start-up 6.30 to 7.00 s. |
| Python 10.48 to 10.88 s (the report says 10.5 to 10.9) | Linux Mint desktop (Intel Core i7-6700K, 4 cores and 8 threads, 32 GB), one process, inside `main()` | Simulation and control, no profiling | Imports (13.91 to 15.58 s with them), dataset saving | `collection/native/MINT-BENCHMARKS.md` text only. The raw record it names (`results/mint/`) is not in the repository. |
| Native 2.29 s per episode | M1, one process running two episodes (4.58 s in all) | Process start-up (counted once), world loading and reset, simulation, recording, saving binary arrays | Mass matrix, file hashing and duplicate removal after the timed part, build time | `results/export_optimised.json` |
| Native 0.646 s per episode | M1, 8 processes, 16 episodes in 10.34 s | As above | As above | `results/export_optimised.json` |
| Native 0.640 s per episode (the report says 0.64) | M1, 8 preloaded processes (`--fork-workers`), 16 episodes in 10.24 s | As above | As above | `results/release_fast.json`, `results/FINAL.json` |
| Native 3.08 and 1.09 s per episode | M1, 1 process with 2 episodes (6.17 s) and 8 processes with 16 episodes (17.45 s), `--fork-workers --diagnostics` | As above, plus the 23×23 mass matrix at every tick | As above | `results/release_full.json` |
| Native 3.381 s per episode (the report says 3.38) | Linux Mint desktop, 1 worker, 2 episodes | Native process start-up, world setup, simulation, recording, file export | Compilation, environment installation, `pixi` launcher time, validation, duplicate cleanup, any forced disk sync, mass matrix | `results/MINT_FINAL.json` (`mint_full.json` entry) |
| Native 0.882 s per episode (the report says 0.88) | Linux Mint desktop, 8 workers, 16 episodes | As above | As above | `results/MINT_FINAL.json`. With the mass matrix: 1.377 s. |

How to read the timings. The Python figure is the time of one process and leaves out imports and saving. The native figures are batch time divided by the episodes saved, and they include start-up and saving. The one-process native rows compare most directly with Python. The Python figure should not be divided by an eight-worker figure to claim a per-process speedup. The audit does not show a single complete expert run under 500 ms.

Result files:

| File | Status | Why |
|---|---|---|
| `results/FINAL.json` | Valid | Final audit summary: 10 runs, 80,010 samples, all exact, mass matrices included, repeat runs equal, asset hashes match. Source fingerprint `839477d353ef3d8a7694f81df3b71fe19ba810904dce0cb04a356665e599cbd7`. |
| `results/verified_summary.json`, `verified_0.json` to `verified_9.json` | Valid | `all_passed` true and `all_exact` true. Each run has `max_table_error` 0.0, `max_tensor_error` 0.0 and `simulation_timestamps_equal` true. |
| `results/fidelity_final_summary.json`, `fidelity_final_0.json` to `fidelity_final_9.json` | Older, superseded | Shows one failed target: target 8 (`run_009`, target 1.0, 0.3, 0.3) has `passed` false and a largest table error of 1608.2. It is from before a correction for low targets. `results/low_target_corrected_comparison.json` shows the same `run_009` passing with zero error (that run did not record the mass matrix). `verified_8.json` is the full audit of it. |
| `results/release_validation_summary.json`, `release_validation_0.json` to `release_validation_9.json` | Do not use | All ten show `simulation_timestamps_equal` false, with zero table error. |
| Other files in `results/` | Development records | Earlier builds and timing sweeps. Quote only the files above. |

Limits. The exact match holds for these ten targets, this engine build and this platform. The Mac and Linux builds of the physics engine differ, and each native port matches the Python run on its own machine. On Mint, two targets (straight ahead and a left turn) were audited against Linux Python captures (`results/linux-audit-ahead.json`, `results/linux-audit-left.json`). No gravity, contact, motor gain or geometry was changed to gain speed, and the original mass and inertia inconsistency is kept.

## Appendix F: Background

We describe behaviour cloning (BC) and DAgger by what each one matches: BC matches the expert's actions on the expert's own states, and DAgger matches them on the states the learner reaches. This framing follows Choudhury's imitation learning lecture series, https://www.youtube.com/watch?v=V38omEpfbjI.

## Appendix G: V2 systems and what differs

Main V2 outcomes: [Notebook](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb#cell-id=overnight-publication-overview) · [saved view](v2_sensors/review/golden_dataset_51_57.md#cell-386). Teammate systems have separate source/result records; no matching teammate notebook was supplied in the reviewed repository.

All three systems use the same 100 test targets and the same sensors. The main baseline, the two-rate system and the target-memory system use one identical driving checkpoint, `driving_seed7.npz`. Outcomes are counts of 100 test attempts, and median distance covers stopped runs. These are descriptive results of separate systems, and no difference here is the measured effect of one change.

| System | Arm training data | Arm before PARK | Arm controller after PARK | STOP | Simulator, renderer | PARK, STOP, success; mm at STOP | Source |
|---|---|---|---|---|---|---|---|
| Main baseline | 100 golden runs, 884,912 rows | Learned torque MLP from reset | Same direct-torque MLP, 18 inputs | MLP distance, 10 mm or less | Native, EGL | 100, 91, 0; 169 | `v2_sensors/arm/results/overnight_20261105/basics_standard/scores.csv` |
| Main combined | Adds 199 noisy runs and DAgger rows on commands | As above | As above, retrained | As above, trained with noisy expert runs | Native, EGL | 87, 66, 4; 34 | `v2_sensors/arm/results/overnight_20261105/dart_dagger/scores.csv` |
| Two-rate system | 4,000 V1 expert runs, with expert reference labels; world inputs removed | Conventional hold using true joint state and gravity | Velocity MLP, bounded integrator, torque MLP | Fixed rule: FK to tracked target, 5 mm for 100 ms | PyBullet, CPU | 100, 68, 1; 17 | `docs/golden_results.md` on branch `experiments/v2-golden-arm` |
| Target-memory system | Reaching rows of 1,350 V1 runs with pushes; ball and stop networks on recorded V2 runs | Conventional hold, until PARK and a first target estimate | Direct-torque MLP, 16 inputs, with target memory | Stop MLP, 9 mm for 15 frames, none in first 300 ms | Native, CPU | 100, 87, 82; 5.9 | `vincent_final/sensor_only/` on branch `vincent/final` |

## Appendix H: Secondary results

These results support the report and sit here so its argument stays short. Some also appear in the report.

V1 controllers:

- An earlier two-part model reached 100 of 120 test targets, with 118 of 120 within 2 cm at closest. The earlier single model and rejected inputs are in the branch `vincent/final`.
- Four early DAgger attempts on an earlier single model reached 0 to 3 of the first 20 test targets.
- Extra history inputs lowered validation error from 0.064 to 0.054 and reached 46 of 50 against 43, too small a difference to call. An IMU input to driving showed no clear benefit across seeds.
- Independent sensor errors at the largest level did not hurt either model (saved model 50 of 50, learned version 44). A constant joint speed error alone gave 15 of 50 on the learned version, while the other sensors gave 43 or 44, and retraining with speed errors did not help.
- Physical pushes matching training: saved model 50 of 50 at half size and 47 of 50 at full size, learned version 45 and 44, against 43 with no pushes.
- Two-rate arm: every miss on new goals by either policy was less than 0.8 m from the start. Outside the training box both reached 279 of 400 and missed different goals. Removing the world goal position alone gave a mean of 106.3 of 120, so input groups interact. At a 10 mm goal offset, 116 of 120 still ended under 2 cm.

Main V2 family:

Analysis sections: [speed inputs](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c030) · [saved view](v2_sensors/review/report_statistics.md#cell-23); [action predictability](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c045) · [saved view](v2_sensors/review/report_statistics.md#cell-38); [state shift](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c039) · [saved view](v2_sensors/review/report_statistics.md#cell-32); [error growth](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c060) · [saved view](v2_sensors/review/report_statistics.md#cell-53); [approach to the target](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c090) · [saved view](v2_sensors/review/report_statistics.md#cell-83); [after STOP](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c095) · [saved view](v2_sensors/review/report_statistics.md#cell-88); [disturbances](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb#cell-id=c076) · [saved view](v2_sensors/review/report_statistics.md#cell-69).

- Cumulative effects used motor damping and elasticity through τ += M(q)(−c·q̇ − k·q) with c = 1 s⁻¹ and k = 1 s⁻², and motor inertia of 0.001 kg m². Median closest distance on parked runs was 184 mm with none, then 574, 556, 662, 641, 764 and 860 mm.
- The tracker's median range error at PARK is 9 mm. A 20% fall in the speed inputs lowered the wheel command by 0.18 to 0.25 rad/s, while 20 cm more range changed it by at most 0.03 rad/s.
- Wen's action predictability diagnostic did not separate the driving models: the better one was the more predictable from its previous commands (log ratio 3.97 against 3.30).
- Larger weights on the STOP loss raised validation torque loss by 11% to 109%. DART-inspired collection gave 199 usable demonstrations from 200. Laskey et al. find isotropic noise poor, while Zhang et al. find fixed noise with clean labels and a clean plus noisy mix enough in theory.
- After DAgger's first round a classifier still separates the model's states from the expert's with an AUC of 0.969, on folds split by run. Growth exponents of joint distance from the expert's path are 0.91 and 0.77, with overlapping ranges.
- Validation and test rank DAgger and the combined model differently: 4 against 1 validation successes, and 2 against 4 test successes.
- Before STOP, the gripper visited the 10 mm band on 7 DAgger and 6 combined runs, and the network stopped inside it on 2 and 4. A different STOP would also change what follows. Encoder noise alone moves the STOP output by 32 mm (SD). Over the 0.5 s after STOP, distance to the target changes by a median 103.1 mm for the baseline and -0.6 mm for DAgger, each on its own stopped runs.
- Under all effects the baseline's 48 PARK uses the same driving network as the cumulative test's 89, so its arm's motion may disturb the base, which was not tested. Retraining on random starts gave PARK on 85 with a 29.5 cm park error, possibly because the expert parks relative to the world origin. DAgger raised the torque noise predicted from encoder noise from 1.6 to 2.9 N m to 4.0 to 8.4 N m.
- Training STOP on recovery rows in a six-target pilot improved its loss on those rows, and it issued STOP on two targets instead of five. Bursty packet loss was planned and not done. The strongest direct weight edit scaled the arm's PARK-input weights by 1.25 and cut median closest distance from 86 to 52 mm on twenty validation targets, with no 10 mm successes.

Targets:

- Training and test targets from the robot's start pose are plotted in [docs/report_target_distribution.png](docs/report_target_distribution.png). Shaded bearings are beyond ±80°, where 52 of the 55 blind training starts lie.

Next steps in more detail:

- A time to PARK on the policy's own states could be defined from the true distance to the expert's parking position.
- After filtering the speed estimate over several readings, we would measure how much encoder noise reaches the STOP output.
- For occlusion we would compare normal tracker updates with holding the last camera estimate before sweeping screen frequency and phase.

Alternative V2 systems:

- Two-rate system: 32 runs had no STOP, 63 were short reaches and 4 were missed stops. On one validation goal the estimated distance at STOP was 4.3 mm with a true distance of 37.1 mm.
- Target-memory system: failed runs parked a median 3.7 s later than the expert, against 0.4 s for successes. Success was similar across target heights (27 of 34 low, 33 of 38 middle, 22 of 28 high).
