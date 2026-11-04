# v2 sensors: driving to the dot without simulator truth

v1 gave the network the robot's world pose and the dot's world position. v2 uses only what the robot could sense: wheel encoders and one camera. From iteration 13 the network sees nothing else. The outputs stay the same as the expert's: four wheel speed commands and PARK. Arm torques are not covered yet.

## Model to build from

Iteration 13, seed 7: `driving_v2/models/camera_flag_ttp_scaled_tail100_encoders.npz`. Wheel encoders and the corner camera only, scaled angle inputs, trained with 100 parked rows per run. PARK at 0.25 s or less. Parked 24 of 25 bearings, median final error 7.4 cm. Over four seeds it parks 11 to 24 of 25 (mean 18), so compare changes over several seeds.

## What is here

- `driving_v2/driving_v2_sensors.ipynb`: every iteration, its training curves and its test results, with plots.
- `driving_30/data/expert_30hz/`: the expert driving dataset. 30 trajectories, one per bearing (`sampled/` at 30 Hz, `full_rate/` at 1 kHz), the split (`split_plan.json`: 21 train, 4 test, 5 holdout) and the run records (`episodes.jsonl`).
- `driving_v2/camera_features/`: the camera inputs computed for each expert row (distance, sin, cos, visible flag).
- `driving_v2/models/`: all seven networks as `.npz`.
- `driving_v2/camera_loop/`: per-bearing test results for the three camera models.
- `driving_v2/closed_loop_native/`: test results for the four earlier models, which were given the true dot.
- `driving_v2/expert_reference.json`: the expert's park time and final pose per bearing, used to score every test.
- `driving_v2/sensors.py`: the simulated wheel encoders. New sensors (an IMU) go in its `Sensors` class.
- `driving_v2/park_tail/`: 100 rows after the expert parks, per run, with camera inputs (`expert_tail.py`).
- `../collection/native/`: the fast simulator. `sensor_parking` steps the physics for the camera tests; `policy_replay` runs a whole network inside the simulator.

## The iterations

All use the same network (2×16 ReLU), Adam, lr 0.01, 300 epochs and the same 21 training runs. Seed 7 unless stated. Iterations 8 to 13 are trained with `train_v2.py`.

| Iteration | Dot input | PARK output | Parked /25 |
|---|---|---|---|
| `wheels` | none | yes/no | 12 |
| `wheels+start_dot` | true direction at the start | yes/no | 11 |
| `wheels+live_dot` | true position every step | yes/no | 23 |
| `wheels+live_dot+skid` | true position every step, skid-steer output | yes/no | 9 |
| `camera+flag` | camera + visible flag | yes/no | 17 |
| `camera+flag+time_to_park` | camera + visible flag | time to park, PARK at 0.25 s | 16 |
| `camera+time_to_park` | camera | time to park, PARK at 0.35 s | 6 |
| 8: `+scaled` | iteration 6, angle inputs standardised | time to park, PARK at 0.25 s | 14 |
| 9 to 11: seeds 1, 2, 3 | iteration 8 with other seeds | same | 0, 4, 21 |
| 12: `+parked_rows` | iteration 8 plus 100 rows after the expert parks | same | mean 5.75 over 4 seeds (0 to 12) |
| 13: `+encoders` | iteration 12 with wheel inputs from simulated encoders | same | mean 18 over 4 seeds (11 to 24) |

The `wheels+live_dot` result uses simulator truth, so it is a ceiling for the camera models, not a v2 result.

## Reproduce

1. Build the simulator: `cd collection/native && pixi run cmake --build build-clang19 --target sensor_parking policy_replay`.
2. Camera inputs for the expert data: `cd v2_sensors/driving_v2 && python camera_features.py camera_features` (any Python with PyBullet).
3. Train: `python train_v2.py NAME --scale-angles --park-tail 100 --encoders --seed 7` (iterations 8 to 13). Iterations 5 to 7 use `train_camera.py` and `train_camera_ttp.py`.
4. Test: `./run_camera_tests.sh models/NAME.npz camera_loop/NAME`.
5. Open the notebook from `v2_sensors/` so its links resolve.

The camera defaults to PyBullet's CPU renderer. `CAMERA_RENDERER=egl` enables Linux/NVIDIA rendering with fixes for unspecified material colours, transparent-dot depth and half-pixel alignment. Training/evaluation renderer provenance and the remaining single-pixel occlusion differences are documented in [the golden-camera replay notes](arm/CAMERA_GOLDEN.md).
