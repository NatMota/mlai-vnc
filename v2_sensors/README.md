# v2 sensors: driving to the dot without simulator truth

v1 gave the network the robot's world pose and the dot's world position. v2 uses only what the robot could sense: wheel encoders and one camera. The outputs stay the same as the expert's: four wheel speed commands and PARK. Arm torques are not covered yet.

## Model to build from

`camera+flag+time_to_park` (`driving_v2/models/camera_flag_ttp.npz`). PARK at 0.25 s or less. Parked 16 of 25 test bearings.

## What is here

- `driving_v2/driving_v2_sensors.ipynb`: every iteration, its training curves and its test results, with plots.
- `driving_30/data/expert_30hz/`: the expert driving dataset. 30 trajectories, one per bearing (`sampled/` at 30 Hz, `full_rate/` at 1 kHz), the split (`split_plan.json`: 21 train, 4 test, 5 holdout) and the run records (`episodes.jsonl`).
- `driving_v2/camera_features/`: the camera inputs computed for each expert row (distance, sin, cos, visible flag).
- `driving_v2/models/`: all seven networks as `.npz`.
- `driving_v2/camera_loop/`: per-bearing test results for the three camera models.
- `driving_v2/closed_loop_native/`: test results for the four earlier models, which were given the true dot.
- `driving_v2/expert_reference.json`: the expert's park time and final pose per bearing, used to score every test.
- `../collection/native/`: the fast simulator. `sensor_parking` steps the physics for the camera tests; `policy_replay` runs a whole network inside the simulator.

## The iterations

All use the same network (2×16 ReLU), Adam, lr 0.01, 300 epochs, seed 7, and the same 21 training runs.

| Iteration | Dot input | PARK output | Parked /25 |
|---|---|---|---|
| `wheels` | none | yes/no | 12 |
| `wheels+start_dot` | true direction at the start | yes/no | 11 |
| `wheels+live_dot` | true position every step | yes/no | 23 |
| `wheels+live_dot+skid` | true position every step, skid-steer output | yes/no | 9 |
| `camera+flag` | camera + visible flag | yes/no | 17 |
| `camera+flag+time_to_park` | camera + visible flag | time to park, PARK at 0.25 s | 16 |
| `camera+time_to_park` | camera | time to park, PARK at 0.35 s | 6 |

The `wheels+live_dot` result uses simulator truth, so it is a ceiling for the camera models, not a v2 result.

## Reproduce

1. Build the simulator: `cd collection/native && pixi run cmake --build build-clang19 --target sensor_parking policy_replay`.
2. Camera inputs for the expert data: `cd v2_sensors/driving_v2 && python camera_features.py camera_features` (any Python with PyBullet).
3. Train: `python train_camera.py` and `python train_camera_ttp.py` (add `FLAG=0` for the model without the flag).
4. Test: `./run_camera_tests.sh models/camera_flag_ttp.npz camera_loop/camera_flag_ttp`.
5. Open the notebook from `v2_sensors/` so its links resolve.

The camera uses PyBullet's CPU renderer. The GPU renderer colours parts that have no colour in the robot files (wheels, arm ring) red, green, blue and yellow, which confuses the red detector.
