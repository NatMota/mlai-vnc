# Copied from Nat's v2-sensor branch (golden test)

Unchanged copies used by `golden_sensor.py`, which tests the exported two-rate arm after Nat's driving network on the
100 golden test dots.

- Repository: NatMota/mlai-vnc, branch `v2-sensor`, commit `d89051b4f971c54d59c81bf71f9f92cb4f8e2833`
- Each file was written with `git show` and checked against its Git blob hash. Do not edit these files.

| File here | Source path | sha256 |
| --- | --- | --- |
| `camera_loop.py` | `v2_sensors/driving_v2/camera_loop.py` (reference only, not imported) | `1fa00e61e61c3b05bd2bc2bd3fb23dd0e6462ee14e728ea5b93b2221ad1a26ec` |
| `sensors.py` | `v2_sensors/driving_v2/sensors.py` | `e8d59c74670ab0643edee7e4d5598bd47d93b1846cc2f16af9dd67235415959e` |
| `wrist_camera.py` | `v2_sensors/driving_v2/wrist_camera.py` | `53b325fd5a8360b8c26436a9454b113704ead71b5f7e51837a3dc5e45357314e` |
| `full_run.py` | `v2_sensors/arm/full_run.py` (reference only, not imported) | `a099cabfe4479cf25bd59665e16717b8634a7feeb1b647f1dc39df9bf674514d` |
| `models/driving_seed7.npz` | `v2_sensors/arm/results/overnight_20261105/basics_standard/models/driving_seed7.npz` (iteration 43) | `458fc825c3d3c1e5d39d3630655eb872bb46f3ddc34e0a850ee4eb5da52a18c3` |
| `expert_test.json` | `.../overnight_20261105/basics_standard/test/expert_test.json` | `12eb02ece040351c7b7fec4f551d99017b77fd4ef81a5565882564764ae87075` |
| `expert_val.json` | `.../overnight_20261105/basics_standard/val/expert_test.json` | `11d0b4da2fee075fd77b7799e80e2fceb7a3dccb00f75d998220b97c4a4b163e` |
| `seed7_metrics.csv` | `.../overnight_20261105/basics_standard/test/seed7_metrics.csv` | `ba2c8b8f162400ebd9460447b52d9f23d8d9f0bf54888fedca2f8d8dfe548e13` |
| `test_config.json` | `.../overnight_20261105/basics_standard/test/config.json` | `b2bf96e7a358ee9a5f4b205c2330d52d22f8efef3514ca159f4ed2e447f24ec1` |
| `publication_scores.csv` | `v2_sensors/arm/results/overnight_20261105/publication_scores.csv` | `97d81b07942156fef0c5ee7de0160ff095721dd7312e1416c640dbe625adf376` |

`camera_loop.py` and `full_run.py` cannot be imported on Windows (they look up Nat's native simulator at import
time). The functions needed from `camera_loop.py` are copied word for word into `golden_sensor.py`; a test checks
they still match. Scoring follows `full_run.py`:
- success = parked, the arm's own STOP issued, and the end-effector within 10 mm of the target at the STOP step;
- the run continues 0.5 s after STOP;
- without a STOP, the run ends at the expert's stop time + 4 s.
