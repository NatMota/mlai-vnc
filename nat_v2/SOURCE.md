# Copied from Nat's v2-sensor branch

Unchanged copies, used by `v2_sensor.py` to run Nat's sensor-only driving with the two-rate arm.

- Repository: NatMota/mlai-vnc, branch `v2-sensor`, commit `6f71a94353ce527414b127a24fda4f385638e875`
- Do not edit these files; copy them again from a newer commit instead.

| File here | Source path | sha256 |
| --- | --- | --- |
| `sensors.py` | `v2_sensors/driving_v2/sensors.py` | `e5273951b411589fa55441a86b7823e31be74fbee5f388ac4d46f10a944ff978` |
| `wrist_camera.py` | `v2_sensors/driving_v2/wrist_camera.py` | `d7beb6e2a96f574bf22070483ea6559928395a3caa2bd8206e9eca88cedc402e` |
| `camera_loop.py` | `v2_sensors/driving_v2/camera_loop.py` (reference only, not imported) | `5184ca4b5985ac429d82e823d5eb6f7dad81d953b1035f6b3b085a642a0978d8` |
| `expert_reference.json` | `v2_sensors/driving_v2/expert_reference.json` | `43d1f888fec75418a4d173efa0eddac9368eb516dd02e49bea0e7cde0c20b716` |
| `split_plan.json` | `v2_sensors/driving_30/data/expert_30hz/split_plan.json` | `7080e648ec663cc729bbf3eee93f82d1ad462fa903b03e3471a761aa29d02771` |
| `models/iter13.npz` | `v2_sensors/driving_v2/models/camera_flag_ttp_scaled_tail100_encoders.npz` (seed 7) | `880d7e0df16c453e5d911c8dce5f22288694d11e4233a8e62271ecef97b2c201` |
| `models/iter13_seed1.npz` | `..._encoders_seed1.npz` | `4c1f38c7a37983e8aa52d5f9359ab743f7846e202e542f823ca3ca659ca72200` |
| `models/iter13_seed2.npz` | `..._encoders_seed2.npz` | `a11c8f08e2d7d01c194eaed021fbecb59cda4de54b160b45b83ac67b2584fceb` |
| `models/iter13_seed3.npz` | `..._encoders_seed3.npz` | `f442af21a21e4721592c04b49ff973745eef5099647171dba94df502c3b18dbf` |

Results of the experiment: `docs/v2_sensor_results.md`.

`camera_loop.py` cannot be imported on Windows: it looks up Nat's native simulator at import time. The few
functions needed from it are copied word for word into `v2_sensor.py`, and a test checks they still match.
