# Code map

[Home](../README.md) · [V1](V1.md) · [V2](../v2_sensors/README.md) · [Iterations](../v2_sensors/ITERATIONS.md)

Follow collection → training → evaluation → analysis. Existing paths are retained so experiment imports and saved configurations continue to refer to the same files.

| Stage | Entry points | Purpose |
|---|---|---|
| Original policy | [Guide](../HOW_TO_USE.md), [train.py](../train.py), [test.py](../test.py) | Initial driving and arm networks |
| Expert data | [Collection guide](../collection/README.md), [native simulator](../collection/native/README.md), [fidelity and timing](../DECISION_LOG.md#appendix-e-normal-against-fast-simulator) | Record demonstrations and check the fast simulator against the course expert |
| Sensors | [Encoders](../v2_sensors/driving_v2/sensors.py), [camera rendering](../v2_sensors/driving_v2/wrist_camera.py), [camera tracker](../v2_sensors/driving_v2/camera_loop.py) | Construct sensor inputs; the filename `wrist_camera.py` also supports chassis mounts |
| Early V2 training | [Driving](../v2_sensors/driving_v2/train_v2.py), [arm](../v2_sensors/arm/train_arm.py), [STOP](../v2_sensors/arm/train_stop.py) | Iterations 1–33; the first oracle models are defined in the early notebook |
| Golden-data training | [train_golden.py](../v2_sensors/arm/train_golden.py), [driving_height.py](../v2_sensors/arm/driving_height.py), [arm_cumulative.py](../v2_sensors/arm/arm_cumulative.py) | Iterations 34–40 |
| Full-episode training | [train_full_episode.py](../v2_sensors/arm/train_full_episode.py) | Learned arm control from reset, starting at iteration 41 |
| Recovery data and training | [DART-style collection](../v2_sensors/arm/collect_recovery.py), [DAgger collection](../v2_sensors/arm/collect_dagger.py), [programme](../v2_sensors/arm/run_recovery_programme.py) | Fixed-noise augmentation and learner-state relabelling |
| Closed-loop evaluation | [full_run.py](../v2_sensors/arm/full_run.py), [golden evaluation](../v2_sensors/arm/evaluate_golden.py), [cumulative comparison](../v2_sensors/arm/evaluate_time_cumulative.py), [disturbances](../v2_sensors/arm/evaluate_curriculum.py) | Drive, PARK, reach and STOP, with the stated conditions |
| Saved-result analysis | [Golden](../v2_sensors/arm/report_golden.py), [curriculum](../v2_sensors/arm/report_curriculum.py), [notebooks](../v2_sensors/README.md#notebooks) | Read existing evidence and produce tables and figures |
| Iteration identity | [Index](../v2_sensors/ITERATIONS.md), [display-label mapping](../v2_sensors/arm/experiment_labels.py), [reference manifest](../v2_sensors/REFERENCE.json) | Resolve report numbering, unchanged condition names and frozen model hashes |

For a runnable review, use the [reproduction guide](REPRODUCE.md), [reported-value checker](reproduction/verify_reported_results.py) and [statistics notebook](../v2_sensors/driving_v2/report_statistics.ipynb).

Teammate implementations have their own dependencies and entry points: see [V1](V1.md) and [V2 extensions](../v2_sensors/README.md#team-sensor-extensions).

The root Pixi environment is for MLflow tracking. Simulation and training need the environments described by the linked guides. Current source links identify the implementation; historical evaluations also record source hashes in their configuration files. A current file is not automatically the exact historical runtime.
