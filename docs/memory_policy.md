# Memory policy v5 (seed 0)

## Results

| Evaluation | Successes within 1 cm |
| --- | --- |
| v5 validation | 34/40 (85.0%) |
| v5 test | 98/120 (81.7%) |
| Expert validation | 120/120 (100%) |
| Expert test | 120/120 (100%) |

The v5 test set was evaluated once after validation-based development, as
reported in the experiment record. All 120 target results are present, with
zero process errors. Of 104 targets that switched to reaching, 98 succeeded
and 6 failed. The remaining 16 failures never switched. Mean final test error
is 5.4067 cm; median final test error is 0.9989 cm. Success measures simulated
end-effector position; grasping and lifting a ball were not evaluated.

These are the existing recorded results, not new simulations performed during
publication. No earlier model checkpoints or exploratory experiments are
included in this release.

## Method

The policy has four learned networks: base, phase, arm velocity and arm torque.
The velocity network predicts a joint reference velocity. Integrating it creates
a joint reference angle, which acts as memory. The torque network uses the joint
reference error and velocity error to control the arm. Velocity is scaled by the
end-effector distance to the target, so corrections shrink near the target.
Reference lead and joint-limit bounds prevent excessive integration.

DAgger labels student-visited states with a shadow expert. Arm labels start from
the student's current reference memory. Base labels restart navigation after
drifting away, and consistently use final-heading alignment near the parking
point. Phase training uses expert-only data, with switch labels moved 200 ms
early. At runtime, switching requires 50 consecutive positive predictions and
is latched after reaching starts.

The published v5 checkpoint contains all four networks, input/output scaling,
command limits and settings. No separate scaler or earlier checkpoint is needed
to run it. The recorded v5 training step updated the base network while reusing
the other three networks; this lineage remains in its original summary metadata.
It does not require distributing an earlier checkpoint for inference.

## Files

- `memory_policy.py`: network loading, NumPy inference and reference memory.
- `sim.py`: simulation and expert controller adapter.
- `parallel_eval.py`, `pool.py`: complete parallel evaluations.
- `collect.py`: expert data collection with reference memory recorded.
- `dagger.py`: shadow-expert collection and base relabelling.
- `hidden_state.py`: reference reconstruction helper and phase-label utility.
- `train_memory.py`: training and checkpoint creation.
- `local_tracking.py`: local-only MLflow tracking during training/evaluation.
- `features.py`: shared robot-view and end-effector features.
- `robotnik_arm_control.py`: shared expert controller functions.
- `configs/robotnik_arm.json`: simulation configuration used by this release.
- `tests/test_memory_policy.py`: policy, reference and labeler checks.

## Environment and assets

Use the existing COMP0245 robot environment with Python 3.11, NumPy, PyTorch,
PyBullet, Pinocchio, `simulation_and_control` and MLflow. The recorded training
environment used PyTorch 2.14.1+cpu and NumPy 2.4.6. The repository's default
`pixi.toml` installs tracking tools only; it does not install the robot stack.

Use the same course robot asset folders under `models/`, including
`summit_xl_arm_description`, `summit_xl_description` and their mesh dependencies.
The release provides the configuration, but does not redistribute the course
assets or raw training/target trajectories. Obtain those from the team's shared
course package. Keep the original `data_dart/split.json` and target `.npy` files
at the paths listed in that split for the same evaluation targets.

On Windows PowerShell, set `$env:KMP_DUPLICATE_LIB_OK = "TRUE"` and use the robot
environment's Python. Run commands from this checkout's root.

## Download and evaluate v5

Download `model/policy.pt` and `model/summary.json` from the new team MLflow run
`8c619cf3c990461fb62da686a31fabe0` into `outputs/memory-v5/`. Download
`model/split.json` into `data_dart/` alongside the original target trajectories.
The checkpoint SHA256 is:

```text
c272f03b626ac086fd09d07780934e8a4999f17cff658aff7b26e9e509a6b147
```

Use fresh filenames when running an evaluation, to preserve existing evidence:

```text
python -B -m unittest tests.test_memory_policy
python -B parallel_eval.py memory:outputs/memory-v5/policy.pt --group val --n 40 --workers 12 --out outputs/eval/recheck_v5_val40.jsonl
python -B parallel_eval.py memory:outputs/memory-v5/policy.pt --group test --n 120 --workers 12 --out outputs/eval/recheck_v5_test120.jsonl
```

Check success counts, final errors, process errors and phase switching. A crashed
or timed-out target counts as a miss. Use fewer workers if memory is limited.
Evaluation writes only to local MLflow through `local_tracking.py`.

## Collect and train

Collection writes to `data_dart2/`. This is the generated dataset, distinct
from the teammate's separately generated folder of the same name. Keep datasets
and checkpoints outside Git.

```text
python -B collect.py --runs 2000 --first-seed 100000 --workers 12
python -B dagger.py memory:outputs/memory-v5/policy.pt --runs 600 --first-seed 300000 --out data_dagger/new_collection --workers 12
python -B dagger.py memory:outputs/memory-v5/policy.pt --first-seed 300000 --relabel-from data_dagger/new_collection --out data_dagger/new_relabel
```

These are example new collection batches, not a reconstruction of every original
training trajectory. For the recorded v5 base-training step, use the original
`data_dart2` and `data_dagger/iter1_relabel2` datasets and settings from
`model/summary.json`. All frozen networks are available in the published v5
checkpoint, so it can supply them for a base-only retraining attempt:

```text
python -B train_memory.py --data data_dart2 data_dagger/iter1_relabel2 --output outputs/memory-v5-retrain --seed 0 --stride 10 --good-cm 2 --epochs 20 --batch 2048 --lr 0.002 --width 256 --depth 3 --threads 20 --early-rows 200 --only base --from outputs/memory-v5/policy.pt
```

Do not claim exact retraining from new random data. Preserve collection manifests,
original training data and environment details when comparing results. The
historical commit in the training summary records training-time HEAD; the new
run's `git_commit` tag identifies this published code snapshot.
