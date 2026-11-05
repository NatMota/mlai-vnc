# Reproduce the final Fused 30 Hz model

## What is released

The policy is `fused30:outputs/ablation/all_world/policy.pt`.
The teammate's base and phase networks run every 33 simulator steps (about
30.303 Hz). The arm velocity network also runs every 33 steps; the torque
network runs at 1,000 Hz. The phase confirmation is two high-level updates.

The arm masks these zero-based feature columns after standardisation:
`[0, 1, 2, 3, 18, 19, 20, 26, 27, 28, 29]`. Network sizes stay unchanged.
The base and phase networks still use their original inputs and scaling.

Recorded results, computed from the saved evaluation files:

| Fixed evaluation | Success within 1 cm |
| --- | --- |
| Validation | 117/120 |
| Test | 120/120; mean final error 0.5885568224 cm |
| 500 new goals, final model | 492/500 (98.4%) |
| Same 500 new goals, expert | 497/500 (99.4%) |
| 400 goals outside the training box, final model | 279/400 (69.75%) |
| Same 400 outside goals, expert | 279/400 (69.75%) |

Success means reaching the goal with the simulated end effector. Grasping and
lifting were not evaluated. The model was selected on validation results; the
original final test evaluation was performed once. A separate telemetry replay
for plots must match that original evaluation exactly and is not model selection.

## Environment and external files

Use the existing `robotnik_arm_task` environment with PyTorch, NumPy, PyBullet,
Pinocchio, `simulation_and_control` and MLflow. The repository's default
`pixi.toml` only provides MLflow tracking tools, not the robot simulation stack.

Copy the existing course package's `models/` folder into this repository root.
Also provide the complete original `data_dart/` folder, including `split.json`
and the train, validation and test trajectories named in it. These external
folders are not stored in Git. The configuration is already at
`configs/robotnik_arm.json`. Run commands from the repository root with the
robot environment's Python.

Windows PowerShell:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
python -B -m unittest tests.test_memory_policy tests.test_fused_policy
```

## Reproduce evaluation results

Connect Tailscale and open the team MLflow experiment `robotnik_arm_task` (2).
Use run `540bdcf942a5448baefdc2802271bbf3`. Download its
`model/policy.pt` into `outputs/ablation/all_world/policy.pt`.

Run:

```text
python -B prepare_team.py
```

This reads the teammate's existing run `f978e3431988470188317b3c6efd9e29`,
downloads its pinned checkpoint, scaler and feature source into
`outputs/teammate_f978e343/`, and computes command limits from the original
`data_dart` training split. It verifies the NumPy implementation against the
teammate's torch networks. Do not run preparation during an evaluation.
The new run also contains `model/limits.npz` as an archive of the limits used.
The teammate's checkpoint and scaler are referenced, not duplicated in this run.

Then evaluate, using fresh output filenames:

```text
python -B parallel_eval.py fused30:outputs/ablation/all_world/policy.pt --group test --n 120 --workers 12 --out outputs/eval/repro_test120.jsonl
python -B parallel_eval.py fused30:outputs/ablation/all_world/policy.pt --group fresh --n 500 --workers 12 --out outputs/eval/repro_fresh500.jsonl
python -B parallel_eval.py fused30:outputs/ablation/all_world/policy.pt --group outside --n 400 --workers 12 --out outputs/eval/repro_outside400.jsonl
```

With the same model, assets, simulator and targets, the expected recorded counts
are 120/120 for test, 492/500 for fresh, and 279/400 for outside. Fresh targets
use seeds 900000 through 900499 and lie inside the training target box.
Outside targets use seeds 800000 through 800399; `target % 4` assigns each goal
to far, farther, higher or lower, respectively, with 100 goals per category.
The model reached 99, 95, 53 and 32 goals in those categories. Failed processes
remain in the denominator. These full-benchmark results come from archived
evaluations. The separate ten-goal integration check is described in
`merge_validation.md`; it does not replace the archived benchmarks.

The training target box is x/y in [-1.5, 1.5] m and z in [0.1, 1.25] m.
For far and farther, max(abs(x), abs(y)) is in [1.5, 2.0) m and [2.0, 3.0) m.
For higher and lower, x/y stay inside the box while z is in [1.25, 1.45) m and
[0.0, 0.1) m. See `fusion_results.md` for model/expert comparisons. These success
counts apply to the saved goal sets; outside-range performance is lower.

The two original fresh500 summary files predate the `dirty`, `source_sha256`
and `first_seed` fields. The two outside400 summaries include `dirty=true`,
`source_sha256` and `first_seed`. Both groups record a Git commit that does not
contain the actual code used for evaluation. All eight JSONL/summary files are
preserved byte-for-byte. The outside hashes name the original local files with
an older personal prefix; the published modules remove that prefix and update their
imports, so published file hashes differ. The run's `git_commit` tag identifies
the latest publication, and `initial_publication_git_commit` preserves the initial
release commit. Do not treat either summary's commit as a complete source record.

## Collect data and retrain the arm

These instructions create a new training attempt. They do not promise identical
weights or scores from freshly generated data.

The expert data collector records hidden joint-reference memory:

```text
python collect_memory.py --runs 4000 --first-seed 100000 --workers 12
```

This collects seeds 100000 through 103999 into `data_dart2/`. The original local
folder actually contains 4,009 `.npz` runs: these 4,000 plus seeds
`1, 2, 200000, 200003, 200005, 200006, 200007, 200008, 200009`.
Simply collecting 4,000 runs therefore does not recreate the original dataset
file for file. Use the archived data if an exact comparison is required; changing
filenames or folders also changes the CRC32 validation split and sampling order.

Download `reproduce/policy.pt` from the new MLflow run into
`outputs/memory-v1/policy.pt`. This is the historical first-round DAgger student,
not the final deployed arm model. Generate the original seed range and relabel:

```text
python dagger.py memory:outputs/memory-v1/policy.pt --runs 400 --first-seed 300000 --out data_dagger/iter1 --workers 12
python dagger.py memory:outputs/memory-v1/policy.pt --first-seed 300000 --relabel-from data_dagger/iter1 --out data_dagger/iter1_relabel2
```

This range is 300000 through 300399, as recorded in the original collection
manifest. The shadow expert uses the student's reference memory for arm labels;
the relabelling step updates base labels from saved states.

Finally run the all-world-input ablation. Keep the final checkpoint intact and
write the new attempt to a separate directory:

```text
python ablation.py --groups all_world --from outputs/ablation/all_world/policy.pt --output outputs/ablation_retrain
```

`--from` supplies unchanged command limits and unused donor base/phase networks.
Both arm networks are freshly trained; deployed base/phase still come from the
teammate via `make_fused`. Using the published final checkpoint as this donor
avoids requiring an older v5 checkpoint or accidentally skipping retraining when
the default output already exists.

The recorded confirmation states that the original ablation invocation supplied no command-line
options and redirected stdout/stderr to `outputs/ablation_train.log`. It trained
all six input groups, including `all_world`, using the script defaults:
seed 0, stride 20, good-run limit 2 cm, phase-label advance 200 ms, 12 epochs,
batch 2048, learning rate 0.002, 3 hidden ReLU layers of width 256, and 20 threads.
The data folders were `data_dart2` and `data_dagger/iter1_relabel2`; the original
donor was `outputs/memory-v5/policy.pt`. These settings are based on the supplied
original command and the confirmed unchanged defaults; no separate argument
snapshot was saved during training. The histories and network shapes provide
additional checks for the 12 epochs and architecture.

The checkpoint's top-level `settings` are inherited donor metadata (20 epochs,
stride 10, base-only training) and must not be presented as the arm's settings.
Any archived learning-rate curve is reconstructed using AdamW with weight decay
`1e-5` and OneCycleLR with max learning rate `0.002`, `pct_start=0.05`, and the
other PyTorch defaults. Each batch calls the scheduler once; the reported value
is taken after the last batch in each epoch. The velocity network has 307
batches per epoch (628,483 training rows); torque has 727 (1,487,332 rows).

All collector, training and evaluation tracking through `local_tracking.py`
stays local. Data, weights, outputs, local MLflow files and environment files are
excluded from Git. See `fusion_results.md` for the experiment comparisons.

Published source and test filenames use neutral names. The historical v5 guide
is at `docs/memory_policy.md`; its commands use the current filenames. Use this
document for the fused release. The original two-phase collector remains at
`collect.py`; the reference-memory collector has its own `collect_memory.py`
entry point so both workflows can be used independently.
