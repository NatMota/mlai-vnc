# Original two-phase and fused policies

The merge keeps the complete two-phase implementation from `b35c559` and adds
the fused policy. All original Python files have the same Git contents as that
commit, including `collect.py`, `drive.py`, `train.py`, `test.py`, `features.py`
and the expert controller. `HOW_TO_USE.md` and the environment files are also
unchanged. README and ignore rules describe the additional workflow.

The new arm uses a velocity network to update its joint reference and a torque
network to follow that reference. The fused controller uses the later split
model's base and phase networks. The original complete model remains a separate
option; see `HOW_TO_USE.md`. See `REPRODUCE.md` for fused weights and setup.

## Resolve the collector conflict

Both branches added a different file named `collect.py`:

| Workflow | Original two-phase | Reference-memory arm |
| --- | --- | --- |
| Entry point after merge | `collect.py` | `collect_memory.py` |
| Arguments | `normal/dart COUNT` | `--runs COUNT --first-seed SEED --workers COUNT` |
| Execution | Sequential course-script subprocesses | Seeded parallel subprocesses with resume |
| Storage | `.npy` rows from the course scripts | `run_<seed>.npz` with state, end-effector position, clean expert labels, phase, joint reference and reference velocity |
| Training | Original `train.py` | `train_memory.py`, then arm ablation if required |

The original collector is kept unchanged. The memory collector is a separate
entry point, with updated documentation and worker self-invocation. For example:

```text
python collect.py normal 10
python collect.py dart 10
python collect_memory.py --runs 10 --first-seed 100000 --workers 4
```

The normal course script writes `data_normal/*.npy`; its DART script writes
`data_dart2/dart_*.npy`. The memory collector writes `data_dart2/run_*.npz`.
The loaders use different file patterns, so the two formats do not mix. Keep
the original datasets and splits when reproducing recorded results.

Five additional unit tests cover both original CLI modes, default dispatch,
seeded collection, worker environment and resume, the hidden-reference storage
contract and reading the new NPZ through the real training loader. Simulation
is mocked in these collector tests; they do not claim new collected datasets.

## Validation procedure

The integration check uses the first ten entries of the original `val` split,
with the same targets and a 30,000-step limit for both policies. It does not use
the test split. The original driver only reads a list named `test`, so a separate
copy of `split.json` routes these ten validation paths through that slot. The
original split and trajectories are never changed; its train list is preserved
for the original driver's command limits.

The original model is the epoch-4 sequential control network, not the later
split model loaded by the evaluator's `team:` option. Its checkpoint SHA256 is
`ddf6742d3f07c579975c66090f1fa65688f11153e3b4c8d663fa92162526186b`.
The fused arm checkpoint SHA256 is
`0fbba5bc46e0b64bf9fbfe7235e0b7b54bffcbdc35cc9c0b585472e86f57f525`.
Its base/phase checkpoint SHA256 is
`50cdda1db063d97f1facb4262b7a4b2060b6ab6da51bd8bded058580edea9efd`.

Use the robot Python environment, course assets and datasets described in the
two guides. The published `configs/robotnik_arm.json` has the same JSON values
as the original course configuration; only whitespace differs. A staging copy
for the original model can be made with this Python snippet from the repo root:

```python
import json
from pathlib import Path
import shutil

source = Path("data_dart")
stage = Path("outputs/merge_check_baseline")
stage.mkdir(parents=True, exist_ok=False)
split = json.loads((source / "split.json").read_text())
split["test"] = split["val"][:10]
(stage / "split.json").write_text(json.dumps(split, indent=2))
for name in ("ee_good_best.pt", "scaler_ee_good.npz"):
    shutil.copy2(source / name, stage / name)
```

PowerShell commands, using the robot environment's `python`:

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:ROBOENV_HEADLESS = "1"
$env:ROBOENV_MAX_STEPS = "30000"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
# Legacy checkpoint loading, only for the trusted checkpoint identified above.
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
python -B -m unittest discover -s tests -v
0..9 | ForEach-Object { python -B drive.py outputs/merge_check_baseline $_ }
python -B parallel_eval.py fused30:outputs/ablation/all_world/policy.pt --group val --n 10 --workers 4 --max-steps 30000 --out outputs/eval/merge_check_fused10.jsonl
```

The measured integration run launches the same single-target driver/evaluator
commands with at most four concurrent subprocesses and saves stdout/stderr per
goal. It does not upload to MLflow. The multi-target evaluator command above
also writes local tracking records only.

The original driver's errors and elapsed simulation time are printed to one
decimal place. Results from that driver retain this precision. These are two
complete-policy checks, not an arm-only ablation: their base and phase weights
also differ. Ten validation goals are a small integration check, not evidence
of a new full-test success rate. Success here means reaching within 1 cm with
the simulated end effector; grasping and lifting are not tested.

## Recorded integration results

All 27 unit tests passed: the 22 existing tests and 5 collector compatibility tests.
The complete suite was run with `python -B -m unittest discover -s tests -v`.
All tracked Python files also passed syntax parsing. `git diff --check` passed.

| Policy | Reached within 1 cm | Switched to reaching | Process errors | Mean final error (cm) |
| --- | ---: | ---: | ---: | ---: |
| Original two-phase | 6/10 | 10/10 | 0 | 1.79 |
| Fused 30 Hz | 10/10 | 10/10 | 0 | 0.46 |

All 20 subprocesses exited successfully. The original data split and all evaluated
source files kept the same SHA256 throughout the run. Source hashes, checkpoint
hashes and per-goal logs were saved locally in `outputs/v1_fused_merge/`;
they are not committed. Source was evaluated during merge preparation and checked
again before commit; the recorded pre-merge HEAD alone is not its source identity.

The original model averages below are calculated from its printed one-decimal
values, so they do not imply extra measurement precision. Success flags come
from each policy runner, not from rounding errors shown in this table.

| Goal index | Original reached | Original final (cm) | Original switch (s) | Fused reached | Fused final (cm) | Fused switch (s) |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| 0 | yes | 0.5 | 5.079 | yes | 0.253 | 5.709 |
| 1 | yes | 1.0 | 6.261 | yes | 1.000 | 7.029 |
| 2 | yes | 0.4 | 1.068 | yes | 0.085 | 1.254 |
| 3 | no | 1.9 | 8.114 | yes | 0.999 | 6.138 |
| 4 | yes | 0.7 | 1.374 | yes | 0.021 | 1.683 |
| 5 | yes | 0.3 | 2.513 | yes | 0.011 | 3.399 |
| 6 | no | 8.9 | 5.206 | yes | 1.000 | 6.303 |
| 7 | no | 1.3 | 1.152 | yes | 0.207 | 1.254 |
| 8 | yes | 1.0 | 7.703 | yes | 1.000 | 6.633 |
| 9 | no | 1.9 | 0.049 | yes | 0.054 | 0.066 |

The four original-model misses (indices 3, 6, 7 and 9) all switched to reaching
and exited normally. Each stopped with `Target not reachable: closest pose
reached`. This is the controller's stop message, not proof that the target is
physically unreachable. In particular, goal 7 prints a closest error of 1.0 cm
after rounding, but the original driver's actual success flag is false.

The validation target files, in this order, were:

```text
data_dart/dart_1792956187.npy
data_dart/dart_1792956003.npy
data_dart/dart_1792954345.npy
data_dart/dart_1792749547.npy
data_dart/dart_1792954604.npy
data_dart/dart_1792954377.npy
data_dart/dart_1792956814.npy
data_dart/dart_1792956693.npy
data_dart/dart_1792955371.npy
data_dart/dart_1792956297.npy
```

Runtime source hashes (local bytes used for these rollouts):

| File | SHA256 |
| --- | --- |
| `drive.py` | `7c855fb83f7476d23453663a2d88082c585b55cbb9e898feb502e03ca5a70034` |
| `features.py` | `abfe5387485cf6f6415dc3037a2d481271dd09e2a92cada30fc212feae8aebeb` |
| `robotnik_arm_control.py` | `f863b91c882f8570c082d4bcbd7879ac7d9d8805a4488be218b15d49d2214dcf` |
| `configs/robotnik_arm.json` | `33f9453e0a142e20e5e90b4b7ac2cced1814c9da154fd44aa7ca3d4fc8660432` |
| `parallel_eval.py` | `d26620eaa66237a99af921cc115795a862c00e65d712a0e9f7a7b96b77810bc8` |
| `fused_policy.py` | `89bff814f0aa50fb348b7860b36658066554c3acc615e4c6bb36043b2b68fe59` |
| `memory_policy.py` | `89d593d3991b04d09a3a7142a9f1a044153a54f9ff37d2243ede76d6c60e18d3` |
| `hidden_state.py` | `b8f019d6b8e33cba451c2a045747d5b891dcc1937588980ec9b480551d191ef2` |
| `sim.py` | `ace3c1d7325b47c8dc1f921e7443029fbe6c318921a80e04a172dd61cdf1c404` |
| `pool.py` | `e8ffeab40c985f876ba2ae4fc4c45fa8af9826fc4504cfb0d2b0b8503e583e41` |
