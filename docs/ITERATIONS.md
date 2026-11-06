# Review and move between iterations

Start at the release tag `comp0245-reproduction-2026-11-06-r2`. Use a clean
checkout for reproduction and keep downloaded result archives separately.
Each new experiment gets a new output directory. Never replace a published model
or result to make a new invocation look like the old experiment.

## Code map

| Entry | Responsibility |
| --- | --- |
| `features.py`, `memory_policy.py` | Features and four-network controller |
| `train_memory.py`, `ablation.py` | Supervised training and removed-input groups |
| `fused_policy.py` | Nat's base/switch with the selected arm |
| `sim.py`, `robotnik_arm_control.py` | Robot simulation and expert controller |
| `parallel_eval.py`, `pool.py` | Per-target workers, results and summaries |
| `iteration_io.py` | Atomic manifests and exclusive output locks |
| `v2_sensor.py`, `v2_eval.py` | Sensor-driven V2 configurations and evaluation |
| `golden_sensor.py`, `golden_eval.py`, `golden_compare.py` | Frozen golden protocol and comparison |

## Experiment index

The tag supplies the current readable entry points for all rows. Historical
scores are evidence in the linked reports; a fresh run can differ by environment.

| Iteration | Model/data | Entry and output | Evidence |
| --- | --- | --- | --- |
| Memory v5 | memory-v5 donor; expert NPZ + DAgger relabel2 | `train_memory.py`; a new `outputs/iterations/v5-*` | [Memory guide](memory_policy.md) |
| Fused30 / ablation | Nat base/switch + one of 12 arms; original split targets | `ablation.py`, `parallel_eval.py`; new training/eval paths | [Fused reproduction](REPRODUCE.md), package 2 result table |
| V2 sensor | exported all_world arm; fixed Nat sensor snapshot | `v2_eval.py`; `--out-dir outputs/iterations/v2-review` | [V2 results](v2_sensor_results.md) |
| Golden arm | all_world arm from run 540bdcf942a5448baefdc2802271bbf3; `nat_golden/SOURCE.md` | S then protocol then P/G; fixed `outputs/golden` in a separate checkout | [Golden reproduction](REPRODUCE_GOLDEN.md), [results](golden_results.md) |

Historical source anchors are `c670a2c` (v5), `80e409277744ac238352bbe3d7babc5650c2d97a`
(V2 publication), and `e407006882467c942ef8da85553a1fd18a6fe36d` (initial golden
publication). They describe source history, not proof that a historical dirty
working tree was identical. Saved result hashes and source provenance are retained.
Nat's golden source snapshot is `d89051b4f971c54d59c81bf71f9f92cb4f8e2833`.

## Enter, stop and resume an ablation iteration

```powershell
python -B ablation.py --groups none all_world --seed 1 --output outputs/iterations/abl-seed1
python -B ablation.py --groups none all_world --seed 1 --output outputs/iterations/abl-seed1 --resume
```

Run the second command only when continuing the first. A completed group is
reused only if its saved configuration matches the requested settings, donor,
input files and executable source hashes. The program prints `SKIP` explicitly.
An unfinished group restarts from its first epoch. Optimizer/scheduler/RNG state
is not saved, so this is not mid-epoch training recovery. Changing seed, input
data, training settings or code requires a new output directory.

The checkpoint now records current `settings`, separate `donor_settings`, and
`training_config`; the same configuration is saved beside it. The historical
released checkpoints retain their original bytes and inherited settings.
Training validation still uses CRC32 of the run name modulo 10: preserve data
folder names and file names when comparing runs. Do not change input files during training.

## Enter, stop and resume an evaluation

```powershell
python -B parallel_eval.py fused30:outputs/ablation/all_world/policy.pt --group val --n 120 --out outputs/iterations/eval-001.jsonl
python -B parallel_eval.py fused30:outputs/ablation/all_world/policy.pt --group val --n 120 --out outputs/iterations/eval-001.jsonl --resume
```

Ctrl+C leaves completed rows on disk. Resume verifies the manifest, source/model
hashes and selected target-file hashes, then runs only missing indices. Process
errors remain failures and are not silently retried. A finished evaluation is
read without rewriting its files or logging it again. The summary appears only
after every target has a row and includes the JSONL hash. `wall_s` measures the
current invocation, not accumulated time across interruptions.

An existing output without a matching manifest is an archive, not resumable state.
Duplicate, malformed or torn JSONL rows fail closed. Preserve the evidence and
start a separate development iteration; do not repair official results in place.
An exclusive `.lock` prevents simultaneous writers. A hard process kill can leave
it behind: verify that the PID recorded in the lock and all its workers have
stopped before removing only that stale lock. Do not delete manifests or receipts.

## V2 and golden boundaries

V2 accepts a dedicated `--out-dir` and reuses matching stamped rows. Avoid
`--fresh` on any output you need to preserve. For golden development, use
`--validation-case 6 --out-dir work/validation6` in a clean checkout.
S/P/G reproduction is documented separately. P/G keep the fixed directory and
once-only attempt ledger. A started case without a durable receipt requires
inspection; the evaluator intentionally refuses automatic replay. Do not delete
attempts, weaken provenance checks or tune the protocol on G.

Before switching code, stop workers, keep the exact command, commit, manifests,
checksums and output directory together, then use a separate clean checkout for
another release. Existing checkpoints and results are not migration inputs.
Training/evaluation tracking is local; team publication must create a new MLflow
run. Never append revised material to a previous published run.
