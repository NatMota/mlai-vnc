# Reproduce the results

[Home](../README.md) · [Decision log](../DECISION_LOG.md) · [Iteration index](../v2_sensors/ITERATIONS.md) · [Notebook access](NOTEBOOKS.md)

Start with **saved-result verification** below. It needs Python and an 8.7 MB evidence archive, and checks results from all three team members without a simulator, GPU, Tailscale or MLflow. The following sections regenerate the statistics notebook and explain how to rerun frozen policies and training.

## Get the repository and data

Clone the `v2-sensor` branch and record its commit. Use a separate checkout for reproduction so historical scripts cannot overwrite working models.

```sh
git clone --branch v2-sensor https://github.com/NatMota/mlai-vnc.git comp0245-review
cd comp0245-review
git rev-parse HEAD
export REPO="$PWD"
export REPRO="$PWD/../comp0245-reproduction"
mkdir -p "$REPRO/downloads" "$REPRO/data"
```

Download from the school OneDrive: [reproduction supplement](https://liveuclac-my.sharepoint.com/my?id=%2Fpersonal%2Fucabdos_ucl_ac_uk%2FDocuments%2FCOMP0245%20Golden%20Dataset%202026-11-04%2FReproducibility%20supplement%202026-11-06), [golden dataset](https://liveuclac-my.sharepoint.com/my?id=%2Fpersonal%2Fucabdos_ucl_ac_uk%2FDocuments%2FCOMP0245%20Golden%20Dataset%202026-11-04), or [original iteration records and replays](https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabdos_ucl_ac_uk/IgCNPEGA3exISLAjJTeC_SOMAUkixVY1PwH50LZCMYRDoFE?e=3sej5t). OneDrive access is separate from Tailscale; sign in with an account granted access to the course folder. Save the ZIPs in `$REPRO/downloads`. Every archive has a SHA-256 hash and byte count in that manifest; the new supplement ZIPs also contain per-file `MANIFEST.json` hashes.

| Archive | Needed for | Approximate size |
|---|---|---:|
| `analysis_evidence_20261106.zip` | Original team result artifacts and the reported-value checker | 8.7 MB |
| `report_analysis_data_20261106.zip` | Statistics-notebook inputs, original analysis scripts and outputs | 1.9 MB |
| `COMP0245_Golden_300_2026-11-04.zip` | Golden expert trajectories, cameras, manifest and frozen train/validation/test split; in the parent folder | 875 MB |
| `historical_runtimes_20261106.zip` | Exact earlier Python/native source snapshots; use these for historical evaluations | 14 MB |
| `recovery_training_20261106.zip` | Fixed-noise and random-start inputs, all six DAgger collections and round models | 2.34 GB |
| `recovery_selection_20261106.zip` | Corrected round-2 combined-recovery export named in the selection record | 1.7 MB |
| `early_v2_training_20261106.zip` | Early full-arm demonstrations, 30 Hz data, IMU/camera caches and parking tails | 377 MB |
| `occlusion_records_20261106.zip` | All 1,200 paired occlusion episodes, frozen runtime, checkpoints, cases and protocol | 432 MB |
| `iteration_41_58_records_models.zip` | Full original trajectory records and models; in the existing Iterations 41–58 folder | 1.33 GB |
| `iteration_41_58_test_replays.zip` | Optional standalone test players with recorded camera images; same existing folder | 2.75 GB |

The **Reproducibility supplement 2026-11-06** folder is inside **COMP0245 Golden Dataset 2026-11-04**. Original archive labels 52–58 correspond to report iterations 51–57; the excluded old iteration 51 remains archived. Resolve models by condition name and the [iteration index](../v2_sensors/ITERATIONS.md), not archive numbers alone.

Check each downloaded file before extracting it, for example:

```sh
python3 docs/reproduction/verify_archives.py --dir "$REPRO/downloads" \
  analysis_evidence_20261106.zip report_analysis_data_20261106.zip
```

Supply other filenames to the same command as needed. [SHA256SUMS.txt](reproduction/SHA256SUMS.txt) is also usable with `shasum -a 256 -c` on macOS or `sha256sum -c` on Linux when all listed files are present.

## Verify saved reported values

```sh
python3 -m zipfile -e "$REPRO/downloads/analysis_evidence_20261106.zip" \
  "$REPRO/data/analysis-evidence"
python3 docs/reproduction/verify_reported_results.py \
  --audit "$REPRO/data/analysis-evidence" --out "$REPRO/recomputed"
```

Expected: **219 metrics; 95 matched stated values; 0 mismatches**. The remaining 124 entries are recomputed aggregates without a hard-coded reported-value assertion. The output records each input path, value, reference value and status. This independently recalculates selected preliminary arm/driving results, full-episode results, team result counts and occlusion aggregates. It does not claim that all 219 entries are separate paper claims, or that every historical model has been retrained.

## Regenerate all statistics-notebook plots

The [frozen statistics notebook](../v2_sensors/driving_v2/report_statistics.ipynb) contains the report workspace's analysis cells with a portable data-directory setting. Its original saved outputs remain available for comparison.

```sh
python3 -m zipfile -e "$REPRO/downloads/report_analysis_data_20261106.zip" \
  "$REPRO/data/report-data"
python3 -m venv "$REPRO/statistics-venv"
"$REPRO/statistics-venv/bin/python" -m pip install \
  -r docs/reproduction/statistics-requirements.txt
"$REPRO/statistics-venv/bin/python" docs/reproduction/run_statistics_notebook.py \
  --data "$REPRO/data/report-data/report-analysis/notebook/data" \
  --out "$REPRO/report_statistics_executed.ipynb"
```

All **32 code cells** were executed successfully on 6 November using the extracted archive, Python 3.14.7 on macOS arm64 and the pinned packages above. The runner creates a new notebook and fails on a cell error. This regenerates figures and statistics computed in those cells from archived analysis inputs; it does not rerun the upstream simulations or every offline diagnostic that produced those inputs.

Recalculate the closed-loop confidence intervals and paired comparisons directly from the repository's per-target CSV records:

```sh
"$REPRO/statistics-venv/bin/python" \
  "$REPRO/data/analysis-evidence/analysis-source/closed_loop_stats.py" \
  "$REPO/v2_sensors/arm/results" "$REPRO/closed-loop-statistics"
```

This regenerated all **24 condition summaries and 18 paired comparisons**, and its output JSON was identical to the notebook's packaged `closed_loop_stats.json`. [validation.json](reproduction/validation.json) records the verification scope and native smoke result.

For interactive local use, install JupyterLab in that environment, set `COMP0245_REPORT_DATA` to the same data directory, then open a copy of the notebook. The [live links](NOTEBOOKS.md) remain restricted to Tailscale.

## Run a frozen V2 policy

Use **iteration 43 / `basics_standard`** as the clean full-episode comparison, or **iteration 56 / `dart_dagger`** as the combined recovery comparison. [REFERENCE.json](../v2_sensors/REFERENCE.json) fixes the model, split and historical runtime hashes. Neither configuration is a claim of stable task success.

Extract the golden data and historical runtime package:

```sh
python3 docs/reproduction/verify_archives.py --dir "$REPRO/downloads" \
  COMP0245_Golden_300_2026-11-04.zip historical_runtimes_20261106.zip
python3 -m zipfile -e "$REPRO/downloads/COMP0245_Golden_300_2026-11-04.zip" "$REPRO/data"
python3 -m zipfile -e "$REPRO/downloads/historical_runtimes_20261106.zip" "$REPRO/data"
export GOLDEN="$REPRO/data/COMP0245_Golden_300_2026-11-04"
export HIST="$REPRO/data/overnight_20261105"
export RUNTIME="$HIST/recovery_runtime"
export MODELS="$REPO/v2_sensors/arm/results/overnight_20261105/basics_standard/models"
mkdir -p "$RUNTIME/v2_sensors/driving_30/data/expert_30hz"
cp "$REPO/v2_sensors/driving_30/data/expert_30hz/split_plan.json" \
  "$RUNTIME/v2_sensors/driving_30/data/expert_30hz/"
```

Install Pixi in its usual `~/.local/bin` or `~/.pixi/bin` location. The native [locked environment and build guide](../collection/native/README.md) target Linux x86-64 and Apple Silicon macOS. Use the archived native source and lock in a **new build copy**:

```sh
mkdir "$REPRO/native"
git archive HEAD collection/native | tar -x -C "$REPRO/native"
export NATIVE_ROOT="$REPRO/native/collection/native"
cp "$RUNTIME/native/pixi.toml" "$RUNTIME/native/pixi.lock" \
  "$RUNTIME/native/CMakeLists.txt" "$NATIVE_ROOT/"
cp "$RUNTIME/native/src/"* "$NATIVE_ROOT/src/"
cp "$RUNTIME/native/sensor_curriculum.cpp" "$NATIVE_ROOT/src/sensor_curriculum.cpp"
(cd "$NATIVE_ROOT" && pixi install --locked && pixi run configure && \
  pixi run cmake --build build-clang19 --target sensor_curriculum sensor_parking expert_compatible -j 1)
export CURRICULUM_BINARY="$NATIVE_ROOT/build-clang19/sensor_curriculum"
export CURRICULUM_SOURCE="$NATIVE_ROOT/src/sensor_curriculum.cpp"
export CAMERA_URDF="$NATIVE_ROOT/assets/summit_xl_arm_description/summit_xl_complete.urdf"
export CAMERA_RENDERER=tiny
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Start with one fixed **validation** target in a new output directory:

```sh
"$NATIVE_ROOT/.pixi/envs/default/bin/python" \
  "$RUNTIME/v2_sensors/arm/evaluate_curriculum.py" \
  --data "$GOLDEN" --models "$MODELS" --split val --ids 6 --jobs 1 \
  --out "$REPRO/iteration43-val6"
```

Inspect `completion.json`, `summary.json` and the episode `status`. An accounted-for episode with `execution_failure` is not a successful reproduction. To evaluate all validation targets, remove `--ids 6` and choose a new output directory. For the full reserved test set, use `--split test`, omit `--ids`, and again choose a new output directory. For iteration 56, change `MODELS` to the `dart_dagger/models` folder. Preserve the recorded condition JSON for a disturbance run and pass it with `--condition`; a clean run is not an equivalent substitute.

The commands above were checked on 6 November using a fresh locked native environment and build directory, extracted historical runtime and the frozen models. All three requested executables built. The one-target validation completed in about 51 seconds: with the Mac Tiny renderer, target 6 stopped at **125.56 mm**, compared with 127.05 mm in the archived Linux run; both failed the 10 mm success criterion. This also matched the earlier independent Mac smoke. Original report evaluations used Linux EGL. CPU, Bullet build and renderer changes can alter trajectories; record those differences and compare outcomes rather than promise bit-identical rollouts. The full historical test suite has not been rerun on a clean external machine.

## Retrain and trace earlier experiments

The [iteration index](../v2_sensors/ITERATIONS.md) links each iteration to its source, model and result. Use the selected model's `config.json`, data audits and history, including its seed, split, input scaling, epoch-selection rule and trainer hash. Historical source is supplied in `baseline/runtime`, `ladder_runtime`, `recovery_runtime` and `recovery_runtime_exportfix1`; match the recorded hash before running. Current branch code can be newer than a reported experiment.

For recovery training, verify and extract both recovery ZIPs into `$REPRO/data`. They supply the exact packed expert/camera inputs consumed by the trainer; duplicate raw collector exports are omitted. DART here means **fixed-noise DART-inspired augmentation**, not the full adaptive algorithm. DAgger has three rounds per family, frozen learner states and expert relabelling, with separate validation-based candidate selection. The corrected round-2 combined model is under `dart_dagger_round2/models_exportfix1`, as named in `dart_dagger_selection.json`.

The full-episode trainer accepts `--data`, `--driving`, `--out`, `--seed`, `--arm-scaling`, `--epochs`, `--stop-epochs` and `--device`. Recovery additionally uses `--extra-training`, `--dagger`, `--train-driving`, `--driving-epochs` and `--allowed-conditions`. Relocate only the paths in the saved configuration; retain the settings and validation-selection rule. PyTorch is needed for training; the recorded runs used CUDA, while `--device cpu` is available as a different compute route. No full retraining was performed as part of this documentation check.

The retained Mint training environment reports Python 3.12.3, NumPy 2.2.6 and PyTorch 2.6.0+cu124. Keep this separate from the locked native simulator environment and the statistics environment above. CUDA 12.4 wheels need compatible NVIDIA hardware/drivers; CPU retraining is a separately recorded alternative and can select a different checkpoint through numerical differences.

| Report reference | Frozen trainer directory | `train_full_episode.py` SHA-256 prefix |
|---|---|---|
| 43 `basics_standard` | `baseline/runtime` | `48ae867e795e` |
| 52 `dart`, 54 `dagger` | `recovery_runtime` | `665cede964df` |
| 56 `dart_dagger` | `recovery_runtime_exportfix1` | `03fae4cc7c5b` |

For example, after extracting the two recovery archives into `$REPRO/data`, these are the final combined-model settings. Use a Python environment with PyTorch and NumPy, set `TRAIN_PYTHON` to its interpreter, and use a new output path:

```sh
export TRAIN_RUNTIME="$HIST/recovery_runtime_exportfix1"
mkdir -p "$TRAIN_RUNTIME/v2_sensors/driving_30/data/expert_30hz"
cp "$REPO/v2_sensors/driving_30/data/expert_30hz/split_plan.json" \
  "$TRAIN_RUNTIME/v2_sensors/driving_30/data/expert_30hz/"
"$TRAIN_PYTHON" "$TRAIN_RUNTIME/v2_sensors/arm/train_full_episode.py" \
  --data "$GOLDEN" \
  --driving "$REPO/v2_sensors/arm/results/overnight_20261105/basics_standard/models/driving_seed7.npz" \
  --out "$REPRO/retrained-56" --seed 7 --device cuda \
  --epochs 1200 --stop-epochs 600 --arm-hidden 128 --arm-scaling standard \
  --lr 0.01 --chunk 32768 --train-driving --driving-epochs 300 \
  --allowed-conditions baseline random_start dart \
  --extra-training "$HIST/dart_data" "$HIST/dart_data_seed2" \
  --dagger "$HIST/dart_dagger_round1/dagger" \
    "$HIST/dart_dagger_round2/dagger" "$HIST/dart_dagger_round3/dagger"
```

This refits the final selected round using its already collected training records. Regenerating those learner-state collections is a longer experiment: use the archived `run_recovery_programme.py` and collection configs, retaining the three rounds and validation-only selection. Do not replace the fixed data with newly collected trajectories and call the result an exact replay of the original fit.

For preliminary iterations, extract `early_v2_training_20261106.zip` into a **fresh repository copy**. It restores the expected `v2_sensors/...` data layout. Run the relevant early-notebook cell or linked training script with a new model name and compare the original metric: closest approach, error at expert STOP, predicted STOP and strict success are different quantities. The 25-bearing experiments and later 100-target golden experiments have different splits and must not be combined.

The original upstream diagnostic scripts are in the analysis archives. Their docstrings/CLI arguments identify their raw inputs. `analysis-source/common_data.py` accepts `REPO_ROOT`; preserve model and data identities when relocating its inputs. The portable statistics notebook is the verified route for regenerating their published plots from the supplied analysis outputs.

## Occlusion and teammate work

The occlusion archive preserves all 1,200 original paired episode records, their hashes, case lists, checkpoints, source and geometry checks. Its historical runner records Steam Deck paths and AMD EGL. Camera PNGs are omitted from this supplement; recorded state/visibility diagnostics remain. The saved-number checker reproduces the published aggregates from the preserved episode table. A cross-platform simulation rerun is a separate check and must record the renderer and technical-gate status.

Check the table against **every raw episode** directly, without extracting the large recordings:

```sh
python3 docs/reproduction/verify_archives.py --dir "$REPRO/downloads" occlusion_records_20261106.zip
python3 docs/reproduction/verify_occlusion.py \
  --zip "$REPRO/downloads/occlusion_records_20261106.zip" \
  --out "$REPRO/occlusion_raw_verification.json"
```

This check passed for all 1,200 original hashes, result flags and distances on 6 November. Matching archived records verifies reproducibility of those measurements; it does not change the original experiment's technical-gate status.

Team implementations are pinned in the [V1 guide](V1.md) and [V2 extensions](../v2_sensors/README.md#team-sensor-extensions), with their own commands and dependencies. The evidence archive contains the original result artifacts used by the checker. Some historical teammate ablation checkpoints and exact experiment-to-model mappings were not present in the reviewed release; the main final implementations do not establish completeness of every preliminary variant. Do not treat absent logs as proof that work was not done. Full retraining coverage remains conditional on those original artifacts being supplied.
