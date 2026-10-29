# Native COMP0245 expert collector

Generate complete expert trajectories on the CPU, with physics, control and recording at **1,000 Hz**. This is a separate Pixi environment and C++ port of the existing mobile-base-plus-arm expert. It leaves the coursework, notebooks and training pipeline unchanged.

The ten-target audit compared 80,010 total samples against the original Python captures. Its reports are included in `results/`; generated datasets remain outside Git. On this M1, eight-worker batch cost was about **0.64 seconds per trajectory without the unused mass-matrix diagnostic**, or **1.09 seconds including every original diagnostic**. These are batch throughput figures, not single-run latency. See [BENCHMARKS.md](BENCHMARKS.md) for measurement boundaries.

The validated route is `expert_compatible`: C++ orchestration, controller, recording and process workers calling the **same native Bullet engine build** as the original simulation. No Python interpreter runs the simulation or controller. The engine is loaded from the pinned PyBullet package; its shared library still requires `libpython` to resolve unused binding symbols. This is not a Python-dependency-free distribution.

## Run

From this folder:

```sh
pixi run build-compatible

# One straight-ahead run, including the original mass-matrix diagnostic.
pixi run collect --target 1.1 0 0.5 --binary --diagnostics --out data/my_first_run

# Eight independent worlds/processes; four complete episodes per worker.
pixi run collect --targets targets.json --workers 8 --repeat 4 --binary --out data/my_batch
```

Output folders must be new. Full runs use the original reached/stalled/step-cap rules. Repeating one target produces duplicate demonstrations; use varied targets when collecting distinct experience. `targets.json` contains the ten original inspected target positions.

`--repeat` is **per worker**: 8 workers × 4 repeats = 32 episodes. Without `--targets`, every episode uses `--target`. With a target file, targets cycle through its entries. Each episode resets to the same original starting state. `--dynamic` optionally shares a work queue among processes. `--fork-workers` loads the world once before forking independent copies, reducing repeated startup work; this path also passes the ten-target full-data audit on this Mac.

For every original diagnostic, add `--diagnostics` to the batch command. Without it, the unused 23×23 mass matrix is neither calculated nor saved; all states, commands, kinematics and controller diagnostics remain at 1,000 Hz. This omission does not change the physics or controller. Use `--help` for other options.

## Driving-only collection

```sh
# Same physics and controller; stop after saving the first navigation_done tick.
pixi run collect-driving --target 1.1 0 0.5 --binary --out data/driving_first

# Independent worlds, each reset to the original starting state.
pixi run collect-driving --targets targets.json --workers 4 --repeat 5 --binary --out data/driving_batch
```

The terminal row is included. `termination: parked_command` means the expert issued its done signal, not that every wheel has physically settled. Its 25-second navigation timeout is labelled `navigation_timeout`; it is not successful parking. A step-capped run may contain no parked label. Full-run `collect` remains available with its original termination rules.

These files retain all eight tables and the normal tensors at 1,000 Hz. This mode changes only when collection ends. On the ten checked Mac targets, every retained sample exactly matches the corresponding full-run prefix. The straight-ahead case contains 1,943 rows instead of 8,001.

There is only one done-labelled row in an episode stopped at its first done command. **Keep this terminal event when resampling to 30 Hz**; a plain every-Nth-row slice can discard it. This collector does not change the notebook's sampling, labels, loss or model.

## Data

Each episode contains eight tables: `target`, `joint_state`, `base_state`, `kinematics`, `base_controller`, `arm_controller`, `commands`, and `timing`. With `--binary`, each is an ordinary float64 `.npy` array plus a `.schema.json` file giving column names and categorical encodings. Without that flag, tables are CSV.

`q_pin.npy` and `arm_jacobian.npy` preserve the original tensor names and shapes; `--diagnostics` adds `mass_matrix.npy`. These separate NPY files replace the original compressed NPZ container. Timing measurements necessarily differ between implementations.

Rows pair the **pre-step observation with the command applied during that step**. `time_s` follows the original floating-point accumulator; the timing table's `post_step_time_s` follows the original wrapper's step-counter × dt clock. A typical 8-second run has 8,001 rows. Recorded state and labels remain float64 to preserve the reference values; convert to float32 later when forming training tensors. No 30 Hz training subsampling happens here.

To inspect a table from the existing notebook environment:

```python
import sys
sys.path.insert(0, "/path/to/mlai-vnc/collection/native/tools")
from dataset import read_table

input_df = read_table("/path/to/mlai-vnc/collection/native/data/my_first_run", "joint_state")
input_df.head()
```

`manifest.json` is written only after all arrays/tables are saved. It records the target, step size, termination reason, parking tick, source fingerprint, compiler and engine identity. Reject an incomplete folder without that manifest. An export error returns a failure status and leaves the partial folder for inspection.

## What changed

- The Python simulation wrapper, expert controller and per-tick recorder became C++.
- One consolidated state read replaces repeated joint/base queries and conversions.
- Constant finger targets are set once; Bullet retains them.
- An unnecessary duplicate Jacobian calculation was removed.
- Native BLAS/LAPACK calls preserve NumPy's calculation order, including rounding-sensitive reductions, SVD and IK solves.
- Collision geometry, masses, damping, contact settings, gains, target-height standoff and stopping rules retain the original effective behaviour, including the existing model quirks.
- Visual meshes are skipped for headless collection. Collision meshes are retained.
- Loaded worlds are reset between episodes, avoiding repeated URDF/mesh parsing.
- Optional preloaded process forks share untouched asset pages through the operating system while keeping each physics world independent.
- Binary export avoids CSV number formatting. Independent processes use one BLAS thread each.

## Validation and measured speed

See [the benchmark report](BENCHMARKS.md) and the raw JSON/logs in `results/`. Validation compares complete trajectories, labels and tensors against the original observer captures, not just final positions. Exact equality is established only for the tested targets, engine build and platform; re-run the audit after changes to any of them.

The portable-from-source `expert` executable (`pixi run standalone`) is retained as an **experimental** alternative. Its floating-point physics differs from the original binary and it is not approved as a drop-in data replacement. Likewise, `--world-experts N` puts multiple isolated robots inside one Bullet world for investigation, but changes solver ordering and did not preserve the reference trajectories. Use separate process worlds for validated collection.

The Pixi lock targets **Apple Silicon macOS and Linux x86-64**. Each platform needs its own equivalence audit; floating-point trajectories need not match across CPUs. See [Mint measurements and further optimisation](MINT-BENCHMARKS.md). GPU hardware is not used. Python scripts in `tools/` are optional validation, timing and notebook-reading utilities, not part of trajectory generation.

## Source and build provenance

`reference/` contains snapshots of the source controller/config. Observer captures are retained outside Git. `assets/` contains copies of the original models, with hashes in `asset_hashes.json`. Original coursework files were not edited.

Bullet is vendored from the official [3.25 release](https://github.com/bulletphysics/bullet3/tree/3.25); its licence remains in `vendor/bullet3-3.25/LICENSE.txt`. `vendor/source.json` records the archive URL and checksum. Its embedded Python metadata says 3.2.5 despite the conda package being Bullet/PyBullet 3.25; using the older PyPI source produces a different engine. Preserve upstream model/code licences and course attribution when sharing this project.

The numeric compatibility implementation follows NumPy's [matrix multiplication dispatch](https://github.com/numpy/numpy/blob/v2.3.4/numpy/_core/src/umath/matmul.c.src), using the same BLAS/LAPACK implementation as the reference on each platform (OpenBLAS on this Mac; BLIS plus Netlib LAPACK on Mint). `-ffast-math` is deliberately not enabled.
