# Collect expert data at 1 kHz and sample it at 30 Hz

`native/` contains the CPU-only C++ expert collector, robot assets, pinned Pixi
environment and numerical-equivalence audits. `sampling/` contains the compact
collection and preprocessing code used for the 30-bearing experiment. These
tools are separate from the team's existing training code and tracking setup.

## Build and collect

The native collector supports Apple Silicon macOS and Linux x86-64. From the
repository root:

```sh
cd collection/native
pixi run build-compatible

# One driving episode, ending on the expert's first PARK command.
pixi run collect-driving --target 1.1 0 0.5 --binary --out data/first_drive

# Eight independent workers, four episodes per worker: 32 actual executions.
pixi run collect-driving --targets targets.json --workers 8 --repeat 4 \
  --fork-workers --binary --out data/driving_batch
```

Output directories must be new. `pixi run collect` runs the complete driving
and reaching expert; `collect-driving` ends after recording the first
`navigation_done` tick. That command does not mean physical drift has stopped.
Timeouts and step caps remain distinct termination reasons.

The control/physics timestep stays at 1 ms. The C++ loop removes per-tick Python
overhead, consolidates state reads, reuses loaded worlds and exports binary
arrays. Process workers have independent physics worlds. `--diagnostics`
retains the otherwise unused mass-matrix diagnostic. See the native
[README](native/README.md), [Mac benchmarks](native/BENCHMARKS.md) and
[Linux benchmarks](native/MINT-BENCHMARKS.md) for the tested equivalence and
throughput boundaries. Benchmarks describe prior measured sessions, not a
speed guarantee for another machine.

## Collect compressed, sampled data

After building the collector, from the repository root:

```sh
cd collection/sampling
pixi run check
pixi run python collect_shard.py --host mac \
  --native-root ../native --output data/mac
# Or run --host mint --output data/mint on Linux.
pixi run python prepare_dataset.py data/mac
```

Each host collects 50 executions at each of 30 bearings, 12 degrees apart:
1,500 executions per host. Running both hosts produces the original 3,000-run
allocation. `--host` selects worker/batch sizing and repetition IDs; it is not
a network connection. Collection resumes from the output's audit log.
Preparing one shard processes that shard only; copying both directories does
not automatically merge them. `dataset.prepare()` also accepts a combined
audit log whose archive paths resolve beneath its output directory.

The pipeline retains:

- A separate audit record for every execution, with source/engine identity.
- Compressed 30 Hz FP32 observations for every execution, deduplicated only
  when retained signals match exactly. Noise is disabled in this experiment.
- The terminal PARK event, even when it falls between regular sample times.
- Four relative cumulative wheel angles, four wheel speeds, four derived
  movement features, and sine/cosine of the requested initial target bearing.
  This bearing is an explicit goal input, not a camera measurement.
- Simulation steps and timestamps as metadata. Inputs are not scaled.
- Wheel-command and PARK recordings for inspection, plus sampled base poses
  and joint positions for replay. No training objective is selected here.
- Full-rate float64 driving fields for the first five repetitions per bearing
  on each host; mass matrices, Jacobians and dynamics diagnostics are omitted
  from these compact archives.

Sampling selects the nearest available 1 ms row; it does not filter or average
the signal. The deterministic split groups whole bearings: 21 training, four
test and five holdout, with five folds inside training. Exact repeats remain
in the audit but enter the prepared fitting table once. Prepared arrays are
NumPy files; PyTorch can convert `x` with `torch.from_numpy()`.

Installed environments, generated data, notebooks, model checkpoints and
unfinished outcome-loss training are excluded from this branch.
