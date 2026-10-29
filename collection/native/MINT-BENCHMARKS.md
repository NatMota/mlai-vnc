# Mint CPU collection and driving-only runs

Measured 29 October 2026 on Linux Mint, Intel Core i7-6700K (4 physical cores, 8 logical CPUs), 32 GB RAM. The GPU was not used. The isolated project is stored on the Fast Drive and is accessible as `~/Projects/comp0245-native-expert`.

**Recommended collection: eight independent, preloaded workers with binary output.** Straight-ahead driving batches took about **0.19–0.20 seconds per saved episode** on disk. RAM-backed output reduced this to **0.175 seconds per episode**. A single fresh driving process took **1.39 seconds**. Batch throughput and individual latency are different measurements.

The machine was busy throughout: existing RViz processes, a Python process and other desktop services remained running. The recorded load average was roughly 8–11. These are measurements under the current workload, not idle-machine specifications. Raw timings, manifests and hashes are in `results/mint/` on the Mac and `results/` on Mint.

## Timing results

The straight-ahead target is `[1.1, 0, 0.5]`. Full trajectories contain 8,001 physics/control samples; driving-only trajectories contain 1,943. Both execute and record at 1,000 Hz.

| Workers | Full trajectory, seconds/episode | Driving only, seconds/episode |
|---:|---:|---:|
| 1 | 3.381 | 0.838 |
| 2 | 1.980 | 0.471 |
| 4 | 1.146 | 0.290 |
| 8 | 0.882 | 0.216 |

Full batches used two episodes per worker; driving batches used four. These figures include native process startup, world setup, simulation, recording and file export, divided by the number of episodes. They exclude compilation, environment installation, Pixi launcher overhead, subsequent validation and duplicate-output cleanup. Files are closed, but no explicit disk `fsync` is forced. The unused mass-matrix diagnostic is omitted in this table.

At eight workers, collecting the full 23×23 mass matrix at every tick increased the full-run figure to **1.377 seconds/episode**. The original Python expert, without profiling or dataset export, took **10.48–10.88 seconds inside `main()`**, or **13.91–15.58 seconds including imports**. That Python measurement is single-process latency and should not be divided directly by the eight-worker throughput figure to claim a per-process speedup.

| Additional experiment | Result | Interpretation |
|---|---:|---|
| Driving, 8 preloaded workers, disk | 0.200 and 0.193 s/episode | Preloading helped modestly. |
| Driving, 8 workers, RAM output without preloading | 0.211 s/episode | Small gain versus 0.216 s; could include workload noise. |
| Driving, 8 preloaded workers, RAM output | 0.1749 and 0.1756 s/episode | About 9–12% faster than the preloaded disk measurements. |
| One driving episode in a fresh process | 1.393 s | Startup is substantial; this is not below 500 ms latency. |
| Ten targets, each repeated twice, 4 preloaded workers | 8.291 s total; 0.415 s/episode | Varied-target throughput; episodes have different durations. |
| Same varied targets with dynamic job allocation | 8.956 s total; 0.448 s/episode | No improvement in this small batch; retain static allocation by default. |

Repeated identical targets are useful for timing and determinism checks, but are duplicate demonstrations. The varied-target test contains ten distinct targets, not twenty distinct scenarios.

## Driving-only stopping rule

`collect-driving` selects `--driving-only`. It runs the same controller, arm holding behaviour and complete robot physics as the full collector. After the first tick where `navigation_done` is true, it saves that tick and exits. It does not wait for the arm to reach the target.

- `parked_command`: the expert completed its orientation criterion and issued the done signal.
- `navigation_timeout`: the original expert forced completion after its navigation timeout; do not count it as successful parking.
- `step_cap`: the original cap ended the run, potentially before parking.

The done command does not prove that physical wheel drift has stopped. The first done tick also activates the original arm controller; its command is preserved in the terminal row. There is no additional settled tail in this mode. `collect` retains the original full-run stopping rules.

Only one row is labelled done in a trajectory stopped at its first done command. Preserve that terminal row explicitly when converting to the notebook's 30 Hz training samples. No notebook, tensor preparation, model, training loss or sampling policy was changed here.

## Verification

- Ten Mac driving trajectories exactly matched their previously verified full-run prefixes: all retained tables, numeric arrays, labels and simulation timestamps.
- The final Mac build still matched the original straight-ahead prefix. A full-run regression also matched the preceding full-run implementation.
- Two complete Linux trajectories, straight ahead and left turn, exactly matched fresh captures from the unchanged Python expert running on Mint. This includes every numeric value in the seven data tables, simulation timestamps, mass matrices, Jacobians and Pinocchio configurations. Wall-clock measurements are naturally different.
- Linux driving prefixes for both those targets exactly matched the Linux reference captures.
- Across 339 benchmark episodes, repeated non-timing arrays were identical for the same target and duration mode, including ordinary workers, preloaded forks, RAM output and dynamic scheduling. This establishes repeatability for these cases, not generalisation to arbitrary targets.
- Original source hashes matched the Mac snapshots. Full physics, gains, contact settings, missing-inertia behaviour and the 1 ms time step were retained.

The Linux and Mac physics binaries differ. A left turn parks at step 4,578 on Linux versus 4,589 on the Mac. Each port matches its own platform's original expert; do not require cross-platform bitwise identity or silently mix these as the same numerical reference.

Audit evidence: `results/linux-audit-ahead.json`, `results/linux-audit-left.json`, `results/linux-driving-prefix.json`, `results/linux-driving-left-prefix.json`, `results/driving-prefix-mac.json`, and `results/mint-validation.json`.

## Run on Mint

```sh
cd ~/Projects/comp0245-native-expert

# Install/build once; the environment is already installed on this machine.
~/.local/bin/pixi run build-compatible

# New folder required. Eight workers × four episodes = 32 episodes.
~/.local/bin/pixi run collect-driving --targets targets.json \
  --workers 8 --repeat 4 --fork-workers --binary --out data/new_driving_batch

# Full-run version remains available.
~/.local/bin/pixi run collect --targets targets.json \
  --workers 8 --repeat 4 --fork-workers --binary --out data/new_full_batch
```

For RAM-backed output, use a new path under `/dev/shm`, for example `--out /dev/shm/comp0245-new-driving-batch`. Copy completed datasets to persistent storage before ending the session or rebooting; RAM output is temporary. Benchmark RAM files have already been copied, hash-verified and preserved under `data/ram-benchmark-saved` on Mint, then removed from RAM.

The Linux build includes two portability fixes: quoting the generated Clang sysroot path when the environment lives on “Fast Drive”, and binding Bullet to its own bundled XML parser rather than incompatible symbols loaded by Pinocchio dependencies. The original coursework environment was not changed. The Linux reference uses BLIS with Netlib LAPACK; the Mac reference uses OpenBLAS. Each native build follows its platform's reference arithmetic.

## Remaining speedups and the literature

**Most remaining time is physics.** In the one-worker Mint driving measurement, the summed physics-step timer was 96.7% of the collection-loop time. Even eliminating all other loop work would improve that loop by only about 3.4%; export and startup are separate costs. This bounds the likely benefit of further vectorising the small controller calculations.

| Approach | Current status and next use |
|---|---|
| Keep loaded worlds and workers alive | Implemented through repeats, reset and optional preloaded forks. Use batches rather than launching one command for every demonstration. |
| Keep samples in memory | Already implemented: observations accumulate in RAM and are exported after the trajectory. There is no CSV write or Python IPC at every tick. |
| RAM output | Tested above. Useful when a trainer consumes data immediately; archive it afterward. |
| Memory-mapped dataset reads | Existing NPY files can be opened with `np.load(path, mmap_mode='r')`, avoiding an immediate whole-file copy. This helps data consumption, not physics speed; no training-speed claim was measured. |
| Direct shared-memory producer/consumer | A possible next step for continuous collection and training: fixed buffers could remove file export and reloading. It needs an explicit consumer API and a benchmark; not implemented here. |
| Vectorise across independent robots | Independent process worlds are already measured. Consecutive 1 ms steps depend on previous states and cannot simply be evaluated together. A SIMD redesign of Bullet's multi-world solver would be a separate numerical-equivalence project. |
| Dynamic scheduling | Available, but did not improve this small mixed-target batch. It may help with a wider distribution of episode lengths; measure before making it the default. |
| A quieter CPU window | Worth testing before another rewrite. Existing desktop CPU work was substantial and was not stopped for these measurements. |

[EnvPool (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html) uses native execution, asynchronous scheduling and batched buffers to reduce environment overhead. Those are relevant design patterns. Its reported gains are for its own Atari/MuJoCo workloads, not a further multiplier that can be applied to this already-native Bullet collector. Our entire feedback loop stays within each process, so it already avoids per-step Python and inter-process messaging.

[MuJoCo's rollout documentation](https://mujoco.readthedocs.io/en/stable/python.html#rollout) provides another concrete implementation of native rollout loops, reusable thread pools and caller-supplied output buffers. Its rollout helper executes supplied control sequences; this expert still requires feedback at every tick. Moving to MuJoCo would also change the contact and dynamics implementation and require new validation. It is an alternative experiment, not an interchangeable acceleration of this dataset.

The next practical improvement is continuous in-memory consumption if collection becomes part of a live training loop. For the current offline notebook, preloaded worker batches and binary output provide the useful gains without adding another service or changing the expert.

## M1 RAM comparison, measured afterward

The same eight-worker, preloaded, driving-only configuration was measured on this 16 GB M1 using a temporary 512 MB HFS+ RAM disk. Two batches of 32 straight-ahead runs took **4.652 s and 4.111 s**, or **145 ms and 128 ms per saved episode** (6.88–7.78 episodes/second). A fresh single process took **847 ms**, including its output. Each episode contains 1,943 samples at 1,000 Hz. All 65 benchmark episodes matched the already validated Mac numerical arrays exactly, excluding wall-clock timing.

Mint's comparable RAM batches took about 175 ms per episode. The machines had different background workloads, so this is a comparison of measured sessions, not an isolated CPU ranking. RAM creation/formatting, subsequent validation and archival are outside the timed interval. Outputs were copied and hash-verified under `data/m1-ram-benchmark-saved` before detaching the temporary RAM disk. Evidence: `results/M1_RAM_FINAL.json` and `results/m1_driving_ram_*.json`.
