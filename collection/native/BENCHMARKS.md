# CPU collection benchmark and equivalence audit

Measured on 29 October 2026, Apple M1, 8 CPU cores, 16 GB RAM, macOS 14.6.1. GPU use: none. The desktop remained in use; Photos background processing and memory pressure were visible during the measurements. These are measured operating results, not an isolated hardware maximum.

## Fidelity comes before speed

Ten original target cases were run through the complete expert, including base turning, height-dependent parking and arm reaching. Each produced 8,001 consecutive samples at the original 0.001-second timestep. The audit compares the seven numerical/controller tables and all three tensor arrays against existing original Python observer captures. String/boolean labels, NaN locations, sample counts and column order are also checked.

The final full-diagnostic audit found **zero numerical differences** for every checked value in all ten runs: joint positions, velocities and backward accelerations; base states; end-effector poses; target geometry; wheel and arm commands; controller states; Pinocchio configuration; Jacobians; and the full mass matrices. Parking transitions occur at the same tick. See `results/verified_summary.json` and `results/verified_0.json` through `verified_9.json`. This final audit also checks the preloaded fork workers and the saved simulation timestamps. The ten complete datasets are in `data/verified/run_0` through `run_9`.

The timing table contains wall-clock performance measurements, which cannot match between implementations. Simulation clocks preserve the original two conventions: accumulated controller time, and step-counter × dt for the wrapper's post-step time. File containers differ: separate uncompressed NPY arrays replace CSV/NPZ for the fast writer. Exact decoded values do not mean byte-identical CSV/NPZ files.

This evidence covers these ten targets and the pinned environment, not every possible initial condition or a different CPU/engine build. The original mass/inertia inconsistency is retained. No gravity, contact, motor gain or physical geometry was corrected to obtain faster results.

## Worker sweep

Every entry includes process/library startup, loading/resetting worlds, a complete eight-second simulation and saving the dataset. The mass-matrix diagnostic is omitted in this sweep; all other output remains at 1,000 Hz. Two complete episodes are generated per worker. File hashing and removal of redundant generated copies happen after the timed section.

| Processes | Complete episodes | Wall time | Seconds per saved episode | Episodes/second |
|---:|---:|---:|---:|---:|
| 1 | 2 | 4.58 s | 2.29 | 0.437 |
| 2 | 4 | 4.67 s | 1.17 | 0.857 |
| 4 | 8 | 7.15 s | 0.894 | 1.119 |
| 6 | 12 | 9.64 s | 0.803 | 1.245 |
| 8 | 16 | 10.34 s | 0.646 | 1.548 |
| 12 | 24 | 16.86 s | 0.703 | 1.423 |
| 16 | 32 | 22.47 s | 0.702 | 1.424 |

Raw records after the final export optimisation: `results/export_optimised.json`. All 98 repeated-run sets of numerical arrays matched exactly across these worker counts. Timing arrays and run metadata were excluded from repeat hashing. The first dataset is retained; identical generated copies were removed to conserve disk space, with hashes and manifests retained in the benchmark JSON. Serial export took about 0.114 seconds per run, down from about 0.5 seconds in the preceding sweep.

Eight processes were fastest in this sweep. The preceding sweep favoured twelve (`results/benchmark_final.json`), demonstrating the effect of background load and export overhead. Eight is a reasonable working default. More processes do not imply more CPU cores, and sixteen did not improve throughput.

Preloading the world before forking measured 0.639 seconds per saved episode with eight workers and 0.672 with twelve (`results/fork_optimised.json`). The eight-worker improvement over ordinary process startup was only about 1% in this sweep, too small to establish a robust throughput advantage. The flag is optional and may be more useful for very short batches or reducing repeated asset loading.

### Full original diagnostic output

The final executable, with both timestamp conventions verified and every mass matrix included, measured:

| Processes | Complete episodes | Wall time | Seconds per saved episode |
|---:|---:|---:|---:|
| 1 | 2 | 6.17 s | 3.08 |
| 8 | 16 | 17.45 s | 1.09 |

These runs used `--fork-workers --diagnostics`; all 18 sets of numerical output arrays matched each other exactly. Raw records: `results/release_full.json`. The extra cost buys the 23×23 mass-matrix record at every tick, not different physics or better expert actions. Use this mode when the requirement is to retain every original diagnostic; omit the matrix when it has no downstream consumer.

The final source fingerprint is `839477d353ef3d8a7694f81df3b71fe19ba810904dce0cb04a356665e599cbd7`. A final fast-mode check on this exact executable saved 16 complete episodes in **10.24 seconds**, or **0.640 seconds per episode**, using eight preloaded workers without mass-matrix logging (`results/release_fast.json`). All 16 sets of numerical arrays matched, and the retained run also matches the corresponding full-diagnostic dataset for all shared numerical arrays. The earlier worker sweeps used the same simulation/control implementation before a reporting-only fix to the post-step timestamp; the final audits and these two release benchmarks include that fix.

An episode's batch-amortised cost is not its latency. This sweep does **not** demonstrate an individual complete expert run below 500 ms. The earlier, lighter-load sweep reached 0.662 seconds per episode with eight workers, but preceded the final numerical-compatibility corrections and is retained only as development evidence (`results/cpu_benchmark.json`). Do not quote it as the final validated result.

## Where time goes

The original Python cProfile recorded 3,549,009 function calls over 8,001 steps. Major costs included its simulation wrapper, repeated observations/conversions and mass-matrix handling. The original unprofiled headless run previously took 5.85 seconds without writing the full dataset; profiling raised that to 7.10 seconds. A fresh three-run check measured 5.46–5.77 seconds for the unchanged expert's main function, or 6.30–7.00 seconds including process startup/imports, still without dataset export (`results/python_reference_timing.json`). Compare those limits explicitly: the native sweep includes saving data and omits the unused mass matrix.

A native sampling profile, taken while executing complete episodes without export, attributed 2,299 of 2,401 main-thread samples (about 96%) to Bullet's physics step. About 1,919 samples (80%) were under constraint solving. These are inclusive stack sample counts, not additive categories. See `results/native_cpu_sample.txt`.

This means replacing the Python hot path has exposed the native contact/constraint solver as the main remaining cost. Reducing solver iterations, enlarging the timestep, merging physical links, dropping contacts or changing the engine could speed it up, but each changes the reference computation. None is included in the validated collector.

## Experiments retained or rejected

| Experiment | Finding | Use |
|---|---|---|
| C++ control and recording | Removed the Python per-tick path | Enabled |
| Consolidated observation | Same values with fewer boundary calls | Enabled |
| Persistent finger servo settings | Same commands/physics without resending constants | Enabled |
| Saved-state resets | Reused parsed assets; repeated output matched | Enabled |
| Preload world before forking | Independent process copies; ten-target audit still matches | Optional `--fork-workers` |
| Skip visual meshes | Same physics output; shorter initialisation | Enabled in Pixi task |
| Binary arrays | Preserves float64 and avoids CSV formatting | Recommended |
| Omit unused mass matrix | Same control/physics; less calculation/storage | Optional; use `--diagnostics` for full original output |
| Independent process worlds | Repeat hashes match across worker counts | Recommended |
| Dynamic work queue | Useful for uneven run lengths; no demonstrated fixed-target speed advantage | Optional |
| Multiple robots inside one world | Changed solver ordering/trajectories; two-robot test did not improve throughput | Experimental, rejected for equivalent data |
| Source-built Bullet | Same release, different numerical trajectory from the original binary | Experimental, rejected for equivalent data |
| Clang 19 versus AppleClang 16 for source engine | Did not resolve engine numerical differences | No fidelity benefit established |
| Eigen-only controller math | One-ULP changes eventually altered parking/contact dynamics | Replaced with the same BLAS/LAPACK operations as NumPy |

## Reproduce

```sh
pixi run build
pixi run collect --binary --diagnostics --out data/new_audit_run
```

Using the existing coursework Python environment for the optional audit/measurement tools:

```sh
python tools/compare.py ORIGINAL_CAPTURE data/new_audit_run --report results/new_audit.json
python tools/audit_suite.py ORIGINAL_INSPECTION_FOLDER data/verified --tag repeated_audit
python tools/benchmark.py --tag new_cpu_sweep --workers 1 2 4 8 12 --repeat 2 --compact
python tools/benchmark.py --tag new_full_sweep --workers 1 8 12 --repeat 2 --diagnostics --compact
```

`tools/compare.py` reports both tight tolerance checks and exact decoded equality. `--compact` retains the first dataset and deletes only newly generated matching repeats after preserving hashes/metadata. Validation scripts require NumPy and pandas from the existing notebook environment; the C++ collector does not.
