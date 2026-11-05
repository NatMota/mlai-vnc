One existing validation episode (target 6, seed 7) was replayed with the frozen evaluator, native binary, models, EGL renderer and random seeds. Every retained original field matched exactly, including 269 camera frames, 897 control-log rows and all outcomes. All 269 camera PNGs had identical hashes. The trace observer does not draw random numbers or alter the control outputs.

The new trace contains 8,967 arm captures and 269 wheel captures. Original joint-state logs were recorded at 30 Hz, so the intermediate 1 kHz states are newly recorded reconstruction evidence, not independently verified original measurements. The original random streams, integer count calculations and velocity differences are verified exactly.

`encoder_trace.npz` contains capture steps, true/noisy angles, integer counts, reported angles, reported velocities and the no-noise encoder velocities on the same states. Truth columns are diagnostic references only. `summary.json` records hashes and measured noise. `course_source.json` records the course configuration and sampling excerpts with hashes. The PNG figures are displayed in iteration 42 of the golden notebook.

Reproduce on the original compatible environment, using a new empty output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg python recover_encoder_trace.py \
  --root /path/to/overnight_20261105 \
  --data /path/to/COMP0245_Golden_300_2026-11-04 \
  --native /path/to/comp0245-native-expert \
  --out /path/to/new_encoder_trace_audit
```

The root must include the frozen `ladder_runtime`, original `encoder_noise/preflight` records and camera PNGs, and the baseline models at their recorded location. The command refuses changed source/model/binary hashes or any mismatch with the saved episode. A different platform or renderer may fail exact reproduction; do not relax the comparison and call that exact.

The course calls the angle setting `joint_cov` but uses it as a standard deviation of 0.0001 rad. This experiment derives velocity from noisy encoder counts; the course wrapper separately supports direct velocity noise. The original wheel calculation uses fixed 1/30 s despite alternating 33/34 ms capture intervals. No sensor equation, network or training setting was changed by this audit.
