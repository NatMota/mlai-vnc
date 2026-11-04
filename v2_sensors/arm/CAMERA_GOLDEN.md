# Golden camera replay (ticket 11108)

Render the camera along the recorded expert trajectory from its first step, then replay both trackers and the wheel encoders. This supplies supervised training observations without rerunning the physics. Evaluation still renders from the student's own trajectory, in a live simulation.

The source state is sampled before each 1 ms physics step. Camera steps are `round(i * 1000 / 30)` while that value is less than the number of recorded steps. In particular, a 67-row recording has frames at 0 and 33, not 67. Do not use a ceiling formula for the count.

## Renderer fixes and portability

`CAMERA_RENDERER=tiny` remains the default and preserves the existing CPU path. `CAMERA_RENDERER=egl` uses Linux EGL, tested on Mint's NVIDIA GTX 1060. The shared `wrist_camera.Scene` implements three GPU corrections, so `full_run.py` and the offline renderer use the same camera:

1. Give robot visuals with **no URDF material** an explicit white colour. EGL otherwise supplies a palette including a red wheel and arm ring, which triggers the red-dot detector. Explicit URDF materials are retained.
2. Render the dot as opaque on EGL. Its original alpha of 0.85 leaves background depth in the GPU depth buffer. TinyRenderer's detector and depth are unchanged by this opacity choice. Geometry, radius, location, physics and controller are unchanged.
3. Offset each GPU cube-face projection by half a pixel in window x/y. TinyRenderer evaluates triangle coverage at integer coordinates (`examples/TinyRenderer/our_gl.cpp`); OpenGL uses half-pixel centres. The projection correction is `P[8] -= 1/FACE; P[9] -= 1/FACE` in OpenGL's column-major matrix. This aligns GPU images with the existing CPU fisheye lookup. [OpenGL pixel-centre convention](https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.60.html).

On dot 0's complete 287-frame recording, the first two corrections alone left a median detection difference of 7.33 mm (maximum 24.29 mm) against CPU. With the pixel correction, the median is 0.0061 mm and p95 0.0402 mm. The maximum is still 10.17 mm, and four frames disagree about visibility: CPU sees a single red pixel around the occluding arm while GPU sees none. That difference can cause up to 51.01 mm divergence in the tracker, which either resets from the detection or continues dead reckoning. This is a remaining rendering/occlusion difference, not different robot motion. The detector has deliberately NOT been changed to hide it.

A second audit of 40 start/middle/PARK/end poses across ten training dots found no visibility disagreements, no false red pixels, and median/p95/maximum detected-position differences of 0.0032 / 0.2295 / 1.2352 mm among the 33 jointly visible samples. These checks do not establish equivalence for every possible pose.

Run CPU and EGL audits in **separate processes**. Loading the EGL plugin can make subsequent requests use that plugin even when the caller requests TinyRenderer; toggling the renderer flag inside one client was not a valid CPU/GPU comparison here.

Saved NumPy features can be used for training on any supported CPU/GPU, including Mac. GPU hardware is only required to reproduce EGL rendering. For comparable live scores, use the same renderer and camera revision, preferably the shared Mint/Powerglide evaluator. CPU is a local fallback, not a promise of identical scores. Powerglide's RTX 4090 has not yet been tested with this change: run the smoke/parity checks there before collecting or reporting results.

## Reproduce

Use Python with NumPy and PyBullet; no Torch or Pillow is required for rendering. The existing native Pixi environment supplies these. Set `CAMERA_URDF` when assets are outside the repository.

```bash
export CAMERA_RENDERER=egl  # use tiny on CPU/macOS
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python v2_sensors/arm/check_render_golden.py --self-test
python v2_sensors/arm/render_golden.py \
  --source /path/to/run_1000/baseline \
  --out /path/to/camera_golden_egl_aligned --split train --renderer nvidia --jobs 2
python v2_sensors/arm/render_golden.py \
  --source /path/to/run_1000/baseline \
  --out /path/to/camera_golden_egl_aligned --split val --renderer nvidia --jobs 2
```

Use `--renderer cpu` for TinyRenderer (including Mac). The CLI overrides `CAMERA_RENDERER` for this process and its workers; omitting it preserves the environment/default. Rendering selection does not change CPU physics. The shared [data package](GOLDEN_DATA.md) contains 300 expert runs but 200 camera traces: test cameras render live along the student's trajectory.

Train IDs come from `collection/native/effects/dots_spread.json`, key `100`; validation IDs come from `split_1000.json`, key `val`. Test IDs are rejected by this renderer CLI. An existing output is reused only when provenance matches. A changed renderer/code/source requires a separate output directory; originals are preserved. Each process handles one run, disconnects PyBullet and exits. Writes are atomic, and `summary.json` is updated after each result.

Build the baseline expert stream for an independent live comparison:

```bash
cd collection/native
pixi run cmake --build build-clang19 --target golden_camera_stream -j 1
cd ../..
python v2_sensors/arm/check_render_golden.py \
  --source /path/to/run_1000/baseline/run_0.npz \
  --camera /path/to/camera_golden_egl_aligned/train/camera_0.npz \
  --native /path/to/collection/native --out /path/to/parity_0.json
```

`golden_camera_stream` uses the same `Physics`, `Expert`, pre-step sampling and termination as the baseline collector. It waits while each camera frame is rendered and then advances physics. It does not substitute the learned arm or add a different holding controller. All ten corrected EGL comparisons passed, with a worst camera/tracker vector difference of 0.0361 mm. The packed source is float32: rounding near an encoder boundary can shift a wheel count by one tick (0.001534 rad, or 0.04602 rad/s over 1/30 s). The parity report retains wheel-feature differences rather than claiming bitwise equality with live float64 states.

`audit_camera_renderer.py` samples poses without advancing physics. Run it separately with `CAMERA_RENDERER=tiny` and `egl`, then compare its JSON detections, pixel counts and depth. Segmentation is used only to count false red pixels in diagnostics, never as input to the dot detector.

## Files and loader

Each `train/camera_ID.npz` or `val/camera_ID.npz` contains:

| Field | Meaning |
|---|---|
| `step` | Actual 1 kHz indices of the 30 Hz camera frames |
| `seen` | Fresh detection in this frame, boolean |
| `detection` | Camera-derived robot-frame xyz, metres; NaN when missing |
| `drive` | Tracked range (m), sin bearing, cos bearing; zeros before first sighting |
| `arm` | Tracked robot-frame xyz (m); NaN before first sighting |
| `wheel_features` | Existing 12 encoder features, float32 |
| `wheel_speed` | Quantised encoder finite-difference speeds, rad/s |
| `false_red_pixels` | Diagnostic count only |
| `dot`, `target`, `parking_step`, `n_steps`, `joint_names` | Source identity, metadata and ordering; not model inputs |
| `provenance`, `summary` | JSON strings with hashes/settings and measured diagnostics |

Use `camera_inputs(split, dot, steps, root=...)`. It returns float32 `drive` (range, sin, cos, visible), `arm` (xyz), and `wheel` (12 features). It selects the latest camera frame **at or before** each requested 1 ms step, zeros the unseen arm input, and never uses future interpolation. The tracker dead-reckons between sightings and its result is held between camera frames. Raw detections remain available to rerun trackers without rendering again.

Every output records the source and code SHA-256s, base Git revision, renderer, dependency versions, URDF hash, camera geometry, GPU pixel alignment and encoder settings. Files also retain joint ordering. When training models, copy the sensor/renderer provenance into checkpoints and use it when choosing the evaluation renderer.

`summary.json` reports PARK error against simulator truth, first sighting, post-PARK visibility and stalled runs. Missing PARK/sightings are explicit. Errors above 100 mm and visibility below 80% are listed as observations; neither causes rejection or tuning to a desired result.

## Current collection location

Mint: `/media/nm/Fast Drive/comp0245-camera-11108/camera_golden_egl_aligned/`.

Mac copy: `v2_sensors/arm/data/camera_golden/` (gitignored, the loader's default location). Its files contain EGL provenance even though training can run on the Mac.

The earlier GPU outputs without the pixel correction are preserved under `camera_golden_egl/` for comparison; do not mix the two. The corrected 100-training-run batch took 389.9 seconds with two workers (6 min 30 s). The preceding GPU batch took 391.5 seconds and used 3.27 MB including PNGs and reports. One reference run took 111.1 seconds on CPU versus 7.8 seconds on GPU. Physics remains on CPU. Mint had about 10–11 GiB available RAM during the batch; rendering workers are limited to 4 GiB combined.

## Verified collection, 4 November 2026

All 200 corrected runs are complete: **100 train + 100 validation, 53,598 camera frames**, with no test runs included. All ten live parity comparisons pass. A full archive check verified source hashes, split membership, exact frame indices, shapes, float32/bool types, causal loader output and one consistent code/renderer fingerprint. The Mac copy matches every Mint NPZ SHA-256.

Validation rendering took **497.7 seconds (8 min 18 s)**; training took 389.9 seconds. Combined output is about **6.6 MB**, including per-run PNGs and reports. PARK-position error median/p95/max was 12.05 / 40.22 / 52.84 mm for training and 11.02 / 35.96 / 51.77 mm for validation. No run lacked an estimate at PARK or stalled. Post-PARK visibility fell below 80% in 27 training and 24 validation runs; these are observations, not filtered failures.

Across the 53,598 frames, the segmentation audit found one red pixel outside the dot mask, in validation dot 687 at step 6000. It is adjacent to the dot (65 dot-mask pixels, 66 red pixels); its segmentation ID is the plane, not the robot. This is consistent with an RGB/segmentation edge discrepancy, not the former red-wheel problem. The detector is unchanged and does not use segmentation.

See `results/camera_11108/` for verification, renderer comparisons and live parity reports. `summary.json` in the dataset contains the per-run diagnostics. Training/checkpoint integration and final student-policy evaluation belong to tickets 11109 and 11110; no model was trained or rescored as part of this camera collection.
