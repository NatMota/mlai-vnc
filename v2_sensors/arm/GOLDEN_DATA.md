# COMP0245 golden data: 300 expert runs and 200 camera traces

This data-only package contains the agreed baseline recordings, with no noise/effects added. Nothing is trained by packaging it. Original NPZ files are copied unchanged and verified by SHA-256. Code, assets and dependencies stay on GitHub.

Code: https://github.com/NatMota/mlai-vnc/tree/CODE_REVISION (branch `v2-sensor`).
The earlier `data/native-collector-and-30hz-sampling` branch does not contain the latest camera pipeline; use this pinned revision for these data.

## Contents

- `expert/train/`: 100 selected expert runs, spread across the 800 training dots.
- `expert/val/`: 100 validation runs for model/epoch selection.
- `expert/test/`: 100 reserved test references, never used for training or model selection.
- `camera/train/` and `camera/val/`: the matching 200 offline camera traces.
- `split.json`: exact dot IDs. IDs are shared between `run_ID.npz` and `camera_ID.npz`.
- `manifest.json`: per-file hashes, sizes, step counts, terminations and code revision.
- `SHA256SUMS.txt`: file integrity checks.

The test cameras are rendered live along the student's own trajectory during closed-loop evaluation. No offline expert-test camera traces are needed for that evaluation. This package is not a new trained checkpoint or a claim of test performance. Integrating the new dataset with the trainers is roadmap step 3.

## Expert files (1 kHz)

Open with `np.load(path, allow_pickle=False)`. Each file contains named arrays, not one anonymous matrix. The `manifest` and `columns` entries are JSON strings; parse with `json.loads(str(data['manifest']))`. One row is the pre-step state and the command applied for that 1 ms tick. Numeric arrays are float32, except timestamps are float64.

| Arrays | Meaning and use |
|---|---|
| `time_s` | Simulation timestamp in seconds; align to camera step indices, not wall-clock time. |
| `q`, `qd` | 17 joint angles (rad) and speeds (rad/s). Order: four wheels FR/FL/BL/BR, seven arm joints, six fixed finger joints. Exact names are in the manifest. These are raw simulator joint records: use `sensors.WheelEncoders` / `ArmEncoders` to model the sensor inputs, rather than treating raw speeds as encoder measurements. |
| `wheel_command` | Four expert wheel velocity commands, rad/s, in FR/FL/BL/BR order. Training labels. |
| `arm_torque_command` | Seven expert arm torque commands, N m. Training labels. |
| `navigation_done`, `reach_active` | Expert phase flags; `parking_step` in the manifest identifies PARK. |
| `hand_distance`, `hand_distance_expert` | End-effector-to-target distance, metres. Reference/stop supervision, not sensor input. |
| `base_position`, `base_quaternion`, `base_linear_velocity`, `base_angular_velocity` | Simulator state for replay and evaluation references. Metres, xyzw unit quaternion, m/s and rad/s; velocity components are world-frame. Not V2 network inputs. |
| `hand_position`, `hand_rotation` and their `_expert` counterparts | Simulator/controller hand poses for reference. Position in metres, rotation stored as nine row-major entries. Not V2 network inputs. |
| `arm_q_des`, `arm_qd_des`, `sent_torque` | Existing controller/debug records retained because the archive is unchanged. Internal reference angles/speeds are not V2 deployment inputs. `sent_torque` records the 11 sent wheel/arm torques. |

Target world coordinates, final state and termination are metadata for reproducing episodes and reference scoring, not direct network inputs. Check termination explicitly; a timeout is not a successful STOP label.

## Camera files (30 Hz)

The camera mount is the existing corner mount, 180-degree fisheye RGB-D, 256 x 256 output from five 192 x 192 faces. Stored data are detections and tracker outputs, not RGB videos or a CNN image-training dataset.

- `step`: exact 1 kHz source indices `round(i * 1000 / 30)`, beginning at zero.
- `detection`: camera-derived robot-frame xyz, metres; NaN when not visible.
- `seen`: fresh-detection flag for the current camera frame.
- `drive`: tracked range in metres, sin bearing, cos bearing.
- `arm`: tracked robot-frame x/y/z in metres; NaN before the first sighting.
- `wheel_features`: the existing 12 wheel-encoder features, already quantised using 4,096 counts/turn; `wheel_speed` records the encoder finite-difference speeds.
- `provenance`: original source hash, camera/code hashes, renderer, dependency versions, sensor settings and geometry.
- `false_red_pixels` and `summary`: diagnostics only, not model inputs.

Use `v2_sensors/arm/render_golden.py:camera_inputs(split, dot, steps, root=...)` with `root` pointing to this package's `camera` folder. It returns float32 inputs, holds the latest camera frame at or before each requested step, and substitutes zeros before the first sighting. It never reads a future frame. Combine its 12 wheel features and four driving camera values for the 16-input driving network. The arm's 17 inputs are seven encoder angles, seven encoder speeds and the three camera/tracker coordinates. Preserve the existing sensor reset/history from the beginning of the episode.

Driving decisions use 30 Hz; arm torque decisions use 1 kHz with held 30 Hz camera estimates. Fit any scaling on training runs only. Validation can choose checkpoints; keep test references out of those decisions. The 100 training dots are an initial data budget, not proof that more data cannot help.

## CPU / NVIDIA

Saved NumPy data can be loaded on CPU, Mac or NVIDIA without rerendering. For camera generation, `render_golden.py --renderer cpu` selects TinyRenderer; `--renderer nvidia` selects Linux EGL. NVIDIA mode fails explicitly if EGL is unavailable. Physics remains on CPU.

For existing live-run scripts, `CAMERA_RENDERER=tiny` selects CPU and `CAMERA_RENDERER=egl` selects NVIDIA. These camera traces were generated on NVIDIA. Use the same renderer revision for comparable live evaluations. Single-pixel occlusions can still differ between backends; see `CAMERA_GOLDEN.md`. RTX 4090 has not yet been validated with these fixes.

To reproduce the package from the original collection, run from the repository root:

```bash
python collection/native/tools/share_golden.py \
  --source /path/to/run_1000/baseline \
  --camera /path/to/camera_golden \
  --out /path/to/COMP0245_Golden_300 \
  --code-revision CODE_REVISION
```

Verify downloaded files from the extracted folder with `shasum -a 256 -c SHA256SUMS.txt` (Mac) or `sha256sum -c SHA256SUMS.txt` (Linux).

## Train the golden models (ticket 11109)

`train_golden.py` opens training and validation trajectories only. It verifies source/camera hashes and timing, preserves the existing encoder conventions, weights each run equally, and selects each checkpoint on validation loss. Seeds 7, 1 and 2 become official iterations 34 to 36 only after their reserved-test evaluations. Training and validation smoke checks do not create numbered MLflow runs. The architectures and losses remain those specified in the notebook plan.

```bash
python v2_sensors/arm/check_train_golden.py
python v2_sensors/arm/train_golden.py \
  --data /path/to/COMP0245_Golden_300 \
  --out /path/to/new_training_output \
  --seeds 7 1 2 --epochs 300 --stop-epochs 600 --device cuda
```

Use `--device cpu` without CUDA. The recorded GPU environment used Python 3.12, PyTorch 2.6.0+cu124 and NumPy 2.2.6 on a GTX 1060. Physics remains on CPU; CUDA training and EGL rendering are separate choices. `models/golden_11109/environment.json` records the GPU driver and native simulator hashes. New output directories are required so earlier runs cannot be overwritten.

The trainer saves folded NumPy weights, resumable PyTorch checkpoints, every epoch's losses, selected-epoch metadata and a data audit. NumPy exports are checked against PyTorch on fixed training and validation rows. Checkpoints are also MLflow artifacts; `.pt` files are not required for NumPy inference. The frozen models and histories are under `models/golden_11109`.

```bash
CAMERA_RENDERER=egl NATIVE_ROOT=/path/to/comp0245-native-expert \
  CAMERA_URDF=/path/to/comp0245-native-expert/assets/summit_xl_arm_description/summit_xl_complete.urdf \
  python v2_sensors/arm/validate_golden.py \
  --data /path/to/COMP0245_Golden_300 \
  --models /path/to/new_training_output --out /path/to/new_validation_output --jobs 2
```

Use a Python environment with PyBullet and the built native `sensor_parking` executable. `CAMERA_RENDERER=tiny` selects CPU rendering explicitly. Validation uses the ten target IDs fixed in the training config, for each seed; the final 100-target test belongs to ticket 11110. The expert is a recorded reference only. Actual distance at the exact STOP step is scored; subsequent drift is separate.

The historical notebook remains unchanged at 361 cells, including its previous validation analysis. Continue in `../driving_v2/golden_dataset.ipynb`. Validation smoke checks appear there as a short table without animations. Their former MLflow entries are archived, with artifacts retained; `results/golden_11109/archived_smoke_runs.json` records the IDs. Training and validation histories are retained with the eventual official test iterations.

The new notebook can be read from saved manifests, previews and results without downloading the full data package. Set `GOLDEN_DATA_ROOT` to the extracted package for new data loading or training. The fixed native PD-plus-model-gravity arm hold during driving remains part of this baseline; learned torque control starts at PARK.

## Frozen full test (ticket 11110)

Evaluate all 100 reserved targets for each of the three frozen seeds, giving 300 attempts on the same 100 targets. Seed 7 was chosen as the representative by validation before simulation outcomes. No checkpoints, thresholds or settings may be chosen from these test outcomes.

```bash
CAMERA_RENDERER=egl NATIVE_ROOT=/path/to/comp0245-native-expert \
  CAMERA_URDF=/path/to/comp0245-native-expert/assets/summit_xl_arm_description/summit_xl_complete.urdf \
  python v2_sensors/arm/evaluate_golden.py \
  --data /path/to/COMP0245_Golden_300 \
  --models /path/to/frozen_models \
  --frozen-hashes v2_sensors/arm/results/golden_11109/config.json \
  --out /path/to/new_test_output --jobs 2
```

Use the simulator's Python environment with NumPy and PyBullet; the CUDA training environment alone is not sufficient. `--preflight` restricts the command to one predetermined validation target and seed 7. It creates no MLflow entry or animation. The reference loader caches decompressed arrays once; the cached replay was checked against the earlier recording and was identical.

Success requires PARK and STOP with actual gripper distance at the exact STOP step at most 10 mm. Closest approach, post-STOP drift and error at PARK are separate measurements. Without STOP the fixed deadline is expert completion plus 4 seconds; after STOP the simulation observes a further 0.5 seconds. No PARK, no STOP, execution failures and early terminations remain in the denominator.

The native physics engine runs on CPU and the camera renderer uses EGL on NVIDIA. Driving and camera update at 30 Hz; arm encoder feedback and torques at 1 kHz. During reaching, this test harness exchanges Python/JSON requests with the native process every millisecond. The checked batching option reduces exchanges during driving while preserving each 1 ms physics tick. It does not inherit all the speedups of the native-only expert collector. There is no new measurement noise, delay, damping, elasticity or motor-inertia correction. Existing model quirks, encoder quantisation, camera error and limits remain.

`report_test_golden.py` reads saved test results, checks the complete target roster, produces per-seed metrics with Wilson 95% intervals and builds the players. Each player contains all available recorded attempts for its seed, with the ghost expert and camera inset. Camera previews stay as external files so the browser does not allocate all images at startup; the MLflow replay ZIP contains the HTML and its camera folders. Extract the whole ZIP before opening the HTML. Failed attempts without frames stay in the score table and have no fabricated replay.

`log_golden.py` refuses to log before all 300 attempts are accounted for. It records official test iterations 34, 35 and 36 with their original training/validation loss histories, frozen models, source/configuration hashes, test metrics, full replay archives and executed notebook. Its run record supports resuming an interrupted upload without creating duplicate runs. Validation smoke checks never consume an iteration number.


### Exact execution speedups

The full renderer is preserved. A validation comparison on seeds 7, 1 and 2 at target IDs 6 and 131 compared hashes of every full RGB, depth and segmentation frame, every policy input/output, recorded trajectories and event metrics. All six cases matched exactly. Cached fisheye pixel mappings avoid recomputing geometry that does not depend on robot pose. Batched driving advances the same 1 ms native physics/holding loop to one tick before each camera update, then advances the final tick separately so the arm encoder retains its exact previous sample at PARK. After PARK, the arm still receives fresh feedback and issues torques at every millisecond.

`check_full_run_speed.py` reproduces the comparison against a saved copy of the original camera module. Six concurrent cases took 28.54 s with the original camera and 24.45 s with caching and batching, including startup and hash checks. That is a 1.17× speedup at the same worker count. The earlier full evaluation used only two workers; increasing to six also improves overall throughput. Neither number establishes a 5–10 minute full evaluation. The final completion file records actual elapsed simulation time.

To resume an interrupted evaluation, use `--resume --batch-driving --parity-summary /path/to/speed_summary.json`. It verifies model, native simulator, split, rates and scoring settings; the checked source hashes must match. Completed attempts are kept regardless of success. Source changes and the execution settings for the resumed segment are recorded separately in `resume_config.json`.

The completed test took 1,192.8 seconds of active execution across both segments (19.9 minutes), excluding the pause. The resumed segment evaluated the remaining 235 attempts with six workers in 686.2 seconds. No execution failures occurred; weak task outcomes remain in the results.

The notebook serves replay HTML and camera files from a read-only directory through a loopback HTTP server at `127.0.0.1:8899`. This avoids Jupyter’s HTML sandbox blocking image authentication. It does not expose a simulator or change Jupyter authentication. Rerun the replay cell to start it; the extracted MLflow ZIP also opens directly as a local HTML file.
