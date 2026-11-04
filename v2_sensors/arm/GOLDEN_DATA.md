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
