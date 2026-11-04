This batch tests three cumulative changes to the arm torque network, with the improved 11-input driving model and the separate STOP-distance network frozen:

1. Increase the training budget from 300 to 1,200 epochs, keeping two 64-unit layers.
2. Keep that budget and widen both layers to 128 units.
3. Keep both changes and add sensor observations from 50 and 100 ms earlier.

All variants train from scratch with seed 7 on the same 100 golden training runs. Each checkpoint is selected by the lowest total loss on the 100 validation runs, with earliest ties. Adam 0.01, FP32, equal run weighting, standardised torque targets and the auxiliary distance loss are unchanged. The auxiliary distance head is trained with the torque trunk; the separately frozen STOP network makes the runtime completion decision.

History contains current, 50 ms old and 100 ms old joint angles, joint speeds and camera-tracked target xyz, giving 51 inputs. The buffer resets at PARK and repeats its first sample until enough history exists. Camera values are held between 30 Hz updates; joint feedback and torque remain at 1 kHz. Training and runtime history were checked for exact equality, including padding and buffer rollover. The frozen STOP network uses only the current 17 inputs.

Use a fresh output directory and the complete previous `results/driving_height_no_speed11` experiment as the baseline:

```bash
python v2_sensors/arm/arm_history.py
python v2_sensors/arm/arm_cumulative.py train --data /path/to/golden --baseline /path/to/driving_height_no_speed11 --out /path/to/new/arm_batch --device cuda
CAMERA_RENDERER=egl NATIVE_ROOT=/path/to/native CAMERA_URDF=/path/to/native/assets/summit_xl_arm_description/summit_xl_complete.urdf python v2_sensors/arm/arm_cumulative.py evaluate --data /path/to/golden --baseline /path/to/driving_height_no_speed11 --out /path/to/new/arm_batch --jobs 3
```

Training uses the existing CUDA environment; evaluation uses the native simulator's NumPy/PyBullet environment. Intermediate PyTorch checkpoints at 300, 600 and 1,200 epochs are retained on Mint, along with the selected checkpoints; existing repository rules exclude `.pt` files from Git. Folded NumPy checkpoints, full loss histories and validation results are committed.

Each variant runs all six fixed validation targets from the normal start through driving and reaching: 673, 506, 830, 720, 16, 333. These are 18 new full simulations. The preceding six no-speed-model runs are the cached baseline. Full rendering, sensor rates, physics, deadlines and scoring are unchanged. There are no isolated PARK resets, search rules or reserved-test evaluations. Results append to the golden notebook without rerunning its earlier analysis or generating validation animations.

This is a cumulative, single-seed development screen on six initially visible targets. It does not measure generalisation across seeds or targets, or separate interactions among the changes. No new official MLflow iteration is created.

Training and loading took 239.7 seconds; the 18 full simulations took 62.3 seconds with three workers. All three variants parked and issued STOP on all six targets; none stopped within 10 mm. Median distance at STOP was 242.1 mm for the cached baseline, 358.8 mm with longer training, 131.7 mm with wider layers and 217.5 mm with history. Median closest distances were 142.8, 93.2, 81.0 and 73.9 mm respectively. Closest distance includes the 0.5-second observation after STOP, during which learned torques continue. It is not a successful-stop score.

The wider model is the strongest candidate on median distance at STOP in this small screen. History further reduced median closest distance but worsened distance at STOP. No result yet supports a claim of reliable stopping or generalisation. The first 300 epochs of the longer variant exactly reproduced the earlier torque training losses. All frozen STOP arrays and all six parking times and errors match the cached baseline.
