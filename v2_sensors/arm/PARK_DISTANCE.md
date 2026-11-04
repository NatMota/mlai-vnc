This comparison replaces time-to-PARK with distance-to-PARK supervision. The wheel-command MSE remains. The label is horizontal distance from the current expert base position to its position at the recorded PARK step, clipped at 3 metres and divided by 3. The same numeric cap and divisor were previously seconds. Both loss terms have weight 1, with each run weighted equally. The new runtime PARK threshold is 5 cm, fixed before evaluation. Actual post-PARK drift remains in the labels.

The network still receives only sensor-derived inputs. Recorded world positions are used for labels and evaluation. The distance head predicts remaining distance; training does not differentiate through PyBullet or optimise actual final parking error directly. Since the two driving heads share a trunk, changing the PARK loss can also change wheel commands.

Four configurations repeat the earlier cumulative sequence:

1. Camera height added to driving: 17 inputs, two 16-unit hidden layers, original arm.
2. Six wheel-speed inputs removed from driving: 11 inputs, original arm. Speeds remain in the tracker.
3. Same driving network, with the previous 1,200-epoch-budget arm, two 64-unit layers.
4. Same driving network, with the previous 1,200-epoch-budget arm, two 128-unit layers.

Only the two driving models need retraining because the arm loss and data are unchanged. Exact arm checkpoints, including the separate STOP network, are reused and verified by hash. Driving uses seed 7, 300 epochs, FP32, Adam 0.01 and the same scaling. Epochs 300 and 297 were selected by offline validation loss. The 500-driving-epoch suggestion and sensor-history variant are outside this comparison.

The same 100 training and 100 validation runs are used. All four configurations run from the normal start through driving and reaching on validation targets 673, 506, 830, 720, 16 and 333. Full rendering, camera/driving at 30 Hz, arm/physics at 1 kHz, clipping and deadlines are unchanged. There is no reserved-test evaluation, validation animation or official MLflow iteration.

| Configuration | PARK / 6 | Median parking error, time -> distance | Median arm STOP error, time -> distance |
|---|---:|---:|---:|
| Height | 4 | 23.3 -> 11.9 cm* | 55.3 -> 38.7 cm* |
| No driving speeds | 6 | 4.4 -> 8.4 cm | 24.2 -> 27.3 cm |
| Longer-trained arm | 6 | 4.4 -> 8.4 cm | 35.9 -> 37.0 cm |
| Wider arm | 6 | 4.4 -> 8.4 cm | 13.2 -> 21.5 cm |

*Height-model distance medians exclude its two non-parks. The old time model parked on all six, so these medians describe different subsets and cannot establish improvement. All parked runs issued arm STOP; none stopped within 10 mm. The stronger no-speed configuration worsened under distance supervision. Do not infer that every distance objective would fail: this screen tests one scaling, one threshold, one seed and six initially visible targets. The threshold was not adjusted after results.

Two trainings including loading took 13.8 seconds; 24 full simulations took 84.6 seconds with three workers. The notebook appends six analysis cells per configuration: setup, losses, paired results, PARK traces, reach traces and interpretation. Closest reach includes the 0.5-second observation after arm STOP, during which learned torques continue.

Reproduce with the native simulator and existing environment:

```bash
python v2_sensors/arm/park_distance_batch.py check
python v2_sensors/arm/check_train_golden.py
python v2_sensors/arm/park_distance_batch.py train --data /path/to/golden --prior-root /path/to/previous/results --out /path/to/fresh/output --device cuda
CAMERA_RENDERER=egl NATIVE_ROOT=/path/to/native CAMERA_URDF=/path/to/native/assets/summit_xl_arm_description/summit_xl_complete.urdf python v2_sensors/arm/park_distance_batch.py evaluate --data /path/to/golden --prior-root /path/to/previous/results --out /path/to/fresh/output --jobs 3
```

The prior results directory must contain `driving_height_no_speed11` and `arm_cumulative`. Training uses the Torch CUDA environment; simulation and the runtime check use the native NumPy/PyBullet environment. NumPy models, losses, metadata and all validation logs are saved. PyTorch checkpoints remain on Mint under the existing Git ignore rule. Runtime logs store `driving_park_distance_m`; distance predictions are never labelled as seconds. Existing time-based checkpoints remain supported.
