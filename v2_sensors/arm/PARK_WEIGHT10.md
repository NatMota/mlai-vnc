This single-change experiment increases the PARK-distance MSE coefficient from 1 to 10. Wheel-command MSE remains at coefficient 1. Raw loss components stay unweighted in the saved history; total loss and checkpoint selection use the weighted sum.

The 11 sensor inputs, two 16-unit driving layers, seed 7, 300 epochs, FP32, Adam 0.01, training/validation split, distance scaling and 5 cm PARK threshold are unchanged. The wider arm and its separate STOP network are reused byte for byte from the weight-1 comparison. There is no arm retraining, extra near-goal weighting, history, new head or threshold adjustment. Both recorded initial raw losses match the preceding model exactly.

Train on the same 100 training runs and choose the epoch using the same 100 validation runs. Evaluate six full validation episodes from the normal start: 673, 506, 830, 720, 16, 333. No reserved-test data is evaluated. Rendering, sensors, physics, clipping and deadlines remain unchanged.

| Measure | PARK weight 1 | PARK weight 10 |
|---|---:|---:|
| Selected driving epoch | 297 | 300 |
| Raw PARK validation MSE | 0.002990 | 0.000919 |
| Wheel-command validation MSE | 0.386144 | 0.396763 |
| PARK / 6 | 6 | 6 |
| Arm STOP / 6 | 6 | 6 |
| Median parking error | 8.4 cm | 11.1 cm |
| Median closest reach | 104.9 mm | 113.2 mm |
| Median distance at arm STOP | 214.6 mm | 220.6 mm |
| Within 10 mm at arm STOP / 6 | 0 | 0 |

PARK validation MSE fell by 69%, while wheel-command validation MSE increased by 2.8%. Actual parking error worsened on every one of the six targets. This does not support the hypothesis that 10x weighting alone improves parking under the fixed conditions. The same shared trunk supplies both driving heads, so increasing the PARK coefficient can also change wheel commands. Six initially visible validation targets and one seed are a development check, not evidence of generalisation. Closest reach includes the 0.5-second observation after arm STOP, during which learned torques continue.

Training including loading took 11.8 seconds. Six full simulations took 22.5 seconds with three workers. The notebook appends raw loss curves, paired outcomes, PARK prediction traces and reach traces without changing earlier cells or creating an official MLflow run.

Reproduce with a fresh output directory:

```bash
python v2_sensors/arm/check_train_golden.py
python v2_sensors/arm/park_distance_batch.py train --data /path/to/golden --prior-root /path/to/previous/results --out /path/to/fresh/output --device cuda --park-weight 10 --configuration wider_arm
CAMERA_RENDERER=egl NATIVE_ROOT=/path/to/native CAMERA_URDF=/path/to/native/assets/summit_xl_arm_description/summit_xl_complete.urdf python v2_sensors/arm/park_distance_batch.py evaluate --data /path/to/golden --prior-root /path/to/previous/results --out /path/to/fresh/output --jobs 3
```

The prior results directory contains `driving_height_no_speed11` and `arm_cumulative`. Training uses the existing Torch environment; simulation uses the native NumPy/PyBullet environment. The gradient check verifies the 10x term against an explicit objective, including chunking. Saved checks also verify weighted totals at every epoch and unchanged arm/source hashes.
