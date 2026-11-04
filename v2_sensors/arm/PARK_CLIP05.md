The 0.5 m target clip improved median parking error from 8.4 to 6.7 cm on the same six validation episodes. Extending training to 1,000 epochs lost one PARK and did not improve the median among the remaining five.

| Setting | Selected epoch | PARK / STOP | Median park error | Median arm error at STOP | Offline first trigger within 5 cm |
|---|---:|---:|---:|---:|---:|
| 3 m clip, weight 1, 300 epochs | 297 | 6/6 | 8.4 cm | 21.5 cm | 38/100 |
| 3 m clip, weight 10, 300 epochs | 300 | 6/6 | 11.1 cm | 22.1 cm | 62/100 |
| 0.5 m clip, weight 1, 300 epochs | 300 | 6/6 | 6.7 cm | 10.8 cm | 60/100 |
| 0.5 m clip, weight 1, 1,000 epochs | 993 | 5/6 | 6.8 cm among five | 18.5 cm among five | 37/100 |

Neither new model achieved the full 10 mm success criterion on any of the six episodes. The 1,000-epoch model failed to park on target 830; its missing PARK and STOP are excluded from distance medians, not counted as successes. The six targets were 673, 506, 830, 720, 16 and 333. These are repeatedly used development cases and one seed, not an independent generalisation result.

## Protocol

Both new models use seed 7, 11 sensor inputs, 16/16 hidden layers, Adam at 0.01, equal weight per expert run, wheel-command MSE plus normalized PARK-distance MSE, and the exact same wider arm and separate STOP checkpoint. The target is horizontal distance to the expert's final PARK base position, clipped at 0.5 m and divided by 0.5. The physical PARK threshold remains 5 cm (0.1 normalized). Expert post-PARK drift remains in the labels.

The second training repeats the same initialisation and updates through epoch 300, verified exactly from all six loss fields, then continues to 1,000. Both select the minimum total offline validation loss. Both use all 100 training runs and all 100 offline validation runs. Only six validation episodes are simulated, with both learned networks, full rendering, 30 Hz camera/driving and 1 kHz arm torque/physics. No test trajectory is opened. There are no official MLflow runs or animations for these checks.

Changing the scale increases the contribution of physical errors below 0.5 m by 36 times relative to the 3 m normalization, while clipping longer distances. It changes loss balance as well as target shape. Normalized PARK losses across different clip scales are not directly comparable. Physical prediction error and first-trigger counts are reported instead.

The predicted improvement to above 70/100 accurate first triggers and about 5 cm live parking was not reached. At 300 epochs, accurate first triggers increased to 60/100 and first triggers beyond 10 cm fell from 17 to 8. At 1,000 epochs, mean error near PARK improved further, but accurate first triggers fell to 37/100. First threshold crossing and average regression error measure different behaviour. These results do not establish a single cause for the remaining errors.

Training including loading took 11.7 and 14.6 seconds; simulations took 20.5 and 24.8 seconds respectively. The notebook contains results, losses, offline prediction checks, PARK traces and reaching traces for each model. Closest reach includes the 0.5-second observation after STOP, during which learned holding torques continue; distance at STOP is measured at the command.

## Reproduce

Use the established golden dataset and environment. `PRIOR` contains the previous experiment folders and `OUT` is a new path. Repeat with `EPOCHS=300` and `EPOCHS=1000`, using separate output folders.

```bash
python park_distance_batch.py check
python park_distance_batch.py train --data "$GOLDEN" --prior-root "$PRIOR" --out "$OUT" --device cuda --configuration wider_arm --park-weight 1 --park-scale-m .5 --epochs "$EPOCHS"
python park_distance_batch.py evaluate --data "$GOLDEN" --prior-root "$PRIOR" --out "$OUT" --jobs 3
python audit_park_clip.py --data "$GOLDEN" --root "$PRIOR"
```

`audit_park_clip.py` compares the saved baseline, weight-10 and two clip experiments. It verifies identical first 300 updates, frozen arm hashes, the exported physical threshold and checkpoint selection, and saves per-target offline first-trigger results.
