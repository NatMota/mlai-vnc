We evaluated the four existing time-to-PARK configurations on the same 100 test targets each, retaining their seed-7 checkpoints and original training histories. The results and animations are appended to the golden notebook as iterations 37 to 40.

| Iteration | Cumulative change | Driving | Arm torque |
|---|---|---|---|
| 37 | Add tracked camera target height | 17 inputs, 16/16 hidden units, 300 epochs | Original 17-input, 64/64, 300 epochs |
| 38 | Remove six wheel-speed inputs | 11 inputs; all other driving settings unchanged | Same original arm |
| 39 | Train the arm longer | Same iteration-38 driving checkpoint | 64/64, up to 1,200 epochs |
| 40 | Widen arm layers | Same iteration-38 driving checkpoint | 128/128, up to 1,200 epochs |

Each driving model outputs four wheel speeds and time remaining until expert PARK. The driving loss is wheel-command MSE plus time-to-PARK MSE, with equal weight per expert run. Time labels are clipped at three seconds and divided by three; PARK latches zero wheels when predicted time is at most 0.25 seconds. The arm torque loss and independent distance-based STOP network retain their existing definitions and thresholds. The STOP checkpoint is identical across all four configurations.

No model was retrained for this evaluation. Checkpoints were selected by their original offline validation losses. The frozen model hashes are recorded before any test trajectory is opened. Only the camera/tracker and encoder features enter the policies. Simulator truth is used for rendering and evaluation; expert records supply the ghost, parking reference and fixed deadline, not student commands.

The six earlier validation targets were initially visible. This evaluation includes all 100 test targets across bearings, ranges and heights, including blind starts, without adding a fixed search rule. These targets had already been evaluated for baseline iterations 34 to 36 and informed diagnosis, so this is a subsequent frozen comparison, not a new blind test. It is one seed per configuration. Do not use these outcomes to pick checkpoints or thresholds.

## Results

All 400 simulations completed in 1,132.6 seconds (18.9 minutes), with no execution failures.

| Iteration | PARK | STOP | Median park error | Median error at STOP | Success within 10 mm |
|---|---:|---:|---:|---:|---:|
| 37: camera height | 87/100 | 80/100 | 18.7 cm | 38.7 cm | 0/100 |
| 38: remove wheel speeds | 100/100 | 92/100 | 5.6 cm | 29.8 cm | 0/100 |
| 39: longer arm training | 100/100 | 98/100 | 5.6 cm | 19.4 cm | 0/100 |
| 40: wider arm | 100/100 | 96/100 | 5.6 cm | 8.5 cm | 2/100 |

The driving improvement persists across the full target set. The last three configurations have identical parking outcomes, as expected from their shared driving checkpoint. Longer arm training improves distance at STOP across the full set, unlike the six-case validation result, but worsens median closest approach from 52.5 to 98.3 mm. Widening brings median closest approach to 59.1 mm and improves STOP distance to 85.2 mm. Two successes out of 100 still indicate weak completion performance. Conditional medians exclude attempts without the relevant event; their counts are shown above.

## Simulation and reporting

The full renderer and 256-pixel camera are preserved. Camera and driving update at 30 Hz; arm encoders, torque control and physics update at 1 kHz. The native arm-hold controller remains active before PARK. The fingers stay fixed. No new noise, delay, motor damping, elasticity or motor-inertia correction is enabled. Existing quantisation, camera error and simulator quirks remain.

The existing batched driving path retains each 1 ms physics and holding tick. The arm still receives fresh feedback every millisecond after PARK. Every attempt keeps its fixed deadline of expert completion plus four seconds, or completes a further 0.5-second observation after STOP. Execution failures and missing PARK/STOP remain in the denominator. Success requires PARK, STOP and actual gripper distance at most 10 mm at the exact STOP step, without early termination.

Parking-error medians include parked attempts only; STOP-error medians include stopped attempts only. Closest distance includes the 0.5 seconds after STOP, during which learned holding torques continue. Passing close to the target is not a successful stop.

Each notebook section contains its score table, original training and validation losses, outcome plots, stopping analysis, distance traces, and a player with all recorded test attempts. The player includes the ghost expert, measured gripper distance, arm torques and a camera inset. Camera previews are 128 pixels, derived from the actual 256-pixel sensor frames; previews are never policy inputs. Images load on demand and players load only when clicked.

## Reproduce and view

Use the established golden dataset, native simulator and EGL environment. `PRIOR` contains the preceding experiment folders. Choose a new output directory to preserve earlier results.

```bash
python evaluate_time_cumulative.py --data "$GOLDEN" --prior-root "$PRIOR" --out "$OUT" --jobs 6
```

A matching interrupted run can be continued with `--resume`; completed attempts are retained. The script verifies the frozen source, model and native hashes on resume. `report_time_cumulative.py --build` creates the tables and players from saved records, without rerunning simulation. The notebook reads `results/time_cumulative_test`, which points to the recordings on Mint.

The existing replay server serves `time_cumulative_test/VARIANT/viewer_golden_seed7.html`, where `VARIANT` is `height`, `no_speed`, `longer_arm` or `wider_arm`. Its root links to the new result folder; existing players remain available. The MLflow replay archive for each iteration can also be extracted and opened locally. Keep the camera folders beside the HTML; the simulator is not needed for playback.

`log_time_cumulative.py` refuses to publish until all 400 attempts are accounted for. It uploads the frozen models, original per-epoch training and validation histories, test measurements, replay archive and executed notebook. It verifies the run status, epoch records and model/replay artifacts after logging. Validation smoke checks remain unnumbered.
