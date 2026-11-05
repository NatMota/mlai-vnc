We are replacing the native pre-PARK holding controller with learned arm torque control from reset. Mint hosts execution, notebook, camera recordings and MLflow. No Powerglide availability is assumed.

## Frozen baseline plan: tickets 11109 and 11110

Seed 7. Keep iteration 40's driving weights unchanged. Train the arm from scratch on all rows in the 100 golden training episodes, including the pre-PARK expert holding commands. Fit outputs on training rows only and select epochs using the 100 validation trajectories. Arm inputs: seven counted joint angles, seven finite-difference encoder speeds, three camera/tracker coordinates and the current PARK flag. At deployment PARK comes from the driving network, never the expert. Torque network 18 -> 128 -> 128 -> 7, with its existing auxiliary distance objective; independent STOP 18 -> 32 -> 16 -> 1. ReLU, FP32, Adam 0.01, 1,200 torque epochs and 600 STOP epochs, validation-selected checkpoints. The camera remains at 30 Hz; learned arm torques run at 1 kHz from the first tick. No PD or added gravity compensation. STOP is only actionable after PARK and learned torques continue during the 0.5 s observation.

Prediction: including pre-PARK rows should produce nonzero learned holding torques and prevent immediate arm collapse. It may still fail because the expert's holding trajectories cover few deviations. That failure is evidence for recovery data, not permission to restore the conventional controller. Preflight on validation only: verify no hidden hold, nonzero learned torque before PARK, finite trajectories, own phase input and recorded command timing. Then all 100 validation and 100 test cases, with all failures counted. Existing test targets have been viewed before; results are subsequent comparisons, not an untouched generalisation test.

## Cumulative queue

1. 11111: encoder angle noise before counting, standard deviation 0.0001 rad; derive velocities from consecutive noisy counts. Verify course sampling semantics.
2. 11112: retain noise; add mass-matrix-scaled damping, c=1 /s.
3. 11113: retain preceding effects; add elasticity, k=1 /s².
4. 11114: retain preceding effects; add previous-acceleration motor correction, J=0.001 kg m².
5. 11115: retain ladder; add causal observation latency, starting with the 3 ms course setting.
6. 11116: retain effects; random starting poses. Collect 100/100/100 new expert references and render train/validation. Compare fixed and retrained models.
7. 11117: retain preceding setup; remove a declared window of prepared training rows around PARK and retrain. Preserve sensor history when masking.
8. 11120, 11121, 11122: compare DART-inspired augmentation, DAgger alone, and DART followed by DAgger, on clean and the cumulative condition. DAgger uses training targets only, with a stateful expert and recorded internal reference. Never invent time-to-PARK labels on student trajectories.

Each frozen condition receives 100 validation and 100 test full episodes. Test outcomes are reporting only. Occlusion 11118 and direct weight editing 11123 are deferred. The old independent-lens stack definition in 11119 is superseded by Nat's cumulative instruction.

## If a stage deteriorates

Check implementation, causal timestamps and phase coverage first. Compare physical versus training-derived scaling, relevant input ablations, and bounded training durations on validation, one change at a time. Record every attempted variant and failed result. Do not weaken effects or select epochs from test outcomes. If those checks do not restore a usable model, branch to recovery learning from the last usable stage, retaining the complete failed cumulative branch. No search rule, PPO, arbitrary weight perturbations, new URDF inputs or conventional arm hold may be introduced.

## Status

Iteration 41 is published at commit c8c46d3 and MLflow run 8ba2707ced684ce5aec12b536ee51a38. It completed 100 validation and 100 test episodes on Mint. Test: 100 PARK, 84 STOP, zero within 10 mm at STOP; median PARK error 5.47 cm and median STOP error 272.7 mm. The five-stage cumulative effect queue is running.

Standardised arm inputs improved full validation and are frozen for testing and recovery training. The 300-epoch check did worse and was not selected. DAgger's oracle and training masks have passed their checks; the three-round recovery experiments are queued, not completed. Read ladder_status.json and recovery_status.json on Mint for current execution state. Completed simulations still need plots, notebook review and publication; the queue does not mark them published.

Random starts use x/y in [-0.15, 0.15] m and heading in [-30, 30] degrees for this cumulative overnight run, rather than the wider independent lens originally in ticket 11116. This is the declared modest severity requested by Nat, chosen before evaluation. The supplied expert's world-origin parking convention remains unchanged. Fresh references are collected under clean physics; the cumulative effects are evaluation stresses. Random-start training uses physical scaling for its matched baseline comparison; the separate recovery branch uses validation-selected standardisation.

The exact remaining execution order is in run_recovery_programme.py: standardised model test; wait for the frozen ladder and collection; second DART seed; random-start fixed/retrained/clean-retention comparisons; PARK-window masking; DART training; three DAgger rounds alone; three rounds starting from DART; matched clean and cumulative evaluations. Test scores are never read by selection. Occlusion and direct weight changes remain excluded. An execution or export error stops the queue and retains its files for diagnosis.
