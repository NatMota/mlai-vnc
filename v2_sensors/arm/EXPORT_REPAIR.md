On 5 November 2026, the combined DART+DAgger round 2 attempt stopped during export, after torque
training reached epoch 1200. One of 1792 checked values exceeded the existing
`atol=1e-4, rtol=1e-5` check. The selected epoch was 1198; its state had not yet
been saved. The epoch-1200 snapshot is retained as diagnostic evidence, not
substituted for the selected model.

Folding input scaling into the first layer can cause FP32 cancellation. On
10,529 training states using the saved epoch-1200 weights, the folded export
had maximum error 0.0002327 Nm and one failing value. Explicit input and output
scaling reduced this to 0.00003815 Nm with no failing values. The tolerance,
network, losses, data and training settings are unchanged.

New NPZ files store `export_input_mean` and `export_input_scale`, and torque
files also store `export_output_mean` and `export_output_scale`. Runtime code
applies these operations in training order. Separate STOP scaling is stored
under `stopnet.export_*`; it must not reuse the torque scaler because STOP
excludes DAgger rows. Existing folded NPZ files retain their previous behavior.
The selected raw checkpoint is now saved before export validation.

Validation completed before restarting:

- A one-update training check on a complete recorded DAgger trajectory passed
  the existing export assertions for driving, torque and STOP. It also checked
  that DAgger rows do not supervise auxiliary PARK/STOP labels.
- `check_export_runtime.py` compared batched exported inference with actual
  single-row simulation inference, in their separate Python environments.
  Both the new explicit format and existing round-1 folded models passed.

Run the runtime check with `--models DIR --dagger RUN.npz --reference REF.npz`.
First add `--write-reference` in the training environment; then run without
that flag in the simulation environment.

The retry uses `run_recovery_programme.py --retry-tag exportfix1`, creating
`recovery_runtime_exportfix1`, `models_exportfix1` for the failed round, and
separate logs. The original runtime, failed models, histories and logs remain.
Completed stages are reused only when their model hashes match. No test scores
select the new checkpoint or DAgger round. The original queue status is copied
to `recovery_status_before_exportfix1.json`.

Three notebook additions also failed because they subtracted a missing PARK
timestamp. The plots now use seconds since reset and label these cases
`No PARK`; parked cases retain seconds after PARK. All failures remain in the
score denominator. The notebook watcher retries these sections, appends to
the original error logs, and records resolution separately. Prior notebook
cells are preserved. New results remain pending review and publication until
their full evaluations and checks are complete.
