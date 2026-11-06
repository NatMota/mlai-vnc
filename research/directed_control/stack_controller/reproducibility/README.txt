SOURCE/MODEL REPRODUCIBILITY PACKAGE
Classification: engineered data-derived controller; learned geometry and torque coefficients plus explicit feedback structure. Not pure BC/direct edits only.

Timing: native physics/control step1ms; camera rendering/detection every round(frame_index*1000/30) ticks at256px. Preview recording downsamples to128px only after rendering; detector remains256px. The3ms delayed camera packet is read on30Hz camera decisions, so accepted observation age is typically33-34ms, not3ms. Capture-to-delivery compensation uses only delivered wheel speeds, constant-twistSE2 over exactstepage, wheelradius.11m/track.66m; z unchanged. Drivingcamera and tracker remain unchanged. Armencoder packets retain3msdelay; velocityEMAalpha1/6 has approximately5ms lowfrequencylag. Native truth is diagnostic only.

Requested torque is clipped to30Nm before existing motor effects. Actual sent torque includes motor_extra and is not clipped again. Both are logged; never infer actual30Nm bound. Controller continues unchanged0.5s after STOP.

Models: position.npz learned seven-angle trig map; torque.npz learned g/Kp/Kd with explicit per-jointKd edits; hybrid_bounds.npz training-derived q2posture/q6reference lower priors. Baseline driving network frozen. Baseline reaching NPZ supplies runtime metadata only; structuredarm/STOP replaces itsoutputs.

This archive is a portable SOURCE/MODEL snapshot plus Linuxbinary, not a self-contained executable environment. Nativebinary has pinned Python/Bullet/library ABI and existingenvironment paths. pixi.lock/pixi.toml, packageversions andldd snapshot are supplied; archivedCMake files are originalbuild metadata, not a newly verified relocatable rebuild. Flatnative source files correspond to originalsrc layout. NativeURDF/assets are simulator scene inputs, never controllergeometry inputs.

Entry point: evaluate_hybrid_geometry.py --camera calibrated_arm_only --split val --record-camera --out <folder> --ids <declaredids> --jobs 2, using pinned nativeenvironmentPython. Test uses explicit--split test and matching hash-verified expert/test records only after freeze; no inheritedval case_for. Data/results are excluded from sourcearchive and require their separate manifest/receipts.

REFERENCE-RECOVERY VARIANT (VALIDATION IN PROGRESS): before integrating ownreference, v=max(v,min(.1,(training_lower-ownreference)/.001)) each1ms fromreset. This is a disclosed rate-limited reference recovery using existing training priors, not physical URDF limits, not instantaneous projection. Joint6Kd is halforiginal, q4half, odds1/16; q2original. Gain2.5/cap.5/Cartesian40.
Source snapshot accompanies stackd589 binary only. It does NOT claim to reproduce historicalclean4fc; root recovered separate historicalclean source and verified bit-identical rebuild.
