# Directed-control research

This is an isolated experimental branch. The original behaviour-cloning models, their results and the original project notebook remain unchanged.

## What changed

The initial experiments inspected and edited the original MLP weights, biases and activations. Those changes did not establish reliable reaching. The successful validation candidate instead changes the arm architecture. It combines learned geometry and identified torque coefficients with an explicitly engineered feedback controller. It must not be reported as a small weight edit, an unchanged neural policy, or a purely end-to-end imitation learner.

The driving network remains frozen. It sends four wheel-speed commands and a predicted time to PARK. The arm runs from reset at 1 kHz. Before PARK, it maintains its own joint reference; after PARK, it moves that reference using the camera target and a learned position Jacobian. STOP comes from the estimated geometric gripper-to-target distance, with a 5 mm threshold. The original independent STOP network and redundant distance output are not controlling this candidate.

## Learned components and supervision

- Position: a tensor-product feature map of `[1, sin(q), cos(q)]` for seven encoder angles, followed by 6,561 learned linear coefficients. Training labels are the expert gripper position expressed in the base frame using recorded simulator poses. Those poses are offline supervision, not deployment inputs.
- Torque: 5,103 trigonometric feedforward coefficients plus seven identified proportional and seven identified derivative gains. Identification uses expert internal reference angles and reference velocities on training demonstrations. Runtime uses the controller's own reference instead. The learned feedforward term provides gravity-like compensation; this is not a claim that the arm works without compensation of any kind.
- Two posture bounds: an observed training minimum for joint 2 and an expert-reference plateau for joint 6. These are priors extracted from expert data, not sensor measurements. Runtime URDF joint limits are not substituted for them.
- Camera: calibrated red-target RGB/depth geometry, with known target shape and camera calibration. This is a model-based visual sensor, not a CNN trained from raw images.

Ridge strengths and controller settings were selected on validation data. The golden test targets have historical project exposure, but the current research candidate is frozen before its test evaluation and is not changed using those test outcomes.

## Explicit control structure

The controller differentiates the learned position map to obtain its Jacobian. A damped least-squares solver produces joint reference velocities; an active-set rule prevents velocities beyond the learned posture bounds. The reference integrates at 1 ms and is limited to 0.15 rad from measured angles. Count-derived velocities use a causal exponential filter. Identified joint feedback and added Cartesian damping produce the torque request.

For the clean candidate, rate-limited reference recovery toward the existing lower bounds starts from reset. It does not instantly project the arm to a new pose. The disturbance candidate is evaluated separately and need not use the same controller gains.

At deployment, the arm reads encoder angles and velocities, camera/tracker target coordinates and the PARK state. Simulator target, base and gripper poses are used only by evaluation diagnostics and rendering. Neither DART nor DAgger recovery data is used for this research branch.

## Evaluation

Canonical success requires PARK, STOP and true gripper distance no more than 10 mm at STOP. Every attempt is counted; missing STOP is a failure. A separate settling check measures the exact 1 kHz maximum distance during the next 0.5 seconds while the same controller continues running. Joint-speed oscillation and requested-torque clipping are audited separately. These are simulation diagnostics, not evidence of physical safety.

The disturbance stack contains encoder noise, damping, elasticity, additional motor inertia, a nominal 3 ms measurement delay and random starting poses. It excludes occlusion and bursty recording loss. Camera delivery occurs on the 30 Hz clock, so capture age can be 33 or 34 ms despite a nominal 3 ms delay. Any timestamp correction must use only causal delivered sensor data.

Requested torque is clipped before motor effects are added. The native field `applied_arm_torque` records that clipped request, while `sent_arm_torque` includes the motor correction. A request inside 30 N m does not guarantee the final simulated torque remains inside that limit.

## Results and reproducibility

The separate notebook loads saved results and distinguishes targeted diagnostics, full validation and frozen test evaluations. Model, source, native binary and dataset hashes accompany the freeze and evaluation records. Replays use the recorded camera frames, not regenerated images. Failed variants remain preserved alongside successful candidates.
