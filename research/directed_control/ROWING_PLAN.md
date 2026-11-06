# Unpowered-wheel locomotion demonstration

This is a separate, non-imitation experiment. It starts only after the clean and disturbance evaluations finish and their frozen results are preserved. It does not replace either evaluated controller or their scores.

The objective is forward base displacement produced by arm-ground contact while all four wheel actuator torques remain exactly zero. A zero wheel-speed command is insufficient because the existing wheel servo would brake the wheels. The isolated simulator must bypass that servo and record the actual four sent torques on every step. Fingers retain the existing fixed-position configuration. No teleportation, external forces, moving floor or hidden powered wheels are allowed.

First compare a stationary arm with a repeated arm stroke using identical physics and initial state. Record base displacement, arm position, contact evidence, all actuator torques and energy. Use the current course friction and geometry. The control trajectory may be explicitly designed: this is a locomotion showpiece, not a claim of learned imitation or a general navigation policy.

Start with a short stroke near the front of the robot, then recover the arm above the floor. Keep the original 1 kHz physics, 30 N m arm request limit and fixed fingers. The learned geometry map is a candidate for generating reference motion, but floor-reaching postures are outside its demonstration distribution. Compare its position estimate against diagnostic simulator geometry before relying on it; do not hide extrapolation error.

Retain unsuccessful strokes and a no-stroke control. A convincing result requires net forward travel beyond the stationary-control drift, verified zero wheel actuator work, and recorded arm-ground contact. Render the actual trajectory, with a clear label that this is an engineered demonstration. If the available arm reach or friction prevents locomotion, report that rather than altering the physics to manufacture it.
