### Iteration 54: full-episode dart stack

**What changed:** Test the DART-trained model under all five disturbances and random starts. Training used action perturbations, not the full evaluation disturbance stack.

**Type:** Frozen-model stress test. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 65/100, STOP 64/100, full success 0/100; validation full success 0/100. Median parking error is 19.06 cm among 65 parked attempts. Median STOP distance is 655.6 mm among 64 stopped attempts. 35 attempts did not PARK; 1 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 53 (dart):** PARK count changes by -35; STOP by -14; successes by -1. On the same 65 targets parked by both, the median paired change in parking error is +8.04 cm. On the same 51 targets stopped by both, the median paired change at STOP is +508.9 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 300, torque 1168, STOP 582. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.
