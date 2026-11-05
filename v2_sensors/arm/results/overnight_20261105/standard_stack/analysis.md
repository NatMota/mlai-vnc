### Iteration 52: full-episode standard stack

**What changed:** Test the clean-trained standardised model under all five disturbances and random starts. This is a separate comparison: the preceding noise ladder used the physical-scale baseline.

**Type:** Frozen-model stress test. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 48/100, STOP 19/100, full success 0/100; validation full success 0/100. Median parking error is 33.98 cm among 48 parked attempts. Median STOP distance is 1143.1 mm among 19 stopped attempts. 52 attempts did not PARK; 29 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 43 (basics_standard):** PARK count changes by -52; STOP by -72; successes by +0. On the same 48 targets parked by both, the median paired change in parking error is +28.72 cm. On the same 19 targets stopped by both, the median paired change at STOP is +1013.6 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 299, torque 1200, STOP 600. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.
