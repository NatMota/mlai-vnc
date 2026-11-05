### Iteration 56: full-episode dagger stack

**What changed:** Test the selected DAgger model under all five disturbances and random starts. This is the same checkpoint as its clean test.

**Type:** Frozen-model stress test. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 64/100, STOP 61/100, full success 0/100; validation full success 0/100. Median parking error is 19.64 cm among 64 parked attempts. Median STOP distance is 495.6 mm among 61 stopped attempts. 36 attempts did not PARK; 3 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 55 (dagger):** PARK count changes by -20; STOP by -8; successes by -2. On the same 61 targets parked by both, the median paired change in parking error is +12.14 cm. On the same 50 targets stopped by both, the median paired change at STOP is +439.7 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 292, torque 1186, STOP 600. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.
