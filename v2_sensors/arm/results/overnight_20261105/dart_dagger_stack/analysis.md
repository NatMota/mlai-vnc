### Iteration 58: full-episode dart dagger stack

**What changed:** Test the selected combined model under all five disturbances and random starts. No bursty-packet loss, occlusion or direct weight perturbation is included.

**Type:** Frozen-model stress test. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 87/100, STOP 86/100, full success 0/100; validation full success 0/100. Median parking error is 19.59 cm among 87 parked attempts. Median STOP distance is 321.9 mm among 86 stopped attempts. 13 attempts did not PARK; 1 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 57 (dart_dagger):** PARK count changes by +0; STOP by +20; successes by -4. On the same 80 targets parked by both, the median paired change in parking error is +12.43 cm. On the same 61 targets stopped by both, the median paired change at STOP is +289.9 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 300, torque 1187, STOP 582. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.
