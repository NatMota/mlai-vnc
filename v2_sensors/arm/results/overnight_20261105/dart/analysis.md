### Iteration 53: full-episode dart

**What changed:** Add two batches of 100 perturbed expert episodes to the original data; 199 reached successfully and contributed training rows. Fixed action-noise standard deviations: wheels 0.025 rad/s, arm 0.5 N m, held for 33 ms. Labels are clean expert commands at visited states. DART-inspired fixed noise, not the full adaptive DART algorithm.

**Type:** Recovery training. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 100/100, STOP 78/100, full success 1/100; validation full success 2/100. Median parking error is 5.46 cm among 100 parked attempts. Median STOP distance is 60.9 mm among 78 stopped attempts. 0 attempts did not PARK; 22 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 43 (basics_standard):** PARK count changes by +0; STOP by -13; successes by +1. On the same 100 targets parked by both, the median paired change in parking error is -0.10 cm. On the same 71 targets stopped by both, the median paired change at STOP is -20.7 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 300, torque 1168, STOP 582. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.

**Interpretation:** compare both approach accuracy and completion. A smaller STOP median can coexist with fewer STOPs. The six-case development scores are not estimates of performance across all 100 test targets.
