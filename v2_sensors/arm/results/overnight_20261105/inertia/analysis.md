### Iteration 46: full-episode inertia

**What changed:** Retain previous effects; add motor inertia J=0.001 kg m² using the previous acceleration. Networks unchanged.

**Type:** Frozen-model stress test. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 100/100, STOP 94/100, full success 0/100; validation full success 0/100. Median parking error is 7.17 cm among 100 parked attempts. Median STOP distance is 779.3 mm among 94 stopped attempts. 0 attempts did not PARK; 6 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 45 (elasticity):** PARK count changes by +1; STOP by +1; successes by +0. On the same 99 targets parked by both, the median paired change in parking error is +0.08 cm. On the same 89 targets stopped by both, the median paired change at STOP is +0.3 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 299, torque 1198, STOP 593. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.

**Interpretation:** this measures cumulative sensitivity of the original physical-scale model. It does not isolate the same effect on the standardised model. Differencing noisy angle readings can amplify velocity noise; this is a plausible mechanism, not a demonstrated sole cause.
