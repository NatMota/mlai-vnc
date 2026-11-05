### Iteration 43: full-episode basics standard

**What changed:** Standardise arm inputs using training-set means and standard deviations. Keep the driving checkpoint and training budgets unchanged.

**Type:** New training. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 100/100, STOP 91/100, full success 0/100; validation full success 0/100. Median parking error is 5.35 cm among 100 parked attempts. Median STOP distance is 168.8 mm among 91 stopped attempts. 0 attempts did not PARK; 9 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 41 (baseline):** PARK count changes by +0; STOP by +7; successes by +0. On the same 100 targets parked by both, the median paired change in parking error is +0.02 cm. On the same 77 targets stopped by both, the median paired change at STOP is -63.5 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 299, torque 1200, STOP 600. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.
