### Iteration 57: full-episode dart dagger

**What changed:** Retain original and both DART datasets, then aggregate three rounds of 100 student rollouts with shadow-expert command labels. Round 3 was selected on the six validation cases. New DAgger rows do not train PARK or STOP. Round 2 export was repaired and rerun without loosening the numerical tolerance; the selected model is round 3.

**Type:** Recovery training. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 87/100, STOP 66/100, full success 4/100; validation full success 1/100. Median parking error is 4.87 cm among 87 parked attempts. Median STOP distance is 34.5 mm among 66 stopped attempts. 13 attempts did not PARK; 21 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 53 (dart):** PARK count changes by -13; STOP by -12; successes by +3. On the same 87 targets parked by both, the median paired change in parking error is -0.38 cm. On the same 59 targets stopped by both, the median paired change at STOP is -41.3 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 300, torque 1187, STOP 582. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.

**Interpretation:** compare both approach accuracy and completion. A smaller STOP median can coexist with fewer STOPs. The six-case development scores are not estimates of performance across all 100 test targets.
