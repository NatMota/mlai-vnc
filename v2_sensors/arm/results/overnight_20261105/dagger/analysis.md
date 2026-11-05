### Iteration 55: full-episode dagger

**What changed:** Three rounds of 100 student rollouts with stateful shadow-expert labels, retaining the original data and all earlier rounds. Round 3 was selected on six fixed validation cases before full testing. New DAgger rows train command outputs only; they do not train PARK, auxiliary distance or STOP targets.

**Type:** Recovery training. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 84/100, STOP 69/100, full success 2/100; validation full success 4/100. Median parking error is 4.23 cm among 84 parked attempts. Median STOP distance is 37.0 mm among 69 stopped attempts. 16 attempts did not PARK; 15 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 43 (basics_standard):** PARK count changes by -16; STOP by -22; successes by +2. On the same 84 targets parked by both, the median paired change in parking error is -1.12 cm. On the same 66 targets stopped by both, the median paired change at STOP is -129.9 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 292, torque 1186, STOP 600. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.

**Interpretation:** compare both approach accuracy and completion. A smaller STOP median can coexist with fewer STOPs. The six-case development scores are not estimates of performance across all 100 test targets.
