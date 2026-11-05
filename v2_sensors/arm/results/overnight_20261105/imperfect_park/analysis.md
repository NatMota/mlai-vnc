### Iteration 51: full-episode imperfect park

**What changed:** Retrain after omitting prepared training rows within ±0.25 s of PARK. This was an unapproved substitution for bursty packet loss. Preserve it as exploratory evidence only; it does not complete the requested packet-loss experiment.

**Type:** Outside agreed scope. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 97/100, STOP 97/100, full success 0/100; validation full success 0/100. Median parking error is 8.99 cm among 97 parked attempts. Median STOP distance is 945.0 mm among 97 stopped attempts. 3 attempts did not PARK; 0 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 49 (random_start_retrained):** PARK count changes by +12; STOP by +12; successes by +0. On the same 82 targets parked by both, the median paired change in parking error is -17.03 cm. On the same 82 targets stopped by both, the median paired change at STOP is +153.8 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 299, torque 1194, STOP 599. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.

**Scope correction:** bursty packets remain untested. Do not describe this row deletion as packet loss or mark that roadmap task complete.
