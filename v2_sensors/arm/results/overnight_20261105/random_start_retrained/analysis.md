### Iteration 49: full-episode random start retrained

**What changed:** Retrain driving and reaching on random-start expert data with physical input scales. Evaluate with random starts and all five disturbances. The expert retains its original world-origin parking convention.

**Type:** New training. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 85/100, STOP 85/100, full success 0/100; validation full success 0/100. Median parking error is 29.51 cm among 85 parked attempts. Median STOP distance is 801.3 mm among 85 stopped attempts. 15 attempts did not PARK; 0 parked without STOP. Neither is silently discarded from success.

**Comparison with iteration 48 (random_start_fixed):** PARK count changes by -4; STOP by -4; successes by +0. On the same 75 targets parked by both, the median paired change in parking error is +13.79 cm. On the same 75 targets stopped by both, the median paired change at STOP is -91.7 mm. Negative changes mean smaller errors. These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.

**Training provenance:** selected epochs: driving 300, torque 1174, STOP 599. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.
