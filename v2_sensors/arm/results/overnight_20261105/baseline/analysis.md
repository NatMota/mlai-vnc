### Iteration 41: full-episode baseline

**What changed:** Learn arm torques from reset, including while driving. Removes the expert holding controller. Physical input scales; frozen iteration-40 driving.

**Type:** New training. Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. No conventional arm holding controller or extra gravity compensation is active.

**Result:** test PARK 100/100, STOP 84/100, full success 0/100; validation full success 0/100. Median parking error is 5.47 cm among 100 parked attempts. Median STOP distance is 272.7 mm among 84 stopped attempts. 0 attempts did not PARK; 16 parked without STOP. Neither is silently discarded from success.

**Training provenance:** selected epochs: driving 299, torque 1198, STOP 593. The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.

**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.
