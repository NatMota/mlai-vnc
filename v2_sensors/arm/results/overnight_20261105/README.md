# Overnight results, 5 November 2026

[Back to V2](../../../README.md) · [Iteration index](../../../ITERATIONS.md)

The report uses iterations **41–57**. Original archive 51 is excluded; original 52–58 map to report 51–57. Folder names and saved metadata are unchanged.

The original archive IDs 41–58 each contain a completed evaluation on 100 validation and 100 test targets. The seven six-target recovery checks are development records, not official iterations. All runs use seed 7; the test set has been inspected repeatedly and is not an untouched generalisation estimate.

This Git folder contains models, training histories, configurations, summaries, data audits and per-run analysis. Full trajectory records and standalone test replays are published separately in the university OneDrive folder `COMP0245 Golden Dataset 2026-11-04 / Iterations 41-58 - 2026-11-05`, and official test records are logged in MLflow experiment 3. The notebook on Mint contains all 100 test replays for each condition.

`ARTIFACT_MANIFEST.json` gives hashes for the full records package. Large `seed7.json.gz` trajectory records listed there are in the OneDrive archive rather than Git. Additional DART and DAgger training collections remain on Mint and are not included in that archive; data audits record their lineage. These result packages alone are not a complete training-data release.

Excluded original archive iteration 51 (`imperfect_park`) removes training rows around PARK. This was outside the agreed scope and does not represent the requested bursty-packet experiment. Bursty packets remain outstanding. The cumulative disturbance ladder uses the original learned-arm baseline; standardised inputs and recovery training are separate branches, each evaluated under the stated conditions.

See `publication_scores.csv`, `publication_plan.json` and each condition's `analysis.md` for denominators, model lineage and interpretation. Issuing PARK or STOP is not task success: the official criterion also requires the gripper within 10 mm at STOP.
