# Golden sensor driving + two-rate arm

The final G evaluation achieved **1/100 successes, 100/100 PARKs and 68/100 STOPs**.
Nat iteration 43 achieved 0/100; iteration 57 achieved 4/100 with a different driving network.
Each full-arm test episode was executed once. No G outcome was used for tuning.

## Experiment and success definition

The experiment attaches the exported two-rate arm to Nat's iteration 43
basics_standard driving network. Nat's source snapshot is commit
d89051b4f971c54d59c81bf71f9f92cb4f8e2833; original files and hashes are recorded in
../nat_golden/SOURCE.md.

- Driving model: ../nat_golden/models/driving_seed7.npz, seed 7, 11 inputs
  (wheel velocities omitted, tracked camera height included), unchanged ttp_thresh.
- Arm: ../outputs/v2_arm/arm_no_world_inputs.pt, source run 540bdcf9. The exported
  arm contains the velocity/torque networks with world-frame inputs removed.
  Velocity network scheduling and the 0.15 rad accumulator bound are inherited
  unchanged from the existing Two-rate arm/V2 arm implementation.
- Physics: 1 ms; driving decisions at round(k * 1000 / 30); wheels locked to zero
  after PARK. The arm is held before PARK and uses encoder angles/speeds,
  robot-frame forward kinematics and the height-preserving camera tracker after PARK.
- Success: PARK occurred, the controller issued its own STOP, the true
  robot_j2s7s300_end_effector distance at that pre-step STOP observation was at most
  10 mm, and 500 normal physics transitions completed afterwards. The arm controller
  continues after STOP. Without STOP, timeout is expert stop_s + 4 seconds.
- closest_mm is the minimum true distance observed after PARK, including continued
  control after STOP. It is not the success criterion. end_mm is the terminal
  distance. park_error_cm compares the first PARK base xy with the expert's first
  parked frame.
- Learned driver/arm observation features and the STOP distance estimate use
  sensor readings and the tracked target. Truth also renders the camera and scores
  errors. By the prescribed existing implementation, the pre-PARK hold controller
  uses simulator q/qd and gravity; inherited reset initializes arm memory from the
  reset joint state. Thus 'sensor-only' describes the learned observations and
  STOP estimate, not every computation in the complete simulation.

## Frozen STOP rule and validation

The original rule is retained: estimated hand-to-tracked-target distance at most
5 mm for 100 consecutive 1 ms steps, checked only after PARK and latched once
issued. An above-threshold sample restarts the count. The only allowed alternative
was 8 mm for 100 ms if STOP almost never occurred in S.

S uses the smallest 12 validation IDs: 6, 16, 23, 45, 55, 80, 86, 93, 104, 114,
121 and 131. It produced 12 PARKs, 10 STOPs and 1 success, with no process errors.
STOP was not rare, so the 8 mm alternative was not used. The frozen decision and
validation-summary SHA256 are in ../outputs/golden/protocol.json; no rule was
changed after this decision.

| S statistic | Measured value |
| --- | ---: |
| Success within 10 mm at STOP | 1/12 |
| PARK | 12/12 |
| STOP | 10/12 |
| Median distance at STOP, stopped cases | 22.6345 mm |
| Median closest post-PARK distance | 21.0264 mm |
| Median parking error | 7.0030 cm |
| stop_outside_10mm | 9 |
| no_stop_timeout | 2 |

Validation point 6 illustrates the difference between the sensor estimate and the
score: estimated distance at STOP was 4.2960 mm, but the true distance was
37.1386 mm. Its last logged camera xy error was 3.8390 cm. The last camera error
is not a measurement captured exactly at STOP, so this is diagnostic evidence,
not a causal isolation experiment.

## Driving-only P check

P uses the smallest 12 test IDs: 11, 17, 33, 35, 40, 43, 44, 48, 52, 56, 65 and
83. The arm remains held throughout; the episode ends 0.5 seconds after PARK.
Its no_stop_timeout status is a consequence of disabling arm STOP, not an arm
performance result.

All 12 points parked in both implementations. Absolute parking-time difference
had a median of 0.066 s and maximum of 0.300 s. Median error relative to each
point's expert parking position was 5.5333 cm for The two-rate simulator and
5.5886 cm in Nat's saved metrics. These two medians are errors to the expert,
not the distance between the two simulators' parking positions.

Per-point P values: ../outputs/golden/P_compare.csv.
Parking-only summary: ../outputs/golden/P_compare_summary.json.

## Final G and pointwise comparison

The single final batch completed all 100 cases with zero process errors.
The only successful point was **646**, at 9.8819 mm when STOP was issued
(estimated distance 4.9805 mm). Its final time was 14.706 s, exactly 0.5 s after
STOP at 14.206 s.

| Metric | Two-rate arm arm + Nat iteration 43 driver | Nat iteration 43 |
| --- | ---: | ---: |
| Success at STOP within 10 mm | 1/100 | 0/100 |
| PARK | 100/100 | 100/100 |
| STOP | 68/100 | 91/100 |
| Median distance at STOP, own stopped cases | 16.9519 mm | 168.8090 mm |
| Median closest post-PARK distance | 16.6033 mm | 75.0662 mm |
| Median parking error, own parked cases | 5.4410 cm | 5.3483 cm |

Two-rate arm statuses: success 1, stop_outside_10mm 67, no_stop_timeout 32,
no_park_timeout 0. Eight cases reached a post-PARK closest distance at most 10 mm;
only one met the actual STOP-step success rule. Closest distance is not substituted
for success. At-STOP medians above use different stopped subsets (68 vs 91).

The 2×2 success table is:

| | Nat success | Nat failure |
| --- | ---: | ---: |
| Two-rate arm success | 0 | 1 |
| Two-rate arm failure | 0 | 99 |

Both implementations parked on all 100 points, so the common-parked success rates
are Two-rate arm 1/100 = 1% and Nat 0/100 = 0%. Parking decisions agree on 100/100;
absolute parking-time difference has median 0.034 s and maximum 0.667 s. On the
12 P/G-overlap points, The two-rate parked flag, park_s and park_error_cm match
exactly between the held-arm diagnostic and the full-arm evaluation.

publication_scores.csv independently supplies these reference rows:

| Iteration | Name | PARK | STOP | Success |
| --- | --- | ---: | ---: | ---: |
| 43 | basics_standard | 100/100 | 91/100 | 0/100 |
| 57 | dart_dagger | 87/100 | 66/100 | 4/100 |

These are descriptive results under the stated training, control and simulator
differences; they do not isolate a single causal improvement. The experiment
shows smaller observed reaching distances than iteration 43, while the strict
10 mm success count remains low.

Measured batch wall times: S 112.65 s, P 69.79 s, G 770.42 s. All 100 G attempts
have one immutable receipt each. All 68 STOP episodes ran exactly 0.5 s more;
all 32 no-STOP episodes ended at their recorded expert-stop-plus-four-second limit.

Full precision values, parking times and statuses are in
../outputs/golden/compare.csv; aggregate machine-readable results are in
../outputs/golden/compare_summary.json.

### All 100 pointwise comparisons

Each paired cell is **Two-rate arm / Nat iteration 43**. A dash means no corresponding
STOP measurement. Parking error is in cm; hand distances are in mm. Values here
are rounded to two decimals; the CSV preserves full precision.

| ID | PARK T/N | Parking error cm T/N | Success T/N | At STOP mm T/N | Closest mm T/N |
| --- | --- | ---: | --- | ---: | ---: |
| 11 | Y/Y | 4.57/4.64 | N/N | 12.73/240.59 | 6.61/87.49 |
| 17 | Y/Y | 5.18/5.21 | N/N | 25.01/— | 24.97/116.88 |
| 33 | Y/Y | 5.32/5.67 | N/N | —/154.87 | 22.55/70.47 |
| 35 | Y/Y | 7.19/6.84 | N/N | —/201.54 | 12.79/22.61 |
| 40 | Y/Y | 5.98/5.91 | N/N | —/255.98 | 16.02/55.56 |
| 43 | Y/Y | 5.17/5.50 | N/N | 13.96/51.91 | 13.16/23.10 |
| 44 | Y/Y | 5.75/4.83 | N/N | 14.86/168.81 | 12.67/67.07 |
| 48 | Y/Y | 5.86/5.85 | N/N | 13.04/113.94 | 10.57/59.52 |
| 52 | Y/Y | 4.80/4.69 | N/N | 38.27/70.89 | 29.70/66.62 |
| 56 | Y/Y | 4.32/4.10 | N/N | —/312.29 | 17.24/213.16 |
| 65 | Y/Y | 24.60/24.90 | N/N | —/113.71 | 21.55/108.07 |
| 83 | Y/Y | 17.19/17.11 | N/N | —/73.05 | 20.11/72.67 |
| 92 | Y/Y | 4.96/4.98 | N/N | —/409.83 | 4.81/314.67 |
| 97 | Y/Y | 4.08/4.07 | N/N | 11.19/309.06 | 9.18/219.31 |
| 98 | Y/Y | 5.32/4.73 | N/N | 40.48/265.39 | 29.02/98.82 |
| 113 | Y/Y | 2.10/2.09 | N/N | 23.06/34.44 | 20.63/23.82 |
| 136 | Y/Y | 3.46/4.38 | N/N | 14.66/93.34 | 14.66/52.32 |
| 147 | Y/Y | 20.06/20.27 | N/N | —/114.32 | 19.49/93.31 |
| 148 | Y/Y | 8.23/8.29 | N/N | 14.00/61.00 | 12.22/26.13 |
| 154 | Y/Y | 3.17/3.12 | N/N | —/144.27 | 27.11/53.03 |
| 162 | Y/Y | 2.15/1.92 | N/N | 29.74/280.37 | 27.24/73.13 |
| 196 | Y/Y | 6.12/6.12 | N/N | 13.39/106.63 | 11.94/87.73 |
| 221 | Y/Y | 6.05/6.39 | N/N | 15.54/142.89 | 10.46/89.84 |
| 229 | Y/Y | 3.70/3.40 | N/N | —/48.94 | 11.68/38.78 |
| 232 | Y/Y | 6.57/6.66 | N/N | 15.52/213.05 | 12.55/67.69 |
| 239 | Y/Y | 6.69/6.68 | N/N | 15.38/224.42 | 13.55/80.50 |
| 248 | Y/Y | 6.64/6.55 | N/N | 12.73/157.10 | 10.76/66.39 |
| 260 | Y/Y | 21.65/21.69 | N/N | —/118.61 | 17.28/87.99 |
| 271 | Y/Y | 5.74/4.60 | N/N | 23.76/245.89 | 23.31/231.94 |
| 274 | Y/Y | 4.67/4.67 | N/N | 37.21/190.28 | 20.16/50.89 |
| 277 | Y/Y | 2.54/2.66 | N/N | 34.33/97.72 | 26.03/89.16 |
| 281 | Y/Y | 8.96/9.74 | N/N | 28.45/— | 28.05/124.70 |
| 285 | Y/Y | 4.37/4.23 | N/N | —/232.29 | 12.26/73.99 |
| 302 | Y/Y | 3.03/2.90 | N/N | 14.76/207.56 | 11.30/47.67 |
| 306 | Y/Y | 3.59/3.72 | N/N | 29.17/— | 28.94/79.30 |
| 311 | Y/Y | 1.79/1.29 | N/N | 18.58/— | 11.57/35.82 |
| 315 | Y/Y | 9.31/9.18 | N/N | 13.26/126.53 | 11.15/76.31 |
| 316 | Y/Y | 5.43/5.92 | N/N | 16.97/274.25 | 14.14/62.59 |
| 317 | Y/Y | 3.58/4.52 | N/N | —/135.31 | 24.75/42.50 |
| 361 | Y/Y | 1.99/2.00 | N/N | 40.78/112.32 | 26.46/99.02 |
| 413 | Y/Y | 5.80/5.65 | N/N | 12.89/252.02 | 12.50/74.34 |
| 427 | Y/Y | 6.84/7.20 | N/N | 25.68/470.68 | 22.05/43.64 |
| 432 | Y/Y | 5.75/6.11 | N/N | 32.77/938.42 | 27.45/108.00 |
| 463 | Y/Y | 5.85/5.50 | N/N | 18.51/152.28 | 18.47/65.39 |
| 478 | Y/Y | 5.45/5.94 | N/N | —/169.94 | 12.48/106.79 |
| 479 | Y/Y | 2.07/1.88 | N/N | —/— | 22.58/44.64 |
| 492 | Y/Y | 8.98/9.31 | N/N | 14.53/178.47 | 11.87/82.36 |
| 502 | Y/Y | 11.13/13.61 | N/N | 35.25/56.40 | 22.15/52.26 |
| 534 | Y/Y | 6.48/6.61 | N/N | 17.29/263.23 | 9.78/103.73 |
| 545 | Y/Y | 2.85/2.52 | N/N | —/126.82 | 7.48/69.19 |
| 554 | Y/Y | 5.22/5.25 | N/N | 13.93/243.81 | 12.93/131.29 |
| 571 | Y/Y | 3.56/3.93 | N/N | —/364.05 | 21.94/218.34 |
| 585 | Y/Y | 6.40/6.26 | N/N | 15.14/213.33 | 12.08/183.28 |
| 586 | Y/Y | 5.15/5.33 | N/N | 12.97/245.96 | 11.82/71.41 |
| 592 | Y/Y | 7.57/7.21 | N/N | 16.89/121.60 | 11.80/114.72 |
| 599 | Y/Y | 4.14/4.70 | N/N | —/289.38 | 30.03/172.42 |
| 600 | Y/Y | 3.73/3.76 | N/N | 32.42/75.22 | 26.90/14.20 |
| 605 | Y/Y | 4.99/4.87 | N/N | 26.37/248.17 | 25.49/75.79 |
| 628 | Y/Y | 3.90/3.49 | N/N | 30.09/32.85 | 26.42/27.71 |
| 632 | Y/Y | 2.90/2.77 | N/N | 13.21/274.10 | 10.46/47.07 |
| 642 | Y/Y | 4.95/4.88 | N/N | 29.87/602.98 | 26.26/118.12 |
| 646 | Y/Y | 3.15/2.83 | Y/N | 9.88/271.37 | 9.53/151.84 |
| 652 | Y/Y | 5.65/5.84 | N/N | 19.51/234.45 | 19.03/174.47 |
| 653 | Y/Y | 9.89/9.91 | N/N | 13.17/197.83 | 11.34/102.43 |
| 664 | Y/Y | 7.13/7.93 | N/N | 12.15/42.59 | 12.12/29.94 |
| 669 | Y/Y | 0.62/0.08 | N/N | 21.13/— | 17.19/43.07 |
| 671 | Y/Y | 3.75/3.83 | N/N | —/241.01 | 23.29/148.35 |
| 683 | Y/Y | 5.39/5.37 | N/N | 29.16/63.05 | 28.07/57.01 |
| 693 | Y/Y | 6.82/6.88 | N/N | —/78.23 | 17.62/37.21 |
| 700 | Y/Y | 4.97/4.95 | N/N | 11.07/455.07 | 10.28/275.32 |
| 707 | Y/Y | 5.76/5.61 | N/N | 26.38/22.94 | 26.35/22.12 |
| 708 | Y/Y | 8.17/8.15 | N/N | 14.22/140.73 | 11.52/89.33 |
| 722 | Y/Y | 3.33/4.01 | N/N | —/91.36 | 15.38/90.63 |
| 729 | Y/Y | 17.22/17.46 | N/N | —/73.36 | 20.39/73.15 |
| 730 | Y/Y | 3.87/3.45 | N/N | 39.91/178.25 | 13.03/106.94 |
| 738 | Y/Y | 0.77/0.62 | N/N | —/136.54 | 5.43/109.17 |
| 741 | Y/Y | 5.48/5.82 | N/N | 15.59/300.43 | 14.18/59.24 |
| 750 | Y/Y | 0.50/1.11 | N/N | —/— | 23.82/57.54 |
| 770 | Y/Y | 4.08/4.35 | N/N | —/235.30 | 19.02/84.27 |
| 779 | Y/Y | 10.04/10.94 | N/N | 11.45/208.96 | 10.08/179.36 |
| 806 | Y/Y | 9.77/9.19 | N/N | 16.93/100.77 | 12.49/37.18 |
| 834 | Y/Y | 7.36/7.44 | N/N | 14.16/204.47 | 11.68/41.30 |
| 845 | Y/Y | 5.50/4.64 | N/N | 16.54/167.18 | 11.62/149.42 |
| 846 | Y/Y | 5.05/4.99 | N/N | 19.29/49.54 | 19.09/36.68 |
| 848 | Y/Y | 10.50/10.84 | N/N | 25.07/117.83 | 24.88/115.52 |
| 854 | Y/Y | 6.71/6.42 | N/N | 29.90/205.19 | 27.67/204.43 |
| 857 | Y/Y | 3.38/2.30 | N/N | —/— | 27.01/17.10 |
| 859 | Y/Y | 3.09/3.18 | N/N | —/179.22 | 11.67/27.52 |
| 872 | Y/Y | 2.04/2.44 | N/N | —/— | 30.60/69.32 |
| 883 | Y/Y | 7.96/6.81 | N/N | 15.40/182.69 | 12.06/102.02 |
| 898 | Y/Y | 4.22/4.49 | N/N | 33.87/901.47 | 29.61/107.66 |
| 903 | Y/Y | 7.64/6.92 | N/N | —/121.83 | 11.56/30.94 |
| 905 | Y/Y | 6.46/6.88 | N/N | 29.14/135.89 | 29.04/133.93 |
| 945 | Y/Y | 7.26/7.46 | N/N | 28.81/53.22 | 25.89/50.31 |
| 952 | Y/Y | 5.74/5.83 | N/N | 37.60/118.03 | 29.59/83.38 |
| 958 | Y/Y | 8.36/6.41 | N/N | 10.48/145.47 | 9.11/93.39 |
| 981 | Y/Y | 5.71/5.30 | N/N | 15.06/194.92 | 12.51/101.07 |
| 985 | Y/Y | 18.86/19.26 | N/N | —/71.27 | 20.84/71.16 |
| 986 | Y/Y | 3.04/3.40 | N/N | —/260.62 | 13.74/180.72 |
| 989 | Y/Y | 13.93/14.36 | N/N | —/69.63 | 23.93/69.51 |

## Data boundaries and limitations

1. According to the provided training history, the arm used The two-rate own 4,000
   expert episodes, not the standard 100 training points. This continuation did
   not re-count the training set. Source run 540bdcf9 is corroborated by local
   v2_eval.py and earlier project documentation; live MLflow training provenance
   was not re-queried. This is not an equal-training-data comparison.
2. Training used expert q_des supervision; the accumulator design and 0.15 rad
   bound also come from the expert design.
3. This run renders with CPU TinyRenderer; Nat's published configuration uses EGL.
   It runs PyBullet, while Nat's rollout uses the native C++ engine. The earlier
   reported median parking-position difference of 0.5 cm belongs to a previous
   check, not a statistic re-established by this experiment.
4. Nat's test_config.json records learned arm control from reset, at 1 kHz,
   without PD hold or added gravity. Two-rate arm holds before PARK, then enables the
   learned arm. P therefore checks agreement across both rendering/physics and
   pre-PARK controller differences; identical driving weights do not isolate only
   the post-PARK reaching arm.
5. STOP is our frozen sensor-distance rule; Nat's STOP is a network output.
6. Nat iteration 57's 4/100 uses a different driving network. It is a reference,
   not a controlled same-driver arm comparison.
7. Each test point's full arm evaluation in G is executed once. P additionally
   uses 12 of those test points for the explicitly requested driving-only check.
   No G outcome is used to change rules, thresholds or model parameters.
   Nat's source test_config.json itself states these targets were previously
   evaluated, so this is not a historically unseen blind test set.
8. Nat's no-PARK rows may contain final-position error in park_error_cm. The
   pointwise CSV preserves source values; reported parking-error medians exclude
   no-PARK rows for each side. Success rates over both-parked points use the exact
   intersection as their denominator.

## Reproducibility and verification

The evaluator launches one fresh child process per case with 12 workers. Source,
model, dataset, URDF/mesh and simulation_and_control package hashes are recorded
before each batch and checked by each child. Immutable attempt/receipt records
prevent automatic re-execution of started G cases. Resume reconstructs derived
reports from matching receipts and runs only unstarted IDs.

Required environment:
KMP_DUPLICATE_LIB_OK=TRUE, PYTHONIOENCODING=utf-8, ROBOENV_HEADLESS=1,
CAMERA_RENDERER=tiny. Worker OMP_NUM_THREADS and MKL_NUM_THREADS are 1.

Interpreter:
Use the Python executable from the robot simulation environment.

Original execution provenance digest:
894db3506b74d4bb174c7acf6f5cc499a07bc459e839c96ddadf4430a09d3618

Original execution HEAD was 57f28a669d1f4787368499af2b9d977775d632e1 on
the original local working branch. The working folder already contained unrelated
changes before this continuation; no existing protected branch has been moved.

ECC code-reviewer and python-reviewer reviewed all three experiment modules and
their tests, then re-reviewed the fixes. Regression tests proved parity arm
disabling and terminal finite-state validation failed before correction. Other
tests cover copied source, per-file hashes, input width, tracker height, STOP
boundaries, timeout and 500-step continuation, process receipts/resume, frozen
protocol, joins, comparisons and exclusion of unparked errors from parking medians.

The original experiment verification contained 79 passing cases: 53 sensor/existing regression tests,
17 evaluator tests, and 9 comparison tests. A combined 76-test run passed before
the last three comparison regression tests were added; the final 9-test comparison
suite passed after those fixes. AST parsing succeeded for all six Python files.
coverage, ruff, mypy, pylint and Black are not installed in the designated interpreter;
no automated coverage percentage, lint or type-check result is claimed.

Protected-file hash audit passed for all 52 files captured before work. Nat source
hashes all match. No nat_golden/, nat_v2/, v2_sensor.py, v2_eval.py or existing shared simulation modules
file has been changed by this continuation.

Publication branch: experiments/v2-golden-arm, based on commit
80e409277744ac238352bbe3d7babc5650c2d97a. Publication adapts module names and
removes personal identifiers; it does not alter measured observations.


## Publication verification

The publication checkout uses neutral module names and starts at
80e409277744ac238352bbe3d7babc5650c2d97a. A single replay of validation point 6
matches all 29 measured fields in the original S result exactly; only run identity,
source hashes, timestamps and wall/render timings are excluded from this equality
check. No G episode was repeated.

Release verification uses 80 policy/evaluator/comparison tests, including the added
archive-source-provenance rejection test, and 17 synthetic uploader tests. The
uploader verifies that published code/document bytes match the Git commit and
checks every uploaded artifact by downloading and hashing it. It creates only a
new run in experiment 3 and checks that pre-existing run metadata stays unchanged.

Publication metadata uses neutral labels and portable paths. Original file hashes
and execution provenance remain explicitly labelled as source evidence; hashes
for exported artifacts are recorded separately. Raw original records are retained
locally. The complete source and publication mapping accompanies the MLflow results.
