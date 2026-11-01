# Fused policy, 30 Hz control and arm input ablation

## Final test result

The final model uses the teammate's base and switch networks at about 30 Hz, and an arm with no world-frame
inputs: velocity network at about 30 Hz, torque network at 1 kHz. It was chosen on validation goals (117/120,
tied with the v5 arm, with fewer inputs and the best mean over three seeds), then run once on the test goals.

| Test goals | Result |
| --- | ---: |
| Reached within 1 cm | 120/120 |
| Switched to reaching | 120/120 |
| Mean final error | 0.59 cm |
| Median final error | 0.61 cm |
| Largest final error | 1.00 cm |
| Mean time to success | 8.78 s |

The expert also scores 120/120 on these goals. Validation gave 97.5%, so 120/120 includes some good luck: a
true success rate near 97% would give a perfect score only a few percent of the time. Report validation and
test together, and treat the expected success rate as about 97-100%. Three seeds were not run on the test goals.

## New random goals

The final model and the expert then ran on 500 new goals (`--group fresh`, seeds from 900000), drawn from the
training goal box. None matches any of the 6009 goals in the data folders (training, DAgger, validation and
test); the closest is 1.19 cm away and the median nearest distance is 7.22 cm. The model was not changed.

| 500 new goals | Final model | Expert |
| --- | ---: | ---: |
| Reached within 1 cm | 492/500 (98.4%) | 497/500 (99.4%) |
| 95% interval (Wilson) | 96.9-99.2% | 98.3-99.8% |
| Final error under 2 cm | 496/500 | 497/500 |
| Mean final error | 0.67 cm | 0.69 cm |
| Mean time to success | 8.68 s | 8.63 s |

| Goal distance from the start, in the floor plane | Final model | Expert |
| --- | ---: | ---: |
| Under 0.8 m | 84/92 (91.3%) | 89/92 (96.7%) |
| 0.8-1.5 m | 281/281 | 281/281 |
| Over 1.5 m | 127/127 | 127/127 |

Every miss, by either policy, was a goal under 0.8 m from the start. The model and the expert missed different
goals. Four of the model's eight misses were close, with final errors of 1.19-1.26 cm. This is a better
estimate than the 120 test goals, but it covers one training seed and goals inside the training box only.

The sections below use the 120 validation goals in `data_dart/split.json`, the simulator state as input, and
the original stop rules: the end-effector must be within 1 cm of the goal after at least 8 s.

## What counts as good enough

The task is modelled as moving the gripper to a small object before picking it up. The red marker in the
simulator has a 3 cm radius. A 1 cm limit keeps the end-effector within one third of that radius, leaving
margin for the gripper to close around the object. This limit is our own choice; other tasks would need
different limits. Fingers stay fixed in this project, so success means reaching the goal, not grasping.

## Fusion

| Policy | Base and switch | Arm | Reached | Switched | Mean final error |
| --- | --- | --- | ---: | ---: | ---: |
| v5 | memory policy | memory policy | 97/120 | 99 | 7.27 cm |
| Team | teammate's split network | teammate's split network | 104/120 | 120 | 1.12 cm |
| Fused | teammate's split network | memory policy | 112/120 | 120 | 1.01 cm |
| Fused 30 Hz | teammate's split network, 30 Hz | memory policy, velocity 30 Hz, torque 1 kHz | 117/120 | 120 | 0.67 cm |

The teammate's checkpoint comes from MLflow run f978e343. Its NumPy re-implementation matched the original
torch networks on 200 random inputs, with a largest command difference of about 6e-6.

Fused 30 Hz has only one training seed for each network. Its switch needs two consecutive updates, 33 ms
apart, instead of 50 consecutive 1 kHz steps, so its switching rule is not identical to the 1 kHz policy.

## Goal-position error

Fused 30 Hz with a fixed offset of the given length, in a random direction chosen for each goal. The policy
sees the offset goal; success is measured at the true goal.

| Offset | Reached | Final error under 2 cm |
| ---: | ---: | ---: |
| 0 | 117/120 | 118/120 |
| 1 mm | 116/120 | 118/120 |
| 5 mm | 115/120 | 118/120 |
| 10 mm | 74/120 | 116/120 |

With a 1 cm success limit, a 10 mm offset leaves the hand about 1 cm from the true goal, so many runs narrowly
miss. A camera goal estimate with a constant error up to 5 mm kept the success rate close to the baseline.
Random frame-to-frame noise was not tested.

## Arm input ablation

Each variant retrains `vel_net` and `torque_net` (every 20th row, 12 epochs) and is evaluated as Fused 30 Hz.
A removed input is set to zero after scaling. The base and switch networks are unchanged.

| Removed from the arm | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Nothing | 112 | 105 | 114 | 110.3 |
| World goal position | 100 | 109 | 110 | 106.3 |
| All world-frame inputs | 117 | 117 | 115 | 116.3 |

One-seed runs, not repeated:

| Removed from the arm | Reached | Note |
| --- | ---: | --- |
| World base pose | 117/120 | |
| Goal relative to the base | 112/120 | one run ended 17.8 m from the goal |
| Joint speeds in the shared state | 113/120 | `torque_net` still sees `v - qd`, so this does not remove joint speed fully |

"All world-frame inputs" means base position and bearing, the goal's world position, and the world x, y
position error. The vertical position error stays. Removing them all gave the best and most consistent arm in
all three seeds, so the arm does not need world coordinates. Removing the world goal alone was worse, which
suggests the groups interact; three seeds cannot explain why. Base and switch networks still use world inputs.
