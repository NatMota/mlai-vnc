# Glossary

[Home](../README.md) · [Code map](https://github.com/NatMota/mlai-vnc/blob/e0e302de5c61a333a56feb71ee99f8d39637f13b/docs/CODE.md) · [Iteration index](https://github.com/NatMota/mlai-vnc/blob/e0e302de5c61a333a56feb71ee99f8d39637f13b/v2_sensors/ITERATIONS.md)

Terms used in the report, notebooks and saved results. Individual experiments may use different inputs or protocols; their configurations give the exact settings.

## Task and measurements

| Term | Meaning here |
|---|---|
| Mobile manipulator | A wheeled robot with an arm, used here to drive towards a target and reach it. |
| Expert / demonstration | The reference controller and a recorded episode of its actions and states, used to teach or compare a learned policy. |
| Policy | A model that maps its inputs to actions, such as wheel speeds or arm torques. |
| PARK | The signal that stops the wheels and starts the reaching phase. |
| STOP | The learned signal that ends the reach. Evaluation continues for 0.5 seconds to observe subsequent motion. |
| Full run / closed-loop evaluation | The learned policies control the robot through an episode; their actions affect the states they encounter next. In the full-episode V2 comparisons, this starts at reset and includes learned arm holding. |
| Offline evaluation | Predictions are compared with recorded labels without letting the model control the robot. A low offline error does not establish successful control. |
| Success, V2 full-episode criterion | The robot issues PARK and STOP and the gripper is within 10 mm of the target at STOP. Earlier checks and teammate comparisons may use other criteria. |
| Distance at STOP | Gripper-to-target distance when STOP is issued. A run without STOP has no value for this measurement and remains a failed attempt in the overall success count. |
| Closest distance | The minimum gripper-to-target distance during the observation window, including up to 0.5 seconds after STOP. It is distinct from distance at STOP. |
| Park error | Distance between the robot's parking position and the expert's reference parking position. |
| Bearing | Target direction relative to the robot's starting orientation, in degrees from straight ahead. |

## Learning and sensors

| Term | Meaning here |
|---|---|
| BC: behaviour cloning | Supervised learning to match the expert's actions on recorded demonstration states. |
| DART-inspired / DART-style | This project's fixed-noise augmentation: perturb expert commands during collection while retaining the expert's clean command as the training label. It is not the full adaptive DART algorithm. |
| DAgger: Dataset Aggregation | Run the learner, label the states it visits with expert actions, add them to the training data and retrain. The reported recovery programme uses three rounds. |
| MLP: multilayer perceptron | A feed-forward neural network built from layers of weighted sums and nonlinear activations. |
| MSE: mean squared error | An average of squared prediction errors. Check the stated scaling and weighting before comparing losses. |
| Encoder | A sensor that measures wheel or joint rotation. V2 quantises angles to encoder counts and estimates speeds from successive readings. |
| RGB-D camera | A camera providing colour images and depth. V2's selected mount is on the front-right chassis corner. |
| IMU: inertial measurement unit | A sensor providing acceleration and angular-velocity measurements; tested in arm-input ablations. |
| FK: forward kinematics | Computing the gripper pose from joint angles and robot geometry. |
| Simulator truth / oracle input | A value read directly from simulator state. Whether an input is available at deployment depends on its sensing source, not simply its coordinate frame. |
| Training label / supervision | The target value a model learns to predict. It can come from the expert or simulator without being an input available to the policy at runtime. |
| Ablation | A controlled comparison that removes or changes a component to examine its effect. |

## Experiment records

| Term | Meaning here |
|---|---|
| Iteration | A numbered configuration and evaluation in the report. Some iterations reuse frozen models rather than train new ones. |
| Condition key | An unchanged experiment name, such as `basics_standard` or `dart_dagger`, identifying its evidence folder. Use the iteration index to resolve historical numbering. |
| Frozen reference | A fixed checkpoint and configuration used for comparison. “Frozen” does not mean the robot solves the task reliably. |
| Checkpoint | Saved model parameters, together with the architecture and settings needed to use them. |
| Seed | A value controlling a pseudorandom sequence, for example model initialisation. A result from one seed does not establish robustness across seeds. |
| Epoch | One pass through the training data during optimisation. |
| Training / validation / test split | Data used to fit parameters / select models and settings / evaluate the selected configuration. Repeatedly inspected test results are identified as reused-test comparisons. |
| Golden dataset | The named reference collection and recorded split used for the later V2 comparisons; “golden” is a dataset label, not a claim of flawless data. |
| Disturbance stack | The declared cumulative combination of encoder noise, damping, elasticity, motor inertia and observation delay. Some conditions also vary the starting pose. |
| MLflow | The experiment tracker holding run metadata, metrics and artifacts. Reading the saved evidence in Git does not require tracker access. |
| SHA-256 / source revision | A file-content fingerprint / Git commit identifier used to identify the exact evidence or source version. |

For definitions tied to particular results, follow the [decision log](https://github.com/NatMota/mlai-vnc/blob/e0e302de5c61a333a56feb71ee99f8d39637f13b/DECISION_LOG.md) and the configuration linked from the [iteration index](https://github.com/NatMota/mlai-vnc/blob/e0e302de5c61a333a56feb71ee99f8d39637f13b/v2_sensors/ITERATIONS.md).
