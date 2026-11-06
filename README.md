# COMP0245 · Learning to drive and reach

Group 02 investigates behaviour cloning for a simulated mobile manipulator: drive to a target, park, then reach and stop. We compare policy designs, onboard sensing and recovery training.

| Start here | What to review |
|---|---|
| **[V1: original and team policies](docs/V1.md)** | Original baseline, fused policies and submitted teammate implementations |
| **[V2: sensor experiments and notebooks](v2_sensors/README.md)** | Main notebooks, saved results and frozen reference configurations |
| **[Code map](docs/CODE.md)** | Collection, training, evaluation and analysis entry points |
| **[Iteration index](v2_sensors/ITERATIONS.md)** | Each report iteration linked to its code, models and evidence |
| **[Reproduce the results](docs/REPRODUCE.md)** | Data downloads, checksums, saved-result checks and simulator reruns |
| **[Glossary](docs/GLOSSARY.md)** | Definitions of project terms, acronyms and result measures |

The [decision log](DECISION_LOG.md) holds the report's appendices. For V2, **iteration 43** is the frozen clean comparison baseline; **iteration 56** is the final combined recovery comparison. These identify configurations, not successful task completion.

[Tracking setup](docs/TRACKING.md) · [Repository and data access](v2_sensors/README.md#running-and-data)

## V2 experimental roadmap

The [V2 roadmap](docs/v2-roadmap.html) preserves the supplied planning document
from 4 November 2026. It describes intended experiments, not a completion record.
For implemented methods, deviations and results, consult the
[decision log](DECISION_LOG.md), experiment records and MLflow.
