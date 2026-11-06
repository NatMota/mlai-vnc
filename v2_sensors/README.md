# V2: learning from onboard sensors

[Home](../README.md) · [V1](../docs/V1.md) · [Code map](../docs/CODE.md) · [Iteration index](ITERATIONS.md)

V2 develops camera and encoder policies for driving, PARK, arm torque and STOP. Early oracle-input experiments establish a reference; later experiments use sensor inputs, learned arm holding and recovery training. The operational camera is on the front-right chassis corner. A wrist-mounted camera was tested during design and rejected.

## Notebooks

| Read in order | Open notebook | Read saved output on GitHub |
|---|---|---|
| 1. Sensor design and initial policies, iterations 1–36 | [driving_v2_sensors.ipynb](driving_v2/driving_v2_sensors.ipynb) | [Reading view](review/driving_v2_sensors.md) |
| 2. Golden dataset, full episodes and recovery, iterations 34–57 | [golden_dataset.ipynb](driving_v2/golden_dataset.ipynb) | [Reading view](review/golden_dataset.md) |
| 3. Report statistics and diagnostic comparisons | [report_statistics.ipynb](driving_v2/report_statistics.ipynb) | [Reading view](review/report_statistics.md) |

[Live notebook access](../docs/NOTEBOOKS.md) requires Tailscale. The [reproduction guide](../docs/REPRODUCE.md) provides the data archives, checksums and local commands.

The reading views retain saved prose, code, tables and plots; interactive players remain in the original notebooks. They require no simulator or tracker access. The original notebooks are 10 MB and 20 MB, so downloading them may be more reliable than GitHub's notebook preview.

## Frozen references

| Reference | Use it for | Evidence |
|---|---|---|
| Iteration 13 | Historical camera-and-encoder driving baseline | [Original guide](driving_v2/HISTORICAL_BASELINE.md), [iteration trace](ITERATIONS.md#iteration-13) |
| Iteration 40 | Wider arm before learned holding from reset | [Configuration](arm/results/time_cumulative_test/config.json), [iteration trace](ITERATIONS.md#iteration-40) |
| **Iteration 43: `basics_standard`** | Clean full-episode baseline for recovery comparisons | [Models](arm/results/overnight_20261105/basics_standard/models), [test configuration](arm/results/overnight_20261105/basics_standard/test/config.json), [scores](arm/results/overnight_20261105/basics_standard/scores.csv) |
| **Iteration 56: `dart_dagger`** | Final combined recovery comparison | [Models](arm/results/overnight_20261105/dart_dagger/models), [test configuration](arm/results/overnight_20261105/dart_dagger/test/config.json), [scores](arm/results/overnight_20261105/dart_dagger/scores.csv) |

These are frozen configurations, not claims of stable task success. The disturbance ladder branches from iteration 41; standardised and recovery models form separate comparisons. Any continuation should name its parent iteration and condition explicitly. [REFERENCE.json](REFERENCE.json) records source revision and hashes for the two main comparison configurations; it does not change any script's defaults.

Use report numbering **1–57**. Older recovery metadata uses **52–58** for report iterations **51–57**. The excluded old iteration 51 remains archived. The [iteration index](ITERATIONS.md) resolves both numbering schemes using unchanged condition names.

## Team sensor extensions

- [Haosheng: golden-data sensor and mixed-input comparisons](https://github.com/NatMota/mlai-vnc/tree/e407006882467c942ef8da85553a1fd18a6fe36d), with [evaluation code](https://github.com/NatMota/mlai-vnc/blob/e407006882467c942ef8da85553a1fd18a6fe36d/golden_eval.py).
- [Vincent: sensor-only arm, training scripts and models](https://github.com/NatMota/mlai-vnc/tree/dc7cca31c8df6cddce0f7493131e620c993f8960/vincent_final/sensor_only), with [submission instructions](https://github.com/NatMota/mlai-vnc/blob/dc7cca31c8df6cddce0f7493131e620c993f8960/vincent_final/README.md).

These links pin the reviewed source revisions. Their individual protocols remain separate from the numbered V2 sequence above.

## Running and data

For a review, start with saved notebook outputs and the [report appendices](../DECISION_LOG.md). For a rerun, use the [code map](../docs/CODE.md) and the chosen iteration's models, split, configuration and recorded source hashes.

Open the original notebooks from `v2_sensors/driving_v2/`; relative imports assume that directory. The early notebook contains training cells. The golden notebook reads saved evidence and uses `GOLDEN_DATA_ROOT` when full trajectories are available. Interactive replays may need the separately hosted records.

Git includes models and compact evaluation records. [Data and reproduction instructions](../docs/REPRODUCE.md) link the school OneDrive archives and distinguish saved-result checks from simulator and training reruns. The [artifact manifest and data notes](arm/results/overnight_20261105/README.md) explain the external trajectory archive and recovery-training collections. Those dependencies must be available for a full retraining run; the repository alone is not a complete training-data release. [MLflow setup](../docs/TRACKING.md) requires separate tracker access, but is not needed to read the saved evidence here.
