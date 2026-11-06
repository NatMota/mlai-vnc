# Experiment tracking

[Home](../README.md) · [V2](https://github.com/NatMota/mlai-vnc/blob/ff947179749653e9827137da712b2e0fcf87205d/v2_sensors/README.md)

Use MLflow to record training and evaluation. Reading the repository does not require tracker access.

## Use the shared tracker

Install [Pixi](https://pixi.sh) and clone the repository (access is required while it remains private):

```sh
git clone https://github.com/NatMota/mlai-vnc.git
cd mlai-vnc
```

Ask for access to the Mint device through Tailscale, then open
https://nm-linmint-mlflow.taile381c0.ts.net/. GitHub repository access alone
does not grant access to MLflow. The tracker is private to the tailnet; do not
use Tailscale Funnel.

Set `MLFLOW_TRACKING_URI` to that URL in the environment where you train. For
example, in a macOS or Linux shell:

```sh
export MLFLOW_TRACKING_URI=https://nm-linmint-mlflow.taile381c0.ts.net
```

On Windows PowerShell, use
`$env:MLFLOW_TRACKING_URI = "https://nm-linmint-mlflow.taile381c0.ts.net"`.
From this repository, run `pixi run check` to verify your connection. It creates
one clearly labelled setup run and verifies a metric and artifact round trip.

## Log a training run

Install `mlflow-skinny==3.17.2` in your existing PyTorch environment. Use one
run for each training configuration and record enough information to reproduce
it:

```python
import os
import mlflow
import torch

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("robot-behaviour-cloning")
with mlflow.start_run() as run:
    mlflow.log_params({"seed": 42, "learning_rate": 0.001, "hidden_sizes": "64,64"})
    mlflow.set_tag("git_commit", "PUT_THE_TRAINING_CODE_COMMIT_HERE")
    # Train and evaluate your model here, then log the actual results.
    # mlflow.log_metric("validation_mse", measured_validation_mse, step=epoch)
    # mlflow.log_metric("rollout_success_rate", successes / trials)
    torch.save(model.state_dict(), "model_state.pt")
    mlflow.log_artifact("model_state.pt", artifact_path="checkpoint")
    print(run.info.run_id)
```

Replace the example values and `model` with your actual training code. Save the
architecture/configuration with the code commit: weights alone are not enough
to recreate a model. Log validation plots and selected checkpoints as artifacts,
then compare runs in the web UI. Keep evaluation targets and data splits fixed
when comparing model variants.

To download a teammate's checkpoint, copy its run ID from the UI and use:

```python
import mlflow

mlflow.set_tracking_uri("https://nm-linmint-mlflow.taile381c0.ts.net")
path = mlflow.MlflowClient().download_artifacts(
    "RUN_ID", "checkpoint/model_state.pt", dst_path="."
)
print(path)
```

This download retrieves the checkpoint held by the MLflow server on Mint's Fast Drive. Selected checkpoints and compact results are also committed in the repository. Large trajectory archives and recovery training data have separate storage; consult the [V2 data notes](https://github.com/NatMota/mlai-vnc/blob/ff947179749653e9827137da712b2e0fcf87205d/v2_sensors/README.md#running-and-data). Keep local copies of important experiment artifacts.

## Storage and collaboration

Metadata and artifacts live outside Git in `~/.local/share/mlai-vnc` by default.
Set `MLAI_TRACKING_DIR` for the database and `MLAI_ARTIFACTS_DIR` for artifacts.
On Mint, keep the SQLite database on its Linux filesystem and set
`MLAI_ARTIFACTS_DIR` to a persistent Fast Drive directory.
Back up both `mlflow.db` and the artifacts directory together.

Commit code, configurations, selected checkpoints and compact evidence. Keep credentials, generated tracking databases and large raw collections out of Git. Record external data locations and hashes with the corresponding experiment.

The [expert collection and sampling tools](https://github.com/NatMota/mlai-vnc/blob/ff947179749653e9827137da712b2e0fcf87205d/collection/README.md) provide a
CPU-only C++ collector at 1 kHz and compact 30 Hz preprocessing. Their Pixi
environments are separate from the tracking and training environments.
