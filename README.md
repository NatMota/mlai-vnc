# MLAI — VNC

Private COMP0245 team workspace for robot learning and reproducible experiments.
Train models in PyTorch; use MLflow to compare runs and GitHub to review code.

## Use the shared tracker

Install [Pixi](https://pixi.sh) and clone this private repository:

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

The checkpoint is stored by the MLflow server on Mint's Fast Drive, not in
GitHub. **University OneDrive is not required to share a model within this
team.** The COMP0245 assignment handout also says the Moodle submission need
not include data or pretrained weights/checkpoints; follow its separate report
and code submission instructions. Do not put raw datasets, checkpoints or
credentials in Git. Everyone granted MLflow access can read and change shared
experiment data, so keep a local copy of important checkpoints.

## Storage and collaboration

Metadata and artifacts live outside Git in `~/.local/share/mlai-vnc` by default.
Set `MLAI_TRACKING_DIR` for the database and `MLAI_ARTIFACTS_DIR` for artifacts.
On Mint, keep the SQLite database on its Linux filesystem and set
`MLAI_ARTIFACTS_DIR` to a persistent Fast Drive directory.
Back up both `mlflow.db` and the artifacts directory together.

Commit code, configurations and small reports. Keep raw data, model checkpoints,
credentials and generated run databases out of Git. Use branches and pull requests.
This initial repository contains tracking infrastructure; existing coursework
and notebooks remain in their original repositories.

The [expert collection and sampling tools](collection/README.md) provide a
CPU-only C++ collector at 1 kHz and compact 30 Hz preprocessing. Their Pixi
environments are separate from the tracking and training environments.

## V2 experimental roadmap

The [V2 roadmap](docs/v2-roadmap.html) preserves the supplied planning document
from 4 November 2026. It describes intended experiments, not a completion record.
For implemented methods, deviations and results, consult the
[decision log](DECISION_LOG.md), experiment records and MLflow.
