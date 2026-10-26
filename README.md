# MLAI — VNC

Private COMP0245 team workspace for robot learning and reproducible experiments.
Train models in PyTorch; use MLflow to compare runs and GitHub to review code.

## Start tracking

Install [Pixi](https://pixi.sh), then:

```sh
git clone https://github.com/NatMota/mlai-vnc.git
cd mlai-vnc
pixi run server
```

Open http://127.0.0.1:5055. In another terminal, run `pixi run check` to
verify metric logging and artifact upload/download. Setup checks are labelled
separately from real training experiments.

The locked environment supports Linux, Apple Silicon macOS and Windows.
Training and simulator dependencies are separate from this tracking environment.

## Log a training run

Install `mlflow-skinny==3.17.2` in your existing PyTorch environment and add:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5055")
mlflow.set_experiment("robot-behaviour-cloning")
with mlflow.start_run():
    mlflow.log_params({"seed": 42, "learning_rate": 0.001, "hidden_sizes": "64,64"})
    # Train and evaluate your model here.
    # mlflow.log_metric("validation_mse", measured_validation_mse, step=epoch)
    # mlflow.log_metric("rollout_success_rate", successes / trials)
    # mlflow.log_artifact("outputs/model.pt")
```

Record the Git commit, data version/split, seed, architecture, optimiser,
training settings, validation errors and closed-loop rollout outcomes.
Save plots and selected model checkpoints as artifacts. Keep evaluation targets
and data splits fixed when comparing model variants.

## Connect to the shared server

The server listens only on localhost. Someone with an authorised SSH account
on the hosting machine can forward it:

```sh
ssh -N -L 5055:127.0.0.1:5055 YOUR_SSH_HOST
```

Then open http://127.0.0.1:5055 or set
`MLFLOW_TRACKING_URI=http://127.0.0.1:5055` in training code.
GitHub collaborator access does not grant an SSH account or MLflow access.
Do not expose the tracking port publicly without authentication.

### Access over the internet

For team access without SSH accounts, install Tailscale on the server and
sign it into the owner's tailnet. Mint does not currently have Tailscale
installed, so the steps below require its owner to complete that install and
sign-in first. Then serve the existing local MLflow port over tailnet HTTPS:

```sh
tailscale serve --bg 5055
tailscale serve status
```

Tailscale may ask the tailnet owner to enable HTTPS certificates. Use the
`https://...ts.net` address reported by `tailscale serve status`. Configure
`MLFLOW_SERVER_ALLOWED_HOSTS` on the server to include that exact hostname
and `localhost`, then restart `mlai-vnc-mlflow.service` and test the URL.

Share **only the Mint device** with each teammate using Tailscale's device
sharing feature, and restrict shared users to the served HTTPS port (443) in
the tailnet policy. Review existing broad grants: adding a narrow grant does
not remove a broad one. GitHub usernames do not identify Tailscale accounts;
each teammate must accept a Tailscale share invitation separately.

Once connected, each teammate opens the HTTPS address in a browser and sets
`MLFLOW_TRACKING_URI` to that same address in their local PyTorch environment.
MLflow in this configuration has shared read/write experiment access rather
than separate per-person permissions. Do not use Tailscale Funnel, which
would publish the dashboard to the open internet.

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
