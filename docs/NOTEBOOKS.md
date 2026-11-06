# Notebook access

[Home](../README.md) · [Decision log](../DECISION_LOG.md) · [Reproduce the results](REPRODUCE.md)

Use the saved views to review the evidence immediately. Live notebooks require an approved device on Nat's Tailscale network and permission to reach the notebook service.

| Notebook | Saved output, no Tailscale needed | Live notebook, Tailscale required |
|---|---|---|
| Sensor design and early policies | [Reading view](../v2_sensors/review/driving_v2_sensors.md) · [download notebook](../v2_sensors/driving_v2/driving_v2_sensors.ipynb) | [Open](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/review-notebooks-20261106/driving_v2_sensors.ipynb) |
| Golden data and full-episode experiments | [Reading view](../v2_sensors/review/golden_dataset.md) · [download notebook](../v2_sensors/driving_v2/golden_dataset.ipynb) | [Open](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/golden/v2_sensors/driving_v2/golden_dataset.ipynb) |
| Report statistics | [Reading view](../v2_sensors/review/report_statistics.md) · [download notebook](../v2_sensors/driving_v2/report_statistics.ipynb) | [Original analysis workspace](https://nm-linmint-mlflow.taile381c0.ts.net:8443/lab/tree/report/COMP0245_report_workspace.ipynb) |

The decision log links to individual cells using JupyterLab's [`#cell-id=` navigation](https://jupyterlab.readthedocs.io/en/4.6.x/user/urls.html#linking-to-notebook-sections). Each link has an adjacent saved-view link to the same analysis. If a JupyterLab version does not scroll to the cell, use that saved view; cell identities are also recorded in [notebook-links.json](reproduction/notebook-links.json).

The early live notebook is a frozen review copy. The live report workspace can continue to change; the public statistics notebook preserves its analysis cells and saved outputs from the 6 November snapshot, omitting working-report prose. Its provenance is recorded in notebook metadata. Execute a downloaded copy with the [reproduction instructions](REPRODUCE.md), rather than running training cells on the shared server.

## Network boundary

Verified on 6 November 2026: Tailscale Serve exposes HTTPS port 8443 to the tailnet only, forwarding to Jupyter on `127.0.0.1:8890`. The replay route forwards to `127.0.0.1:8899`; MLflow on HTTPS port 443 forwards to `127.0.0.1:5055`. All three backend services listen on loopback. No Funnel/public serving entry is enabled.

A GitHub link does not grant Tailscale membership. Downloading the school OneDrive archives needs OneDrive permission separately, but local reproduction then needs neither Tailscale nor the live MLflow server. The repository and OneDrive retain their own access permissions.
