"""Verify a real run, metric, tag and artifact through the tracking server."""
import os
import subprocess
from pathlib import Path
import mlflow

uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5055')
mlflow.set_tracking_uri(uri)
mlflow.set_experiment('setup-checks')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
with mlflow.start_run(run_name='tracking-roundtrip', tags={'purpose': 'setup-check', 'git_commit': commit}) as run:
    mlflow.log_param('check_version', 1)
    mlflow.log_metric('connection_ok', 1.0)
    mlflow.log_text('Tracking and artifact upload succeeded. This is not a training result.\n', 'check.txt')
    run_id = run.info.run_id
client = mlflow.MlflowClient()
stored = client.get_run(run_id)
assert stored.data.metrics['connection_ok'] == 1.0
assert stored.data.tags['git_commit'] == commit
assert Path(client.download_artifacts(run_id, 'check.txt')).read_text().startswith('Tracking and artifact upload succeeded.')
print(f'PASS: run {run_id}; metric, Git tag and artifact round trip verified at {uri}')
