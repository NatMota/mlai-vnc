"""Run a local MLflow server with persistent metadata and proxied artifacts."""
import os
from pathlib import Path
import subprocess
import sys

state = Path(os.environ.get('MLAI_TRACKING_DIR', Path.home() / '.local/share/mlai-vnc')).expanduser().resolve()
state.mkdir(parents=True, exist_ok=True)
artifacts = Path(os.environ.get('MLAI_ARTIFACTS_DIR', state / 'artifacts')).expanduser().resolve()
artifacts.mkdir(parents=True, exist_ok=True)
command = [sys.executable, '-m', 'mlflow', 'server',
           '--host', '127.0.0.1', '--port', os.environ.get('MLAI_TRACKING_PORT', '5055'),
           '--backend-store-uri', f'sqlite:///{state / "mlflow.db"}',
           '--artifacts-destination', str(artifacts)]
print(f'MLflow metadata: {state}; artifacts: {artifacts}', flush=True)
raise SystemExit(subprocess.call(command))
