"""Download the teammate's model and prepare the files fused_policy.py needs.

    python prepare_team.py

Downloads checkpoint.pt and scaler.npz from the team MLflow run f978e343 (Tailscale must be connected)
into outputs/teammate_f978e343/model/, writes the command limits their drive.py computes from the
data_dart training runs, and checks the NumPy networks against their torch networks. Files whose hash
already matches are not downloaded again. Do not run this while an evaluation is running.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from pool import save_npz_atomic

ROOT = Path(__file__).resolve().parent
TEAM_URI = "https://nm-linmint-mlflow.taile381c0.ts.net"
TEAM_RUN = "f978e3431988470188317b3c6efd9e29"
MODEL_NAME = "ee_good_split_early200_plus_dart2"
TEAM_DIR = ROOT / "outputs/teammate_f978e343/model"
CODE_DIR = ROOT / "outputs/teammate_f978e343/code"
MAX_ALLOWED_DIFF = 1e-4
# Pinned because features.py is executed and the run could change: a different file stops the script.
EXPECTED_SHA256 = {"checkpoint.pt": "50cdda1db063d97f1facb4262b7a4b2060b6ab6da51bd8bded058580edea9efd",
                   "scaler.npz": "397fc4e5d52c60e12e22db463fc8289f855e61093fd9c5cb004ab86d506806c3",
                   "features.py": "6032b40fac3b3e802ee55d0249fc1f5e7eeb90fdb42f54e993cffd83a062c239"}


def is_expected(path):
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_SHA256[path.name]


def download():
    wanted = {TEAM_DIR / "checkpoint.pt": f"model/{MODEL_NAME}_best_seed0.pt",
              TEAM_DIR / "scaler.npz": f"model/scaler_{MODEL_NAME}_seed0.npz",
              CODE_DIR / "features.py": "code/features.py"}
    missing = {path: artifact for path, artifact in wanted.items() if not is_expected(path)}
    if not missing:
        return
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=TEAM_URI)
    for path, artifact in missing.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            got = Path(client.download_artifacts(TEAM_RUN, artifact, tmp))
            if hashlib.sha256(got.read_bytes()).hexdigest() != EXPECTED_SHA256[path.name]:
                raise SystemExit(f"{artifact} differs from the pinned file; nothing was replaced")
            os.replace(got, path)


def write_limits():
    """The smallest and biggest expert command, computed the same way as the teammate's drive.py."""
    split = json.loads((ROOT / "data_dart/split.json").read_text())
    low, high = np.full(11, np.inf), np.full(11, -np.inf)
    for name in split["train"]:
        commands = np.load(ROOT / name, mmap_mode="r")[:, 21:]
        low, high = np.minimum(low, commands.min(axis=0)), np.maximum(high, commands.max(axis=0))
    save_npz_atomic(TEAM_DIR / "limits.npz", y_min=low, y_max=high)


def check_networks():
    from fused_policy import PHASE_COLS, TeamNets
    spec = importlib.util.spec_from_file_location("team_features", CODE_DIR / "features.py")
    team_features = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(team_features)
    if list(team_features.PHASE_COLS) != list(PHASE_COLS):
        raise SystemExit("The teammate's PHASE_COLS differ from the local ones")
    saved = torch.load(TEAM_DIR / "checkpoint.pt", map_location="cpu", weights_only=True)
    control = team_features.SplitControlNet(35)
    control.load_state_dict(saved["control_net"])
    phase = torch.nn.Sequential(torch.nn.Linear(14, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                                torch.nn.ReLU(), torch.nn.Linear(64, 1))
    phase.load_state_dict(saved["phase_net"])
    nets = TeamNets()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        x = rng.normal(size=(1, 34)).astype(np.float32)
        flag = float(rng.integers(2))
        with torch.no_grad():
            y = control(torch.cat([torch.tensor(x), torch.tensor([[flag]], dtype=torch.float32)], 1))[0, :11].numpy()
            p_reach = torch.sigmoid(phase(torch.tensor(x[:, PHASE_COLS]))).item()
        y = np.clip(y * nets.y_std + nets.y_mean, nets.y_min, nets.y_max)
        worst = max(worst, float(np.abs(y - nets.commands(x, flag)).max()), abs(p_reach - nets.p_reach(x)))
    if worst > MAX_ALLOWED_DIFF:
        raise SystemExit(f"NumPy networks differ from the torch networks: {worst}")
    print(f"PREPARED {TEAM_DIR} largest difference {worst:.2e}")


if __name__ == "__main__":
    os.chdir(ROOT)
    download()
    write_limits()
    check_networks()
