# =============================================================================
# train_arm_no_maths.py
# This is for: two arm designs that need no adding-up step while the robot runs.
#              a = plan network that also predicts how far each joint should be from the plan.
#                  It works with the torque network from train_arm_torque.py.
#              b = one arm network that goes straight to the torques. It uses no plan at all.
# How to run:  python3 train_arm_no_maths.py a      (or b)
#              ROBOENV_SEED=1 python3 train_arm_no_maths.py a      (choose the seed)
#              ROBOENV_TRAIN_NOISE=0.0001 python3 train_arm_no_maths.py b      (train b on noisy encoder readings)
# Result:      saves arm_option_model/option_<a or b>_seed<seed>.pt and a NumPy copy (.npz) for test_full.py.
#              With training noise, the name is option_b_noisy_seed<seed>.
#              Your other models are not touched.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   tip gap      target minus end-effector, seen from the robot: ahead, left, up (m).
#   angle gap    the expert's planned joint angle minus the measured joint angle (rad).
#   plan         the joint speeds the expert wants the arm to have (rad/s).
#   reaching     the rows after the expert's wheels have stopped for the last time.
#   MSE          Mean Squared Error. Lower is better.
#   encoder noise  a small random error in each measured joint angle (rad).
# -----------------------------------------------------------------------------

import glob
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from arm_plan_net import make_inputs


# SETTING: which design, typed after the command
#   a: 9 inputs (tip gap, joint angles 1 to 6), 14 outputs (7 angle gaps, 7 planned speeds), 30 decisions per second.
#   b: 16 inputs (the same 9, plus 7 joint speeds), 7 outputs (arm torques), 1000 decisions per second.
OPTION = sys.argv[1]
assert OPTION in ("a", "b"), "Type a or b after the command"

# SETTING: seed. Type it before the command, for example: ROBOENV_SEED=1 python3 train_arm_no_maths.py a
SEED = int(os.environ.get("ROBOENV_SEED", "0"))

# SETTING: encoder noise during training, in rad. Option b only. 0 means no noise.
#   Type it before the command, for example: ROBOENV_TRAIN_NOISE=0.0001 python3 train_arm_no_maths.py b
TRAIN_NOISE = float(os.environ.get("ROBOENV_TRAIN_NOISE", "0"))
assert TRAIN_NOISE == 0 or OPTION == "b", "Training noise is only for option b"

# SETTING: training data. The same runs and the same held-back runs as train_arm_plan.py.
DATA_FOLDER = "data_v2/balanced"
VAL_RUNS = 150

# SETTING: extra training data, added to the runs above. Empty means none. Option b only.
#   Type it before the command, for example: ROBOENV_EXTRA_DATA=data_v2/finish_5mm python3 train_arm_no_maths.py b
#   The model's name then has "_finish" in it. Some of these runs are held back for the check.
EXTRA_FOLDER = os.environ.get("ROBOENV_EXTRA_DATA", "")
EXTRA_VAL_RUNS = 30

# SETTING: DAgger rows from dagger_labels.py, added to the training data. Empty means none. Option b only.
#   Type it before the command: ROBOENV_DAGGER_FILE=dagger_round1.npy python3 train_arm_no_maths.py b
#   The rows come from 30 camera frames per second, so they are few. Each one is used DAGGER_REPEAT times.
#   The model's name then has "_dagger" in it.
DAGGER_FILE = os.environ.get("ROBOENV_DAGGER_FILE", "")
DAGGER_REPEAT = int(os.environ.get("ROBOENV_DAGGER_REPEAT", "10"))

# SETTING: row step. a runs 30 times per second, so every 33rd row. b runs 1000 times per second,
#   so every 5th row, only to make the data smaller, as in train_arm_torque.py.
ROW_STEP = 33 if OPTION == "a" else 5

# SETTING: training length and step size
EPOCHS = 80 if OPTION == "a" else 20
BATCH_SIZE = 256 if OPTION == "a" else 1024
LEARNING_RATE = 1e-3

# SETTING: near-target weight, the same as train_arm_plan.py
NEAR_TARGET_M = 0.03
NEAR_WEIGHT = 5.0

# Where the models are saved. test_full.py reads the NumPy copy from its own models folder.
MODEL_FOLDER = "arm_option_model"
TEST_MODELS = os.path.expanduser("~/mlai-vnc-golden/v2_sensors/arm/models")
HIDDEN = 128


def load(file_list):
    """Reads every ROW_STEP-th reaching row of each run."""
    X, Y = [], []
    for f in file_list:
        d = np.load(f, mmap_mode="r")
        moving_rows = np.nonzero(np.any(np.asarray(d[:, 21:25]) != 0, axis=1))[0]
        first_reach = moving_rows[-1] + 1 if len(moving_rows) > 0 else 0
        rows = np.asarray(d[first_reach::ROW_STEP])
        x = make_inputs(rows[:, :21], rows[:, 36:39])          # tip gap and joint angles 1 to 6
        if OPTION == "a":
            X.append(x)
            Y.append(np.c_[rows[:, 39:46] - rows[:, 4:11], rows[:, 46:53]].astype(np.float32))
        else:
            X.append(np.c_[x, rows[:, 11:18]].astype(np.float32))
            Y.append(rows[:, 25:32].astype(np.float32))
    return np.concatenate(X), np.concatenate(Y)


files = sorted(glob.glob(f"{DATA_FOLDER}/*.npy"))
np.random.default_rng(0).shuffle(files)
train_files, val_files = files[VAL_RUNS:], files[:VAL_RUNS]
if EXTRA_FOLDER:
    extra = sorted(glob.glob(f"{EXTRA_FOLDER}/*.npy"))
    np.random.default_rng(0).shuffle(extra)
    train_files, val_files = train_files + extra[EXTRA_VAL_RUNS:], val_files + extra[:EXTRA_VAL_RUNS]
X_train, Y_train = load(train_files)
X_val, Y_val = load(val_files)
if DAGGER_FILE:
    # DAgger rows: true tip gap (3), joint angles 1 to 6, joint speeds (7), expert torques (7), target number.
    dagger = np.load(DAGGER_FILE)
    X_train = np.r_[X_train, np.repeat(dagger[:, :16], DAGGER_REPEAT, axis=0)]
    Y_train = np.r_[Y_train, np.repeat(dagger[:, 16:23], DAGGER_REPEAT, axis=0)]
    print(f"DAgger rows {len(dagger)}, each used {DAGGER_REPEAT} times", flush=True)

# Print results: how much data was loaded.
print(f"option {OPTION}, seed {SEED}. rows: train {len(X_train)}, val {len(X_val)}", flush=True)

# Standardise with the training data only.
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0) + 1e-8
Y_min, Y_max = Y_train.min(axis=0), Y_train.max(axis=0)


def row_weights(X):
    """NEAR_WEIGHT for rows near the target, 1 for the others. The first 3 inputs are the tip gap (m)."""
    near = np.linalg.norm(X[:, :3], axis=1) < NEAR_TARGET_M
    return torch.from_numpy(np.where(near, NEAR_WEIGHT, 1.0).astype(np.float32)).unsqueeze(1)


x_train, y_train = torch.from_numpy((X_train - X_mean) / X_std), torch.from_numpy((Y_train - Y_mean) / Y_std)
x_val, y_val = torch.from_numpy((X_val - X_mean) / X_std), torch.from_numpy((Y_val - Y_mean) / Y_std)
weight_train, weight_val = row_weights(X_train), row_weights(X_val)

# How much noise each input gets, in standardised units. The inputs are: tip gap (3), joint angles 1 to 6, joint speeds (7).
#   Tip gap: none, so only the joint readings change.
#   Joint angles: the encoder noise itself.
#   Joint speeds: a speed is the change between two noisy angles over 1 ms, so its noise is sqrt(2) x noise / 0.001.
input_noise = np.r_[np.zeros(3), np.full(6, TRAIN_NOISE), np.full(7, np.sqrt(2) * TRAIN_NOISE / 1e-3)] if TRAIN_NOISE else None
if TRAIN_NOISE:
    input_noise = torch.from_numpy((input_noise / X_std).astype(np.float32))
    # The held-back runs get one fixed set of noise, so every epoch is compared on the same check.
    x_val = x_val + torch.randn(x_val.shape, generator=torch.Generator().manual_seed(0)) * input_noise


def mse(out, y, weight):
    """The error for each row, multiplied by that row's weight."""
    return (((out - y) ** 2) * weight).sum() / (weight.sum() * y.shape[1])


class ArmNet(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_out))

    def forward(self, x):
        return self.layers(x)


torch.manual_seed(SEED)
net = ArmNet(x_train.shape[1], y_train.shape[1])
optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

os.makedirs(MODEL_FOLDER, exist_ok=True)
name = f"option_{OPTION}{'_noisy' if TRAIN_NOISE else ''}{'_finish' if EXTRA_FOLDER else ''}{'_dagger' if DAGGER_FILE else ''}_seed{SEED}"
scaling = {"X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std, "Y_min": Y_min, "Y_max": Y_max}
best_val = float("inf")

for epoch in range(EPOCHS):
    # Learn from the training data.
    net.train()
    order = torch.randperm(len(x_train))
    for start in range(0, len(x_train), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        x_batch = x_train[idx]
        if TRAIN_NOISE:
            x_batch = x_batch + torch.randn_like(x_batch) * input_noise     # fresh noise every time
        loss = mse(net(x_batch), y_train[idx], weight_train[idx])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()

    # Check on the held-back runs (no learning here).
    net.eval()
    with torch.no_grad():
        val_mse = mse(net(x_val), y_val, weight_val).item()
    # Print results: one line per epoch.
    print(f"epoch {epoch + 1}/{EPOCHS}  val MSE {val_mse:.5f}", flush=True)

    # Save the best model so far, as a PyTorch file and a NumPy copy for test_full.py.
    if val_mse < best_val:
        best_val = val_mse
        torch.save({"net": net.state_dict(), "epoch": epoch + 1, "seed": SEED, **scaling}, f"{MODEL_FOLDER}/{name}.pt")
        np.savez(f"{TEST_MODELS}/{name}.npz", **{k: v.numpy() for k, v in net.state_dict().items()}, **scaling)

# Print results: where the model was saved.
print(f"Best val MSE {best_val:.5f}. Saved as {MODEL_FOLDER}/{name}.pt")
