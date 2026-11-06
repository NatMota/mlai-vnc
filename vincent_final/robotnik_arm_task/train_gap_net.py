# =============================================================================
# train_gap_net.py
# This is for: a third arm network that replaces the fixed adding-up between the plan and torque networks.
#              It learns how far each joint should be from the plan (the angle gap), from what the robot measures now.
# How to run:  python3 train_gap_net.py      (ROBOENV_SEED=1 python3 train_gap_net.py to choose the seed)
# Result:      saves arm_gap_model/gap_seed<seed>.pt. Used by test_drive_net.py with ROBOENV_GAP_NET=1.
#              Prints the error for each epoch. Your other models are not touched.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   tip gap    target minus end-effector, seen from the robot: ahead, left, up (m).
#   plan       the joint speeds the expert wants the arm to have (rad/s).
#   angle gap  the expert's planned joint angle minus the measured joint angle (rad). This is what the network learns.
#   reaching   the rows after the expert's wheels have stopped for the last time.
#   MSE        Mean Squared Error. Lower is better.
# -----------------------------------------------------------------------------

import glob
import os
import numpy as np
import torch
from arm_plan_net import make_inputs as plan_inputs
from gap_net import GapNet, MODEL_FOLDER, WINDOW_MS, WINDOW_STEPS, make_inputs, window_inputs

# SETTING: seed. Type it before the command, for example: ROBOENV_SEED=1 python3 train_gap_net.py
SEED = int(os.environ.get("ROBOENV_SEED", "0"))

# SETTING: training data. The same runs and the same held-back runs as train_arm_plan.py.
DATA_FOLDER = "data_v2/balanced"
VAL_RUNS = 150

# SETTING: random error added to the joint speed inputs during training, in rad/s. 0 means none.
#   It teaches the network not to rely too much on the exact speed. The model's name then has "_robust" in it.
#   Type it before the command, for example: ROBOENV_SPEED_NOISE=0.05 python3 train_gap_net.py
SPEED_NOISE = float(os.environ.get("ROBOENV_SPEED_NOISE", "0"))
SPEED_COLS = list(range(9, 16))         # the 7 joint speeds in the 23 inputs

# SWITCH: stretch of time.
#   ON  (ROBOENV_GAP_WINDOW=1): the network also reads the last 100 ms of joint movement and plan. The model's name has "_window" in it.
#   OFF (not set): the network only reads the current moment.
USE_WINDOW = os.environ.get("ROBOENV_GAP_WINDOW", "0") == "1"

# SETTING: every Nth reaching row, only to make the data smaller.
ROW_STEP = 5

# SETTING: training length and step size
EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

# SETTING: near-target weight, the same as train_arm_plan.py
NEAR_TARGET_M = 0.03
NEAR_WEIGHT = 5.0


def load(file_list):
    """Reads every ROW_STEP-th reaching row of each run."""
    X, Y = [], []
    for f in file_list:
        d = np.load(f, mmap_mode="r")
        moving_rows = np.nonzero(np.any(np.asarray(d[:, 21:25]) != 0, axis=1))[0]
        first_reach = moving_rows[-1] + 1 if len(moving_rows) > 0 else 0
        rows = np.asarray(d[first_reach::ROW_STEP])
        x = make_inputs(plan_inputs(rows[:, :21], rows[:, 36:39])[:, :3], rows[:, 4:11], rows[:, 11:18], rows[:, 46:53])
        if USE_WINDOW:
            row_numbers = np.arange(first_reach, len(d), ROW_STEP)
            past = np.stack([np.maximum(row_numbers - j * WINDOW_MS, 0) for j in range(1, WINDOW_STEPS + 1)], axis=1)
            all_q, all_plan = np.asarray(d[:, 4:11]), np.asarray(d[:, 46:53])
            x = np.c_[x, window_inputs(rows[:, 4:11], all_q[past], all_plan[past])]
        X.append(x)
        Y.append((rows[:, 39:46] - rows[:, 4:11]).astype(np.float32))
    return np.concatenate(X), np.concatenate(Y)


files = sorted(glob.glob(f"{DATA_FOLDER}/*.npy"))
np.random.default_rng(0).shuffle(files)
X_train, Y_train = load(files[VAL_RUNS:])
X_val, Y_val = load(files[:VAL_RUNS])
# Print results: how much data was loaded.
print(f"seed {SEED}. rows: train {len(X_train)}, val {len(X_val)}", flush=True)

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


def mse(out, y, weight):
    """The error for each row, multiplied by that row's weight."""
    return (((out - y) ** 2) * weight).sum() / (weight.sum() * y.shape[1])


torch.manual_seed(SEED)
torch.set_num_threads(2)
net = GapNet(X_train.shape[1])
optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

os.makedirs(MODEL_FOLDER, exist_ok=True)
model_file = f"{MODEL_FOLDER}/gap{'_window' if USE_WINDOW else ''}{'_robust' if SPEED_NOISE else ''}_seed{SEED}.pt"
speed_noise = torch.zeros(X_std.shape, dtype=torch.float32)
speed_noise[SPEED_COLS] = torch.from_numpy((SPEED_NOISE / X_std[SPEED_COLS]).astype(np.float32))
best_val = float("inf")
for epoch in range(EPOCHS):
    # Learn from the training data.
    net.train()
    order = torch.randperm(len(x_train))
    for start in range(0, len(x_train), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        x_batch = x_train[idx]
        if SPEED_NOISE:
            x_batch = x_batch + torch.randn_like(x_batch) * speed_noise     # fresh error every time
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
    if val_mse < best_val:
        best_val = val_mse
        torch.save({"net": net.state_dict(), "epoch": epoch + 1, "seed": SEED, "X_mean": X_mean, "X_std": X_std,
                    "Y_mean": Y_mean, "Y_std": Y_std, "Y_min": Y_min, "Y_max": Y_max}, model_file)

# Print results: where the model was saved.
print(f"Best val MSE {best_val:.5f}. Saved as {model_file}")
