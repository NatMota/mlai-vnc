# =============================================================================
# train_arm_plan.py
# This is for: teaching the arm plan network to copy the expert's joint speed plan
#              at 30 decisions per second, from 9 inputs. It only learns from reaching rows.
# How to run:  python3 train_arm_plan.py
# Result:      saves arm_plan_model/arm_plan_seed<seed>.pt (weights and scaling numbers).
#              Prints the error for each epoch. Your other models are not touched.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   X        the 9 inputs. One row per decision.
#   Y        the expert's plan: 7 joint speeds (rad/s), columns 46 to 52 of the data.
#   reaching the rows after the expert's wheels have stopped for the last time.
#   MSE      Mean Squared Error. Lower is better.
# -----------------------------------------------------------------------------

import glob
import os
import numpy as np
import torch
import torch.nn as nn
from arm_plan_net import DECIDE_EVERY, MODEL_FOLDER, ArmPlanNet, make_inputs


# SETTING: seed
#   Fixes the starting weights and the shuffling. Try 0, 1, 2 to see if a result is luck.
#   Type it before the command, for example: ROBOENV_SEED=1 python3 train_arm_plan.py
SEED = int(os.environ.get("ROBOENV_SEED", "0"))

# SETTING: training data
#   data_dart cannot be used to check this network, because it has no plan columns.
#   So VAL_RUNS runs of the balanced data are held back to pick the best epoch.
#   The robot test still uses the data_dart validation targets.
DATA_FOLDER = "data_v2/balanced"
VAL_RUNS = 150

# SETTING: training length and step size
#   30 epochs reached 113 of 120 validation targets, and its error was still falling at the end.
#   That model is backed up as arm_plan_model/arm_plan_seed0_30epochs.pt. This backup is not submitted.
EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# SETTING: use less data, to test how much data is needed. 1 means all the runs.
#   4 uses a quarter of the training runs and 4 times the epochs, so the number of weight updates stays about the same.
#   The model's name then ends in "_div4", so the normal model is not touched.
#   Type it before the command, for example: ROBOENV_RUNS_DIVIDE=4 python3 train_arm_plan.py
RUNS_DIVIDE = int(os.environ.get("ROBOENV_RUNS_DIVIDE", "1"))
RUNS_TAG = f"_div{RUNS_DIVIDE}" if RUNS_DIVIDE > 1 else ""
EPOCHS = EPOCHS * RUNS_DIVIDE

# SETTING: near-target weight
#   Rows where the tip is within NEAR_TARGET_M of the target count NEAR_WEIGHT times in the error.
#   The planned speeds are small there, so without this their errors hardly count, even though
#   the last centimetre decides if a target is reached.
#   1 = all rows count the same. The 80-epoch models with weight 1 are backed up as
#   arm_plan_model/arm_plan_seed<seed>_80epochs.pt (seed 0: 117 of 120, seed 1: 112 of 120). These backups are not submitted.
NEAR_TARGET_M = 0.03
NEAR_WEIGHT = 5.0


def load(file_list):
    """Reads every 33rd reaching row of each run."""
    X, Y = [], []
    for f in file_list:
        d = np.load(f, mmap_mode="r")
        moving_rows = np.nonzero(np.any(np.asarray(d[:, 21:25]) != 0, axis=1))[0]
        first_reach = moving_rows[-1] + 1 if len(moving_rows) > 0 else 0
        rows = d[first_reach::DECIDE_EVERY]
        X.append(make_inputs(np.asarray(rows[:, :21]), np.asarray(rows[:, 36:39])))
        Y.append(np.asarray(rows[:, 46:53], dtype=np.float32))
    return np.concatenate(X), np.concatenate(Y)


files = sorted(glob.glob(f"{DATA_FOLDER}/*.npy"))
np.random.default_rng(0).shuffle(files)
train_files = files[VAL_RUNS:]
train_files = train_files[:len(train_files) // RUNS_DIVIDE]
X_train, Y_train = load(train_files)
X_val, Y_val = load(files[:VAL_RUNS])

# Print results: how much data was loaded.
print(f"runs: train {len(train_files)}, val {VAL_RUNS}")
print(f"rows: train {len(X_train)}, val {len(X_val)}")

# Standardise with the training data only.
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
Y_mean, Y_std = Y_train.mean(axis=0), Y_train.std(axis=0) + 1e-8
Y_min, Y_max = Y_train.min(axis=0), Y_train.max(axis=0)

x_train = torch.from_numpy((X_train - X_mean) / X_std)
y_train = torch.from_numpy((Y_train - Y_mean) / Y_std)
x_val = torch.from_numpy((X_val - X_mean) / X_std)
y_val = torch.from_numpy((Y_val - Y_mean) / Y_std)


def row_weights(X):
    """NEAR_WEIGHT for rows near the target, 1 for the others. The first 3 inputs are the tip gap (m)."""
    near = np.linalg.norm(X[:, :3], axis=1) < NEAR_TARGET_M
    return torch.from_numpy(np.where(near, NEAR_WEIGHT, 1.0).astype(np.float32)).unsqueeze(1), near.mean() * 100


weight_train, near_percent = row_weights(X_train)
weight_val, _ = row_weights(X_val)
# Print results: how many rows get the extra weight.
print(f"rows within {NEAR_TARGET_M * 100:.0f} cm of the target: {near_percent:.1f}%, weight {NEAR_WEIGHT}")


def mse(out, y, weight):
    """The error for each row, multiplied by that row's weight."""
    return (((out - y) ** 2) * weight).sum() / (weight.sum() * y.shape[1])


torch.manual_seed(SEED)
net = ArmPlanNet()
optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

os.makedirs(MODEL_FOLDER, exist_ok=True)
model_file = f"{MODEL_FOLDER}/arm_plan_seed{SEED}{RUNS_TAG}.pt"
best_val = float("inf")

for epoch in range(EPOCHS):
    # Learn from the training data.
    net.train()
    order = torch.randperm(len(x_train))
    for start in range(0, len(x_train), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        loss = mse(net(x_train[idx]), y_train[idx], weight_train[idx])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()

    # Check on the held-back runs (no learning here).
    net.eval()
    with torch.no_grad():
        val_mse = mse(net(x_val), y_val, weight_val).item()

    # Print results: one line per epoch.
    print(f"epoch {epoch + 1}/{EPOCHS}  val MSE {val_mse:.4f}")

    # Save the best model so far, with the scaling numbers and the command limits.
    if val_mse < best_val:
        best_val = val_mse
        torch.save({"net": net.state_dict(), "epoch": epoch + 1, "seed": SEED,
                    "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std,
                    "Y_min": Y_min, "Y_max": Y_max}, model_file)

# Print results: where the model was saved.
print(f"Best val MSE {best_val:.4f}. Saved as {model_file}")
