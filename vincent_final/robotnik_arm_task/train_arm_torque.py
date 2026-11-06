# =============================================================================
# train_arm_torque.py
# This is for: teaching the torque network to copy the expert's arm torques, from the
#              plan and the measured joints. It learns from driving and reaching rows,
#              because it must also hold the arm still while the robot drives.
# How to run:  python3 train_arm_torque.py
# Result:      saves arm_torque_model/arm_torque_seed<seed>.pt (weights and scaling numbers).
#              Prints the error for each epoch. Your other models are not touched.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   X        the 20 input numbers (see make_inputs in arm_torque_net.py).
#   Y        the expert's 7 arm torques (N m), columns 25 to 31 of the data.
#   MSE      Mean Squared Error. Lower is better.
# -----------------------------------------------------------------------------

import glob
import os
import numpy as np
import torch
import torch.nn as nn
from arm_torque_net import MODEL_FOLDER, ArmTorqueNet, make_inputs


# SETTING: seed
#   Fixes the starting weights and the shuffling. Try 0, 1, 2 to see if a result is luck.
#   Type it before the command, for example: ROBOENV_SEED=1 python3 train_arm_torque.py
SEED = int(os.environ.get("ROBOENV_SEED", "0"))

# SETTING: training data
#   The same held-back runs as train_arm_plan.py.
DATA_FOLDER = "data_v2/balanced"
VAL_RUNS = 150

# SETTING: row step
#   5 = learn from every 5th row, to fit in memory. This only makes the training data smaller.
#   The network still runs at 1000 decisions per second on the robot, so nothing is held.
ROW_STEP = 5

# SETTING: training length and step size
EPOCHS = 20
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3

# SETTING: use less data, to test how much data is needed. 1 means all the runs.
#   4 uses a quarter of the training runs and 4 times the epochs, so the number of weight updates stays about the same.
#   The model's name then ends in "_div4", so the normal model is not touched.
#   Type it before the command, for example: ROBOENV_RUNS_DIVIDE=4 python3 train_arm_torque.py
RUNS_DIVIDE = int(os.environ.get("ROBOENV_RUNS_DIVIDE", "1"))
RUNS_TAG = f"_div{RUNS_DIVIDE}" if RUNS_DIVIDE > 1 else ""
EPOCHS = EPOCHS * RUNS_DIVIDE


def load(file_list):
    """Reads every ROW_STEP-th row of each run."""
    X, Y = [], []
    for f in file_list:
        rows = np.load(f, mmap_mode="r")[::ROW_STEP]
        X.append(make_inputs(np.asarray(rows[:, 39:46]), np.asarray(rows[:, 46:53]),
                             np.asarray(rows[:, 4:11]), np.asarray(rows[:, 11:18])))
        Y.append(np.asarray(rows[:, 25:32], dtype=np.float32))
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

torch.manual_seed(SEED)
net = ArmTorqueNet()
mse = nn.MSELoss()
optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

os.makedirs(MODEL_FOLDER, exist_ok=True)
model_file = f"{MODEL_FOLDER}/arm_torque_seed{SEED}{RUNS_TAG}.pt"
best_val = float("inf")

for epoch in range(EPOCHS):
    # Learn from the training data.
    net.train()
    order = torch.randperm(len(x_train))
    for start in range(0, len(x_train), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        loss = mse(net(x_train[idx]), y_train[idx])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()

    # Check on the held-back runs (no learning here).
    net.eval()
    with torch.no_grad():
        val_mse = mse(net(x_val), y_val).item()

    # Print results: one line per epoch.
    print(f"epoch {epoch + 1}/{EPOCHS}  val MSE {val_mse:.5f}", flush=True)

    # Save the best model so far, with the scaling numbers and the command limits.
    if val_mse < best_val:
        best_val = val_mse
        torch.save({"net": net.state_dict(), "epoch": epoch + 1, "seed": SEED,
                    "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std,
                    "Y_min": Y_min, "Y_max": Y_max}, model_file)

# Print results: where the model was saved.
print(f"Best val MSE {best_val:.5f}. Saved as {model_file}")
