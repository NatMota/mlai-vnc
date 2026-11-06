# =============================================================================
# train_drive_net.py
# This is for: teaching the driving network (wheel commands and stop decision) at
#              30 decisions per second, from 4 inputs. The arm is not part of this model.
# How to run:  python3 train_drive_net.py
# Result:      saves drive_net_model/drive_net_seed<seed>.pt (weights and scaling numbers).
#              Prints the error for each epoch. Your other models are not touched.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   X        the 4 inputs. One row per decision.
#   Y        the expert's 2 wheel commands: right wheels, left wheels (rad/s).
#   driving  1 while the expert's wheels are still moving, 0 once it has parked.
#   stop     the label for the stop decision: 1 = parked, 0 = keep driving.
#   MSE      Mean Squared Error, for the wheel commands. Lower is better.
#   BCE      Binary Cross Entropy, for the stop decision. Lower is better.
# -----------------------------------------------------------------------------

import glob
import json
import os
import numpy as np
import torch
import torch.nn as nn
from drive_net import DECIDE_EVERY, MODEL_FOLDER, MODEL_TAG, USE_IMU, DriveNet, make_inputs


# SETTING: seed
#   Fixes the starting weights and the shuffling. Try 0, 1, 2 to see if a result is luck.
#   Type it before the command, for example: ROBOENV_SEED=1 python3 train_drive_net.py
SEED = int(os.environ.get("ROBOENV_SEED", "0"))

# SETTING: training data
#   Starting from the map centre, the turn still needed at the parking spot is almost always near
#   0 or 180 degrees, so the balanced data never shows the robot facing the spot with any other turn.
#   Validation target 79 got lost there. data_v2/random_start shows the expert recovering from such places.
#   The models trained on the balanced data alone are backed up as
#   drive_net_model/drive_net_seed<seed>_balanced.pt (both seeds: 117 of 120). This backup is not submitted.
TRAIN_FOLDERS = ["data_v2/balanced", "data_v2/random_start"]

# SETTING: validation data
#   The same validation runs as train.py, so the numbers can be compared. Only good runs are used.
#   With USE_IMU ON, data_dart cannot be used, because it has no IMU columns. So IMU_VAL_RUNS runs from
#   data_v2/medium_push are used to pick the best epoch. They are not used for training.
SPLIT_FILE = "data_dart/split.json"
IMU_VAL_FOLDER = "data_v2/medium_push"
IMU_VAL_RUNS = 100

# SETTING: early switch label
#   The stop label starts this many rows (1 row = 1 ms) before the expert stopped, as in features.py.
EARLY_ROWS = 200

# SETTING: training length and step size
EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

# SETTING: use less data, to test how much data is needed. 1 means all the runs.
#   4 uses a quarter of the training runs and 4 times the epochs, so the number of weight updates stays about the same.
#   The model's name then ends in "_div4", so the normal model is not touched.
#   Type it before the command, for example: ROBOENV_RUNS_DIVIDE=4 python3 train_drive_net.py
RUNS_DIVIDE = int(os.environ.get("ROBOENV_RUNS_DIVIDE", "1"))
RUNS_TAG = f"_div{RUNS_DIVIDE}" if RUNS_DIVIDE > 1 else ""
EPOCHS = EPOCHS * RUNS_DIVIDE


def expert_reached(f):
    """True if the expert ended within 2 cm of the target (the same rule as train.py)."""
    ee = np.load(os.path.join(os.path.dirname(f) + "_ee", os.path.basename(f)), mmap_mode="r")
    target = np.load(f, mmap_mode="r")[-1, 18:21]
    return np.linalg.norm(ee[-1] - target) * 100 < 2.0


def load(file_list):
    """Reads every 33rd row of each run. The labels are worked out on the full run first."""
    X, Y, driving, stop = [], [], [], []
    for f in file_list:
        d = np.load(f, mmap_mode="r")
        wheels = np.asarray(d[:, 21:25])
        moving_rows = np.nonzero(np.any(wheels != 0, axis=1))[0]
        last_moving = moving_rows[-1] if len(moving_rows) > 0 else -1
        rows = np.arange(0, len(d), DECIDE_EVERY)
        if USE_IMU:
            r = np.asarray(d[rows])
            X.append(make_inputs(r[:, :21], r[:, 32:36], r[:, 53:56], r[:, 56:59]))
        else:
            X.append(make_inputs(np.asarray(d[rows, :21])))
        Y.append(wheels[rows][:, [0, 1]])                       # right wheels, left wheels
        driving.append(rows <= last_moving)
        stop.append(rows >= max(0, last_moving + 1 - EARLY_ROWS))
    return (np.concatenate(X), np.concatenate(Y).astype(np.float32),
            np.concatenate(driving), np.concatenate(stop).astype(np.float32))


train_files = sorted(f for folder in TRAIN_FOLDERS for f in glob.glob(f"{folder}/*.npy"))
if RUNS_DIVIDE > 1:
    np.random.default_rng(0).shuffle(train_files)
    train_files = train_files[:len(train_files) // RUNS_DIVIDE]
if USE_IMU:
    val_files = sorted(glob.glob(f"{IMU_VAL_FOLDER}/*.npy"))[:IMU_VAL_RUNS]
else:
    with open(SPLIT_FILE) as f:
        val_files = [v for v in json.load(f)["val"] if expert_reached(v)]

X_train, Y_train, drive_train, stop_train = load(train_files)
X_val, Y_val, drive_val, stop_val = load(val_files)

# Print results: how much data was loaded.
print(f"runs: train {len(train_files)}, val {len(val_files)}")
print(f"rows: train {len(X_train)}, val {len(X_val)}, driving rows in train {drive_train.mean() * 100:.1f}%")

# Standardise with the training data only. The wheels use the driving rows only,
# because after parking the wheel commands are not used (the test sets them to 0).
X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
Y_mean, Y_std = Y_train[drive_train].mean(axis=0), Y_train[drive_train].std(axis=0) + 1e-8
Y_min, Y_max = Y_train[drive_train].min(axis=0), Y_train[drive_train].max(axis=0)


def to_tensors(X, Y, driving, stop):
    return (torch.from_numpy((X - X_mean) / X_std), torch.from_numpy((Y - Y_mean) / Y_std),
            torch.from_numpy(driving), torch.from_numpy(stop).unsqueeze(1))


x_train, y_train, d_train, s_train = to_tensors(X_train, Y_train, drive_train, stop_train)
x_val, y_val, d_val, s_val = to_tensors(X_val, Y_val, drive_val, stop_val)

torch.manual_seed(SEED)
net = DriveNet()
mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

os.makedirs(MODEL_FOLDER, exist_ok=True)
model_file = f"{MODEL_FOLDER}/drive_net_seed{SEED}{MODEL_TAG}{RUNS_TAG}.pt"
best_val = float("inf")

for epoch in range(EPOCHS):
    # Learn from the training data. The wheel error only counts while the expert is driving.
    net.train()
    order = torch.randperm(len(x_train))
    for start in range(0, len(x_train), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        out = net(x_train[idx])
        keep = d_train[idx]
        loss_stop = bce(out[:, 2:], s_train[idx])
        loss_wheels = mse(out[keep, :2], y_train[idx][keep]) if keep.any() else torch.tensor(0.0)
        loss = loss_wheels + loss_stop
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()

    # Check on the validation data (no learning here).
    net.eval()
    with torch.no_grad():
        out = net(x_val)
        val_wheels = mse(out[d_val, :2], y_val[d_val]).item()
        val_stop = bce(out[:, 2:], s_val).item()
        accuracy = ((out[:, 2:] > 0).float() == s_val).float().mean().item() * 100

    # Print results: one line per epoch.
    print(f"epoch {epoch + 1}/{EPOCHS}  val wheel MSE {val_wheels:.4f}  val stop BCE {val_stop:.4f}  "
          f"stop accuracy {accuracy:.1f}%")

    # Save the best model so far, with the scaling numbers and the command limits.
    if val_wheels + val_stop < best_val:
        best_val = val_wheels + val_stop
        torch.save({"net": net.state_dict(), "epoch": epoch + 1, "seed": SEED, "EARLY_ROWS": EARLY_ROWS,
                    "X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std,
                    "Y_min": Y_min, "Y_max": Y_max}, model_file)

# Print results: where the model was saved.
print(f"Best val error {best_val:.4f}. Saved as {model_file}")
