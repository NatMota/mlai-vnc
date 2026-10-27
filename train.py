# =============================================================================
# train.py
# This is for: teaching the model to copy the expert, from the saved runs.
# How to run:  python3 train.py
# Result:      saves the model in data_dart, ready for drive.py and test.py.
#
# The model has two parts:
#   phase_net   decides "keep driving" or "start reaching".
#   control_net gives the 11 commands (4 wheel speeds + 7 arm torques).
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   X        the inputs (what the robot measures). One row per moment.
#   Y        the outputs (the expert's 11 commands). One row per moment.
#   P        the phase label: 0 = driving, 1 = reaching.
#   _train   rows used for learning.
#   _val     rows used for checking (validation). The model never learns from them.
#   _t       the same table, turned into PyTorch format (a "tensor").
#   batch    a small group of rows (1024) the model learns from in one go.
#   epoch    one full pass through all the training rows.
#   MSE      Mean Squared Error. How far the model's commands are from the expert's.
#            Lower is better.
#   BCE      Binary Cross Entropy. How wrong a yes/no answer is (drive or reach?).
#            Lower is better.
#   AdamW    the tool that adjusts the model after each batch.
#   lr       learning rate: how big each adjustment is.
#   ReLU     a simple rule inside the network: keep positive numbers, turn negatives into 0.
#   Linear(a, b)   one layer: takes a numbers in, gives b numbers out.
# -----------------------------------------------------------------------------

import numpy as np
import glob
import json
import os
import torch
import torch.nn as nn
from features import add_relative, add_ee, N_INPUTS, N_INPUTS_EE, PHASE_COLS, USE_EE, ONLY_GOOD_RUNS, MODEL_NAME


# =============================================================================
# SETTINGS
# The switches (USE_EE, ONLY_GOOD_RUNS) are in features.py, not here.
# =============================================================================

# SETTING: seed
#   Fixes the random choices (the run split, the starting weights and the shuffling),
#   so training again on the same computer gives the same model.
SEED = 0

# SETTING: data folder
#   Which saved runs to learn from.
#   "data_dart"   = runs with pushes (used for the results so far).
#   "data_normal" = plain expert runs.
DATA_FOLDER = "data_dart"

# SETTING: learning rate
#   How big each learning step is. Smaller = slower but steadier.
LEARNING_RATE = 1e-3

# SETTING: batch size
#   How many rows the model looks at in one go.
BATCH_SIZE = 1024

# SETTING: epochs
#   How many times the model goes through all the training data.
#   10 and 20 were tested. 10 gave better driving.
EPOCHS = 10

# SETTING: early switch
#   phase_net learns to say "reach" this many rows before the expert did. 500 rows = 0.5 seconds.
#   This helps the robot stop when it is NEARLY parked, instead of driving past.
#   0 = switch at the same time as the expert.
EARLY_ROWS = 500

# Number of inputs, worked out from the USE_EE switch. Do not change.
N_IN = N_INPUTS_EE if USE_EE else N_INPUTS   # 34 or 28


# =============================================================================
# STEP 1: Split the runs into train, validation and test (70% / 15% / 15%)
# This is for: a fair test. Each whole run goes into one group only.
# The seed 0 makes the split the same every time.
# =============================================================================

files = sorted(glob.glob(f"{DATA_FOLDER}/*.npy"))
rng = np.random.default_rng(SEED)
rng.shuffle(files)

n = len(files)
n_train = int(0.70 * n)
n_val = int(0.15 * n)
train_files = files[:n_train]
val_files = files[n_train:n_train + n_val]
test_files = files[n_train + n_val:]

# Save the split, so drive.py and test.py use the same test runs.
split = {"train": train_files, "val": val_files, "test": test_files}
with open(f"{DATA_FOLDER}/split.json", "w") as f:
    json.dump(split, f, indent=2)

# Keep only the runs where the expert really reached the dot (only if ONLY_GOOD_RUNS = True).
# The test runs are NOT changed, so test.py still uses the same test dots as before.
GOOD_RUN_CM = 2.0   # the expert must end within 2 cm of the dot

def expert_reached(f):
    ee = np.load(os.path.join(DATA_FOLDER + "_ee", os.path.basename(f)), mmap_mode="r")   # made by add_ee.py
    target = np.load(f, mmap_mode="r")[-1, 18:21]
    return np.linalg.norm(ee[-1] - target) * 100 < GOOD_RUN_CM

if ONLY_GOOD_RUNS:
    before = (len(train_files), len(val_files))
    train_files = [f for f in train_files if expert_reached(f)]
    val_files = [f for f in val_files if expert_reached(f)]
    # Print results: how many runs were kept.
    print(f"Good runs kept: train {len(train_files)} of {before[0]}, val {len(val_files)} of {before[1]}")


# =============================================================================
# STEP 2: Label each row as "driving" (0) or "reaching" (1)
# This is for: teaching phase_net when to switch.
# The expert's wheel commands are exactly 0 once it has finished driving.
# =============================================================================

def phase_flags(d, early_rows=0):
    wheels = d[:, 21:25]                         # the 4 expert wheel commands
    moving = np.any(wheels != 0, axis=1)         # True while any wheel is moving
    moving_rows = np.nonzero(moving)[0]
    last_moving = moving_rows[-1] if len(moving_rows) > 0 else -1
    flag = np.zeros(len(d), dtype=np.float32)
    start = max(0, last_moving + 1 - early_rows)  # early_rows moves the switch earlier
    flag[start:] = 1.0
    return flag


# =============================================================================
# STEP 3: Load the runs, one file at a time
# This is for: saving memory. Only one file is open at once.
# =============================================================================

def load(file_list):
    n_rows = sum(np.load(f, mmap_mode="r").shape[0] for f in file_list)
    X = np.empty((n_rows, N_IN), dtype=np.float32)   # inputs
    Y = np.empty((n_rows, 11), dtype=np.float32)     # the 11 expert commands
    P = np.empty((n_rows, 2), dtype=np.float32)      # phase: column 0 = true, column 1 = early
    row = 0
    for f in file_list:
        d = np.load(f)
        if USE_EE:
            ee = np.load(os.path.join(DATA_FOLDER + "_ee", os.path.basename(f)))   # made by add_ee.py
            X[row:row + len(d)] = add_ee(add_relative(d[:, :21]), ee)
        else:
            X[row:row + len(d)] = add_relative(d[:, :21])
        Y[row:row + len(d)] = d[:, 21:]
        P[row:row + len(d), 0] = phase_flags(d)
        P[row:row + len(d), 1] = phase_flags(d, EARLY_ROWS)
        row += len(d)
    return X, Y, P

X_train, Y_train, P_train = load(train_files)
X_val, Y_val, P_val = load(val_files)

# Print results: how much data was loaded.
print("files:", len(train_files), len(val_files), len(test_files))
print("train:", X_train.shape, " val:", X_val.shape)
print("rows in reaching phase:", round(float(P_train[:, 0].mean()) * 100, 1), "%")


# =============================================================================
# STEP 4: Standardise (shift each column to average 0, usual change 1)
# This helps the model: big and small numbers count equally.
# Uses the training data only, and saves the numbers for drive.py.
# =============================================================================

X_mean = X_train.mean(axis=0, dtype=np.float64).astype(np.float32)
X_std = (X_train.std(axis=0, dtype=np.float64) + 1e-8).astype(np.float32)   # + 1e-8 stops dividing by 0
Y_mean = Y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
Y_std = (Y_train.std(axis=0, dtype=np.float64) + 1e-8).astype(np.float32)
np.savez(f"{DATA_FOLDER}/scaler_{MODEL_NAME}.npz", X_mean=X_mean, X_std=X_std, Y_mean=Y_mean, Y_std=Y_std)

X_train -= X_mean; X_train /= X_std
X_val -= X_mean; X_val /= X_std
Y_train -= Y_mean; Y_train /= Y_std
Y_val -= Y_mean; Y_val /= Y_std

X_train_t, Y_train_t, P_train_t = torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(P_train)
X_val_t, Y_val_t, P_val_t = torch.from_numpy(X_val), torch.from_numpy(Y_val), torch.from_numpy(P_val)


# =============================================================================
# STEP 5: Build the two networks
# =============================================================================

torch.manual_seed(SEED)   # fixes the starting weights and the shuffling

# phase_net: sees only the base and the dot (not the arm), then says "drive" or "reach".
# This helps the model: it must decide from "am I parked?", not from "is my arm moving?".
phase_net = nn.Sequential(
    nn.Linear(len(PHASE_COLS), 64), nn.ReLU(),   # 14 inputs in, 64 neurons
    nn.Linear(64, 64), nn.ReLU(),                # 64 neurons
    nn.Linear(64, 1),                            # 1 number out: above 0 = reach
)

# control_net: sees all the inputs plus the phase, then gives the 11 commands.
control_net = nn.Sequential(
    nn.Linear(N_IN + 1, 128), nn.ReLU(),   # inputs + phase in, 128 neurons
    nn.Linear(128, 128), nn.ReLU(),        # 128 neurons
    nn.Linear(128, 11),                    # 11 commands out
)
print(phase_net)
print(control_net)


# =============================================================================
# STEP 6: Train
# MSE checks the commands. BCE checks the yes/no phase answer.
# AdamW and learning rate decay help reduce overfitting.
# =============================================================================

mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
params = list(phase_net.parameters()) + list(control_net.parameters())
optimiser = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)   # weight_decay 1e-4 = 0.0001: stops numbers growing too big
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.8)   # learning rate x 0.8 each epoch

best_val = float("inf")   # best validation MSE so far (starts as "infinitely bad")
best_epoch = 0            # which epoch gave the best model
n_rows = X_train_t.shape[0]   # number of training rows

for epoch in range(EPOCHS):
    # Learn from the training data, 1024 rows at a time.
    phase_net.train(); control_net.train()
    order = torch.randperm(n_rows)   # shuffle the rows
    total_cmd = 0.0   # total training error for this epoch
    for start in range(0, n_rows, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]   # row numbers of the next batch
        x_batch, y_batch, p_batch = X_train_t[idx], Y_train_t[idx], P_train_t[idx]   # inputs, expert commands, phase
        loss_phase = bce(phase_net(x_batch[:, PHASE_COLS]), p_batch[:, 1:])            # phase guess vs early label
        loss_cmd = mse(control_net(torch.cat([x_batch, p_batch[:, :1]], dim=1)), y_batch)  # commands vs expert
        loss = loss_cmd + loss_phase   # total error: commands + phase
        optimiser.zero_grad()   # clear the old adjustments
        loss.backward()         # work out how to adjust
        optimiser.step()        # adjust the model
        total_cmd += loss_cmd.item() * len(idx)

    # Check on the validation data (no learning here).
    phase_net.eval(); control_net.eval()
    v_cmd, v_correct = 0.0, 0.0   # total validation error, number of correct phase guesses
    with torch.no_grad():
        for start in range(0, X_val_t.shape[0], 65536):   # 65536 rows at a time, to save memory
            x_chunk = X_val_t[start:start + 65536]
            y_chunk = Y_val_t[start:start + 65536]
            p_chunk = P_val_t[start:start + 65536]
            v_cmd += mse(control_net(torch.cat([x_chunk, p_chunk[:, :1]], dim=1)), y_chunk).item() * len(x_chunk)
            guess = (phase_net(x_chunk[:, PHASE_COLS]) > 0).float()   # above 0 means "reach"
            v_correct += (guess == p_chunk[:, 1:]).float().sum().item()
    val_cmd = v_cmd / X_val_t.shape[0]
    val_acc = v_correct / X_val_t.shape[0] * 100

    # Print results: one line per epoch. Lower MSE is better.
    print(f"epoch {epoch + 1}/{EPOCHS}  train MSE {total_cmd / n_rows:.4f}  val MSE {val_cmd:.4f}  "
          f"phase accuracy {val_acc:.1f}%  lr {scheduler.get_last_lr()[0]:.6f}")

    # Save the best model so far, with everything needed to check it or carry on training.
    # The optimiser and scheduler are saved BEFORE this epoch's scheduler.step().
    if val_cmd < best_val:
        best_val = val_cmd
        best_epoch = epoch + 1
        torch.save({
            "phase_net": phase_net.state_dict(),
            "control_net": control_net.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": best_epoch,
            "val_mse": best_val,
            "settings": {"SEED": SEED, "DATA_FOLDER": DATA_FOLDER, "LEARNING_RATE": LEARNING_RATE,
                         "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "EARLY_ROWS": EARLY_ROWS,
                         "USE_EE": USE_EE, "ONLY_GOOD_RUNS": ONLY_GOOD_RUNS, "N_IN": N_IN},
        }, f"{DATA_FOLDER}/{MODEL_NAME}_best.pt")

    scheduler.step()

# Print results: the best score, and where the model was saved.
print("best val MSE:", round(best_val, 4), " at epoch", best_epoch)
print(f"Saved as {DATA_FOLDER}/{MODEL_NAME}_best.pt")
