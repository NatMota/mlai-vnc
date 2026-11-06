# =============================================================================
# train.py
# This is for: teaching the model to copy the expert, from the saved runs.
# How to run:  python3 train.py
# Result:      saves the model in data_dart, ready for drive.py and test.py.
#              Saves the training results in a new folder every run:
#              results/<model>_seed<seed>/train_<date>_<time>/
#              (training.txt and training_curve.png).
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
#            The best epoch is picked with these.
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
import datetime
import torch
import torch.nn as nn
from features import (add_relative, add_ee, N_INPUTS, N_INPUTS_EE, PHASE_COLS, USE_EE, ONLY_GOOD_RUNS,
                      SPLIT_HEADS, START_STOP, EARLY_ROWS, USE_DAGGER, DAGGER_FOLDER, EXTRA_FOLDERS,
                      USE_IMU, N_IMU, add_imu, INPUT_SIZE, PHASE_SIZE, KEEP_COLS, CONTROL_COLS, N_KEEP, N_IN,
                      ACTION_MASK, DRIVE_ROW_STEP, NEAR_STOP_ROWS, DECIDE_EVERY, CLOSE_REPEAT, CLOSE_DISTANCE,
                      ONLY_EXTRA_FOLDERS, USE_PARK, add_park,
                      MODEL_NAME, build_control_net)


# =============================================================================
# SETTINGS
# The switches (USE_EE, ONLY_GOOD_RUNS, SPLIT_HEADS, START_STOP) are in features.py, not here.
# =============================================================================

# SETTING: seed
#   Fixes the random choices (the starting weights and the shuffling),
#   so training again on the same computer gives the same model.
#   Try 0, 1, 2 to see if a result is luck. Back up the model first: training overwrites it.
SEED = 0

# SETTING: split seed
#   Decides which runs go into train, val and test. Keep it at 0,
#   so every model is tested on the same test targets.
SPLIT_SEED = 0

# SETTING: data folder
#   Which saved runs to learn from. The model is saved in this folder too.
#   "data_dart"   = runs with big pushes (used for the results so far).
#   "data_dart2"  = runs with medium pushes (fewer failed runs). Needs add_ee.py data_dart2 first.
#   "data_normal" = plain expert runs.
#   The test targets always come from data_dart (set in test.py), so scores can be compared.
#   To learn from more folders at once, use EXTRA_FOLDERS in features.py.
DATA_FOLDER = "data_dart"
#   The best model so far used "data_dart" (with EXTRA_FOLDERS = ["data_dart2"] in features.py).
#   For the data_v2 tests: "data_v2/big_push" (with EXTRA_FOLDERS = ["data_v2/medium_push"]).

# SETTING: row step
#   1 = learn from every row (1000 per second). 2 = every 2nd row (half the memory), and so on.
#   Use 2 if training stops with "Killed" (out of memory).
ROW_STEP = 1

# Name used for the results folder. "_dart2" is added when not using data_dart,
# so results from different data are kept apart.
RUN_NAME = MODEL_NAME + ("" if DATA_FOLDER == "data_dart" else "_" + DATA_FOLDER.replace("data_", "").replace("/", "_"))

# SETTING: learning rate
#   How big each learning step is. Smaller = slower but steadier.
LEARNING_RATE = 1e-3

# SETTING: batch size
#   How many rows the model looks at in one go.
BATCH_SIZE = 1024

# SETTING: epochs
#   How many times the model goes through all the training data.
#   10 and 20 were tested at 1000 rows per second. 10 gave better driving.
#   With DECIDE_EVERY above 1 (features.py) there are N times fewer rows, so each epoch is N times
#   shorter. 3 x N epochs then gives enough practice, and training is still about 3 times faster
#   than with every row. For example: DECIDE_EVERY 10 = 30 epochs, DECIDE_EVERY 33 = 99 epochs.
EPOCHS = 10 if DECIDE_EVERY == 1 else 3 * DECIDE_EVERY

# The learning rate shrinks a little after each epoch. It is set so that, after the last epoch,
# it has shrunk by the same total amount as 10 epochs x 0.8 (as before). Do not change.
LR_DECAY = 0.8 ** (10 / EPOCHS)

# The early switch setting (EARLY_ROWS) is now in features.py, with the switches.

# Number of inputs: N_IN comes from features.py (INPUT_SIZE, plus 5 if USE_IMU is ON).


# =============================================================================
# STEP 1: Split the runs into train, validation and test (70% / 15% / 15%)
# This is for: a fair test. Each whole run goes into one group only.
# SPLIT_SEED makes the split the same every time.
# =============================================================================

files = sorted(glob.glob(f"{DATA_FOLDER}/*.npy"))
rng = np.random.default_rng(SPLIT_SEED)
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

# Keep only the runs where the expert really reached the target (only if ONLY_GOOD_RUNS = True).
# The test runs are NOT changed, so test.py still uses the same test targets as before.
GOOD_RUN_CM = 2.0   # the expert must end within 2 cm of the target

def expert_reached(f):
    # The end-effector file has the same name, in the folder with "_ee" on the end (made by add_ee.py).
    ee = np.load(os.path.join(os.path.dirname(f) + "_ee", os.path.basename(f)), mmap_mode="r")
    target = np.load(f, mmap_mode="r")[-1, 18:21]
    return np.linalg.norm(ee[-1] - target) * 100 < GOOD_RUN_CM

if ONLY_GOOD_RUNS:
    before = (len(train_files), len(val_files))
    train_files = [f for f in train_files if expert_reached(f)]
    val_files = [f for f in val_files if expert_reached(f)]
    # Print results: how many runs were kept.
    print(f"Good runs kept: train {len(train_files)} of {before[0]}, val {len(val_files)} of {before[1]}")

# ONLY_EXTRA_FOLDERS ON (features.py): do not train on DATA_FOLDER's runs at all. DATA_FOLDER is then
# only used for the validation runs (above) and the test targets (test.py).
if ONLY_EXTRA_FOLDERS:
    train_files = []
    # Print results: say that only the extra folders are used.
    print(f"Training only on EXTRA_FOLDERS: {EXTRA_FOLDERS}. Validation runs still from {DATA_FOLDER}.")

# Add the runs from the extra folders to the training runs (EXTRA_FOLDERS in features.py).
# All their runs can be used for training, because the test and validation targets come from DATA_FOLDER.
for folder in EXTRA_FOLDERS:
    extra_files = sorted(glob.glob(f"{folder}/*.npy"))
    if ONLY_GOOD_RUNS:
        extra_files = [f for f in extra_files if expert_reached(f)]
    train_files = train_files + extra_files
    # Print results: how many extra runs were added.
    print(f"Extra runs added to training from {folder}: {len(extra_files)}")

# Add the DAgger runs to the training runs (only if USE_DAGGER = True).
# They are ALL kept, even when the model missed: those are the mistakes the expert corrects.
# They are only used for training, never for validation or testing.
if USE_DAGGER:
    dagger_files = sorted(glob.glob(f"{DAGGER_FOLDER}/*.npy"))
    train_files = train_files + dagger_files
    # Print results: how many DAgger runs were added.
    print(f"DAgger runs added to training: {len(dagger_files)}")

# Repeat the close-target training runs (CLOSE_REPEAT in features.py), so the model practises them more.
# A run is "close" when the target starts within CLOSE_DISTANCE of the robot (on the floor).
def start_distance(f):
    d = np.load(f, mmap_mode="r")
    return float(np.hypot(*(np.asarray(d[0, 18:20]) - np.asarray(d[0, 0:2]))))

if CLOSE_REPEAT > 1:
    close_files = [f for f in train_files if start_distance(f) < CLOSE_DISTANCE]
    train_files = train_files + close_files * (CLOSE_REPEAT - 1)
    # Print results: how many close-target runs were repeated.
    print(f"Close-target runs (within {CLOSE_DISTANCE} m): {len(close_files)}, "
          f"each used {CLOSE_REPEAT} times. Training runs now: {len(train_files)}")


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

def rows_to_keep(d):
    """Row numbers to keep from one run (ROW_STEP here; DRIVE_ROW_STEP, NEAR_STOP_ROWS and
    DECIDE_EVERY in features.py). step = ROW_STEP x DECIDE_EVERY.
    Reaching rows, and the last NEAR_STOP_ROWS driving rows: every step-th row.
    Other driving rows: every (step x DRIVE_ROW_STEP)th row."""
    step = ROW_STEP * DECIDE_EVERY
    row_numbers = np.arange(len(d))
    reaching = phase_flags(d) == 1.0
    near_stop = phase_flags(d, NEAR_STOP_ROWS) == 1.0   # reaching rows + the last NEAR_STOP_ROWS driving rows
    keep = (row_numbers % step == 0) & (reaching | near_stop | (row_numbers % (step * DRIVE_ROW_STEP) == 0))
    return np.nonzero(keep)[0]

def load(file_list):
    n_rows = sum(len(rows_to_keep(np.load(f, mmap_mode="r"))) for f in file_list)
    X = np.empty((n_rows, N_KEEP), dtype=np.float32)   # inputs (the columns in KEEP_COLS)
    Y = np.empty((n_rows, 11), dtype=np.float32)     # the 11 expert commands
    P = np.empty((n_rows, 2), dtype=np.float32)      # phase: column 0 = true, column 1 = early
    row = 0
    for f in file_list:
        d_full = np.load(f)
        keep = rows_to_keep(d_full)
        flags_true = phase_flags(d_full)                # worked out on the full run,
        flags_early = phase_flags(d_full, EARLY_ROWS)   # then the rows are thinned out below
        d = d_full[keep]
        if USE_EE:
            # The end-effector file has the same name, in the folder with "_ee" on the end.
            # data_dart_ee is made by add_ee.py. data_dagger_ee is made by dagger.py.
            ee = np.load(os.path.join(os.path.dirname(f) + "_ee", os.path.basename(f)))[keep]
            x = add_ee(add_relative(d[:, :21]), ee)
        else:
            x = add_relative(d[:, :21])
        if USE_IMU:
            assert d.shape[1] >= 59, f"{f} has no IMU columns. USE_IMU needs the data_v2 folders."
            x = add_imu(x, d[:, 3], d[:, 32:36], d[:, 53:56], d[:, 56:59])
        if USE_PARK:
            x = add_park(x)   # the 4 parking-spot inputs (features.py)
        X[row:row + len(d)] = x[:, KEEP_COLS]   # keep only the inputs chosen by INPUT_SIZE
        Y[row:row + len(d)] = d[:, 21:32]   # the 11 expert commands (columns 21 to 31)
        P[row:row + len(d), 0] = flags_true[keep]
        P[row:row + len(d), 1] = flags_early[keep]
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

# phase_net: sees only the base and the target (not the arm), then says "drive" or "reach".
# This helps the model: it must decide from "am I parked?", not from "is my arm moving?".
phase_net = nn.Sequential(
    nn.Linear(len(PHASE_COLS), 64), nn.ReLU(),   # 14 inputs in (PHASE_SIZE with INPUT_SIZE 20 or 13), 64 neurons
    nn.Linear(64, 64), nn.ReLU(),                # 64 neurons
    nn.Linear(64, 1),                            # 1 number out: above 0 = reach
)

# control_net: sees all the inputs plus the phase, then gives the 11 commands.
# Its shape depends on the SPLIT_HEADS switch (see features.py).
# With START_STOP ON, it also gives the start and stop commands,
# and only its arm end is told the phase (so stop cannot just copy it).
control_net = build_control_net(N_IN + 1)   # inputs + phase in

def control_input(x, p):
    """The inputs for control_net: its columns of x (CONTROL_COLS) + the true phase."""
    return torch.cat([x[:, CONTROL_COLS], p[:, :1]], dim=1)
print(phase_net)
print(control_net)


# =============================================================================
# STEP 6: Train
# MSE checks the commands. BCE checks the yes/no phase answer.
# AdamW and learning rate decay help reduce overfitting.
# =============================================================================

mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
cross_entropy = nn.CrossEntropyLoss()   # for start/stop: which of the 2 is right?

# Which of the 11 commands count in the error, for each row (ACTION_MASK in features.py).
# ACTION_MASK OFF: all 11 count. ON: driving rows count the 4 wheels only, reaching rows the 7 arm torques only.
def command_mask(p):
    """p: the phase labels (column 0 = true phase). Gives back 0 or 1 for each command, shape (rows, 11)."""
    if not ACTION_MASK:
        return torch.ones(len(p), 11)
    reaching = p[:, :1]                                        # 1 while reaching, 0 while driving
    return torch.cat([(1 - reaching).expand(-1, 4), reaching.expand(-1, 7)], dim=1)

# The command error (MSE), counting only the commands that the mask keeps.
def command_error(out, y, p):
    mask = command_mask(p)
    return ((out[:, :11] - y) ** 2 * mask).sum() / mask.sum()

# Checks the model on rows it does not learn from. Gives back the MSE and the phase accuracy (%).
def check(X_t, Y_t, P_t):
    total, count, correct = 0.0, 0.0, 0.0   # total squared error, number of commands counted, correct phase guesses
    with torch.no_grad():
        for start in range(0, X_t.shape[0], 65536):   # 65536 rows at a time, to save memory
            x_chunk = X_t[start:start + 65536]
            y_chunk = Y_t[start:start + 65536]
            p_chunk = P_t[start:start + 65536]
            out = control_net(control_input(x_chunk, p_chunk))
            mask = command_mask(p_chunk)
            total += (((out[:, :11] - y_chunk) ** 2) * mask).sum().item()
            count += mask.sum().item()
            if START_STOP:
                guess = out[:, 12:13] > out[:, 11:12]   # stop bigger than start means "stop"
            else:
                guess = (phase_net(x_chunk[:, PHASE_COLS]) > 0)   # above 0 means "reach"
            guess = guess.float()
            correct += (guess == p_chunk[:, 1:]).float().sum().item()
    return total / count, correct / X_t.shape[0] * 100

# With START_STOP ON, phase_net is not used, so only control_net learns.
params = list(control_net.parameters()) if START_STOP else list(phase_net.parameters()) + list(control_net.parameters())
optimiser = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)   # weight_decay 1e-4 = 0.0001: stops numbers growing too big
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=LR_DECAY)   # learning rate x 0.8 each epoch (10 epochs)

best_val = float("inf")   # best validation MSE so far (starts as "infinitely bad")
best_epoch = 0            # which epoch gave the best model
train_history, val_history = [], []   # MSE for each epoch, for the graph
log_lines = []   # the printed lines, saved to training.txt at the end

# A new folder for this training run, inside the folder for this model and seed.
# The date and time are in the name, so every run keeps its own results.
STAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")   # for example 2026-10-27_23-15
RESULTS = f"results/{RUN_NAME}_seed{SEED}/train_{STAMP}"
os.makedirs(RESULTS, exist_ok=True)
n_rows = X_train_t.shape[0]   # number of training rows

for epoch in range(EPOCHS):
    # Learn from the training data, 1024 rows at a time.
    phase_net.train(); control_net.train()
    order = torch.randperm(n_rows)   # shuffle the rows
    total_cmd = 0.0   # total training error for this epoch
    for start in range(0, n_rows, BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]   # row numbers of the next batch
        x_batch, y_batch, p_batch = X_train_t[idx], Y_train_t[idx], P_train_t[idx]   # inputs, expert commands, phase
        out = control_net(control_input(x_batch, p_batch))
        loss_cmd = command_error(out, y_batch, p_batch)   # the commands vs expert (only the ones the mask keeps)
        if START_STOP:
            # start/stop vs early label (0 = start, 1 = stop)
            loss_phase = cross_entropy(out[:, 11:13], p_batch[:, 1].long())
        else:
            loss_phase = bce(phase_net(x_batch[:, PHASE_COLS]), p_batch[:, 1:])   # phase guess vs early label
        loss = loss_cmd + loss_phase   # total error: commands + phase
        optimiser.zero_grad()   # clear the old adjustments
        loss.backward()         # work out how to adjust
        optimiser.step()        # adjust the model
        total_cmd += loss_cmd.item() * len(idx)

    # Check on the validation data (no learning here).
    phase_net.eval(); control_net.eval()
    val_cmd, val_acc = check(X_val_t, Y_val_t, P_val_t)

    train_history.append(total_cmd / n_rows)
    val_history.append(val_cmd)

    # Print results: one line per epoch. Lower MSE is better.
    line = (f"epoch {epoch + 1}/{EPOCHS}  train MSE {total_cmd / n_rows:.4f}  val MSE {val_cmd:.4f}  "
            f"phase accuracy {val_acc:.1f}%  lr {scheduler.get_last_lr()[0]:.6f}")
    print(line)
    log_lines.append(line)

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
            "settings": {"SEED": SEED, "SPLIT_SEED": SPLIT_SEED, "DATA_FOLDER": DATA_FOLDER, "LEARNING_RATE": LEARNING_RATE,
                         "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "EARLY_ROWS": EARLY_ROWS, "USE_DAGGER": USE_DAGGER, "USE_IMU": USE_IMU, "ROW_STEP": ROW_STEP, "EXTRA_FOLDERS": EXTRA_FOLDERS,
                         "USE_EE": USE_EE, "ONLY_GOOD_RUNS": ONLY_GOOD_RUNS, "SPLIT_HEADS": SPLIT_HEADS, "START_STOP": START_STOP,
                         "INPUT_SIZE": INPUT_SIZE, "PHASE_SIZE": PHASE_SIZE, "N_IN": N_IN,
                         "ACTION_MASK": ACTION_MASK, "DRIVE_ROW_STEP": DRIVE_ROW_STEP, "NEAR_STOP_ROWS": NEAR_STOP_ROWS, "DECIDE_EVERY": DECIDE_EVERY,
                         "CLOSE_REPEAT": CLOSE_REPEAT, "CLOSE_DISTANCE": CLOSE_DISTANCE,
                         "ONLY_EXTRA_FOLDERS": ONLY_EXTRA_FOLDERS, "USE_PARK": USE_PARK},
        }, f"{DATA_FOLDER}/{MODEL_NAME}_best.pt")

    scheduler.step()

# Print results: the best score, and where the model was saved.
log_lines.append(f"best val MSE: {round(best_val, 4)}  at epoch {best_epoch}")
log_lines.append(f"model: {MODEL_NAME}  seed: {SEED}  saved as {DATA_FOLDER}/{MODEL_NAME}_best.pt")
print(log_lines[-2])
print(log_lines[-1])

# Save the printed results as a text file in the results folder.
with open(f"{RESULTS}/training.txt", "w") as f:
    f.write("\n".join(log_lines) + "\n")
print(f"Results saved in {RESULTS}/")


# =============================================================================
# STEP 7: Graph: the training curve
# This is for: seeing if the model is still learning, or starting to overfit
# (validation MSE going up while training MSE keeps going down).
# Two lines, like the learning curves in Lecture 6: training and validation.
# The best epoch is picked with validation.
# Saved in the results folder. Open it in VS Code to see it.
# =============================================================================

# Graph colours and style (shared look for all graphs)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

def graph_style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

try:
    import matplotlib
    matplotlib.use("Agg")   # draw to a file, no window
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_history) + 1))
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#fcfcfb")
    graph_style(ax)
    ax.plot(epochs, train_history, color=BLUE, linewidth=2, marker="o", markersize=5, label="Training")
    ax.plot(epochs, val_history, color=ORANGE, linewidth=2, marker="o", markersize=5, label="Validation")
    ax.axvline(best_epoch, color=INK_2, linewidth=1, linestyle="--")
    ax.text(best_epoch, max(train_history + val_history), f"  best: epoch {best_epoch}",
            color=INK_2, va="top", fontsize=9)
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel("MSE (lower is better)", color=INK)
    if len(epochs) <= 20:
        ax.set_xticks(epochs)   # one tick per epoch (with many epochs, matplotlib picks the ticks)
    ax.set_title(f"Training curve ({MODEL_NAME}, seed {SEED})", color=INK, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    graph_file = f"{RESULTS}/training_curve.png"
    fig.savefig(graph_file, dpi=150)
    print(f"Graph saved as {graph_file}")
except ImportError:
    print("No graph: matplotlib is missing. Run with the pixi line to get the graph.")