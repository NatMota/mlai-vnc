# =============================================================================
# train_tip_net.py
# This is for: a network that learns where the gripper tip is, seen from the robot, from the 7 arm angles.
#              It replaces the robot model's maths for the plan's input. The answers come from the recorded
#              expert runs, so they include the small tilt of the real base.
# How to run:  python3 train_tip_net.py
# Result:      saves input_net_model/tip_net.npz and prints its error on held-back runs.
#              Used by test_drive_net.py with ROBOENV_TIP_NET=1. Your other models are not touched.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   tip         the end of the gripper, seen from the robot: ahead, left, up (m).
#   reaching    the rows after the expert's wheels have stopped for the last time.
#   held-back   runs the network does not learn from, used to check it.
# -----------------------------------------------------------------------------

import glob
import os
import numpy as np
import torch
import torch.nn as nn

# SETTING: training data, and the same held-back runs as train_arm_plan.py.
DATA_FOLDER = "data_v2/balanced"
VAL_RUNS = 150

# SETTING: take every Nth reaching row, about 30 per second.
ROW_STEP = 33

# SETTING: network size and training length.
HIDDEN = 256
EPOCHS = 60
BATCH_SIZE = 512
SEED = 1

MODEL_FOLDER = "input_net_model"


def load(file_list):
    """The 7 arm angles and the tip seen from the robot, for the reaching rows of each run."""
    angles, tips = [], []
    for f in file_list:
        d = np.load(f, mmap_mode="r")
        moving_rows = np.nonzero(np.any(np.asarray(d[:, 21:25]) != 0, axis=1))[0]
        first_reach = moving_rows[-1] + 1 if len(moving_rows) > 0 else 0
        rows = np.asarray(d[first_reach::ROW_STEP])
        from_base = rows[:, 36:39] - rows[:, :3]
        c, s = np.cos(rows[:, 3]), np.sin(rows[:, 3])
        angles.append(rows[:, 4:11])
        tips.append(np.c_[c * from_base[:, 0] + s * from_base[:, 1], -s * from_base[:, 0] + c * from_base[:, 1], from_base[:, 2]])
    return np.concatenate(angles), np.concatenate(tips)


def inputs(angles):
    """Each joint angle, with its sin and cos, because the tip moves in circles as a joint turns."""
    return np.c_[angles, np.sin(angles), np.cos(angles)]


files = sorted(glob.glob(f"{DATA_FOLDER}/*.npy"))
np.random.default_rng(0).shuffle(files)
angle_train, tip_train = load(files[VAL_RUNS:])
angle_back, tip_back = load(files[:VAL_RUNS])
x_train, x_back = inputs(angle_train), inputs(angle_back)
# Print results: how much data was loaded.
print(f"rows: train {len(x_train)}, held back {len(x_back)}", flush=True)

x_mean, x_std, y_mean, y_std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8, tip_train.mean(axis=0), tip_train.std(axis=0)
X = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32)
Y = torch.tensor((tip_train - y_mean) / y_std, dtype=torch.float32)
X_back = torch.tensor((x_back - x_mean) / x_std, dtype=torch.float32)

torch.manual_seed(SEED)
torch.set_num_threads(2)
net = nn.Sequential(nn.Linear(21, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 3))
optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)


def held_back_error():
    with torch.no_grad():
        return net(X_back).numpy() * y_std + y_mean - tip_back


for epoch in range(EPOCHS):
    order = torch.randperm(len(X))
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        loss = ((net(X[batch]) - Y[batch]) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()
    if (epoch + 1) % 15 == 0:
        # Print results: the error on held-back runs every 15 epochs.
        print(f"epoch {epoch + 1}/{EPOCHS}: median {1000 * np.median(np.linalg.norm(held_back_error(), axis=1)):.2f} mm", flush=True)

os.makedirs(MODEL_FOLDER, exist_ok=True)
np.savez(f"{MODEL_FOLDER}/tip_net.npz", **{k: v.numpy() for k, v in net.state_dict().items()},
         x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

# Print results: the final error and where the model was saved.
error = held_back_error()
size = 1000 * np.linalg.norm(error, axis=1)
print(f"Held-back error: median {np.median(size):.2f} mm, 90% under {np.percentile(size, 90):.2f}, "
      f"99% under {np.percentile(size, 99):.2f}, average direction {np.round(1000 * error.mean(axis=0), 2)} mm")
print(f"Saved {MODEL_FOLDER}/tip_net.npz")
