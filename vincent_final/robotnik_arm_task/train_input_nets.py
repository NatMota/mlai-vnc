# =============================================================================
# train_input_nets.py
# This is for: networks that replace analytical calculations in the inputs of the three-network model.
#              park   = base pose and target to the 4 parking inputs (replaces the parking formula).
#              target = base pose and target to the target's position seen from the robot
#                       (replaces the rotation into the robot's frame during the reach).
#              The analytical calculations are only used here, to give the training answers.
#              They do not run on the robot when these networks are switched ON.
# How to run:  python3 train_input_nets.py park      (or target)
# Result:      saves input_net_model/<park or target>_net.pt and prints its error on separate check rows.
#              Used by test_drive_net.py with ROBOENV_PARK_NET=1 or ROBOENV_TARGET_NET=1.
#              Your other models are not touched.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   target        the red ball the gripper has to reach (world frame, m).
#   base pose     where the base is and which way it faces.
#   parking spot  where the expert parks the base before it starts reaching.
#   X21           one row of a run, columns 0 to 20 (see INPUT LAYOUT in features.py).
#   check rows    random rows the network does not learn from.
# -----------------------------------------------------------------------------

import os
import sys
import numpy as np
import torch
from features import STANDOFF_POINTS, STANDOFF_Z_POINTS, add_park, add_relative
from input_nets import InputNet, MODEL_FOLDER, make_inputs

# SETTING: which network, typed after the command.
KIND = sys.argv[1]
assert KIND in ("park", "target"), "Type park or target after the command"

# SETTING: seed. Type it before the command, for example: ROBOENV_SEED=1 python3 train_input_nets.py park
SEED = int(os.environ.get("ROBOENV_SEED", "0"))

# SETTING: training length. New random rows are made every epoch, so the network never runs out of data.
EPOCHS = 60
ROWS_PER_EPOCH = 400000
CHECK_ROWS = 200000
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

# SETTING: the box the task's targets come from (m), the same as the "new" targets in test_drive_net.py.
TARGET_LOW, TARGET_HIGH = [-1.5, -1.5, 0.1], [1.5, 1.5, 1.25]

# SETTING: how far the base can be from the expert's parking spot (m) and heading (rad) in the "target" rows.
#   The "target" network only runs after parking, so it only needs to learn poses near the spot.
SPOT_SPREAD_M = 0.35
HEADING_SPREAD = 0.6


def make_rows(rng, n):
    """Random base poses and targets, in the places the robot really visits. Returns X21 with only the used columns filled."""
    target = rng.uniform(TARGET_LOW, TARGET_HIGH, size=(n, 3))
    standoff = np.interp(target[:, 2], STANDOFF_Z_POINTS, STANDOFF_POINTS)
    line = np.arctan2(target[:, 1], target[:, 0])
    spot = target[:, :2] - standoff[:, None] * np.c_[np.cos(line), np.sin(line)]
    if KIND == "target":
        # Near the parking spot, roughly facing along the line.
        base = spot + rng.normal(0, SPOT_SPREAD_M / 2, size=(n, 2))
        bearing = line + rng.normal(0, HEADING_SPREAD / 2, size=n)
    else:
        # Half: on the way from the map centre to the spot. Half: anywhere nearby, to cover wrong turns.
        along = rng.uniform(0, 1.05, size=(n, 1)) * spot + rng.normal(0, 0.15, size=(n, 2))
        anywhere = rng.uniform(-2.2, 2.2, size=(n, 2))
        base = np.where(rng.random((n, 1)) < 0.5, along, anywhere)
        bearing = rng.uniform(-np.pi, np.pi, size=n)
    X21 = np.zeros((n, 21))
    X21[:, :2], X21[:, 3], X21[:, 18:21] = base, bearing, target
    return X21


def answers(X21):
    """The training answers, from the analytical calculations."""
    if KIND == "park":
        return add_park(X21)[:, -4:]
    relative = add_relative(X21)
    return np.c_[relative[:, 21], relative[:, 22], X21[:, 20]]     # ahead, left, target height


rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(2)

check_rows = make_rows(rng, CHECK_ROWS)
check_y = answers(check_rows)
sample = make_rows(rng, CHECK_ROWS)
x_mean, x_std = make_inputs(sample).mean(axis=0), make_inputs(sample).std(axis=0) + 1e-8
y_mean, y_std = answers(sample).mean(axis=0).astype(np.float32), answers(sample).std(axis=0).astype(np.float32) + 1e-8

net = InputNet(check_y.shape[1])
optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)


def check():
    """Error on the check rows, in the same units as the answers."""
    with torch.no_grad():
        out = net(torch.from_numpy((make_inputs(check_rows) - x_mean) / x_std)).numpy() * y_std + y_mean
    return out - check_y


for epoch in range(EPOCHS):
    rows = make_rows(rng, ROWS_PER_EPOCH)
    x = torch.from_numpy((make_inputs(rows) - x_mean) / x_std)
    y = torch.from_numpy((answers(rows).astype(np.float32) - y_mean) / y_std)
    order = torch.randperm(len(x))
    for start in range(0, len(x), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]
        loss = ((net(x[idx]) - y[idx]) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        size = 1000 * np.linalg.norm(check()[:, :2], axis=1)
        # Print results: the position error on the check rows every 10 epochs.
        print(f"epoch {epoch + 1}/{EPOCHS}: position error median {np.median(size):.2f} mm, 99% under {np.percentile(size, 99):.2f} mm", flush=True)

os.makedirs(MODEL_FOLDER, exist_ok=True)
torch.save({"net": net.state_dict(), "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
           f"{MODEL_FOLDER}/{KIND}_net.pt")

# Print results: the final error and where the model was saved.
error = check()
size = 1000 * np.linalg.norm(error[:, :2], axis=1)
print(f"Final position error: median {np.median(size):.2f} mm, 90% under {np.percentile(size, 90):.2f} mm, "
      f"99% under {np.percentile(size, 99):.2f} mm")
if KIND == "park":
    print(f"Turn input error (sin, cos): median {np.median(np.abs(error[:, 2:])):.4f}, 99% under {np.percentile(np.abs(error[:, 2:]), 99):.4f}")
else:
    print(f"Height error: median {1000 * np.median(np.abs(error[:, 2])):.2f} mm")
print(f"Saved {MODEL_FOLDER}/{KIND}_net.pt")
