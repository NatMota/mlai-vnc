# =============================================================================
# train_stop_net.py
# This is for: retraining only the stop network, on runs recorded with the ball network and memory, on training targets
#              the ball network has not seen. The stop network then learns from readings like the robot will really get.
# How to run:  python train_stop_net.py results/my_arm_record_train50b_nets_frames.npy
#              MY_ARM_STOP_SPEED=0 python train_stop_net.py FRAMES.npy     (the version without joint speeds)
# Result:      saves models/ball_stop_net_final.npz: the ball network from ball_stop_net_memory.npz and the new stop network.
#              This is the file the reported robot uses, with MY_ARM_MEMORY=1 and MY_ARM_MEMORY_NETS=1.
#              With MY_ARM_STOP_SPEED=0 it saves models/ball_stop_net_nospeed.npz instead. That version scored 22 of 37
#              against 23, so it is not used. True positions are only training answers here.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   target        the red ball the gripper has to reach.
#   tip           the end of the gripper, from the arm encoders and the robot model.
#   tip gap       remembered target minus tip, in the robot's frame.
#   true distance how far the tip really is from the target. This is what the network learns.
# -----------------------------------------------------------------------------

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

# SWITCH: joint speeds as stop network inputs.
#   ON  (not set, or MY_ARM_STOP_SPEED=1): the 7 joint speeds are inputs. This makes the reported ball_stop_net_final.npz.
#   OFF (MY_ARM_STOP_SPEED=0): no joint speeds, because a wrist joint can spin fast while the tip stays still.
#        How far the tip moved over the last 3 camera frames is added instead. Saves ball_stop_net_nospeed.npz.
USE_SPEED = os.environ.get('MY_ARM_STOP_SPEED', '1') == '1'
OUTPUT_FILE = 'models/ball_stop_net_final.npz' if USE_SPEED else 'models/ball_stop_net_nospeed.npz'

# SETTING: rows to skip after PARK (ms), while the base is still settling.
SKIP_MS = 300

# SETTING: network size and training length.
HIDDEN = 64
EPOCHS = 60
SEED = 1

frames = np.load(sys.argv[1])
target_id, ms, true_distance, tip, speed, memory = frames[:, 0], frames[:, 1], frames[:, 2], frames[:, 3:6], frames[:, 16:23], frames[:, 35:38]

# How far the tip moved over the last 10 and the last 3 camera frames.
moved = np.zeros(len(frames))
moved_short = np.zeros(len(frames))
for one_run in np.unique(target_id):
    rows = np.where(target_id == one_run)[0]
    for i, row in enumerate(rows):
        moved[row] = np.linalg.norm(tip[row] - tip[rows[max(0, i - 10)]])
        moved_short[row] = np.linalg.norm(tip[row] - tip[rows[max(0, i - 3)]])

gap = memory - tip
if USE_SPEED:
    x = np.c_[gap, np.linalg.norm(gap, axis=1), np.abs(speed), moved * 100]
else:
    x = np.c_[gap, np.linalg.norm(gap, axis=1), moved * 100, moved_short * 100]
usable = (ms > SKIP_MS) & ~np.isnan(gap[:, 0])
x, y = x[usable], np.minimum(true_distance[usable], 0.1)[:, None]

x_mean, x_std, y_std = x.mean(axis=0), x.std(axis=0) + 1e-8, y.std(axis=0) + 1e-8
inputs, answers = torch.tensor((x - x_mean) / x_std, dtype=torch.float32), torch.tensor(y / y_std, dtype=torch.float32)
torch.manual_seed(SEED)
net = nn.Sequential(nn.Linear(x.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))
optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
for _ in range(EPOCHS):
    order = torch.randperm(len(inputs))
    for start in range(0, len(order), 128):
        batch = order[start:start + 128]
        loss = ((net(inputs[batch]) - answers[batch]) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()

# Keep the ball network and replace the stop network.
final = {k: v for k, v in np.load(HERE / 'models/ball_stop_net_memory.npz').items() if k.startswith('ball')}
final.update({f'stop.{k}': v.numpy() for k, v in net.state_dict().items()})
final.update(stop_x_mean=x_mean, stop_x_std=x_std, stop_y_std=y_std)
np.savez(HERE / OUTPUT_FILE, **final)

# Print results: this error is on the training frames, so it flatters the network.
with torch.no_grad():
    error = np.abs(net(inputs).numpy()[:, 0] * y_std[0] - y[:, 0])
print(f'stop frames {len(x)}, training error median {1000 * np.median(error):.1f} mm. Saved {OUTPUT_FILE}')
