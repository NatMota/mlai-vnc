# =============================================================================
# train_ball_stop.py
# This is for: two small networks trained on the arm's own recorded runs (my_arm.py with MY_ARM_RECORD=1).
#                ball network = where the target is, from a summary of the red pixels. Replaces the ball fit.
#                stop network = how far the tip really is from the target. Replaces the fixed stop rule.
#              True positions are only used here, as training answers. The robot never sees them.
# How to run:  python train_ball_stop.py FRAMES.npy
#              python train_ball_stop.py FRAMES.npy DRIVING.npy   (recordings made with memory: also learns from driving views,
#                                                                  and the stop network reads the remembered target)
# Result:      saves models/ball_stop_net.npz, or models/ball_stop_net_memory.npz for recordings made with memory.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   target     the red ball the gripper has to reach.
#   tip        the end of the gripper, from the arm encoders and the robot model.
#   tip gap    target minus tip, in the robot's frame.
#   frame      one camera picture (30 per second).
# -----------------------------------------------------------------------------

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

# SETTING: fewest red pixels the ball network will read. With fewer, the robot keeps its last answer.
MIN_PIXELS = 8

# SETTING: rows to skip after PARK when training the stop network (ms), while the base is still settling.
SKIP_MS = 300

# SETTING: network size and training length.
HIDDEN = 64
BALL_EPOCHS, STOP_EPOCHS = 150, 60
SEED = 1

frames = np.load(sys.argv[1])
target_id, ms, true_distance = frames[:, 0], frames[:, 1], frames[:, 2]
tip, true_target, speed, summary = frames[:, 3:6], frames[:, 6:9], frames[:, 16:23], frames[:, 23:32]
seen = ~np.isnan(summary[:, 0])
with_memory = len(sys.argv) > 2


def ball_inputs(s):
    """Average point (3), spread (3, multiplied by 30 so it has a similar size), median and smallest distance, log of the pixel count."""
    return np.c_[s[:, :3], s[:, 3:6] * 30, s[:, 6:8], np.log(np.maximum(s[:, 8], 1))]


def train(x, y, epochs):
    """Trains one network. Returns its predictions for every row, and its weights and scaling."""
    x_mean, x_std, y_std = x.mean(axis=0), x.std(axis=0) + 1e-8, y.std(axis=0) + 1e-8
    inputs, answers = torch.tensor((x - x_mean) / x_std, dtype=torch.float32), torch.tensor(y / y_std, dtype=torch.float32)
    torch.manual_seed(SEED)
    net = nn.Sequential(nn.Linear(x.shape[1], HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, y.shape[1]))
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    for _ in range(epochs):
        order = torch.randperm(len(inputs))
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            loss = ((net(inputs[batch]) - answers[batch]) ** 2).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        scheduler.step()
    return {k: v.numpy() for k, v in net.state_dict().items()}, x_mean, x_std, y_std


def run(saved, x):
    w, x_mean, x_std, y_std = saved
    out = (x - x_mean) / x_std
    for layer in ('0', '2'):
        out = np.maximum(out @ w[f'{layer}.weight'].T + w[f'{layer}.bias'], 0)
    return (out @ w['4.weight'].T + w['4.bias']) * y_std


# The ball network learns the distance from the average red point to the ball's true centre.
enough = seen & (np.nan_to_num(summary[:, 8]) >= MIN_PIXELS)
ball_x = ball_inputs(np.nan_to_num(summary))
ball_data_x, ball_data_y = ball_x[enough], (true_target - summary[:, :3])[enough]
if with_memory:
    # Driving views: columns are target number, true target (3), red pixel summary (9).
    driving = np.load(sys.argv[2])
    driving = driving[driving[:, 12] >= MIN_PIXELS]
    ball_data_x = np.r_[ball_data_x, ball_inputs(driving[:, 4:13])]
    ball_data_y = np.r_[ball_data_y, driving[:, 1:4] - driving[:, 4:7]]
ball = train(ball_data_x, ball_data_y, BALL_EPOCHS)

# Its answers, keeping the last good answer in each run, as on the robot.
target_estimate = np.full((len(frames), 3), np.nan)
answer = run(ball, ball_x) + np.nan_to_num(summary[:, :3])
for one_run in np.unique(target_id):
    last = None
    for row in np.where(target_id == one_run)[0]:
        last = answer[row] if enough[row] else last
        target_estimate[row] = last if last is not None else np.nan

# The stop network learns the true distance from the estimated gap, the speeds and how much the tip moved.
moved = np.zeros(len(frames))
for one_run in np.unique(target_id):
    rows = np.where(target_id == one_run)[0]
    for i, row in enumerate(rows):
        moved[row] = np.linalg.norm(tip[row] - tip[rows[max(0, i - 10)]])
gap = (frames[:, 35:38] if with_memory else target_estimate) - tip
stop_x = np.c_[gap, np.linalg.norm(gap, axis=1), np.abs(speed), moved * 100]
usable = (ms > SKIP_MS) & ~np.isnan(gap[:, 0])
stop = train(stop_x[usable], np.minimum(true_distance[usable], 0.1)[:, None], STOP_EPOCHS)

out_name = 'ball_stop_net_memory.npz' if with_memory else 'ball_stop_net.npz'
np.savez(HERE / 'models' / out_name, **{f'ball.{k}': v for k, v in ball[0].items()}, ball_x_mean=ball[1], ball_x_std=ball[2], ball_y_std=ball[3],
         **{f'stop.{k}': v for k, v in stop[0].items()}, stop_x_mean=stop[1], stop_x_std=stop[2], stop_y_std=stop[3])

# Print results: errors on the training frames. These flatter the networks, so the robot test is what counts.
errors = 1000 * np.linalg.norm(answer - true_target, axis=1)[enough]
print(f'frames {len(frames)}, ball frames {int(enough.sum())}, stop frames {int(usable.sum())}')
print(f'ball error on training frames: median {np.median(errors):.1f} mm')
print(f'stop error on training frames: median {1000 * np.median(np.abs(run(stop, stop_x[usable])[:, 0] - np.minimum(true_distance[usable], 0.1))):.1f} mm')
print('Saved models/' + out_name)
