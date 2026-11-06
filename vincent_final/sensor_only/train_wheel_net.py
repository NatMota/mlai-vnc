# =============================================================================
# train_wheel_net.py
# This is for: a small network that learns how the base really moved from the 4 wheel encoder speeds, instead of the fixed
#              wheel formula. The robot uses it to move its memory of the target while the target is hidden.
# How to run:  python train_wheel_net.py ~/COMP0245_Golden_300
# Result:      saves models/wheel_net.npz, for my_arm.py with MY_ARM_WHEEL_NET=1. True base poses are only training answers.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   target   the red ball the gripper has to reach.
#   frame    one camera picture (30 per second).
#   move     how the base moved in one frame, seen from where it was: ahead (m), left (m), turn (rad).
# -----------------------------------------------------------------------------

import json
import sys
from pathlib import Path
import numpy as np
import pybullet as p
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

# SETTING: network size and training length.
HIDDEN = 32
EPOCHS = 30
SEED = 1

root = Path(sys.argv[1]).expanduser()
wheel_speeds, moves = [], []
for dot in json.loads((root / 'split.json').read_text())['train']:     # training targets only
    camera = np.load(root / f'camera/train/camera_{dot}.npz', allow_pickle=False)
    expert = np.load(root / f'expert/train/run_{dot}.npz', allow_pickle=False)
    steps, speed = camera['step'], camera['wheel_speed']
    position, rotation = expert['base_position'][steps], expert['base_quaternion'][steps]
    for i in range(1, len(steps)):
        before = np.array(p.getMatrixFromQuaternion(rotation[i - 1])).reshape(3, 3)
        after = np.array(p.getMatrixFromQuaternion(rotation[i])).reshape(3, 3)
        moved = before.T @ (position[i] - position[i - 1])
        turn = before.T @ after
        wheel_speeds.append(speed[i])
        moves.append([moved[0], moved[1], np.arctan2(turn[1, 0], turn[0, 0])])
x, y = np.array(wheel_speeds, float), np.array(moves, float)

x_mean, x_std, y_std = x.mean(axis=0), x.std(axis=0) + 1e-8, y.std(axis=0) + 1e-8
inputs, answers = torch.tensor((x - x_mean) / x_std, dtype=torch.float32), torch.tensor(y / y_std, dtype=torch.float32)
torch.manual_seed(SEED)
net = nn.Sequential(nn.Linear(4, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 3))
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

np.savez(HERE / 'models/wheel_net.npz', **{k: v.numpy() for k, v in net.state_dict().items()}, x_mean=x_mean, x_std=x_std, y_std=y_std)
# Print results: how much data, and the move it predicts when the wheels are still (should be close to zero).
with torch.no_grad():
    still = net(torch.tensor(-x_mean / x_std, dtype=torch.float32)).numpy() * y_std
print(f'rows {len(x)}. With still wheels it predicts {1000 * still[0]:.3f} mm, {1000 * still[1]:.3f} mm, {np.degrees(still[2]):.4f} deg per frame')
print('Saved models/wheel_net.npz')
