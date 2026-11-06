# =============================================================================
# train_tip_net.py
# This is for: a network that learns where the gripper tip is from the 7 arm encoder angles, instead of the robot model's
#              fixed maths while the robot runs. The robot model is only used here, to give the training answers.
# How to run:  python train_tip_net.py
# Result:      saves models/tip_net.npz, for my_arm.py with MY_ARM_TIP_NET=1, and prints its error on held-back runs.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   tip         the end of the gripper, in the robot's frame (m).
#   pose        one set of 7 arm joint angles.
#   held-back   runs the network does not learn from, used to check it.
# -----------------------------------------------------------------------------

import glob
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import sensors

# SETTING: where the arm poses come from: expert runs and the arm's own recorded runs on training targets.
EXPERT_FOLDERS = [os.path.expanduser('~/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/data_v2/' + name) for name in ('balanced', 'finish_5mm')]
RECORDINGS = [HERE / 'results/my_arm_record_train50_memory_frames.npy', HERE / 'results/my_arm_record_train50b_nets_frames.npy']

# SETTING: take every Nth row of each expert run, so the training set stays small.
ROW_STEP = 40

# SETTING: fraction of expert runs held back for the check.
HELD_BACK = 0.1

# SETTING: network size and training length.
HIDDEN = 256
EPOCHS = 80
BATCH_SIZE = 512
SEED = 1

hand = sensors.HandPose()


def tip_positions(angles):
    """The training answers, from the robot model."""
    out = np.zeros((len(angles), 3))
    q = np.zeros(17)
    for i, a in enumerate(angles):
        q[4:11] = a
        out[i] = hand([0, 0, 0], [0, 0, 0, 1], q)[0]
    return out


def inputs(angles):
    """Each joint angle, with its sin and cos, because the tip moves in circles as a joint turns."""
    return np.c_[angles, np.sin(angles), np.cos(angles)].astype(np.float32)


def expert_angles(files):
    return np.concatenate([np.asarray(np.load(f, mmap_mode='r')[::ROW_STEP, 4:11]) for f in files])


files = sum([sorted(glob.glob(folder + '/*.npy')) for folder in EXPERT_FOLDERS], [])
np.random.default_rng(0).shuffle(files)
n_back = int(len(files) * HELD_BACK)
train_angles = expert_angles(files[n_back:])
back_angles = expert_angles(files[:n_back])

# The recorded runs: the last 10 targets of each recording are held back.
for path in RECORDINGS:
    frames = np.load(path)
    back_ids = np.unique(frames[:, 0])[-10:]
    is_back = np.isin(frames[:, 0], back_ids)
    train_angles = np.r_[train_angles, frames[~is_back, 9:16]]
    recorded_back = frames[is_back, 9:16]
    back_angles = np.r_[back_angles, recorded_back]

x_train, y_train = inputs(train_angles), tip_positions(train_angles)
x_back, y_back = inputs(back_angles), tip_positions(back_angles)
print(f'poses: training {len(x_train)}, held back {len(x_back)}', flush=True)

x_mean, x_std, y_mean, y_std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8, y_train.mean(axis=0), y_train.std(axis=0)
to_x = lambda x: torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
to_y = lambda y: torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
X, Y = to_x(x_train), to_y(y_train)

torch.manual_seed(SEED)
net = nn.Sequential(nn.Linear(21, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 3))
optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)
for epoch in range(EPOCHS):
    order = torch.randperm(len(X))
    for start in range(0, len(order), BATCH_SIZE):
        batch = order[start:start + BATCH_SIZE]
        loss = ((net(X[batch]) - Y[batch]) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    scheduler.step()
    if (epoch + 1) % 20 == 0:
        with torch.no_grad():
            err = 1000 * np.linalg.norm(net(to_x(x_back)).numpy() * y_std + y_mean - y_back, axis=1)
        # Print results: the error on held-back poses every 20 epochs.
        print(f'epoch {epoch + 1}: held-back error median {np.median(err):.2f} mm, 99% under {np.percentile(err, 99):.2f} mm', flush=True)

with torch.no_grad():
    err = 1000 * np.linalg.norm(net(to_x(x_back)).numpy() * y_std + y_mean - y_back, axis=1)
np.savez(HERE / 'models/tip_net.npz', **{k: v.numpy() for k, v in net.state_dict().items()}, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
print(f'Final held-back error: median {np.median(err):.2f} mm, 90% under {np.percentile(err, 90):.2f}, 99% under {np.percentile(err, 99):.2f}, largest {err.max():.2f}')
print('Saved models/tip_net.npz')
