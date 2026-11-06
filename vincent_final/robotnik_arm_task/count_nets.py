# =============================================================================
# count_nets.py
# This is for: the efficiency numbers for the report: how many parameters each network has, how big its saved
#              file is, and how long one decision takes on one CPU thread.
# How to run:  ROBOENV_SEED=1 python3 count_nets.py
# Result:      prints one table for the saved model and one for the learned version. Nothing is saved or changed.
#              The times change a little between runs.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   parameter  one number the network learned in training (a weight or a bias).
#   decision   one time a network is asked for its output.
# -----------------------------------------------------------------------------

import os
import time
import numpy as np
import torch
import torch.nn as nn

import arm_plan_net
import arm_torque_net
import drive_net
import gap_net
import input_nets

# SETTING: seed of the networks to measure. Type it before the command.
SEED = int(os.environ.get("ROBOENV_SEED", "1"))

# SETTING: how many decisions to time for each network.
TIMING_RUNS = 2000

torch.set_num_threads(1)


def tip_net_from_file(path):
    """Builds the tip network from its NumPy file, so it can be timed the same way as the others."""
    weights = np.load(path)
    net = nn.Sequential(nn.Linear(21, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 3))
    net.load_state_dict({k: torch.from_numpy(weights[k]) for k in net.state_dict()})
    return net


def load(net, path):
    net.load_state_dict(torch.load(path, weights_only=False)["net"])
    return net


# Each row: name, decisions per second on the robot, the network, its number of inputs, its saved file.
PLAN_FILE = f"{arm_plan_net.MODEL_FOLDER}/arm_plan_seed{SEED}.pt"
TORQUE_FILE = f"{arm_torque_net.MODEL_FOLDER}/arm_torque_seed{SEED}.pt"
DRIVE_FILE = f"{drive_net.MODEL_FOLDER}/drive_net_seed{SEED}{drive_net.MODEL_TAG}.pt"
SAVED = [("Driving", 30, load(drive_net.DriveNet(), DRIVE_FILE), drive_net.N_INPUTS, DRIVE_FILE),
         ("Plan", 30, load(arm_plan_net.ArmPlanNet(), PLAN_FILE), 9, PLAN_FILE),
         ("Torque", 1000, load(arm_torque_net.ArmTorqueNet(), TORQUE_FILE), 20, TORQUE_FILE)]
PARK_FILE, TARGET_FILE, TIP_FILE = f"{input_nets.MODEL_FOLDER}/park_net.pt", f"{input_nets.MODEL_FOLDER}/target_net.pt", f"{input_nets.MODEL_FOLDER}/tip_net.npz"
GAP_FILE = f"{gap_net.MODEL_FOLDER}/gap_seed{SEED}.pt"
FULL_ML = SAVED + [("Parking", 30, load(input_nets.InputNet(4), PARK_FILE), 7, PARK_FILE),
                   ("Target", 30, load(input_nets.InputNet(3), TARGET_FILE), 7, TARGET_FILE),
                   ("Tip", 30, tip_net_from_file(TIP_FILE), 21, TIP_FILE),
                   ("Angle gap", 1000, load(gap_net.GapNet(), GAP_FILE), 23, GAP_FILE)]


def show(title, rows):
    print(f"\n{title}")
    print(f"{'Network':<11}{'Parameters':>12}{'File kB':>9}{'Decisions/s':>13}{'ms each':>9}{'CPU share':>11}")
    total_parameters, total_kb, total_share = 0, 0, 0
    for name, rate, net, n_in, path in rows:
        net.eval()
        x = torch.zeros(1, n_in)
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(TIMING_RUNS):
                net(x)
        ms = 1000 * (time.perf_counter() - start) / TIMING_RUNS
        parameters = sum(p.numel() for p in net.parameters())
        kb = os.path.getsize(path) / 1024
        share = ms * rate / 1000          # the fraction of one second this network needs
        total_parameters, total_kb, total_share = total_parameters + parameters, total_kb + kb, total_share + share
        # Print results: one row per network.
        print(f"{name:<11}{parameters:>12}{kb:>9.0f}{rate:>13}{ms:>9.3f}{100 * share:>10.1f}%")
    print(f"{'Total':<11}{total_parameters:>12}{total_kb:>9.0f}{'':>13}{'':>9}{100 * total_share:>10.1f}%")


show("Saved three-network model", SAVED)
show("Learned version", FULL_ML)
