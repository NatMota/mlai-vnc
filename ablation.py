"""Input ablation for the arm: retrain vel_net and torque_net without groups of inputs.

    python ablation.py --output outputs/ablation

A removed input is set to zero after scaling, in training and at run time, so the network has no
information from it and the parameter count stays the same. The base and switch networks in each
saved checkpoint are copied from the --from checkpoint; evaluate with the 'fused:' policy, which uses
the teammate's base and switch, so only the arm changes between variants.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import zlib

import numpy as np
import torch

from train_memory import ROOT, load_runs, tables, train_net

# Column numbers in features(); see the HOW_TO_USE input table (columns there start at 1).
GROUPS = {
    "none": [],
    "world_base_pose": [0, 1, 2, 3, 26, 27],             # base x, y, z, bearing, sin/cos bearing
    "world_goal": [18, 19, 20],                          # goal x, y, z in the world
    "all_world": [0, 1, 2, 3, 18, 19, 20, 26, 27, 28, 29],   # both above + world x, y position error
    "goal_from_base": [21, 22, 23, 24, 25],              # goal relative to the base
    "joint_velocity": list(range(11, 18)),               # state speeds only: torque_net still sees v - qd
}


def mask_for(size, removed):
    mask = np.ones(size, np.float32)
    mask[removed] = 0.0
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data_dart2", "data_dagger/iter1_relabel2"])
    parser.add_argument("--from", dest="donor", type=Path, default=Path("outputs/memory-v5/policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs/ablation"))
    parser.add_argument("--groups", nargs="+", default=list(GROUPS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--good-cm", type=float, default=2.0)
    parser.add_argument("--early-rows", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--threads", type=int, default=20)
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    runs = load_runs(args.data, args.stride, args.good_cm, np.random.default_rng(args.seed), args.early_rows)
    is_val = [zlib.crc32(r["name"].encode()) % 10 == 0 for r in runs]
    train = tables([r for r, v in zip(runs, is_val) if not v])
    val = tables([r for r, v in zip(runs, is_val) if v])
    del runs
    donor = torch.load(args.donor, map_location="cpu", weights_only=False)
    report = {}
    for group in args.groups:
        out = args.output / group
        if (out / "policy.pt").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)
        saved = dict(donor)
        for name in ("vel", "torque"):
            size = train[name][0].shape[1]
            mask = mask_for(size, GROUPS[group])
            saved[name] = train_net(name, train[name], val[name], args, x_mask=mask)
        saved["ablation"] = {"group": group, "removed": GROUPS[group], "seed": args.seed}
        tmp = out / "policy.tmp.pt"
        torch.save(saved, tmp)
        tmp.replace(out / "policy.pt")
        report[group] = {name: saved[name]["best_val"] for name in ("vel", "torque")}
        print("TRAINED", group, json.dumps(report[group]), flush=True)
    summary = args.output / "training_summary.json"
    old = json.loads(summary.read_text()) if summary.exists() else {}
    summary.write_text(json.dumps({**old, **report}, indent=2), encoding="utf-8")
    print("ABLATION_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
