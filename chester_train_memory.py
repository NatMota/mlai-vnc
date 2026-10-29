"""Train the four networks of chester_memory_policy from runs that record the expert's hidden reference.

    python chester_train_memory.py --data data_dart2 --output outputs/memory-seed0 --seed 0
    python chester_train_memory.py --data data_dart2 data_dagger/iter1 --output outputs/dagger1-seed0

Inputs are .npz runs written by chester_collect.py or chester_dagger.py. Expert runs are kept only if
the expert itself ended within --good-cm of the dot (like ONLY_GOOD_RUNS); DAgger runs are always kept,
because their labels are the expert's correction of the learner's own mistakes.
Rows are subsampled (1 kHz rows are nearly identical) with a random offset per run.
About 10% of runs (chosen by file-name hash) are held out for validation MSE; split.json val/test targets are never used here.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time
import zlib

import numpy as np
import torch
from torch import nn

from chester_memory_policy import features, torque_inputs, vel_scale
from chester_pool import parking_goal
from features import PHASE_COLS

ROOT = Path(__file__).resolve().parent
NETS = ("base", "phase", "vel", "torque")
AMBIGUOUS_BASE_M = (0.08, 0.12)   # expert base pos_tol 0.08 m, up to ~one braking distance beyond


def mlp(n_in, n_out, width, depth):
    layers, size = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(size, width), nn.ReLU()]
        size = width
    layers.append(nn.Linear(size, n_out))
    return nn.Sequential(*layers)


def load_runs(folders, stride, good_cm, rng, early_rows=0):
    """Return per-run dicts of the four training tables, subsampled."""
    runs = []
    for folder in folders:
        files = sorted(p for p in (ROOT / folder).glob("run_*.npz") if not p.name.endswith(".tmp.npz"))
        kept = 0
        for path in files:
            with np.load(path) as d:
                # DAgger runs (they carry an expert phase label) are kept even when the learner failed.
                dagger = "phase_label" in d.files
                if not dagger and float(d["final_error_cm"]) > good_cm:
                    continue
                reach_all = d["reach"]
                rows = np.arange(int(rng.integers(stride)), len(reach_all), stride)
                state, ee, label = d["state"][rows], d["ee"][rows], d["label"][rows]
                q_des, qd_des = d["q_des"][rows], d["qd_des"][rows]
                if dagger:
                    weight = d["weight"][rows]
                else:
                    # Expert rows 8-12 cm from the spot are ambiguous for the base (see BaseShadow):
                    # the expert may be driving to the spot or turning to the final bearing.
                    goal, _ = parking_goal(d["state"][0, 18:21])
                    dist = np.linalg.norm(d["state"][rows, :2] - goal, axis=1)
                    weight = ~((dist > AMBIGUOUS_BASE_M[0]) & (dist < AMBIGUOUS_BASE_M[1]))
                if dagger:
                    phase = d["phase_label"][rows]
                else:
                    # Switch label moved early_rows steps (ms) before the expert's base is fully parked,
                    # so the learner also switches when it parks nearly, not exactly, on the spot.
                    start = int(np.argmax(reach_all > 0)) if reach_all.any() else len(reach_all)
                    phase = (np.arange(len(reach_all)) >= start - early_rows)[rows].astype(np.float32)
            reach = reach_all[rows].astype(np.float32)
            q, qd = state[:, 4:11], state[:, 11:18]
            runs.append({
                "x": features(state, ee).astype(np.float32), "reach": reach,
                "phase": phase.astype(np.float32),
                "wheels": label[:, :4].astype(np.float32), "torque": label[:, 4:].astype(np.float32),
                "z": (q_des - q).astype(np.float32), "ev": (qd_des - qd).astype(np.float32),
                "qd_des": qd_des.astype(np.float32), "base_valid": weight.astype(np.float32),
                # The switch is learned from expert runs only: on DAgger runs the shadow's strict
                # "parked" test is rarely met, which taught the phase net never to switch.
                "phase_valid": np.full(len(rows), 0.0 if dagger else 1.0, np.float32),
                "name": f"{folder}/{path.name}",
            })
            kept += 1
        print(f"DATA {folder}: kept {kept} of {len(files)} runs", flush=True)
    return runs


def tables(runs):
    cat = lambda key: np.concatenate([r[key] for r in runs])
    x, reach = cat("x"), cat("reach")
    driving = (reach == 0) & (cat("base_valid") > 0)
    phase_rows = cat("phase_valid") > 0
    reaching = reach == 1
    return {
        "base": (x[driving], cat("wheels")[driving]),
        "phase": (x[phase_rows][:, PHASE_COLS], cat("phase")[phase_rows][:, None]),
        "vel": (x[reaching], cat("qd_des")[reaching] / vel_scale(x[reaching])[:, None]),
        "torque": (torque_inputs(x, cat("z"), cat("ev"), reach), cat("torque")),
    }


def standardiser(a, skip_last=False):
    mean, std = a.mean(axis=0), a.std(axis=0) + 1e-6
    if skip_last:                    # the binary phase flag stays 0/1
        mean[-1], std[-1] = 0.0, 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def train_net(name, train, val, args):
    x_tr, y_tr = train
    x_va, y_va = val
    classify = name == "phase"
    x_mean, x_std = standardiser(x_tr, skip_last=name == "torque")
    if classify:
        y_mean, y_std = np.zeros(1, np.float32), np.ones(1, np.float32)
    else:
        y_mean, y_std = standardiser(y_tr)
    xs = torch.from_numpy((x_tr - x_mean) / x_std)
    ys = torch.from_numpy((y_tr - y_mean) / y_std)
    xv = torch.from_numpy((x_va - x_mean) / x_std)
    yv = torch.from_numpy((y_va - y_mean) / y_std)
    net = mlp(x_tr.shape[1], y_tr.shape[1], args.width, args.depth)
    loss_fn = nn.BCEWithLogitsLoss() if classify else nn.MSELoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * math.ceil(len(xs) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    best, best_state, history = float("inf"), None, []
    gen = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        net.train()
        order = torch.randperm(len(xs), generator=gen)
        total = 0.0
        for i in range(0, len(xs), args.batch):
            idx = order[i:i + args.batch]
            loss = loss_fn(net(xs[idx]), ys[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(net(xv[i:i + 65536]), yv[i:i + 65536]).item() * len(xv[i:i + 65536])
                           for i in range(0, len(xv), 65536)) / max(1, len(xv))
        history.append({"net": name, "epoch": epoch, "train": total / len(xs), "val": val_loss,
                        "seconds": round(time.perf_counter() - started, 1)})
        print(json.dumps(history[-1]), flush=True)
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return {"weights": best_state, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
            "best_val": best, "rows": len(xs), "history": history,
            "parameters": sum(p.numel() for p in net.parameters())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=["data_dart2"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--good-cm", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--early-rows", type=int, default=0, help="switch label this many ms early")
    parser.add_argument("--only", nargs="+", choices=NETS, help="train only these nets ...")
    parser.add_argument("--from", dest="from_ckpt", type=Path, help="... and copy the others from here")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if (out / "policy.pt").exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if bool(args.only) != bool(args.from_ckpt):
        raise SystemExit("--only and --from go together")
    runs = load_runs(args.data, args.stride, args.good_cm, np.random.default_rng(args.seed), args.early_rows)
    # Held-out runs are chosen by a hash of the file name, so the split does not move when
    # --seed, --stride or --good-cm change and experiments stay comparable.
    is_val = [zlib.crc32(r["name"].encode()) % 10 == 0 for r in runs]
    val_names = [r["name"] for r, v in zip(runs, is_val) if v]
    train = tables([r for r, v in zip(runs, is_val) if not v])
    val = tables([r for r, v in zip(runs, is_val) if v])
    n_train, n_val = len(runs) - len(val_names), len(val_names)
    del runs
    torque_all = train["torque"][1]
    wheels_all = train["base"][1]
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    settings.update({"train_runs": n_train, "val_runs": n_val, "torch": torch.__version__, "numpy": np.__version__,
                     "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                              text=True).stdout.strip()})
    saved = {"settings": settings,
             "wheel_limits": (wheels_all.min(axis=0), wheels_all.max(axis=0)),
             "torque_limits": (torque_all.min(axis=0), torque_all.max(axis=0)), "vel_scaled": True}
    donor = torch.load(args.from_ckpt, map_location="cpu", weights_only=False) if args.from_ckpt else None
    for name in NETS:
        if donor is not None and name not in args.only:
            saved[name] = donor[name]           # reused unchanged from --from
            train.pop(name), val.pop(name)
            continue
        saved[name] = train_net(name, train.pop(name), val.pop(name), args)   # free each table after use
    tmp = out / "policy.tmp.pt"
    torch.save(saved, tmp)
    tmp.replace(out / "policy.pt")
    summary = {name: {"best_val": saved[name]["best_val"], "rows": saved[name]["rows"],
                      "parameters": saved[name]["parameters"]} for name in NETS}
    (out / "summary.json").write_text(json.dumps({"settings": settings, "nets": summary,
                                                  "val_run_names": val_names}, indent=2), encoding="utf-8")
    print("TRAINED " + json.dumps(summary), flush=True)
    from chester_tracking import log_training
    log_training(out / "policy.pt", settings, summary, files=[out / "summary.json"])


if __name__ == "__main__":
    main()
