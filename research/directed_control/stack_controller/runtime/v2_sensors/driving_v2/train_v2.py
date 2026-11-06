"""Train one v2 iteration. Builds on camera+flag+time_to_park (16 inputs, 4 wheel speeds + time to park).

    python train_v2.py NAME [--scale-angles] [--seed N] [--park-tail N] [--encoders]

--scale-angles  standardise the six angle inputs (4 wheel angles, mean angle, right minus left angle) with the
                training-set mean and standard deviation. The scaling is folded into the first layer when saving,
                so the saved network takes raw inputs like every other model.
--seed N        random seed for the initial weights (default 7).
--park-tail N   add N rows per run after the expert parks (from datasets made by expert_tail.py).
--encoders      wheel inputs from the simulated wheel encoders (sensors.py) instead of the simulator's joint values.

Writes models/NAME.npz and models/NAME_history.json (loss per epoch).
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch import nn

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'driving_30/data/expert_30hz'
EPOCHS, LR, CLIP, THRESH = 300, 0.01, 3.0, 0.25
ANGLES = [0, 1, 2, 3, 8, 9]          # wheel angles FR, FL, BL, BR, mean angle, right minus left angle
torch.set_num_threads(1)

ap = argparse.ArgumentParser()
ap.add_argument('name'); ap.add_argument('--scale-angles', action='store_true'); ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--park-tail', type=int, default=0); ap.add_argument('--encoders', action='store_true')
args = ap.parse_args()


def retrack(cam, speeds):
    """Re-run the tracker with different wheel speeds. When the dot is visible the tracker takes the camera
    reading (saved in cam), so only the hidden stretches change and nothing needs rendering again."""
    from camera_loop import Tracker
    out, tr = cam.copy(), Tracker()
    for i, (d, s_, c, v) in enumerate(cam):
        seen = np.array([d * c, d * s_]) if v > 0 else None
        dot = tr.update(seen, speeds[i])
        if dot is None:
            out[i, :3] = 0
        else:
            a = np.arctan2(dot[1], dot[0]); out[i, :3] = [np.hypot(*dot), np.sin(a), np.cos(a)]
    return out


def load_runs():
    records = [json.loads(l) for l in open(DATA / 'episodes.jsonl')]
    split = {q['bearing_deg']: q['split'] for q in json.loads((DATA / 'split_plan.json').read_text())}
    unique = {}
    for r in records:
        unique.setdefault(r['trajectory_hash'], r)
    runs = []
    for r in sorted(unique.values(), key=lambda r: r['bearing_deg']):
        b = r['bearing_deg']
        if split[b] == 'holdout':
            continue
        with np.load(DATA / r['archive']) as a:
            wheel_x, wheels, t, q = a['x'][:, :12], a['wheels'], a['time_s'], a['joint_positions'][:, :4]
        cam = np.load(HERE / 'camera_features' / f'camera_{b}.npy')
        ttp = np.clip(t[-1] - t, 0, CLIP) / CLIP
        if args.park_tail:
            tail = np.load(HERE / 'park_tail' / f'tail_{b}.npz')
            n = args.park_tail
            wheel_x = np.r_[wheel_x, tail['x'][:n]]; cam = np.r_[cam, tail['camera'][:n]]; q = np.r_[q, tail['wheel_q'][:n]]
            wheels = np.r_[wheels, np.zeros((n, 4), np.float32)]; ttp = np.r_[ttp, np.zeros(n)]     # parked: wheels 0, no time left
        if args.encoders:
            import sensors
            wheel_x = sensors.WheelEncoders.features_for_run(q)
            cam = retrack(cam, wheel_x[:, 4:8])
        runs.append(dict(bearing=b, split=split[b], X=np.c_[wheel_x, cam].astype(np.float32), wheels=wheels.astype(np.float32),
                         ttp=ttp.astype(np.float32)))
    return runs


class Policy(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.wheels = nn.Linear(16, 4); self.park = nn.Linear(16, 1)
    def forward(self, x):
        h = self.shared(x); return self.wheels(h), self.park(h).squeeze(1)


runs = load_runs()
train = [e for e in runs if e['split'] == 'train']; test = [e for e in runs if e['split'] == 'test']
mean = np.zeros(16, np.float32); std = np.ones(16, np.float32)
if args.scale_angles:
    Xall = np.concatenate([e['X'] for e in train])
    mean[ANGLES] = Xall[:, ANGLES].mean(0); std[ANGLES] = Xall[:, ANGLES].std(0)


def stack(eps):
    X = torch.from_numpy((np.concatenate([e['X'] for e in eps]) - mean) / std)
    W = torch.from_numpy(np.concatenate([e['wheels'] for e in eps]))
    P = torch.from_numpy(np.concatenate([e['ttp'] for e in eps]))
    w = torch.from_numpy(np.concatenate([np.full(len(e['X']), 1 / (len(e['X']) * len(eps))) for e in eps]).astype(np.float32))
    return X, W, P, w


def losses(m, X, W, P, w):
    pw, pt = m(X)
    return (w * ((pw - W) ** 2).mean(1)).sum(), (w * (pt - P) ** 2).sum()


torch.manual_seed(args.seed)
tr, te = stack(train), stack(test)
model = Policy(tr[0].shape[1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
history = []
for epoch in range(EPOCHS + 1):
    wm, pm = losses(model, *tr)
    with torch.no_grad():
        twm, tpm = losses(model, *te)
    history.append(dict(epoch=epoch, train_wheel_mse=wm.item(), test_wheel_mse=twm.item(),
                        train_time_to_park_mse=pm.item(), test_time_to_park_mse=tpm.item()))
    if epoch < EPOCHS:
        opt.zero_grad(); (wm + pm).backward(); opt.step()

sd = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
# Fold the input scaling into the first layer: W x' + b with x' = (x - mean) / std  ==  (W / std) x + (b - W mean / std).
W0 = sd['shared.0.weight']
sd['shared.0.bias'] = (sd['shared.0.bias'] - W0 @ (mean / std)).astype(np.float32)
sd['shared.0.weight'] = (W0 / std).astype(np.float32)
with torch.no_grad():
    x = tr[0][:200]; raw = x * torch.from_numpy(std) + torch.from_numpy(mean)
    folded = Policy(16); folded.load_state_dict({k: torch.from_numpy(v) for k, v in sd.items()})
    diff = max(float((a - b).abs().max()) for a, b in zip(model(x), folded(raw)))
sd.update(ttp_clip=np.float32(CLIP), ttp_thresh=np.float32(THRESH), info_input_mean=mean, info_input_std=std,
          info_seed=np.int64(args.seed), info_park_tail=np.int64(args.park_tail), info_encoders=np.int64(args.encoders))
np.savez(HERE / f'models/{args.name}.npz', **sd)
(HERE / f'models/{args.name}_history.json').write_text(json.dumps(history))
h = history[-1]
print(f"{args.name}: train rows {len(tr[0])}, wheel MSE train {h['train_wheel_mse']:.4f} test {h['test_wheel_mse']:.4f}, "
      f"time-to-park MSE train {h['train_time_to_park_mse']:.4f} test {h['test_time_to_park_mse']:.4f}, folding check {diff:.1e}")
