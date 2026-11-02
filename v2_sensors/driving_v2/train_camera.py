"""Rung 2: same network and loss as rung1 wheels+live_dot; only the dot inputs change (oracle -> camera + tracker + visible flag)."""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'driving_30/data/expert_30hz'
FEATS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'camera_features'
SEED, EPOCHS, LR, PARK_WEIGHT = 7, 300, 0.01, 0.1
torch.set_num_threads(1)

records = [json.loads(l) for l in open(DATA / 'episodes.jsonl')]
split = {q['bearing_deg']: q['split'] for q in json.loads((DATA / 'split_plan.json').read_text())}
unique = {}
for r in records:
    unique.setdefault(r['trajectory_hash'], r)
rows = []
for r in sorted(unique.values(), key=lambda r: r['bearing_deg']):
    if split[r['bearing_deg']] == 'holdout':
        continue
    with np.load(DATA / r['archive']) as a:
        cam = np.load(FEATS / f"camera_{r['bearing_deg']}.npy")
        assert len(cam) == len(a['x'])
        rows.append(dict(bearing=r['bearing_deg'], split=split[r['bearing_deg']], X=np.c_[a['x'][:, :12], cam],
                         wheels=a['wheels'], park=a['park'].ravel()))

class Policy(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.wheels = nn.Linear(16, 4); self.park = nn.Linear(16, 1)
    def forward(self, x):
        h = self.shared(x); return self.wheels(h), self.park(h).squeeze(1)

def stack(eps):
    X = torch.from_numpy(np.concatenate([e['X'] for e in eps]).astype(np.float32))
    W = torch.from_numpy(np.concatenate([e['wheels'] for e in eps]).astype(np.float32))
    P = torch.from_numpy(np.concatenate([e['park'] for e in eps]).astype(np.float32))
    w = torch.from_numpy(np.concatenate([np.full(len(e['X']), 1 / (len(e['X']) * len(eps))) for e in eps]).astype(np.float32))
    return X, W, P, w

def loss_fn(m, X, W, P, w, pw_):
    pw, pl = m(X)
    wm = (w * ((pw - W) ** 2).mean(1)).sum(); pb = F.binary_cross_entropy_with_logits(pl, P, pos_weight=pw_)
    return wm + PARK_WEIGHT * pb, wm, pb

train = [e for e in rows if e['split'] == 'train']; test = [e for e in rows if e['split'] == 'test']
torch.manual_seed(SEED)
Xtr, Wtr, Ptr, wtr = stack(train); Xte, Wte, Pte, wte = stack(test)
pos_weight = (len(Ptr) - Ptr.sum()) / Ptr.sum()
model = Policy(Xtr.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
for epoch in range(EPOCHS + 1):
    total, wm, pb = loss_fn(model, Xtr, Wtr, Ptr, wtr, pos_weight)
    with torch.no_grad():
        _, te, _ = loss_fn(model, Xte, Wte, Pte, wte, pos_weight)
    if epoch < EPOCHS:
        opt.zero_grad(); total.backward(); opt.step()
print(f'inputs={Xtr.shape[1]} train rows={len(Xtr)} train wheel MSE={wm.item():.4f} test={te.item():.4f} park BCE={pb.item():.3f}')
print('visible fraction train %.2f test %.2f' % (Xtr[:, 15].mean(), Xte[:, 15].mean()))
torch.save(model.state_dict(), HERE / 'models/camera_flag.pt')
np.savez(HERE / 'models/camera_flag.npz', **{k: v.numpy() for k, v in model.state_dict().items()})
