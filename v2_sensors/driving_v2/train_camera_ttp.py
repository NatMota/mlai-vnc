"""Rung 3: rung 2 with one change - the PARK head predicts time-to-park instead of a one-row PARK class.

Target: seconds until the expert parks, clipped at CLIP and divided by CLIP (so 0..1). Loss: equal-trajectory MSE,
same weighting as the wheels. Command: PARK when predicted time-to-park <= THRESH seconds (fixed before testing)."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'driving_30/data/expert_30hz'
FEATS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'camera_features'
SEED, EPOCHS, LR, TTP_WEIGHT = 7, 300, 0.01, 1.0
FLAG = os.environ.get('FLAG', '1') == '1'     # FLAG=0: drop the visible flag (15 inputs, same as the oracle model)
CLIP, THRESH = 3.0, 0.25   # PARK when predicted time-to-park <= 0.25 s (fixed)
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
        rows.append(dict(bearing=r['bearing_deg'], split=split[r['bearing_deg']], X=np.c_[a['x'][:, :12], cam if FLAG else cam[:, :3]],
                         wheels=a['wheels'], park=a['park'].ravel(),
                         ttp=np.minimum(a['time_s'][-1] - a['time_s'], CLIP) / CLIP))
        assert a['park'].ravel()[-1] == 1 and a['park'].sum() == 1

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
    P = torch.from_numpy(np.concatenate([e['ttp'] for e in eps]).astype(np.float32))
    w = torch.from_numpy(np.concatenate([np.full(len(e['X']), 1 / (len(e['X']) * len(eps))) for e in eps]).astype(np.float32))
    return X, W, P, w

def loss_fn(m, X, W, P, w, _=None):
    pw, pt = m(X)
    wm = (w * ((pw - W) ** 2).mean(1)).sum(); pb = (w * (pt - P) ** 2).sum()
    return wm + TTP_WEIGHT * pb, wm, pb

train = [e for e in rows if e['split'] == 'train']; test = [e for e in rows if e['split'] == 'test']
torch.manual_seed(SEED)
Xtr, Wtr, Ptr, wtr = stack(train); Xte, Wte, Pte, wte = stack(test)
pos_weight = None
model = Policy(Xtr.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
for epoch in range(EPOCHS + 1):
    total, wm, pb = loss_fn(model, Xtr, Wtr, Ptr, wtr, pos_weight)
    with torch.no_grad():
        _, te, _ = loss_fn(model, Xte, Wte, Pte, wte, pos_weight)
    if epoch < EPOCHS:
        opt.zero_grad(); total.backward(); opt.step()
print(f'inputs={Xtr.shape[1]} train rows={len(Xtr)} train wheel MSE={wm.item():.4f} test={te.item():.4f} time-to-park MSE (scaled)={pb.item():.4f}')
# Offline check of the fixed rule on each split: rows that would PARK early, and whether the final row parks.
with torch.no_grad():
    for name, eps in (('train', train), ('test', test)):
        early = hit = 0
        for e in eps:
            t = model(torch.from_numpy(e['X'].astype(np.float32)))[1].numpy() * CLIP
            early += int((t[:-1] <= THRESH).sum()); hit += int(t[-1] <= THRESH)
        print(f'{name}: final row parks {hit}/{len(eps)}, early PARK rows {early}')
sd = {k: v.numpy() for k, v in model.state_dict().items()}
sd['ttp_clip'] = np.float32(CLIP); sd['ttp_thresh'] = np.float32(THRESH)
torch.save(model.state_dict(), HERE / ('models/camera_flag_ttp.pt' if FLAG else 'models/camera_ttp_noflag.pt'))
np.savez(HERE / ('models/camera_flag_ttp.npz' if FLAG else 'models/camera_ttp_noflag.npz'), **sd)
