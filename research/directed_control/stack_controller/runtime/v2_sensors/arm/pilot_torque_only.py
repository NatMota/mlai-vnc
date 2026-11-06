"""Validation-only torque ablation, using the frozen golden loader and optimizer loop."""
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from torch import nn
import train_golden as tg
import train_full_episode as tf

BASE = Path('/media/nm/Fast Drive/comp0245-golden11109/overnight_20261105/dagger_round3/models')

class TorqueOnly(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.arm = nn.Linear(hidden, 7)
    def forward(self, x):
        torque = self.arm(self.shared(x))
        return torque, torch.zeros(len(x), device=x.device)

def objective(model, data, kind, chunk, backward=False, aux_weight=1.):
    assert kind == 'torque'
    total = torch.zeros(2, device=data[0].device)
    for begin in range(0, len(data[0]), chunk):
        x,y,_,w = [a[begin:begin+chunk] for a in data[:4]]
        loss = (w * (model(x)[0] - y).square().mean(1)).sum()
        if backward: loss.backward()
        total[0] += loss.detach()
    return total.cpu().numpy()

def infer_export(w, x, kind='torque'):
    assert kind == 'torque'
    x = np.asarray(x, np.float32)
    x = (x-w['export_input_mean'])/w['export_input_scale']
    for layer in ('shared.0','shared.2'):
        x = np.maximum(x @ w[layer+'.weight'].T+w[layer+'.bias'], 0)
    command = (x @ w['arm.weight'].T+w['arm.bias'])*w['export_output_scale']+w['export_output_mean']
    return command, np.zeros(len(x), np.float32)

from pilot_torque_runtime import policy

original_train = tg.train_one

def train_one(records, kind, seed, args):
    if kind != 'torque':
        for suffix in ('.npz','_summary.json','_history.json'):
            shutil.copy2(BASE/f'{kind}_seed7{suffix}', args.out/f'{kind}_seed7{suffix}')
        return dict(np.load(BASE/f'{kind}_seed7.npz')), json.loads((BASE/f'{kind}_seed7_summary.json').read_text())
    baseline = dict(np.load(BASE/'reaching_seed7.npz'))
    stripped = {k:v for k,v in baseline.items() if not k.startswith('stop.')}
    raw,_,_,_ = tg.arrays(records['train'], 'torque')
    samples = raw[np.linspace(0,len(raw)-1,512,dtype=int)]
    from train_arm import policy as old_policy
    for row in samples:
        old,new = old_policy(baseline,row),policy(stripped,row)
        np.testing.assert_array_equal(old[0],new[0]); assert old[1] == new[1]
    tg.write_json(args.out/'deletion_parity.json',dict(samples=512,torque_bit_equal=True,stop_bit_equal=True,baseline_sha256=tg.sha(BASE/'reaching_seed7.npz')))
    np.savez(args.out/'deletion_only_reaching_seed7.npz', **stripped)
    tg.ArmPolicy, tg.objective, tg.infer_export = TorqueOnly, objective, infer_export
    weights, meta = original_train(records,kind,seed,args)
    assert not any(k.startswith('stop.') for k in weights)
    meta.update(loss_terms=['standardized_torque_mse'],auxiliary_loss_weight=0.,architecture='18->128->128->7',reference_model_sha256=tg.sha(BASE/'reaching_seed7.npz'))
    weights['info_auxiliary_loss_weight']=np.float32(0)
    tg.write_json(args.out/'torque_seed7_summary.json',meta)
    np.savez(args.out/'torque_seed7.npz', **weights)
    return weights,meta

if __name__ == '__main__':
    tf.train_one = train_one
    tf.main()
