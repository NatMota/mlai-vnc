"""Train a separate stop network for an existing arm network, and save the two together as one model.

The torque network (TORQUE_MODEL, from train_arm.py) is kept exactly as it is. The stop network is
INPUTS -> 32 ReLU -> 16 ReLU -> 1 on the same raw inputs, with the same input scaling (physical sizes) and the same
target as train_arm.py --stop distance: the expert's gripper-to-dot distance, clipped at 0.25 m and divided by 0.25 m.
No torque loss reaches it.

Its number of epochs is chosen by 5-fold validation on the training bearings (split_plan.json folds): for each fold,
train on the other 4 for up to MAX epochs and record the held-out stop loss every epoch; the chosen number is the
epoch with the lowest held-out loss averaged over the 5 folds. The test bearings are not used for the choice.

Writes models/NAME.npz (the torque network's weights plus stopnet.*), models/NAME_history.json (torque losses copied
from the torque network, stop losses from the stop network) and results/NAME_epoch_choice.json.

usage: python train_stop.py NAME TORQUE_MODEL [--max-epochs 600] [--seed 7]
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch import nn
from train_arm import load_run, select_inputs, physical_scale, SPLIT, FOLD, LR, DIST_CLIP, policy

HERE = Path(__file__).resolve().parent
torch.set_num_threads(4)


def stop_net(n_in):
    return nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))


def stack(runs, xs):
    X = torch.from_numpy((np.concatenate([r['X'] for r in runs]) / xs).astype(np.float32))
    P = torch.from_numpy(np.concatenate([r['ttp'] for r in runs]))
    w = torch.from_numpy(np.concatenate([np.full(len(r['X']), 1 / (len(r['X']) * len(runs))) for r in runs]).astype(np.float32))
    return X, P, w


def loss(m, X, P, w):
    return (w * (m(X).squeeze(1) - P) ** 2).sum()


def train(train_runs, eval_sets, epochs, xs, seed):
    """Full-batch Adam; returns the network and the loss on each evaluation set at every epoch (epoch 0 = before training)."""
    torch.manual_seed(seed)
    tr = stack(train_runs, xs); ev = {k: stack(v, xs) for k, v in eval_sets.items()}
    m = stop_net(tr[0].shape[1]); opt = torch.optim.Adam(m.parameters(), lr=LR); hist = []
    for epoch in range(epochs + 1):
        l = loss(m, *tr)
        with torch.no_grad():
            hist.append(dict(epoch=epoch, train=l.item(), **{k: loss(m, *v).item() for k, v in ev.items()}))
        if epoch < epochs:
            opt.zero_grad(); l.backward(); opt.step()
    return m, hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('torque_model')
    ap.add_argument('--max-epochs', type=int, default=600); ap.add_argument('--seed', type=int, default=7)
    a = ap.parse_args()
    w = dict(np.load(HERE / f'models/{a.torque_model}.npz'))
    assert str(w['info_scaling']) == 'physical_inputs' and str(w['info_stop_target']) == 'distance'
    flags = dict(imu=bool(w.get('info_imu', 0)), encoders=bool(w.get('info_encoders', 1)), time=bool(w.get('info_time', w['shared.0.weight'].shape[1] == 19)))
    runs = [load_run(b, str(w['info_target']), int(w['info_hz']), 'distance', imu=flags['imu']) for b in sorted(SPLIT) if SPLIT[b] != 'holdout']
    for r in runs:
        r['X'] = select_inputs(r['X'], flags['imu'], flags['encoders'], flags['time'])
    xs = physical_scale(flags['imu'], flags['encoders'], flags['time'])
    train_runs = [r for r in runs if r['split'] == 'train']; test_runs = [r for r in runs if r['split'] == 'test']

    folds = []
    for f in range(5):
        _, h = train([r for r in train_runs if FOLD[r['bearing']] != f], {'val': [r for r in train_runs if FOLD[r['bearing']] == f]}, a.max_epochs, xs, a.seed)
        folds.append([e['val'] for e in h]); print(f'fold {f}: best epoch {int(np.argmin(folds[-1]))}')
    mean = np.mean(folds, 0); chosen = int(np.argmin(mean))
    print(f'chosen {chosen} epochs (mean held-out stop loss {mean[chosen]:.5f}; at {a.max_epochs}: {mean[-1]:.5f})')

    net, h = train(train_runs, {'test': test_runs}, chosen, xs, a.seed)
    sd = {k: v.detach().numpy().astype(np.float64) for k, v in net.state_dict().items()}
    sd['0.weight'] = sd['0.weight'] / xs                       # fold the input scaling into the first layer
    out = dict(w); out.update({f'stopnet.{k}': v.astype(np.float32) for k, v in sd.items()})
    out.update(ttp_clip=np.float32(DIST_CLIP), info_stopnet='17->32->16->1 relu', info_stopnet_epochs=np.int64(chosen),
               info_stopnet_seed=np.int64(a.seed), info_torque_model=a.torque_model)
    with torch.no_grad():
        X = test_runs[0]['X'][:50]; pt = net(torch.from_numpy((X / xs).astype(np.float32))).squeeze(1).numpy()
    diff = max(abs(policy(out, X[i])[1] / DIST_CLIP - pt[i]) for i in range(50))
    np.savez(HERE / f'models/{a.name}.npz', **out)

    th = json.loads((HERE / f'models/{a.torque_model}_history.json').read_text())
    n = max(len(th), len(h)); hist = []
    for e in range(n):
        row = dict(epoch=e)
        if e < len(th):
            row.update(train_torque_mse=th[e]['train_torque_mse'], test_torque_mse=th[e]['test_torque_mse'])
        if e < len(h):
            row.update(train_distance_mse=h[e]['train'], test_distance_mse=h[e]['test'])
        hist.append(row)
    (HERE / f'models/{a.name}_history.json').write_text(json.dumps(hist))
    (HERE / f'results/{a.name}_epoch_choice.json').write_text(json.dumps(dict(
        max_epochs=a.max_epochs, chosen=chosen, mean_val=mean.tolist(), fold_val=[list(map(float, f)) for f in folds],
        rule='epoch with the lowest held-out stop loss averaged over the 5 folds of the training bearings')))
    print(f"{a.name}: stop train {h[-1]['train']:.5f} test {h[-1]['test']:.5f} | fold check {diff:.1e}")
