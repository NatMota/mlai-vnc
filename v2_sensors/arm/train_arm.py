"""Train an arm network that outputs the 7 arm torques directly, at 30 Hz or 1 kHz, from PARK to the expert's stop.

Data: the expert's full runs (data/expert_full, recorded every 1 ms by the fast simulator's expert collector).
--hz 30:   rows at the 30 Hz decision steps round(k * 1000 / 30), the same steps the stepping server uses.
--hz 1000: a row for every 1 ms step, the rate at which the expert itself outputs its torques.

Inputs (19):
  - arm encoder angles (7) and speeds (7): sensors.ArmEncoders read at every decision from the start of the run
    (speed over 1/30 s at 30 Hz, over 1 ms at 1 kHz)
  - dot position in the robot frame x, y, z (3): from the simulator for now, not the camera, so the arm is tested alone
  - time since PARK and time since the start, in seconds (2): the expert stops only after 8 s from the start
Outputs:
  - 7 arm torques (N m), standardised for the loss and unstandardised in the saved weights
  - time to stop: seconds to the expert's stop, clipped at 3 s and scaled to 0..1; stop when it is 0.25 s or less

--target sample: the expert's torque at the decision step.
--target mean:   the expert's torque averaged over the 33 ms until the next decision (the same as sample at 1 kHz).

--val-fold F: hold out the training bearings of fold F (split_plan.json) as validation; their losses are saved per epoch.
--epochs N: number of full-batch epochs (default 300).
--stop time:     the stop head learns the time to the expert's stop (s, clipped at 3 s); stop when it is 0.25 s or less.
--stop distance: the stop head learns the distance from the expert's end effector to the dot centre (m, clipped at
                 0.25 m); stop when it is 1 cm or less. The saved ttp_clip and ttp_thresh hold these values.

usage: python train_arm.py NAME [--hz 30|1000] [--target sample|mean] [--seed N] [--bearings B ...] [--epochs N] [--val-fold F]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import sensors

DATA = HERE / 'data/expert_full'
PLAN = json.loads((HERE.parent / 'driving_30/data/expert_30hz/split_plan.json').read_text())
SPLIT = {q['bearing_deg']: q['split'] for q in PLAN}
FOLD = {q['bearing_deg']: q['fold'] for q in PLAN}
EPOCHS, LR, CLIP, THRESH, HIDDEN = 300, 0.01, 3.0, 0.25, 64
DIST_CLIP, DIST_THRESH = 0.25, 0.01     # --stop distance: target clipped at 0.25 m; stop when the prediction is 1 cm or less


def table(run, name):
    cols = json.loads((run / f'{name}.schema.json').read_text())['columns']
    return dict(zip(cols, np.load(run / f'{name}.npy').T))


def yaw_of(qx, qy, qz, qw):
    return np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def dot_in_robot(target, base_xyz, yaw):
    """Dot relative to the robot base: x forward, y left, z up."""
    dx, dy, dz = target[0] - base_xyz[..., 0], target[1] - base_xyz[..., 1], target[2] - base_xyz[..., 2]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * dx + s * dy, -s * dx + c * dy, dz], -1)


def load_run(bearing, target_mode='sample', hz=30, stop_target='time'):
    run = DATA / f'run_{bearing // 12 - 1}'
    m = json.loads((run / 'manifest.json').read_text())
    arm, base = table(run, 'arm_controller'), table(run, 'base_state')
    q = np.c_[[arm[f'q_input_rad_arm{j}'] for j in range(1, 8)]].T
    tau = np.c_[[arm[f'tau_command_Nm_arm{j}'] for j in range(1, 8)]].T
    pos = np.c_[base['position_m_x'], base['position_m_y'], base['position_m_z']]
    yaw = yaw_of(base['quaternion_x'], base['quaternion_y'], base['quaternion_z'], base['quaternion_w'])
    park, stop = m['parking_step'], m['steps'] - 1
    steps = list(range(m['steps'])) if hz == 1000 else [round(k * 1000 / 30) for k in range(10 ** 5) if round(k * 1000 / 30) < m['steps']]
    angle, speed = sensors.ArmEncoders.readings_for_run(q[steps], 1 / hz)
    rows = [i for i, d in enumerate(steps) if park <= d <= stop]
    d = np.array(steps)[rows]
    ends = [steps[i + 1] if i + 1 < len(steps) else m['steps'] for i in rows]
    y = tau[d] if target_mode == 'sample' else np.array([tau[a:b].mean(0) for a, b in zip(d, ends)])
    t = d / 1000
    X = np.c_[angle[rows], speed[rows], dot_in_robot(m['target_position_world_m'], pos[d], yaw[d]), t - park / 1000, t]
    if stop_target == 'distance':
        dist = table(run, 'kinematics')['ee_distance_error_m'][d]
        ttp = np.clip(dist, 0, DIST_CLIP) / DIST_CLIP
    else:
        ttp = np.clip((stop - d) / 1000, 0, CLIP) / CLIP
    return dict(bearing=bearing, split=SPLIT[bearing], X=X.astype(np.float32), tau=y.astype(np.float32), ttp=ttp.astype(np.float32))


class ArmPolicy(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.arm = nn.Linear(HIDDEN, 7); self.stop = nn.Linear(HIDDEN, 1)
    def forward(self, x):
        h = self.shared(x); return self.arm(h), self.stop(h).squeeze(1)


def policy(w, x):
    """NumPy forward pass of saved weights: raw inputs -> (torques in N m, time to stop in s)."""
    for layer in ('shared.0', 'shared.2'):
        x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    return x @ w['arm.weight'].T + w['arm.bias'], float((x @ w['stop.weight'].T + w['stop.bias'])[0]) * float(w['ttp_clip'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--target', choices=['sample', 'mean'], default='sample')
    ap.add_argument('--seed', type=int, default=7); ap.add_argument('--bearings', type=int, nargs='*')
    ap.add_argument('--hz', type=int, choices=[30, 1000], default=30)
    ap.add_argument('--epochs', type=int, default=EPOCHS); ap.add_argument('--val-fold', type=int)
    ap.add_argument('--stop', choices=['time', 'distance'], default='time')
    args = ap.parse_args()
    SN = 'time_to_stop' if args.stop == 'time' else 'distance'
    runs = [load_run(b, args.target, args.hz, args.stop) for b in sorted(SPLIT) if SPLIT[b] != 'holdout']
    val = [r for r in runs if r['split'] == 'train' and args.val_fold is not None and FOLD[r['bearing']] == args.val_fold]
    train = [r for r in runs if r['split'] == 'train' and (not args.bearings or r['bearing'] in args.bearings) and r not in val]
    test = [r for r in runs if r['split'] == 'test']
    Xall, Yall = np.concatenate([r['X'] for r in train]), np.concatenate([r['tau'] for r in train])
    xm, xs = Xall.mean(0), Xall.std(0) + 1e-6
    ym, ys = Yall.mean(0), Yall.std(0) + 1e-6

    def stack(rs):
        X = torch.from_numpy((np.concatenate([r['X'] for r in rs]) - xm) / xs)
        Y = torch.from_numpy((np.concatenate([r['tau'] for r in rs]) - ym) / ys)
        P = torch.from_numpy(np.concatenate([r['ttp'] for r in rs]))
        w = torch.from_numpy(np.concatenate([np.full(len(r['X']), 1 / (len(r['X']) * len(rs))) for r in rs]).astype(np.float32))
        return X, Y, P, w

    def losses(m, X, Y, P, w):
        py, pt = m(X)
        return (w * ((py - Y) ** 2).mean(1)).sum(), (w * (pt - P) ** 2).sum()

    torch.manual_seed(args.seed)
    tr, te = stack(train), stack(test)
    va = stack(val) if val else None
    model = ArmPolicy(tr[0].shape[1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
    history = []
    for epoch in range(args.epochs + 1):
        am, pm = losses(model, *tr)
        with torch.no_grad():
            tam, tpm = losses(model, *te)
        history.append(dict(epoch=epoch, train_torque_mse=am.item(), test_torque_mse=tam.item(),
                            **{f'train_{SN}_mse': pm.item(), f'test_{SN}_mse': tpm.item()}))
        if va is not None:
            with torch.no_grad():
                vam, vpm = losses(model, *va)
            history[-1].update(val_torque_mse=vam.item(), **{f'val_{SN}_mse': vpm.item()})
        if epoch < args.epochs:
            opt.zero_grad(); (am + pm).backward(); opt.step()

    sd = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    # Fold the scaling into the weights: inputs into the first layer, torques into the torque head.
    W0 = sd['shared.0.weight']
    sd['shared.0.bias'] = sd['shared.0.bias'] - W0 @ (xm / xs); sd['shared.0.weight'] = W0 / xs
    sd['arm.weight'] = sd['arm.weight'] * ys[:, None]; sd['arm.bias'] = sd['arm.bias'] * ys + ym
    sd = {k: v.astype(np.float32) for k, v in sd.items()}
    with torch.no_grad():
        py, pt = model(tr[0][:50])
    x_raw = tr[0][:50].numpy() * xs + xm
    sd.update(ttp_clip=np.float32(CLIP if args.stop == 'time' else DIST_CLIP), ttp_thresh=np.float32(THRESH if args.stop == 'time' else DIST_THRESH))
    diff = max(float(np.abs(policy(sd, x_raw[i])[0] - (py[i].numpy() * ys + ym)).max()) for i in range(50))
    sd.update(info_stop_target=args.stop, info_epochs=np.int64(args.epochs), info_val_fold=np.int64(-1 if args.val_fold is None else args.val_fold), info_hz=np.int64(args.hz), info_seed=np.int64(args.seed), info_target=args.target, info_train_bearings=np.array([r['bearing'] for r in train]))
    out = HERE / 'models'; out.mkdir(exist_ok=True)
    np.savez(out / f'{args.name}.npz', **sd)
    (out / f'{args.name}_history.json').write_text(json.dumps(history))
    h = history[-1]
    print(f"{args.name}: {len(train)} train runs, {len(tr[0])} rows | train torque {h['train_torque_mse']:.4f} test {h['test_torque_mse']:.4f}"
          f" | stop ({SN}) train {h[f'train_{SN}_mse']:.4f} test {h[f'test_{SN}_mse']:.4f} | fold check {diff:.1e} N m")
