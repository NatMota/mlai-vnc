"""Train the fixed driving/reaching architectures on the 100/100 golden split.

Test trajectories are never opened. Each epoch is one equal-run, full-batch Adam
update; --chunk limits activation memory without adding optimizer steps.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch import nn

from render_golden import camera_inputs, frame_steps
from train_arm import ArmPolicy, physical_scale, policy, sensors
from train_stop import stop_net
from parking_target import parking_distance, DISTANCE_SCALE_M, PARK_THRESHOLD_M

HERE = Path(__file__).resolve().parent
ANGLE_COLS = [0, 1, 2, 3, 8, 9]
SEEDS = [7, 1, 2]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def arm_readings(q):
    """Same floor-count and backward difference as ArmEncoders at 1 kHz."""
    angles = np.floor(q.astype(np.float64) / sensors.ARM_TICK) * sensors.ARM_TICK
    speed = np.diff(angles, axis=0, prepend=angles[:1]) / .001
    return angles, speed


def load_data(root, driving_height=False, park_target='time', park_scale_m=DISTANCE_SCALE_M, full_arm=False, allowed_conditions=('baseline',)):
    if park_target not in ('time', 'distance'):
        raise ValueError('PARK target must be time or distance')
    splits = json.loads((root / 'split.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    files = {v['path']: v['sha256'] for v in manifest['files']}
    if set(splits) != {'train', 'val', 'test'} or any(len(set(v)) != 100 for v in splits.values()):
        raise ValueError('Expected the frozen 100/100/100 split')
    if len(set().union(*map(set, splits.values()))) != 300:
        raise ValueError('Split overlap')
    records, audit = {'train': [], 'val': []}, []
    for split in records:
        for dot in splits[split]:
            source = root / 'expert' / split / f'run_{dot}.npz'
            camera = root / 'camera' / split / f'camera_{dot}.npz'
            source_hash = sha(source)
            for p, h in ((source, source_hash), (camera, sha(camera))):
                if h != files[str(p.relative_to(root))]:
                    raise ValueError(f'Hash mismatch: {p}')
            with np.load(source, allow_pickle=False) as a, np.load(camera, allow_pickle=False) as c:
                m = json.loads(str(a['manifest']))
                n, park = int(m['steps']), int(m['parking_step'])
                prov = json.loads(str(c['provenance']))
                if prov['source_sha256'] != source_hash or int(c['dot']) != dot or int(c['n_steps']) != n:
                    raise ValueError(f'Camera pairing failed: {split}/{dot}')
                if int(c['parking_step']) != park or m['condition'] not in allowed_conditions or m['dt_s'] != .001:
                    raise ValueError(f'Phase/condition mismatch: {split}/{dot}')
                np.testing.assert_array_equal(c['step'], frame_steps(n))
                np.testing.assert_allclose(a['time_s'], np.arange(n) * .001, atol=1e-8, rtol=0)
                if a['q'].shape != (n, 17) or a['arm_torque_command'].shape != (n, 7):
                    raise ValueError(f'Wrong joint order/shape: {split}/{dot}')
                row = dict(dot=int(dot), split=split, steps=n, parking_step=park,
                           termination=m['termination'], source_sha256=source_hash,
                           camera_sha256=sha(camera), renderer=prov.get('renderer'))
                if not 0 <= park < n:
                    row.update(excluded='no valid PARK'); audit.append(row); continue
                # Do not relabel a timeout as a successful terminal demonstration.
                if 'timeout' in str(m['termination']).lower() or 'limit' in str(m['termination']).lower():
                    row.update(excluded='expert timeout'); audit.append(row); continue
                nav = np.asarray(a['navigation_done']).astype(bool)
                active = np.asarray(a['reach_active']).astype(bool)
                if not nav[park] or not active[park]:
                    raise ValueError(f'PARK flags disagree at {split}/{dot}:{park}')
                frames = c['step'].copy()
                before = frames[frames < park]
                after = frames[frames >= park][:100]
                drive_steps = np.r_[before, after]
                dcam = camera_inputs(split, dot, drive_steps, root=root / 'camera')
                xdrive = np.c_[dcam['wheel'], dcam['drive']].astype(np.float32)
                if driving_height:
                    xdrive = np.c_[xdrive, dcam['arm'][:, 2]].astype(np.float32)
                wheel = a['wheel_command'][drive_steps].copy()
                parked = drive_steps >= park
                # Keep iteration 13's parked labels, checking the recorded commands first.
                if np.max(np.abs(wheel[parked]), initial=0) > 1e-6:
                    raise ValueError(f'Nonzero recorded wheel command after PARK: {split}/{dot}')
                wheel[parked] = 0
                ttp = np.clip((park - drive_steps) / 1000, 0, 3) / 3
                if park_target == 'distance':
                    ttp = parking_distance(a['base_position'], park, drive_steps, park_scale_m)
                    row.update(park_distance_max_m=float(ttp.max() * park_scale_m),
                               parked_distance_max_m=float(ttp[parked].max(initial=0) * park_scale_m))
                astep = np.arange(0 if full_arm else park, n)
                angles, speed = arm_readings(a['q'][:, 4:11])
                acam = camera_inputs(split, dot, astep, root=root / 'camera')
                xarm = np.c_[angles[astep], speed[astep], acam['arm']].astype(np.float32)
                if full_arm:
                    xarm = np.c_[xarm, (astep >= park).astype(np.float32)]
                torque = a['arm_torque_command'][astep].copy()
                distance = np.clip(a['hand_distance_expert'][astep], 0, .25) / .25
                for x in (xdrive, wheel, ttp, xarm, torque, distance):
                    if not np.isfinite(x).all():
                        raise ValueError(f'Non-finite model data: {split}/{dot}')
                # Test vectorization against the runtime sensor, not an independent formula.
                check_a, check_v = sensors.ArmEncoders.readings_for_run(a['q'][:40, 4:11].astype(float), .001)
                np.testing.assert_array_equal(angles[:40], check_a)
                np.testing.assert_array_equal(speed[:40], check_v)
                row.update(drive_rows=len(drive_steps), parked_rows=len(after), arm_rows=len(astep), arm_prepark_rows=int(np.sum(astep < park)),
                           first_seen_step=int(c['step'][np.flatnonzero(c['seen'])[0]]) if c['seen'].any() else None,
                           final_expert_mm=float(1000 * a['hand_distance_expert'][-1]),
                           max_hand_distance_array_difference_m=float(np.max(np.abs(a['hand_distance'] - a['hand_distance_expert']))))
                audit.append(row)
                records[split].append(dict(dot=dot,drive_steps=drive_steps,arm_steps=astep,parking_step=park, drive=(xdrive, wheel, ttp.astype(np.float32)),
                                           arm=(xarm, torque, distance.astype(np.float32))))
        print(f'Loaded {split}: {len(records[split])} runs', flush=True)
    if any(not v for v in records.values()):
        raise ValueError('Empty training/validation set')
    return records, dict(split_sha256=sha(root / 'split.json'), manifest_sha256=sha(root / 'manifest.json'),
                         source_revision=manifest['code_revision'], rows=audit,
                         park_target=park_target, park_distance_scale_m=park_scale_m,
                         distance_label='hand_distance_expert: aligned pre-step kinematics',
                         wheel_speed_dt_s=sensors.DT,
                         wheel_timing_note='Saved sensor features use the runtime 1/30 s denominator at rounded frame steps; no rescaling.',
                         test_trajectories_opened=0)


class Driving(nn.Module):
    def __init__(self, n_in=16):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.wheels, self.park = nn.Linear(16, 4), nn.Linear(16, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.wheels(h), self.park(h).squeeze(1)


def arrays(runs, kind):
    data = [r['drive' if kind == 'driving' else 'arm'] for r in runs]
    x, y, p = (np.concatenate([r[i] for r in data]) for i in range(3))
    w = np.concatenate([np.full(len(r[0]), 1 / len(r[0]) / len(data), np.float32) for r in data])
    return x, y, p, w


def tensors(runs, kind, scale, device):
    x, y, p, w = arrays(runs, kind)
    xm, xs, ym, ys = scale
    return [torch.as_tensor(a, dtype=torch.float32, device=device) for a in ((x - xm) / xs, (y - ym) / ys, p, w)]


def objective(model, data, kind, chunk, backward=False, aux_weight=1.):
    totals = torch.zeros(2, device=data[0].device)
    for begin in range(0, len(data[0]), chunk):
        x, y, p, w = [a[begin:begin + chunk] for a in data]
        if kind == 'stop':
            first = (w * (model(x).squeeze(1) - p).square()).sum()
            second = first * 0
        else:
            commands, auxiliary = model(x)
            first = (w * (commands - y).square().mean(1)).sum()
            second = (w * (auxiliary - p).square()).sum()
        if backward:
            (first + aux_weight * second).backward()
        totals += torch.stack([first.detach(), second.detach()])
    return totals.cpu().numpy()


def infer_export(w, x, kind):
    x = np.asarray(x, np.float32)
    if kind == 'stop':
        for layer in ('0', '2'):
            x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
        return (x @ w['4.weight'].T + w['4.bias']).squeeze(1)
    for layer in ('shared.0', 'shared.2'):
        x = np.maximum(x @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    head = 'wheels' if kind == 'driving' else 'arm'
    aux = 'park' if kind == 'driving' else 'stop'
    return x @ w[head + '.weight'].T + w[head + '.bias'], (x @ w[aux + '.weight'].T + w[aux + '.bias']).squeeze(1)


def train_one(records, kind, seed, args):
    distance_park = kind == 'driving' and getattr(args, 'park_target', 'time') == 'distance'
    park_scale_m = float(getattr(args, 'park_scale_m', DISTANCE_SCALE_M))
    if not np.isfinite(park_scale_m) or park_scale_m <= 0:
        raise ValueError('Distance scale must be finite and positive')
    aux_weight = float(getattr(args, 'park_weight', 1.)) if kind == 'driving' else 1.
    if not np.isfinite(aux_weight) or aux_weight <= 0:
        raise ValueError('PARK loss weight must be finite and positive')
    torch.manual_seed(seed)
    x, y, _, _ = arrays(records['train'], kind)
    xm, xs = np.zeros(x.shape[1], np.float32), np.ones(x.shape[1], np.float32)
    ym, ys = np.zeros(y.shape[1], np.float32), np.ones(y.shape[1], np.float32)
    if kind == 'driving':
        angle_cols = list(range(6)) if x.shape[1] == 11 else ANGLE_COLS
        xm[angle_cols] = x[:, angle_cols].mean(0)
        xs[angle_cols] = np.maximum(x[:, angle_cols].std(0), 1e-6)
        model = Driving(x.shape[1])
    else:
        if x.shape[1] not in (17, 18, 51):
            raise ValueError('Arm inputs must be current observation or current/50ms/100ms history')
        xs = np.r_[physical_scale(False, True, False), 1.].astype(np.float32) if x.shape[1] == 18 else np.tile(physical_scale(False, True, False), x.shape[1] // 17).astype(np.float32)
        if getattr(args, 'arm_scaling', 'physical') == 'standard':
            xm, xs = x.mean(0), np.maximum(x.std(0), 1e-6)
            if x.shape[1] == 18: xm[-1], xs[-1] = 0., 1.
        if kind == 'torque':
            ym, ys = y.mean(0), y.std(0) + 1e-6
        model = ArmPolicy(x.shape[1], hidden=getattr(args, 'arm_hidden', 64)) if kind == 'torque' else stop_net(x.shape[1])
    scale = (xm, xs, ym, ys)
    model.to(args.device)
    train = tensors(records['train'], kind, scale, args.device)
    val = tensors(records['val'], kind, scale, args.device)
    opt = torch.optim.Adam(model.parameters(), lr=getattr(args, 'lr', .01))
    epochs = args.stop_epochs if kind == 'stop' else args.epochs
    history, best, started = [], None, time.monotonic()
    prefix = args.out / f'{kind}_seed{seed}'
    for epoch in range(epochs + 1):
        tick = time.monotonic()
        # Epoch zero is before any optimizer update. Each later record is after one update.
        if epoch:
            opt.zero_grad(set_to_none=True)
            objective(model, train, kind, args.chunk, backward=True, aux_weight=aux_weight)
            opt.step()
        with torch.no_grad():
            tr = objective(model, train, kind, args.chunk)
            va = objective(model, val, kind, args.chunk)
        row = dict(epoch=epoch, train_command=float(tr[0]), train_aux=float(tr[1]), train_total=float(tr[0]+aux_weight*tr[1]),
                   val_command=float(va[0]), val_aux=float(va[1]), val_total=float(va[0]+aux_weight*va[1]),
                   seconds=time.monotonic() - tick)
        history.append(row)
        if not np.isfinite(list(row.values())).all():
            raise ValueError(f'Non-finite loss: {kind}/{seed}/{epoch}')
        if best is None or row['val_total'] < best['loss']:
            best = dict(epoch=epoch, loss=row['val_total'], state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        optimizer=copy.deepcopy(opt.state_dict()))
        if epoch in getattr(args, 'save_epochs', ()):
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            optimizer=opt.state_dict(), epoch=epoch, input_mean=xm, input_scale=xs,
                            output_mean=ym, output_scale=ys),
                       prefix.with_name(prefix.name + f'_epoch{epoch}.pt'))
        if epoch < 3 or epoch % 25 == 0 or epoch == epochs:
            print(f'{kind} seed {seed} epoch {epoch}/{epochs}: train {row["train_total"]:.6g}, val {row["val_total"]:.6g}, {row["seconds"]:.3f}s', flush=True)
            write_json(prefix.with_name(prefix.name + '_history.json'), history)
    model.load_state_dict(best['state'])
    w = {k:v.numpy().astype(np.float64) for k,v in best['state'].items()}
    first = '0' if kind == 'stop' else 'shared.0'
    w[first + '.bias'] -= w[first + '.weight'] @ (xm / xs)
    w[first + '.weight'] /= xs
    if kind == 'torque':
        w['arm.weight'] *= ys[:, None]
        w['arm.bias'] = w['arm.bias'] * ys + ym
    w = {k:v.astype(np.float32) for k,v in w.items()}
    checks = []
    for split in ('train', 'val'):
        raw, _, _, _ = arrays(records[split], kind)
        raw = raw[np.linspace(0, len(raw)-1, min(256, len(raw)), dtype=int)]
        with torch.no_grad():
            pred = model(torch.as_tensor((raw-xm)/xs, device=args.device))
        saved = infer_export(w, raw, kind)
        if kind == 'stop':
            expected = pred.squeeze(1).cpu().numpy()
            np.testing.assert_allclose(saved, expected, atol=1e-4, rtol=1e-5)
            diff = float(np.max(np.abs(saved-expected)))
        else:
            expected = (pred[0].cpu().numpy() * ys + ym, pred[1].cpu().numpy())
            for got, want in zip(saved, expected):
                np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-5)
            diff = max(float(np.max(np.abs(a-b))) for a,b in zip(saved, expected))
        checks.append(dict(split=split, max_abs_difference=diff, atol=1e-4, rtol=1e-5))
    meta = dict(kind=kind, seed=seed, selected_epoch=best['epoch'], selected_val_loss=best['loss'],
                auxiliary_loss_weight=aux_weight,
                elapsed_s=time.monotonic()-started, export_checks=checks, train_rows=len(train[0]), val_rows=len(val[0]),
                loss_terms=['distance_mse'] if kind=='stop' else ['wheel_mse', 'park_distance_mse' if distance_park else 'time_to_park_mse'] if kind=='driving' else ['standardized_torque_mse','auxiliary_distance_mse'],
                run_weighting='equal mean per run', checkpoint_rule='minimum validation total, earliest tie',
                input_mean=xm.tolist(), input_scale=xs.tolist(), output_mean=ym.tolist(), output_scale=ys.tolist())
    w.update(info_arm_history_ms=np.array([0, 50, 100] if kind == 'torque' and len(xm) == 51 else [0]),
             info_seed=np.int64(seed), info_epochs=np.int64(best['epoch']), info_encoders=np.int64(1), info_auxiliary_loss_weight=np.float32(aux_weight),
             info_time=np.int64(0), info_imu=np.int64(0), info_dot_input='camera_tracker', info_full_episode=np.int64(len(xm)==18 and kind!='driving'),
             info_park_input=np.int64(len(xm)==18 and kind!='driving'),
             info_inputs=np.int64(len(xm)), info_camera_height=np.int64(kind == 'driving' and len(xm) in (11, 17)), info_wheel_velocity=np.int64(len(xm) != 11), info_target='sample', info_hz=np.int64(30 if kind=='driving' else 1000),
             info_input_mean=xm, info_input_std=xs,
             info_scaling='standardised_angles' if kind=='driving' else ('standardised_training_inputs' if getattr(args,'arm_scaling','physical')=='standard' else 'physical_inputs'),
             info_stop_target='time' if kind=='driving' else 'distance',
             ttp_clip=np.float32(3 if kind=='driving' else .25), ttp_thresh=np.float32(.25 if kind=='driving' else .01))
    if distance_park:
        w.pop('ttp_clip'); w.pop('ttp_thresh')
        w.update(info_stop_target='distance', park_distance_scale_m=np.float32(park_scale_m),
                 park_distance_threshold_m=np.float32(PARK_THRESHOLD_M))
        meta.update(park_target='distance', park_distance_scale_m=park_scale_m,
                    park_distance_threshold_m=PARK_THRESHOLD_M,
                    label='horizontal Euclidean distance to recorded expert PARK base position')
    np.savez(prefix.with_suffix('.npz'), **w)
    torch.save(dict(model=best['state'], optimizer=best['optimizer'], config=meta, epoch=best['epoch']), prefix.with_suffix('.pt'))
    write_json(prefix.with_name(prefix.name+'_summary.json'), meta)
    return w, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--stop-epochs', type=int, default=600)
    ap.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--chunk', type=int, default=32768)
    ap.add_argument('--code-revision', help='Source revision when running a staged copy without .git')
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Preserve previous results; choose a new --out: {args.out}')
    args.out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    records, audit = load_data(args.data)
    audit['load_seconds'] = time.monotonic()-started
    write_json(args.out/'data_audit.json',audit)
    # Chosen before training: ten evenly spaced validation IDs, not favourable outcomes.
    val_ids = sorted(r['dot'] for r in records['val'])
    cases = [val_ids[i] for i in np.linspace(0,len(val_ids)-1,10,dtype=int)]
    config = dict(seeds=args.seeds, epochs=args.epochs, stop_epochs=args.stop_epochs, chunk=args.chunk,
                  lr=.01, device=args.device, threads=args.threads, dtype='float32', validation_cases=cases,
                  torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
                  git_commit=args.code_revision or subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
                  trainer_sha256=sha(__file__), sensor_sha256=sha(sensors.__file__),
                  representative_seed_rule='median of summed component validation objectives; seed order breaks ties',
                  test_status='reserved for ticket 11110; not evaluated here')
    write_json(args.out/'config.json',config)
    summaries=[]
    for seed in args.seeds:
        _, d = train_one(records,'driving',seed,args)
        torque, a = train_one(records,'torque',seed,args)
        stop, s = train_one(records,'stop',seed,args)
        combined=dict(torque)
        combined.update({'stopnet.'+k:v for k,v in stop.items() if k.endswith(('.weight','.bias'))})
        combined.update(info_stopnet='17->32->16->1 relu',info_stopnet_epochs=np.int64(s['selected_epoch']))
        np.savez(args.out/f'reaching_seed{seed}.npz',**combined)
        summaries.append(dict(seed=seed,driving=d,torque=a,stop=s,
                              validation_composite=sum(m['selected_val_loss'] for m in (d,a,s))))
        write_json(args.out/'summary.json',dict(seeds=summaries, status='training', elapsed_s=time.monotonic()-started))
    ranked=sorted(summaries,key=lambda r:(r['validation_composite'],args.seeds.index(r['seed'])))
    write_json(args.out/'summary.json',dict(seeds=summaries,status='trained; simulator validation pending',
               representative_seed=ranked[len(ranked)//2]['seed'],elapsed_s=time.monotonic()-started))
    print('Training and export checks complete; simulator validation and reporting remain.',flush=True)


if __name__ == '__main__':
    main()
