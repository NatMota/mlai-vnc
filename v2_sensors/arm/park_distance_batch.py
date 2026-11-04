"""Redo four cumulative configurations with distance-based PARK supervision."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from parking_target import parking_distance, DISTANCE_SCALE_M, PARK_THRESHOLD_M
from render_golden import sha
from driving_height import save

LINEUP = [('height', 'height17', 'original'), ('no_speed', 'no_speed11', 'original'),
          ('longer_arm', 'no_speed11', 'longer'), ('wider_arm', 'no_speed11', 'wider')]


def check():
    # A 3-4-5 triangle, independent of height, translation and rotation.
    xyz = np.array([[.3, .4, 8], [0, 0, 1], [.01, 0, 2], [4, 0, 0]])
    got = parking_distance(xyz, 1, np.arange(4))
    np.testing.assert_allclose(got, [.5/3, 0, .01/3, 1], atol=1e-8)
    rotated = xyz.copy(); rotated[:, :2] = xyz[:, :2] @ np.array([[0, -1], [1, 0]]) + [8, -5]
    np.testing.assert_allclose(parking_distance(rotated, 1, np.arange(4)), got, atol=1e-8)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'driving_v2'))
    from camera_loop import policy
    w = {'shared.0.weight': np.zeros((2,17)), 'shared.0.bias': np.zeros(2),
         'shared.2.weight': np.zeros((2,2)), 'shared.2.bias': np.zeros(2),
         'park.weight': np.zeros((1,2)), 'park.bias': np.zeros(1),
         'wheels.weight': np.zeros((4,2)), 'wheels.bias': np.ones(4),
         'park_distance_scale_m': DISTANCE_SCALE_M, 'park_distance_threshold_m': PARK_THRESHOLD_M}
    for distance, expected in ((.049, 1), (.051, 0)):
        w['park.bias'][0] = distance / DISTANCE_SCALE_M
        cmd, parked = policy(w, np.zeros(17))
        assert parked == expected and policy.ttp is None
        np.testing.assert_allclose(policy.park_distance_m, distance)
        np.testing.assert_array_equal(cmd, np.ones(4))
    del w['park_distance_scale_m']; del w['park_distance_threshold_m']
    w.update(ttp_clip=3., ttp_thresh=.25); w['park.bias'][0] = .2/3
    assert policy(w, np.zeros(17))[1] == 1 and policy.park_distance_m is None
    assert abs(policy.ttp-.2) < 1e-8
    print('PASS: distance labels, frame invariance, post-PARK drift, scaling, 5 cm trigger, legacy time path')


def train(a):
    import torch
    from train_golden import load_data, train_one, sensors
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    start = time.monotonic(); torch.set_num_threads(4)
    previous = json.loads((a.prior_root/'driving_height_no_speed11/config.json').read_text())
    lineup = [row for row in LINEUP if a.configuration is None or row[0] == a.configuration]
    arm_sources = {'original': a.prior_root/'driving_height_no_speed11/models/reaching_seed7.npz',
                   **{n: a.prior_root/f'arm_cumulative/models/{n}/reaching_seed7.npz' for n in ('longer','wider')}}
    arm_sources = {n:p for n,p in arm_sources.items() if n in {row[2] for row in lineup}}
    cfg = dict(seed=7, driving_epochs=300, lr=.01, dtype='float32', chunk=32768,
               validation_cases=previous['validation_cases'], cases=previous['cases'], split='val',
               lineup=lineup, park_target='distance', park_distance_scale_m=DISTANCE_SCALE_M,
               park_distance_threshold_m=PARK_THRESHOLD_M, park_weight=a.park_weight,
               loss=f'wheel MSE + {a.park_weight:g} * normalized PARK-distance MSE',
               label='2D distance to expert base position at parking_step; retain measured post-PARK drift',
               arm_sha256={n:sha(p) for n,p in arm_sources.items()},
               arm_retraining=False, arm_note='Reuse unchanged original, 1200-epoch 64/64 and 1200-epoch 128/128 torque checkpoints; same separate STOP network',
               checkpoint_selection='minimum total offline validation loss; earliest tie', test_trajectories_opened=0,
               source_sha256={n:sha(Path(__file__).with_name(n)) for n in ('train_golden.py','parking_target.py','park_distance_batch.py','full_run.py')},
               policy_sha256=sha(Path(__file__).parent.parent/'driving_v2/camera_loop.py'))
    save(a.out/'config.json', cfg)
    arms=a.out/'arms'; arms.mkdir()
    for n,p in arm_sources.items(): shutil.copy2(p, arms/f'{n}.npz')
    records,audit = load_data(a.data, driving_height=True, park_target='distance')
    save(a.out/'data_audit.json',audit)
    summaries=[]
    for name,remove in [('height17',False),('no_speed11',True)]:
        if name not in {row[1] for row in lineup}: continue
        data=records
        if remove:
            data={split:[dict(r,drive=(r['drive'][0][:,sensors.DRIVING_NO_SPEED_COLS],*r['drive'][1:])) for r in rs] for split,rs in records.items()}
        dest=a.out/'models'/name; dest.mkdir(parents=True)
        args=argparse.Namespace(out=dest,device=a.device,epochs=300,stop_epochs=600,chunk=32768,park_target='distance',park_weight=a.park_weight)
        _,meta=train_one(data,'driving',7,args)
        summaries.append(dict(variant=name,**meta))
        save(a.out/'training_summary.json',dict(variants=summaries,elapsed_s=time.monotonic()-start,torch=torch.__version__,numpy=np.__version__))
        print('TRAINED',name,meta['selected_epoch'],meta['selected_val_loss'],flush=True)


def attempt(job):
    from full_run import one
    variant,dot,case,wd,wa=job; start=time.monotonic()
    r=one(wd,wa,dot,case=case,batch_driving=True)
    r.update(variant=variant,dot=dot,seed=7,elapsed_s=time.monotonic()-start,
             expert_park_base_xy=case['reference']['park_base_xy'])
    return r


def evaluate(a):
    from validate_golden import case_for
    cfg=json.loads((a.out/'config.json').read_text()); split=json.loads((a.data/'split.json').read_text())
    assert set(cfg['validation_cases']) <= set(split['val']) and not set(cfg['validation_cases']) & set(split['test'])
    out=a.out/'results'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    for name,digest in cfg['arm_sha256'].items():assert sha(a.out/'arms'/f'{name}.npz')==digest
    drives={n:dict(np.load(a.out/'models'/n/'driving_seed7.npz',allow_pickle=False)) for n in dict.fromkeys(row[1] for row in cfg['lineup'])}
    arms={n:dict(np.load(a.out/'arms'/f'{n}.npz',allow_pickle=False)) for n in cfg['arm_sha256']}
    for w in drives.values():
        assert 'ttp_clip' not in w and str(w['info_stop_target'])=='distance'
    save(out/'config.json',dict(validation_cases=cfg['validation_cases'],jobs=a.jobs,
        driving_hashes={n:sha(a.out/'models'/n/'driving_seed7.npz') for n in drives},arm_hashes=cfg['arm_sha256'],
        full_rendering=True,camera_hz=30,driving_hz=30,arm_hz=1000,park_distance_threshold_m=PARK_THRESHOLD_M))
    jobs=[]
    for dot in cfg['validation_cases']:
        case,_=case_for(a.data,dot,'val')
        for name,drive,arm in cfg['lineup']:jobs.append((name,dot,case,drives[drive],arms[arm]))
    rows=[]; start=time.monotonic()
    with ProcessPoolExecutor(a.jobs) as pool:
        for r in pool.map(attempt,jobs):
            save(out/f"{r['variant']}_dot{r['dot']}.json",r)
            rows.append({k:v for k,v in r.items() if k not in ('frames','log')})
            save(out/'summary.json',rows)
            print(f"{len(rows)}/{len(jobs)} {r['variant']} dot{r['dot']}: PARK={r['parked']}, park={100*r['park_error_vs_expert_m']:.1f} cm, closest={r['min_mm']}, STOP={r['at_network_stop_mm']}",flush=True)
    save(out/'completion.json',dict(attempts=len(rows),elapsed_s=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['check','train','evaluate'])
    for key in ('data','prior-root','out'):p.add_argument('--'+key,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--park-weight',type=float,default=1.)
    p.add_argument('--configuration',choices=[r[0] for r in LINEUP])
    args=p.parse_args()
    if args.stage=='check':check()
    else:
        if any(getattr(args,k) is None for k in ('data','prior_root','out')):p.error('--data, --prior-root and --out are required')
        globals()[args.stage](args)
