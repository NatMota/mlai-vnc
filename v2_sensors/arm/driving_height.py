"""One controlled driving-height experiment; unchanged reaching, six validation full runs.

Run prepare, train, then evaluate using the same --data, --baseline and --out paths.
Training uses the CUDA environment; evaluation uses the simulator Python environment.
No reserved test trajectories are opened and no official MLflow iteration is created.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from render_golden import sha


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(a):
    if a.out.exists():
        raise FileExistsError(a.out)
    split = json.loads((a.data / 'split.json').read_text())
    candidates = []
    for dot in split['val']:
        with np.load(a.data / 'camera/val' / f'camera_{dot}.npz', allow_pickle=False) as c:
            target = c['target']
            candidates.append(dict(dot=dot, range_m=float(np.hypot(*target[:2])), height_m=float(target[2]),
                                   initially_visible=bool(c['seen'][0])))
    cases = []
    for lo, hi, centre in ((.1, .5, .3), (.5, 1.1, .8), (1.1, 1.26, 1.2)):
        for near in (True, False):
            pool = [r for r in candidates if r['initially_visible'] and lo <= r['height_m'] < hi
                    and (r['range_m'] < 1.5) == near]
            if not pool:
                raise ValueError('No initially visible validation target in required stratum')
            chosen = min(pool, key=lambda r: ((r['range_m'] - (1.25 if near else 1.75))**2
                                               + (r['height_m'] - centre)**2, r['dot']))
            cases.append(chosen)
    assert len({r['dot'] for r in cases}) == 6
    a.out.mkdir(parents=True)
    cfg = dict(seed=7, epochs=300, lr=.01, chunk=32768, threads=4, dtype='float32',
               cases=cases, split='val', validation_cases=[r['dot'] for r in cases],
               selection='one initially-visible validation target per low/mid/high height and near/far stratum; nearest stratum centre; no outcome selection',
               baseline_hashes={name:sha(a.baseline/name) for name in ('driving_seed7.npz','reaching_seed7.npz')},
               split_sha256=sha(a.data/'split.json'), test_trajectories_opened=0,
               change='append camera-tracked robot-frame height in metres; 16 to 17 driving inputs; unchanged reaching',
               expectation='lower paired median parking error, particularly low/high targets; worse or equal error does not support the benefit on these six cases')
    cfg['without_wheel_velocity'] = a.without_wheel_velocity
    if a.without_wheel_velocity:
        cfg.update(change='retain camera height; remove six wheel-speed inputs: 17 to 11 inputs; unchanged reaching and tracker',
                   expectation='test whether omitting speed inputs reduces paired parking error; no assumed benefit',
                   candidate_variant='height_no_speed')
    save(a.out/'config.json',cfg)
    print(json.dumps(cfg, indent=2), flush=True)


def train(a):
    import torch
    from train_golden import load_data, train_one
    cfg=json.loads((a.out/'config.json').read_text())
    models=a.out/'models'
    if models.exists(): raise FileExistsError(models)
    models.mkdir()
    torch.set_num_threads(cfg['threads'])
    start=time.monotonic()
    records,audit=load_data(a.data,driving_height=True)
    audit['load_seconds']=time.monotonic()-start
    save(models/'data_audit.json',audit)
    # Height comes from the existing causal tracker, including zeros before first sighting.
    for split in ('train','val'):
        assert all(r['drive'][0].shape[1] == 17 for r in records[split])
    if cfg.get('without_wheel_velocity', False):
        from train_golden import sensors
        for split in records:
            for r in records[split]:
                x,y,p = r['drive']
                r['drive'] = (x[:, sensors.DRIVING_NO_SPEED_COLS], y, p)
    args=argparse.Namespace(out=models,device=a.device,epochs=cfg['epochs'],stop_epochs=600,chunk=cfg['chunk'])
    _,summary=train_one(records,'driving',cfg['seed'],args)
    arm=a.baseline/'reaching_seed7.npz'
    assert sha(arm)==cfg['baseline_hashes'][arm.name]
    shutil.copy2(arm,models/arm.name)
    save(models/'summary.json',dict(driving=summary,reaching_sha256=sha(models/arm.name),elapsed_s=time.monotonic()-start,
        torch=torch.__version__,numpy=np.__version__,device=a.device,trainer_sha256=sha(Path(__file__).with_name('train_golden.py'))))


def attempt(job):
    from full_run import one
    variant,dot,case,wd,wa=job
    started=time.monotonic()
    row=one(wd,wa,dot,case=case,camera_dir=None,batch_driving=True)
    row.update(variant=variant,dot=dot,seed=7,elapsed_s=time.monotonic()-started)
    return row


def evaluate(a):
    from validate_golden import case_for
    cfg=json.loads((a.out/'config.json').read_text())
    result=a.out/'results'
    if result.exists(): raise FileExistsError(result)
    result.mkdir()
    split=json.loads((a.data/'split.json').read_text())
    assert set(cfg['validation_cases']) <= set(split['val'])
    assert not set(cfg['validation_cases']) & set(split['test'])
    for name,digest in cfg['baseline_hashes'].items(): assert sha(a.baseline/name)==digest
    arm=a.baseline/'reaching_seed7.npz'
    assert sha(a.out/'models'/arm.name)==sha(arm)
    wa=dict(np.load(arm,allow_pickle=False))
    jobs=[];ghosts={}
    models={'baseline':a.baseline/'driving_seed7.npz',cfg.get('candidate_variant','height'):a.out/'models/driving_seed7.npz'}
    for dot in cfg['validation_cases']:
        case,ghost=case_for(a.data,dot,'val');ghosts[str(dot)]=ghost
        for variant,model in models.items():
            jobs.append((variant,dot,case,dict(np.load(model,allow_pickle=False)),wa))
    save(result/'expert_validation.json',ghosts)
    save(result/'config.json',dict(cases=cfg['validation_cases'],split='val',jobs=a.jobs,
        model_hashes={k:sha(v) for k,v in models.items()},reaching_sha256=sha(arm),
        full_run_sha256=sha(Path(__file__).with_name('full_run.py')),
        full_rendering=True,camera_hz=30,driving_hz=30,arm_hz=1000,batch_driving=True,save_camera_png=False))
    start=time.monotonic();rows=[]
    with ProcessPoolExecutor(a.jobs) as pool:
        for row in pool.map(attempt,jobs):
            save(result/f"{row['variant']}_dot{row['dot']}.json",row);rows.append(row)
            print(f"{len(rows)}/12 {row['variant']} dot {row['dot']}: park={row['parked']}, error={row['park_error_vs_expert_m']}, closest={row['min_mm']}, {row['elapsed_s']:.1f}s",flush=True)
    save(result/'summary.json',[{k:v for k,v in r.items() if k not in ('frames','log')} for r in rows])
    save(result/'completion.json',dict(attempts=len(rows),elapsed_s=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','train','evaluate'])
    for name in ('data','baseline','out'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--without-wheel-velocity',action='store_true')
    args=p.parse_args()
    globals()[args.stage](args)
