"""Three cumulative torque-network experiments; fixed driving/STOP and six full validation cases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from render_golden import sha
from driving_height import save

VARIANTS=[('longer',64,False),('wider',128,False),('history',128,True)]


def train(a):
    import torch
    from train_golden import load_data,train_one
    from arm_history import stack_history
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    prior=json.loads((a.baseline/'config.json').read_text())
    cfg=dict(seed=7,epochs=1200,save_epochs=[300,600,1200],variants=VARIANTS,
             validation_cases=prior['validation_cases'],cases=prior['cases'],split='val',
             lr=.01,chunk=32768,history_lags_ms=[0,50,100],history_reset='PARK; first-sample padding',
             fixed_models={name:sha(a.baseline/'models'/name) for name in ('driving_seed7.npz','reaching_seed7.npz')},
             checkpoint_selection='minimum total validation loss, earliest tie; no simulation selection',test_trajectories_opened=0)
    save(a.out/'config.json',cfg)
    torch.set_num_threads(4);started=time.monotonic()
    records,audit=load_data(a.data)
    save(a.out/'data_audit.json',audit)
    arm=dict(np.load(a.baseline/'models/reaching_seed7.npz',allow_pickle=False))
    stop={k:v for k,v in arm.items() if k.startswith('stopnet.')}
    assert stop
    shutil.copy2(a.baseline/'models/driving_seed7.npz',a.out/'driving_seed7.npz')
    summaries=[]
    for name,width,history in VARIANTS:
        data=records
        if history:
            data={split:[dict(r,arm=(stack_history(r['arm'][0]),*r['arm'][1:])) for r in rs] for split,rs in records.items()}
        dest=a.out/'models'/name;dest.mkdir(parents=True)
        args=argparse.Namespace(out=dest,device=a.device,epochs=1200,stop_epochs=600,chunk=32768,arm_hidden=width,save_epochs=[300,600,1200])
        w,summary=train_one(data,'torque',7,args)
        w.update(stop)
        for key in ('info_stopnet','info_stopnet_epochs'):
            if key in arm:w[key]=arm[key]
        for k,v in stop.items():np.testing.assert_array_equal(w[k],v)
        np.savez(dest/'reaching_seed7.npz',**w)
        summaries.append(dict(variant=name,width=width,history=history,**summary))
        save(a.out/'training_summary.json',dict(variants=summaries,elapsed_s=time.monotonic()-started,torch=torch.__version__,numpy=np.__version__))
        print('FINISHED',name,summary['elapsed_s'],flush=True)


def attempt(job):
    from full_run import one
    variant,dot,case,wd,wa=job;started=time.monotonic()
    r=one(wd,wa,dot,case=case,batch_driving=True)
    r.update(variant=variant,dot=dot,seed=7,elapsed_s=time.monotonic()-started)
    return r


def evaluate(a):
    from validate_golden import case_for
    cfg=json.loads((a.out/'config.json').read_text());split=json.loads((a.data/'split.json').read_text())
    assert set(cfg['validation_cases']) <= set(split['val']) and not set(cfg['validation_cases']) & set(split['test'])
    for k,v in cfg['fixed_models'].items():assert sha(a.baseline/'models'/k)==v
    result=a.out/'results'
    if result.exists():raise FileExistsError(result)
    result.mkdir()
    wd=dict(np.load(a.out/'driving_seed7.npz',allow_pickle=False));jobs=[];ghosts={}
    arms={name:dict(np.load(a.out/'models'/name/'reaching_seed7.npz',allow_pickle=False)) for name,_,_ in VARIANTS}
    for dot in cfg['validation_cases']:
        case,ghost=case_for(a.data,dot,'val');ghosts[str(dot)]=ghost
        for name,wa in arms.items():jobs.append((name,dot,case,wd,wa))
    save(result/'expert_validation.json',ghosts)
    save(result/'config.json',dict(cases=cfg['validation_cases'],split='val',jobs=a.jobs,driving_sha256=sha(a.out/'driving_seed7.npz'),
        arm_hashes={name:sha(a.out/'models'/name/'reaching_seed7.npz') for name in arms},
        full_run_sha256=sha(Path(__file__).with_name('full_run.py')),camera_hz=30,driving_hz=30,arm_hz=1000,full_rendering=True))
    rows=[];start=time.monotonic()
    with ProcessPoolExecutor(a.jobs) as pool:
        for r in pool.map(attempt,jobs):
            save(result/f"{r['variant']}_dot{r['dot']}.json",r);rows.append(r)
            print(f"{len(rows)}/18 {r['variant']} dot{r['dot']}: closest={r['min_mm']:.1f}, STOP={r['at_network_stop_mm']}",flush=True)
    save(result/'summary.json',[{k:v for k,v in r.items() if k not in ('frames','log')} for r in rows])
    save(result/'completion.json',dict(attempts=len(rows),elapsed_s=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['train','evaluate'])
    for key in ('data','baseline','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--device',default='cuda',choices=['cpu','cuda']);p.add_argument('--jobs',default=3,type=int)
    args=p.parse_args();globals()[args.stage](args)
