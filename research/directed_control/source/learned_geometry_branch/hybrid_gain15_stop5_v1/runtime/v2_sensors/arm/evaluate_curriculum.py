"""Frozen full-episode evaluation. Split is explicit; every attempted target is retained."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from full_run import one
from validate_golden import case_for, write_json
from render_golden import sha


def attempt(job):
    seed,dot,case,wd,wa,camera_dir,condition=job;started=time.monotonic()
    try:
        row=one(wd,wa,dot,True,case=case,camera_dir=camera_dir,condition=condition)
        row.update(seed=seed,dot=dot,elapsed_s=time.monotonic()-started,execution_mode='1ms-exchanges')
        return row
    except Exception as e:
        return dict(seed=seed,dot=dot,bearing=dot,split=case['split'],condition=condition,success=False,parked=False,
                    stop_s=None,at_network_stop_mm=None,min_mm=None,park_error_vs_expert_m=None,
                    status='numerical_failure' if isinstance(e,FloatingPointError) else 'execution_failure',
                    error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started,frames=[],log=[])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','models','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--ids',type=int,nargs='+')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--condition',type=Path)
    a=p.parse_args();split=json.loads((a.data/'split.json').read_text());ids=a.ids or split[a.split]
    condition=json.loads(a.condition.read_text()) if a.condition else {}
    assert set(ids)<=set(split[a.split]) and len(set(ids))==len(ids)
    if a.ids and a.split=='test':raise ValueError('Test evaluation must include every reserved target')
    a.out.mkdir(parents=True,exist_ok=a.resume)
    wd=dict(np.load(a.models/f'driving_seed{a.seed}.npz',allow_pickle=False));wa=dict(np.load(a.models/f'reaching_seed{a.seed}.npz',allow_pickle=False))
    assert wa['info_full_episode'] and wa['info_park_input']
    native=Path(os.environ['NATIVE_ROOT'])
    config=dict(cases=ids,split=a.split,seed=a.seed,jobs=a.jobs,models=str(a.models),condition=condition,
        model_hashes={n:sha(a.models/n) for n in (f'driving_seed{a.seed}.npz',f'reaching_seed{a.seed}.npz')},
        native_hashes={str(n):sha(n) for n in (Path(os.environ.get('CURRICULUM_SOURCE',native/'src/sensor_curriculum.cpp')),Path(os.environ.get('CURRICULUM_BINARY',native/'build-clang19/sensor_curriculum')))},
        source_hashes={n:sha(Path(__file__).with_name(n)) for n in ('full_run.py','evaluate_curriculum.py','curriculum_sensors.py')},
        split_sha256=sha(a.data/'split.json'),renderer=os.environ.get('CAMERA_RENDERER','tiny'),
        arm_control='learned from reset, 1 kHz, no PD hold or gravity addition',
        selection='checkpoints frozen on offline validation; no fitting in evaluator',
        test_reuse='existing targets previously evaluated; not a fresh blind test')
    if a.resume:assert json.loads((a.out/'config.json').read_text())==config
    else:write_json(a.out/'config.json',config)
    rows=[];jobs=[];ghosts={}
    for dot in ids:
        case,ghosts[str(dot)]=case_for(a.data,dot,a.split)
        dest=a.out/f'seed{a.seed}_dot{dot}.json'
        if a.resume and dest.exists():rows.append(json.loads(dest.read_text()));continue
        jobs.append((a.seed,dot,case,wd,wa,a.out/f'seed{a.seed}_dot{dot}_camera',condition))
    write_json(a.out/f'expert_{a.split}.json',ghosts)
    # Existing analysis code expects this name; config and rows retain the real split.
    write_json(a.out/'expert_test.json',ghosts)
    started=time.monotonic();restored=len(rows)
    with ProcessPoolExecutor(a.jobs) as pool:
        futures=[pool.submit(attempt,j) for j in jobs]
        for f in as_completed(futures):
            r=f.result();rows.append(r);write_json(a.out/f"seed{a.seed}_dot{r['dot']}.json",r)
            elapsed=time.monotonic()-started
            write_json(a.out/'progress.json',dict(completed=len(rows),total=len(ids),elapsed_s=elapsed,eta_s=(len(ids)-len(rows))*elapsed/(len(rows)-restored)))
            print(f"{a.split} {len(rows)}/{len(ids)} dot {r['dot']}: {r['status']}",flush=True)
    rows.sort(key=lambda r:ids.index(r['dot']))
    write_json(a.out/f'seed{a.seed}.json',rows)
    write_json(a.out/'summary.json',[{k:v for k,v in r.items() if k not in ('frames','log')} for r in rows])
    write_json(a.out/'completion.json',dict(attempts=len(rows),expected=len(ids),all_accounted={r['dot'] for r in rows}==set(ids),elapsed_s=time.monotonic()-started))


if __name__=='__main__':main()
