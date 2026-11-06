"""Evaluate frozen golden models once on every reserved target; no fitting or selection."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from validate_golden import case_for, write_json
from full_run import one
from render_golden import sha


def attempt(job):
    started=time.monotonic()
    try:
        seed,dot,case,wd,wa,camera_dir,batch=job
        row=one(wd,wa,dot,True,case=case,camera_dir=camera_dir,batch_driving=batch)
        row.update(seed=seed,dot=dot,elapsed_s=time.monotonic()-started,execution_mode='batched-driving' if batch else '1ms-exchanges')
        return row
    except Exception as e:
        seed,dot,case,_,_,_,_=job
        return dict(seed=seed,dot=dot,bearing=dot,split=case['split'],success=False,parked=False,
                    stop_s=None,at_network_stop_mm=None,min_mm=None,park_error_vs_expert_m=None,
                    status='numerical_failure' if isinstance(e,FloatingPointError) else 'execution_failure',
                    error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-started,frames=[],log=[])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','models','out','frozen-hashes'): ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--preflight',action='store_true',help='One predetermined validation target, seed 7; no test files')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--batch-driving',action='store_true')
    ap.add_argument('--parity-summary',type=Path)
    a=ap.parse_args()
    if a.out.exists() and not a.resume: raise FileExistsError(a.out)
    if a.resume and not (a.out/'config.json').exists(): raise FileNotFoundError('No frozen evaluation to resume')
    if a.batch_driving:
        if a.parity_summary is None: raise ValueError('Require exact validation parity before enabling batching')
        parity=json.loads(a.parity_summary.read_text());assert parity['all_exact']
        for p in (Path(__file__).with_name('full_run.py'),Path(__file__).resolve().parent.parent/'driving_v2/wrist_camera.py'):
            assert sha(p)==parity['source_hashes'][p.name], 'Source differs from parity check'
    train=json.loads((a.models/'config.json').read_text())
    split=json.loads((a.data/'split.json').read_text())
    frozen=json.loads(a.frozen_hashes.read_text())['model_hashes']
    seeds=[7] if a.preflight else [7,1,2]
    which='val' if a.preflight else 'test'
    ids=[train['validation_cases'][0]] if a.preflight else split['test']
    if not a.preflight:
        assert len(ids)==len(set(ids))==100 and not set(ids)&set(split['train']+split['val'])
    assert train['seeds']==[7,1,2]
    models={};hashes={}
    for seed in seeds:
        dp,arm=a.models/f'driving_seed{seed}.npz',a.models/f'reaching_seed{seed}.npz'
        hashes[str(seed)]=dict(driving=sha(dp),reaching=sha(arm))
        assert hashes[str(seed)]==frozen[str(seed)], f'Frozen model changed: {seed}'
        models[seed]=(dict(np.load(dp,allow_pickle=False)),dict(np.load(arm,allow_pickle=False)))
    native=Path(os.environ['NATIVE_ROOT']); env=json.loads((a.models/'environment.json').read_text())
    native_hashes={name:sha(native/name) for name in env['native_files']}
    assert native_hashes==env['native_files'], 'Native simulator changed since validation'
    a.out.mkdir(parents=True,exist_ok=a.resume)
    config=dict(split=which,cases=ids,seeds=seeds,model_hashes=hashes,native_hashes=native_hashes,
        batch_driving=a.batch_driving,representative_seed=7,renderer=os.environ.get('CAMERA_RENDERER','tiny'),jobs=a.jobs,
        evaluation_sha256=sha(Path(__file__)),full_run_sha256=sha(Path(__file__).with_name('full_run.py')),
        data_split_sha256=sha(a.data/'split.json'),expert_commands=False,success_threshold_mm=10,
        timeout='expert completion + 4 s; complete 0.5 s observation after STOP',
        driving_hz=30,camera_hz=30,arm_hz=1000,added_noise=False,added_delay=False,
        added_motor_damping=False,added_elasticity=False,added_motor_inertia=False,
        prepark_arm_control='native PD hold plus model gravity',checkpoint_selection='validation only')
    if a.resume:
        previous=json.loads((a.out/'config.json').read_text())
        for k in ('split','cases','seeds','model_hashes','native_hashes','renderer','data_split_sha256','success_threshold_mm','timeout','driving_hz','camera_hz','arm_hz'):
            assert previous[k]==config[k], f'Evaluation contract changed: {k}'
        write_json(a.out/'resume_config.json',config)
    else: write_json(a.out/'config.json',config)
    if a.parity_summary: write_json(a.out/'speed_parity.json',parity)
    cases,ghosts={},{}
    # References score the attempt and set its fixed deadline; they are never student inputs.
    for dot in ids: cases[dot],ghosts[str(dot)]=case_for(a.data,dot,which)
    write_json(a.out/f'expert_{which}.json',ghosts)
    jobs=[(seed,dot,cases[dot],*models[seed],a.out/f'seed{seed}_dot{dot}_camera',a.batch_driving) for seed in seeds for dot in ids]
    rows=[];completed=set();prior_elapsed=0.
    if a.resume:
        for p in a.out.glob('seed*_dot*.json'):
            r=json.loads(p.read_text());key=(r['seed'],r['dot'])
            assert r['seed'] in seeds and r['dot'] in ids and key not in completed
            rows.append(r);completed.add(key)
        prior_elapsed=json.loads((a.out/'progress.json').read_text())['elapsed_s']
    pending_jobs=[j for j in jobs if (j[0],j[1]) not in completed]
    restored=len(rows);started=time.monotonic()
    with ProcessPoolExecutor(a.jobs) as pool:
        pending=[pool.submit(attempt,job) for job in pending_jobs]
        for future in as_completed(pending):
            row=future.result()
            write_json(a.out/f'seed{row["seed"]}_dot{row["dot"]}.json',row)
            rows.append(row)
            elapsed=time.monotonic()-started
            write_json(a.out/'progress.json',dict(completed=len(rows),total=len(jobs),elapsed_s=prior_elapsed+elapsed,resumed_elapsed_s=elapsed,restored=restored,
                         eta_s=(len(jobs)-len(rows))*elapsed/(len(rows)-restored),last_seed=row['seed'],last_dot=row['dot']))
            print(f'{len(rows)}/{len(jobs)} seed {row["seed"]} dot {row["dot"]}: {row["status"]}, {row["elapsed_s"]:.1f}s',flush=True)
    rows.sort(key=lambda r:(seeds.index(r['seed']),ids.index(r['dot'])))
    for seed in seeds: write_json(a.out/f'seed{seed}.json',[r for r in rows if r['seed']==seed])
    write_json(a.out/'summary.json',[{k:v for k,v in r.items() if k not in ('frames','log')} for r in rows])
    write_json(a.out/'completion.json',dict(attempts=len(rows),expected=len(jobs),elapsed_s=prior_elapsed+time.monotonic()-started,
                resumed_elapsed_s=time.monotonic()-started,restored=restored,
                all_accounted=len({(r['seed'],r['dot']) for r in rows})==len(jobs)))


if __name__=='__main__': main()
