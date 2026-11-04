"""Predetermined validation runs only; final test evaluation is a separate ticket."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np

from full_run import one
from render_golden import frame_steps, sha


def write_json(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,allow_nan=False)+'\n')
    tmp.replace(path)


def case_for(root, dot):
    with np.load(root/'expert/val'/f'run_{dot}.npz',allow_pickle=False) as a:
        m=json.loads(str(a['manifest'])); park=int(m['parking_step'])
        reference=dict(park_time_s=park/1000,park_base_xy=a['base_position'][park,:2].tolist(),
                       final_base_xy=a['base_position'][-1,:2].tolist())
        frames=[]
        for k in frame_steps(m['steps']):
            frames.append(dict(time_s=float(a['time_s'][k]),base=a['base_position'][k].tolist(),
                 quaternion=a['base_quaternion'][k].tolist(),joints=a['q'][k].tolist(),
                 ee=a['hand_position'][k].tolist(),parked=bool(k>=park),
                 tau=a['arm_torque_command'][k].tolist(),ee_mm=float(1000*a['hand_distance'][k])))
        ghost=dict(target=m['target_position_world_m'],park_s=park/1000,stop_s=(m['steps']-1)/1000,
                   stop_mm=float(1000*a['hand_distance'][-1]),frames=frames)
        return dict(manifest=m,reference=reference,split='val'),ghost


def run_case(job):
    seed,dot,case,wd,wa,camera_dir=job
    started=time.monotonic()
    row=one(wd,wa,dot,True,case=case,camera_dir=camera_dir)
    row.update(seed=seed,dot=dot,elapsed_s=time.monotonic()-started)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=2)
    a=ap.parse_args()
    if a.out.exists(): raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    config=json.loads((a.models/'config.json').read_text())
    ids=config['validation_cases']
    split=json.loads((a.data/'split.json').read_text())
    if not set(ids)<=set(split['val']) or set(ids)&set(split['test']): raise ValueError('Only validation cases allowed')
    cases,ghosts={},{}
    for dot in ids: cases[dot],ghosts[str(dot)]=case_for(a.data,dot)
    write_json(a.out/'expert_validation.json',ghosts)
    jobs=[]; hashes={}
    for seed in config['seeds']:
        dp,ap_=a.models/f'driving_seed{seed}.npz',a.models/f'reaching_seed{seed}.npz'
        wd,wa=dict(np.load(dp,allow_pickle=False)),dict(np.load(ap_,allow_pickle=False))
        hashes[str(seed)]={'driving':sha(dp),'reaching':sha(ap_)}
        jobs.extend((seed,dot,cases[dot],wd,wa,a.out/f'seed{seed}_dot{dot}_camera') for dot in ids)
    write_json(a.out/'config.json',dict(cases=ids,split='val',model_hashes=hashes,expert_commands=False))
    rows=[]
    with ProcessPoolExecutor(a.jobs) as pool:
        for row in pool.map(run_case,jobs):
            write_json(a.out/f'seed{row["seed"]}_dot{row["dot"]}.json',row)
            rows.append(row)
            print(f'seed {row["seed"]} dot {row["dot"]}: parked={row["parked"]}, stop_mm={row["at_network_stop_mm"]}, {row["elapsed_s"]:.1f}s',flush=True)
    for seed in config['seeds']: write_json(a.out/f'seed{seed}.json',[r for r in rows if r['seed']==seed])
    write_json(a.out/'summary.json',[{k:v for k,v in r.items() if k not in ('frames','log')} for r in rows])


if __name__=='__main__': main()
