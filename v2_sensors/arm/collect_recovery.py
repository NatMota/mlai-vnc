"""Collect and render fresh training data without changing the golden split.

Random starts collect new references in every split. DART-inspired collection
perturbs actions on training targets only and reuses the original validation and
test references. All attempts, including expert failures, remain in the package.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from render_golden import one, sha


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def render_job(job):
    return one(*job)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--golden',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['random_start','dart'],required=True)
    p.add_argument('--seed',type=int,default=70000)
    p.add_argument('--jobs',type=int,default=1)
    p.add_argument('--effects',type=Path)
    p.add_argument('--wheel-noise',type=float,default=.025)
    p.add_argument('--arm-noise',type=float,default=.5)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();native=Path(os.environ['NATIVE_ROOT'])
    split=json.loads((a.golden/'split.json').read_text())
    shutil.copy2(a.golden/'split.json',a.out/'split.json')
    collect_splits=list(split) if a.mode=='random_start' else ['train']
    plan=[];mapping={};rng=np.random.default_rng(a.seed)
    for name in collect_splits:
        for dot in split[name]:
            with np.load(a.golden/'expert'/name/f'run_{dot}.npz',allow_pickle=False) as z:
                m=json.loads(str(z['manifest']))
            row=dict(id=dot,target=m['target_position_world_m'],seed=a.seed+dot,max_steps=40000)
            if a.mode=='random_start':
                # Modest reset variation, fixed before any student evaluation.
                row['start_pose']=[*rng.uniform(-.15,.15,2).tolist(),float(rng.uniform(-np.pi/6,np.pi/6))]
            else:row.update(wheel_noise_sd=a.wheel_noise,arm_noise_sd=a.arm_noise,noise_hold_steps=33)
            plan.append(row);mapping[dot]=name
    payload=dict(runs=plan)
    if a.effects:payload['effects_file']=str(a.effects.resolve())
    write_json(a.out/'collection_plan.json',payload)
    write_json(a.out/'collection_config.json',dict(mode=a.mode,seed=a.seed,split_sha256=sha(a.golden/'split.json'),
        source_golden=str(a.golden),native_sha256=sha(native/'build-clang19/collect_curriculum'),
        label_rule='clean stateful expert at actual pre-step state; action noise never used as label',
        test_used_for_training=False,rendering='full original pixels, 30 Hz, deployment detector and tracker'))
    with (a.out/'collection.log').open('w') as log:
        subprocess.run([str(native/'build-clang19/collect_curriculum'),str(native/'assets'),str(a.out/'collection_plan.json'),str(a.out/'raw')],stdout=log,stderr=subprocess.STDOUT,check=True)
    sys.path.insert(0,str(native/'tools'))
    from pack_runs import pack,TruePose
    pose=TruePose(native/'assets/summit_xl_arm_description/summit_xl_complete.urdf')
    attempts=[]
    for name,ids in split.items():
        (a.out/'expert'/name).mkdir(parents=True)
        for dot in ids:
            dest=a.out/'expert'/name/f'run_{dot}.npz'
            if name in collect_splits:
                raw=a.out/'raw'/f'run_{dot}';m=json.loads((raw/'manifest.json').read_text())
                pack(raw,dest,pose,a.mode)
                attempts.append(dict(dot=dot,split=name,termination=m['termination'],steps=m['steps'],parking_step=m['parking_step']))
            else:dest.symlink_to((a.golden/'expert'/name/f'run_{dot}.npz').resolve())
    write_json(a.out/'collection_attempts.json',attempts)
    jobs=[];camera=a.out/'camera';camera.mkdir()
    for name in ('train','val'):
        if name not in collect_splits:
            (camera/name).symlink_to((a.golden/'camera'/name).resolve(),target_is_directory=True)
        else:jobs.extend((a.out/'expert'/name,camera,name,dot) for dot in split[name])
    rendered=[]
    with ProcessPoolExecutor(a.jobs) as pool:
        futures=[pool.submit(render_job,job) for job in jobs]
        for f in as_completed(futures):
            rendered.append(f.result())
            write_json(a.out/'render_progress.json',dict(completed=len(rendered),total=len(jobs),elapsed_s=time.monotonic()-started))
    files=[]
    for name,ids in split.items():
        for dot in ids:
            paths=[a.out/'expert'/name/f'run_{dot}.npz']
            if name!='test':paths.append(camera/name/f'camera_{dot}.npz')
            for path in paths:files.append(dict(path=str(path.relative_to(a.out)),sha256=sha(path),bytes=path.stat().st_size))
    write_json(a.out/'manifest.json',dict(code_revision=os.environ.get('CAMERA_CODE_COMMIT','see collection_config hashes'),files=files,
        condition=a.mode,split_sha256=sha(a.out/'split.json'),attempts=attempts))
    write_json(a.out/'completion.json',dict(status='collected and rendered; not trained',attempts=len(attempts),
        reached=sum(r['termination']=='reached' for r in attempts),rendered=len(rendered),elapsed_s=time.monotonic()-started))


if __name__=='__main__':main()
