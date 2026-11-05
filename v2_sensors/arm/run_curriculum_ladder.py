"""Run the five declared cumulative effects, with frozen models and complete splits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def freeze(source,out,native):
    out.mkdir(parents=True)
    for directory in ('arm','driving_v2'):
        target=out/'v2_sensors'/directory;target.mkdir(parents=True)
        for p in (source/directory).glob('*.py'):shutil.copy2(p,target/p.name)
    for rel in ('driving_v2/expert_reference.json','driving_v2/closed_loop_native/result_wheels_live_dot.json','driving_30/data/expert_30hz/split_plan.json'):
        target=out/'v2_sensors'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source/rel,target)
    target=out/'native';target.mkdir()
    for p in (native/'src/sensor_curriculum.cpp',native/'build-clang19/sensor_curriculum'):shutil.copy2(p,target/p.name)
    shutil.copytree(native/'src',target/'src')
    for name in ('CMakeLists.txt','pixi.toml','pixi.lock'):
        shutil.copy2(native/name,target/name)
    save(out/'hashes.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','models','out','baseline','native'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    source=Path(__file__).resolve().parent.parent
    a.out.mkdir(exist_ok=True,parents=True)
    frozen=a.out/'ladder_runtime';freeze(source,frozen,a.native)
    env=dict(os.environ,NATIVE_ROOT=str(a.native),CURRICULUM_BINARY=str(frozen/'native/sensor_curriculum'),
             CURRICULUM_SOURCE=str(frozen/'native/sensor_curriculum.cpp'),CAMERA_RENDERER='egl',
             CAMERA_URDF=str(a.native/'assets/summit_xl_arm_description/summit_xl_complete.urdf'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    python=a.native/'.pixi/envs/default/bin/python'
    script=frozen/'v2_sensors/arm/evaluate_curriculum.py'
    stages=['encoder_noise','damping','elasticity','inertia','delay']
    for name in stages:shutil.copy2(source/'arm/curriculum_conditions'/f'{name}.json',frozen/f'{name}.json')
    save(a.out/'ladder_status.json',dict(status='waiting_for_baseline',stages=stages))
    # Baseline executes in its own immutable snapshot; never kill or modify it.
    deadline=time.monotonic()+3600
    while not all((a.baseline/s/'completion.json').exists() for s in ('val','test')):
        if time.monotonic()>deadline:raise TimeoutError('Baseline did not complete within one hour')
        time.sleep(10)
    for name in stages:
        for split in ('val','test'):
            save(a.out/'ladder_status.json',dict(status='running',stage=name,split=split,stages=stages))
            out=a.out/name/split
            # A preflight may already exist, but a frozen full split must never be overwritten.
            out.parent.mkdir(exist_ok=True)
            with (out.parent/(split+'.log')).open('w') as log:
                subprocess.run([str(python),str(script),'--data',str(a.data),'--models',str(a.models),
                                '--out',str(out),'--split',split,'--jobs',str(a.jobs),
                                '--condition',str(frozen/(name+'.json'))],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            rows=json.loads((out/'summary.json').read_text())
            if any(r['status']=='execution_failure' for r in rows):
                save(a.out/'ladder_status.json',dict(status='blocked_on_implementation',stage=name,split=split))
                raise RuntimeError('Execution failure retained; fix before continuing the stack')
            print(name,split,len(rows),'attempts',sum(r['parked'] for r in rows),'PARK',flush=True)
        save(a.out/name/'review_pending.json',dict(status='completed_simulations_not_yet_published',
            next='result review, plots, player, notebook and MLflow; test outcomes cannot tune the next model'))
    save(a.out/'ladder_status.json',dict(status='five_effect_stages_complete',
        next='Review validation, basics and random-pose/DART/DAgger preparation; those stages have not been launched'))


if __name__=='__main__':main()
