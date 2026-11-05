"""Durable overnight queue. Failed execution stops the queue; physical failures stay.

No test result is read by checkpoint, round, or severity selection. Completed
evaluations are queued for notebook/publication review, never labelled published.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from run_curriculum_ladder import save,freeze
from render_golden import sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','golden','native','cuda'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--retry-tag',default='',help='New runtime, model and log suffix; retains a failed attempt unchanged')
    a=p.parse_args()
    if a.retry_tag and not a.retry_tag.isalnum():p.error('Retry tag must be alphanumeric')
    frozen=a.root/('recovery_runtime'+('_'+a.retry_tag if a.retry_tag else ''))
    if not frozen.exists():freeze(Path(__file__).resolve().parent.parent,frozen,a.native)
    arm=frozen/'v2_sensors/arm';py=a.native/'.pixi/envs/default/bin/python'
    env=dict(os.environ,NATIVE_ROOT=str(a.native),CURRICULUM_BINARY=str(frozen/'native/sensor_curriculum'),
        CURRICULUM_SOURCE=str(frozen/'native/sensor_curriculum.cpp'),CAMERA_RENDERER='egl',
        CAMERA_URDF=str(a.native/'assets/summit_xl_arm_description/summit_xl_complete.urdf'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    baseline=a.root/'baseline/models';standard=a.root/'basics_standard/models'
    condition=a.root/'ladder_runtime/delay.json';random=a.root/'random_start_data'
    dart=[a.root/'dart_data',a.root/'dart_data_seed2']
    started=time.monotonic();completed=[]

    def status(stage,**extra):save(a.root/'recovery_status.json',dict(status='running',stage=stage,elapsed_s=time.monotonic()-started,completed=completed,**extra))
    def wait(path,seconds=14400):
        status('waiting',path=str(path));deadline=time.monotonic()+seconds
        while not path.exists():
            if time.monotonic()>deadline:raise TimeoutError(str(path))
            time.sleep(15)
    def command(name,args):
        status(name);log=a.root/(name+('_'+a.retry_tag if a.retry_tag else '')+'.log')
        with log.open('x') as f:subprocess.run([str(v) for v in args],env=env,cwd=arm,stdout=f,stderr=subprocess.STDOUT,check=True)
    def train(name,data,extras=(),daggers=(),scaling='standard',mask=0):
        out=a.root/name/'models'
        if (out/'summary.json').exists():return out
        if out.exists() and a.retry_tag:
            out=out.with_name('models_'+a.retry_tag)
            if (out/'summary.json').exists():return out
        command(name+'_train',[a.cuda,arm/'train_full_episode.py','--data',data,'--out',out,'--driving',standard/'driving_seed7.npz',
            '--train-driving','--driving-epochs','300','--epochs','1200','--stop-epochs','600','--arm-scaling',scaling,
            '--mask-park-seconds',str(mask),'--allowed-conditions','baseline','random_start','dart',
            *(['--extra-training',*extras] if extras else []),*(['--dagger',*daggers] if daggers else [])])
        return out
    def evaluate(name,models,data,stack=False,ids=None,splits=('val','test')):
        for split in splits:
            out=a.root/name/split
            if (out/'completion.json').exists():
                cfg=json.loads((out/'config.json').read_text())
                assert cfg['model_hashes']=={n:sha(models/n) for n in ('driving_seed7.npz','reaching_seed7.npz')}
                continue
            command(name+'_'+split,[py,arm/'evaluate_curriculum.py','--data',data,'--models',models,'--out',out,
                '--split',split,'--jobs',a.jobs,*(['--condition',condition] if stack else []),*(['--ids',*ids] if ids else [])])
            rows=json.loads((out/'summary.json').read_text())
            if any(r['status']=='execution_failure' for r in rows):raise RuntimeError(f'{name}: retained execution failure')
        if not ids:
            completed.append(name);save(a.root/name/'review_pending.json',dict(status='simulations complete; needs plots, notebook, result review and publication'))
    def smoke(name,models):
        evaluate(name,models,a.golden,ids=[6,16,23,45,55,80],splits=('val',))
        rows=json.loads((a.root/name/'val/summary.json').read_text())
        # Every attempt counts. Missing PARK or STOP incurs a fixed 1 m penalty.
        values=[(r['at_network_stop_mm'] if r['at_network_stop_mm'] is not None else r['end_mm'])+
                (1000 if not r['parked'] or r['stop_s'] is None else 0) for r in rows]
        return float(np.median(values))
    def dagger_rounds(name,start,extras):
        model=start;roots=[];candidates=[]
        for round_ in range(1,4):
            label=f'{name}_round{round_}';data=a.root/label/'dagger'
            if not (data/'completion.json').exists():
                command(label+'_collect',[py,arm/'collect_dagger.py','--data',a.golden,'--models',model,'--out',data,'--jobs',a.jobs])
            roots.append(data);model=train(label,a.golden,extras,roots)
            score=smoke(label+'_smoke',model)
            candidates.append(dict(round=round_,models=str(model),validation_score_mm=score))
            save(a.root/(name+'_rounds.json'),candidates)
        selected=min(candidates,key=lambda r:(r['validation_score_mm'],r['round']))
        save(a.root/(name+'_selection.json'),dict(selected=selected,candidates=candidates,
            selection='median STOP error; missing PARK/STOP uses final error +1000 mm, six fixed validation targets',test_read=False))
        return Path(selected['models'])
    try:
        # Standardisation was selected using all 100 validation cases, before this test.
        later_jobs=a.jobs;a.jobs=min(2,a.jobs)
        evaluate('basics_standard',standard,a.golden,splits=('test',))
        wait(a.root/'ladder_status.json')
        status('waiting_for_frozen_ladder')
        deadline=time.monotonic()+14400
        while True:
            state=json.loads((a.root/'ladder_status.json').read_text())
            if state['status']=='five_effect_stages_complete':break
            if state['status']=='blocked_on_implementation':raise RuntimeError('Frozen ladder needs an implementation fix')
            if time.monotonic()>deadline:raise TimeoutError('Frozen ladder')
            time.sleep(15)
        a.jobs=later_jobs
        wait(dart[0]/'completion.json');wait(random/'completion.json')
        if not (dart[1]/'completion.json').exists():
            command('dart_seed2_collection',[py,arm/'collect_recovery.py','--golden',a.golden,'--out',dart[1],
                '--mode','dart','--seed','80000','--jobs','2'])
        # Cumulative stress branch, preserved even if its outcome is poor.
        evaluate('random_start_fixed',baseline,random,True)
        random_model=train('random_start_retrained',random,scaling='physical')
        evaluate('random_start_retrained',random_model,random,True)
        evaluate('random_start_retention',random_model,a.golden)
        imperfect=train('imperfect_park',random,scaling='physical',mask=.25)
        evaluate('imperfect_park',imperfect,random,True)
        # Recovery branches from the validation-selected, usable clean setup.
        dart_model=train('dart',a.golden,extras=dart)
        smoke('dart_smoke',dart_model)
        pure=dagger_rounds('dagger',standard,[])
        combined=dagger_rounds('dart_dagger',dart_model,dart)
        for name,model in [('standard_stack',standard),('dart',dart_model),('dagger',pure),('dart_dagger',combined)]:
            if name!='standard_stack':evaluate(name,model,a.golden)
            evaluate(name if name=='standard_stack' else name+'_stack',model,random,True)
        save(a.root/'recovery_status.json',dict(status='simulations complete; publication review pending',completed=completed,elapsed_s=time.monotonic()-started))
    except Exception as e:
        save(a.root/'recovery_status.json',dict(status='stopped_on_error',error=repr(e),completed=completed,elapsed_s=time.monotonic()-started))
        raise


if __name__=='__main__':main()
