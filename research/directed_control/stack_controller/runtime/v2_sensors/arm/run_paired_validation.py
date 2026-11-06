"""Evaluate the four remaining paired models on the preselected 50 validation targets."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from run_curriculum_ladder import freeze, save


def summarise(root, ids):
    records=[]
    for seed in (7,1,2):
        for short in ('A','B'):
            path=(root.parent/'overnight_20261105/basics_standard/val/summary.json'
                  if (seed,short)==(7,'A') else root/f'validation_{short}_seed{seed}_50/summary.json')
            rows=[r for r in json.loads(path.read_text()) if r['dot'] in ids]
            assert len(rows)==50 and {r['dot'] for r in rows}==set(ids)
            stopped=[r['at_network_stop_mm'] for r in rows if r.get('at_network_stop_mm') is not None]
            closest=[r['min_mm'] for r in rows if r.get('min_mm') is not None]
            records.append(dict(seed=seed,architecture=short,attempts=50,
                parked=sum(r['parked'] for r in rows),stopped=len(stopped),
                success_10mm=sum(r['success'] for r in rows),
                closest_recorded=len(closest),
                median_closest_mm=statistics.median(closest) if closest else None,
                median_stop_mm=statistics.median(stopped) if stopped else None,
                source=str(path),existing_reference=(seed,short)==(7,'A')))
    save(root/'paired_validation_50_comparison.json',records)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','data','native'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();started=time.monotonic();completed=[]
    ids=json.loads((a.root/'validation_50_plan.json').read_text())['ids']
    assert len(ids)==50 and len(set(ids))==50
    runtime=a.root/'validation_runtime'
    freeze(Path(__file__).resolve().parent.parent,runtime,a.native)
    env=dict(os.environ,NATIVE_ROOT=str(a.native),CURRICULUM_BINARY=str(runtime/'native/sensor_curriculum'),
        CURRICULUM_SOURCE=str(runtime/'native/sensor_curriculum.cpp'),CAMERA_RENDERER='egl',
        CAMERA_URDF=str(a.native/'assets/summit_xl_arm_description/summit_xl_complete.urdf'),
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')

    def status(stage,**extra):
        save(a.root/'validation_queue_status.json',dict(stage=stage,completed=completed,
             elapsed_s=time.monotonic()-started,notebook_writes=False,mlflow_writes=False,**extra))

    try:
        status('waiting_for_gpu_lock')
        with open('/tmp/comp0245-validation-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            for seed in (1,2):
                for architecture,short in (('A_auxiliary','A'),('B_torque_only','B')):
                    models=a.root/architecture/f'seed{seed}'
                    assert (models/'summary.json').exists()
                    out=a.root/f'validation_{short}_seed{seed}_50'
                    status('running',architecture=architecture,seed=seed,out=str(out))
                    cmd=[a.native/'.pixi/envs/default/bin/python',runtime/'v2_sensors/arm/evaluate_curriculum.py',
                         '--data',a.data,'--models',models,'--out',out,'--split','val',
                         '--seed',seed,'--jobs',2,'--ids',*ids]
                    with (a.root/f'validation_{short}_seed{seed}_50.log').open('x') as log:
                        subprocess.run([str(v) for v in cmd],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                    rows=json.loads((out/'summary.json').read_text())
                    assert len(rows)==50 and {r['dot'] for r in rows}==set(ids)
                    assert all(r['seed']==seed for r in rows)
                    if any(r['status'] in ('execution_failure','numerical_failure') for r in rows):
                        raise RuntimeError(f'Implementation or numerical failure retained in {out}')
                    completed.append(dict(seed=seed,architecture=architecture,out=str(out)))
            summarise(a.root, ids)
            status('complete',completed_new_batches=4,targets_per_model=50)
    except Exception as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__':
    main()
