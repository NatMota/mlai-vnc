"""Compare unchanged full rendering and control against exact caching/batching speedups.

Only predetermined validation targets are opened. Camera RGB/depth/segmentation,
network inputs/outputs, recorded trajectories and event metrics must agree exactly.
No MLflow runs, training, or test-set selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('data','models','out','reference-camera'): ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--jobs',type=int,default=6)
    ap.add_argument('--one',nargs=3,metavar=('VARIANT','SEED','DOT'))
    a=ap.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    if a.one:
        variant,seed,dot=a.one;seed,dot=int(seed),int(dot)
        if variant=='reference':
            spec=importlib.util.spec_from_file_location('wrist_camera',a.reference_camera)
            module=importlib.util.module_from_spec(spec);sys.modules['wrist_camera']=module
            module.__file__=str(Path(__file__).resolve().parent.parent/'driving_v2/wrist_camera.py')
            exec(compile(a.reference_camera.read_text(),str(a.reference_camera),'exec'),module.__dict__)
        import numpy as np
        import full_run
        from validate_golden import case_for
        split=json.loads((a.data/'split.json').read_text());assert dot in split['val'] and dot not in split['test']
        case,_=case_for(a.data,dot)
        wd=dict(np.load(a.models/f'driving_seed{seed}.npz'));wa=dict(np.load(a.models/f'reaching_seed{seed}.npz'))
        hashes={k:hashlib.sha256() for k in ('camera','driving_policy','arm_policy')}
        counts={k:0 for k in hashes}
        old_camera=full_run.Scene.fisheye
        def camera(self,*args):
            result=old_camera(self,*args)
            for v in result:hashes['camera'].update(np.ascontiguousarray(v).tobytes())
            counts['camera']+=1
            return result
        full_run.Scene.fisheye=camera
        def wrap(fn,key):
            def call(w,x):
                out=fn(w,x)
                hashes[key].update(x.tobytes())
                for v in out:hashes[key].update(np.asarray(v).tobytes())
                counts[key]+=1;return out
            return call
        full_run.drive_policy=wrap(full_run.drive_policy,'driving_policy')
        full_run.arm_policy=wrap(full_run.arm_policy,'arm_policy')
        started=time.monotonic()
        row=full_run.one(wd,wa,dot,True,case=case,
                        camera_dir=a.out/f'{variant}_seed{seed}_dot{dot}_camera',batch_driving=variant=='fast')
        elapsed=time.monotonic()-started
        # Paths are different between variants; image pixels are checked above.
        for f in row['frames']:f.pop('camera_src',None)
        h=hashlib.sha256(json.dumps(row,sort_keys=True,allow_nan=False).encode()).hexdigest()
        result=dict(variant=variant,seed=seed,dot=dot,elapsed_s=elapsed,trajectory_and_metrics_sha256=h,
                    hashes={k:v.hexdigest() for k,v in hashes.items()},counts=counts)
        (a.out/f'{variant}_seed{seed}_dot{dot}.json').write_text(json.dumps(result,indent=2)+'\n')
        return
    pairs=[(seed,dot) for seed in (7,1,2) for dot in (6,131)]
    wall={}
    for variant in ('reference','fast'):
        def run(pair):
            seed,dot=pair
            cmd=[sys.executable,__file__,'--data',str(a.data),'--models',str(a.models),'--out',str(a.out),
                 '--reference-camera',str(a.reference_camera),'--one',variant,str(seed),str(dot)]
            with (a.out/f'{variant}_seed{seed}_dot{dot}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=f,check=True)
        started=time.monotonic()
        with ThreadPoolExecutor(a.jobs) as pool:list(pool.map(run,pairs))
        wall[variant]=time.monotonic()-started
        print(variant,wall[variant],flush=True)
    rows=[]
    for seed,dot in pairs:
        r=json.loads((a.out/f'reference_seed{seed}_dot{dot}.json').read_text())
        f=json.loads((a.out/f'fast_seed{seed}_dot{dot}.json').read_text())
        for key in ('trajectory_and_metrics_sha256','hashes','counts'):
            assert r[key]==f[key],f'Mismatch seed {seed} target {dot}: {key}'
        rows.append(dict(seed=seed,dot=dot,reference_s=r['elapsed_s'],fast_s=f['elapsed_s'],exact_match=True,counts=f['counts']))
    summary=dict(jobs=a.jobs,cases=rows,wall_s=wall,speedup=wall['reference']/wall['fast'],
                 projected_300_s=wall['fast']/len(pairs)*300,all_exact=True,
                 source_hashes={str(q.name):hashlib.sha256(q.read_bytes()).hexdigest() for q in
                   (Path(__file__).with_name('full_run.py'),Path(__file__).resolve().parent.parent/'driving_v2/wrist_camera.py')})
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
