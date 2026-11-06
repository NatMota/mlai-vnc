"""Full frozen test of four cumulative time-to-PARK configurations, seed 7."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from evaluate_golden import attempt
from validate_golden import case_for, write_json
from render_golden import sha

LINEUP = [('height',37,'driving_height17','original'),
          ('no_speed',38,'driving_height_no_speed11','original'),
          ('longer_arm',39,'driving_height_no_speed11','longer'),
          ('wider_arm',40,'driving_height_no_speed11','wider')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','prior-root','out'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--jobs',type=int,default=6)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.out.exists() and not a.resume:raise FileExistsError(a.out)
    split=json.loads((a.data/'split.json').read_text());ids=split['test']
    assert len(set(ids))==100 and not set(ids)&set(split['train']+split['val'])
    a.out.mkdir(parents=True,exist_ok=a.resume)
    original=Path(__file__).parent/'models/golden_11109'
    native=Path(os.environ['NATIVE_ROOT'])
    env=json.loads((original/'environment.json').read_text())
    assert {n:sha(native/n) for n in env['native_files']}==env['native_files']
    sources={n:sha(Path(__file__).with_name(n)) for n in ('evaluate_time_cumulative.py','full_run.py','evaluate_golden.py','validate_golden.py')}
    sources.update({n:sha(Path(__file__).parent.parent/'driving_v2'/n) for n in ('sensors.py','camera_loop.py','wrist_camera.py')})
    config=dict(split='test',cases=ids,seed=7,jobs=a.jobs,lineup=LINEUP,source_hashes=sources,
        native_hashes=env['native_files'],split_sha256=sha(a.data/'split.json'),
        renderer=os.environ.get('CAMERA_RENDERER','tiny'),camera_hz=30,driving_hz=30,arm_hz=1000,
        park_target='time',park_threshold_s=.25,stop_threshold_m=.01,batch_driving=True,
        full_rendering=True,added_noise=False,added_delay=False,added_motor_corrections=False,
        timeout='expert completion + 4 s; observe 0.5 s after STOP',
        selection='Existing seed-7 checkpoints selected by offline validation; no retraining or test-based selection',
        test_reuse_note='These 100 targets were previously evaluated for baseline seeds 7/1/2; this is a subsequent frozen comparison, not a fresh blind test.')
    if a.resume:
        assert json.loads((a.out/'config.json').read_text())==config
    else:write_json(a.out/'config.json',config)
    models={}
    for name,iteration,drive,arm in LINEUP:
        dest=a.out/name;dest.mkdir(exist_ok=a.resume);md=dest/'models';md.mkdir(exist_ok=a.resume)
        drive_src=a.prior_root/drive/'models'
        arm_src=original if arm=='original' else a.prior_root/'arm_cumulative/models'/arm
        paths={'driving_seed7.npz':drive_src/'driving_seed7.npz',
               'reaching_seed7.npz':a.prior_root/drive/'models/reaching_seed7.npz' if arm=='original' else arm_src/'reaching_seed7.npz'}
        for kind,source in [('driving',drive_src),('torque',arm_src),('stop',original)]:
            for suffix in ('_history.json','_summary.json'):
                paths[f'{kind}_seed7{suffix}']=source/f'{kind}_seed7{suffix}'
        for f,src in paths.items():
            if a.resume:assert sha(md/f)==sha(src)
            else:shutil.copy2(src,md/f)
        hashes={f:sha(md/f) for f in paths}
        wd=dict(np.load(md/'driving_seed7.npz',allow_pickle=False));wa=dict(np.load(md/'reaching_seed7.npz',allow_pickle=False))
        assert 'park_distance_scale_m' not in wd and float(wd['ttp_clip'])==3.
        np.testing.assert_allclose(wd['ttp_thresh'],.25)
        cfg=dict(config,variant=name,iteration=iteration,seeds=[7],model_hashes=hashes,
                 drive_inputs=int(wd['shared.0.weight'].shape[1]),arm_hidden=int(wa['shared.0.weight'].shape[0]))
        if not a.resume:write_json(dest/'config.json',cfg)
        models[name]=(wd,wa)
    # Freeze all models before opening any test trajectory.
    cases={};ghosts={}
    for dot in ids:cases[dot],ghosts[str(dot)]=case_for(a.data,dot,'test')
    for name,*_ in LINEUP:
        if not a.resume:write_json(a.out/name/'expert_test.json',ghosts)
    rows={name:[] for name,*_ in LINEUP};jobs=[]
    for dot in ids:
        for name,*_ in LINEUP:
            out=a.out/name;path=out/f'seed7_dot{dot}.json'
            if a.resume and path.exists():rows[name].append(json.loads(path.read_text()));continue
            jobs.append((name,(7,dot,cases[dot],*models[name],out/f'seed7_dot{dot}_camera',True)))
    restored=sum(map(len,rows.values()));start=time.monotonic()
    prior=json.loads((a.out/'progress.json').read_text())['elapsed_s'] if a.resume else 0.
    with ProcessPoolExecutor(a.jobs) as pool:
        futures={pool.submit(attempt,job):name for name,job in jobs}
        for future in as_completed(futures):
            name=futures[future];r=future.result();r['variant']=name
            write_json(a.out/name/f"seed7_dot{r['dot']}.json",r);rows[name].append(r)
            count=sum(map(len,rows.values()));elapsed=time.monotonic()-start
            write_json(a.out/'progress.json',dict(completed=count,total=400,elapsed_s=prior+elapsed,eta_s=(400-count)*elapsed/(count-restored),last_variant=name,last_dot=r['dot']))
            print(f"{count}/400 {name} dot{r['dot']}: {r['status']}, {r['elapsed_s']:.1f}s",flush=True)
    for name,*_ in LINEUP:
        rr=sorted(rows[name],key=lambda r:ids.index(r['dot']));assert len(rr)==100 and {r['dot'] for r in rr}==set(ids)
        write_json(a.out/name/'seed7.json',rr)
        write_json(a.out/name/'summary.json',[{k:v for k,v in r.items() if k not in ('frames','log')} for r in rr])
        write_json(a.out/name/'completion.json',dict(attempts=100,expected=100,all_accounted=True,sum_case_wall_s=sum(r['elapsed_s'] for r in rr)))
    write_json(a.out/'completion.json',dict(attempts=400,expected=400,all_accounted=True,elapsed_s=prior+time.monotonic()-start))


if __name__=='__main__':main()
