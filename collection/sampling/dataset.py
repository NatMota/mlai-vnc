"""Compact data collection for the 360-degree driving experiment."""
import gzip
import hashlib
import html
import io
import json
import os
import signal
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

RAW_FEATURES = [f'{kind}_j{j:02}' for kind in ('q_rad', 'qd_rad_s') for j in range(1, 5)]
FEATURES = RAW_FEATURES + ['forward_turn_rad', 'right_minus_left_turn_rad',
    'forward_speed_rad_s', 'right_minus_left_speed_rad_s', 'goal_sin_bearing', 'goal_cos_bearing']
WHEELS = ['front_right', 'front_left', 'back_left', 'back_right']
LABELS = [f'wheel_velocity_command_rad_s_{w}' for w in WHEELS]
SEED = 7


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)
    if path.name=='status.json':
        done=value.get('completed',0); total=value.get('requested',3000)
        eta=value.get('eta_seconds',0)/3600
        details=html.escape(json.dumps({k:value[k] for k in
            ('fit','epoch','unique_trajectories','successful_episodes','error') if k in value}))
        page=f'''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="10">
        <style>body{{font:16px system-ui;padding:16px;color:#18283b;background:#f5f8fc}}
        progress{{width:100%;height:24px}}pre{{white-space:pre-wrap}}a{{color:#1263bd}}</style>
        <h2>Driving 30 · {html.escape(value.get('stage','pending'))}</h2>
        <progress value="{done}" max="{total}"></progress><p>{done:,} / {total:,} actual executions</p>
        <p>Estimated collection time remaining: {eta:.1f} hours · retained archives: {value.get('archive_bytes',0)/1e6:.1f} MB</p>
        <pre>{details}</pre><small>Refreshes every 10 seconds. Last collection update: {html.escape(value.get('updated_utc','pending'))}</small>'''
        p=path.with_name('progress.html');t=p.with_suffix('.tmp');t.write_text(page);t.replace(p)


BEARINGS = list(range(12, 361, 12))


def split_plan():
    """30 whole-bearing groups: 21 train, 4 test, 5 final holdout."""
    rng=np.random.default_rng(SEED); records=[];cursor=0
    for sector in range(6):
        angles=rng.permutation(BEARINGS[sector*5:(sector+1)*5])
        labels=['train']*3+['test','holdout'] if sector<3 else ['train']*4+['test' if sector==3 else 'holdout']
        for angle,split in zip(angles,labels):
            fold=cursor%5 if split=='train' else -1
            if split=='train':cursor+=1
            t=np.deg2rad(angle)
            records.append(dict(bearing_deg=int(angle),split=split,fold=fold,target=[float(1.1*np.cos(t)),float(1.1*np.sin(t)),.5]))
    records.sort(key=lambda r:r['bearing_deg'])
    assert [sum(r['split']==s for r in records) for s in ['train','test','holdout']]==[21,4,5]
    return records


def load_records(output):
    path = Path(output)/'episodes.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def arrays_from_run(folder, bearing):
    """Extract only observed wheel state, labels and poses. No dynamics retained."""
    folder=Path(folder)
    def read(name):
        return np.load(folder/(name+'.npy')),json.loads((folder/(name+'.schema.json')).read_text())
    joint,js=read('joint_state');expert,es=read('base_controller');base,bs=read('base_state')
    def columns(a,s,names):return a[:,[s['columns'].index(n) for n in names]]
    step=columns(joint,js,['step']).flatten().astype(np.int32)
    time_s=columns(joint,js,['time_s']).flatten()
    assert np.array_equal(step,columns(expert,es,['step']).flatten())
    assert np.array_equal(time_s,columns(expert,es,['time_s']).flatten())
    assert np.array_equal(step,np.arange(len(step)))
    state=columns(joint,js,RAW_FEATURES)
    commands=columns(expert,es,LABELS)
    park=columns(expert,es,['navigation_done']).flatten()==es['categories']['navigation_done'].index('True') if 'True' in es['categories']['navigation_done'] else np.zeros(len(step),dtype=bool)
    ids=np.unique(np.r_[np.rint(np.arange(0,step[-1]*.001,1/30)/.001).astype(int),int(step[-1])])
    x=state[ids].copy();x[:,:4]-=state[0,:4]
    right,left=x[:,[0,3]].mean(1),x[:,[1,2]].mean(1)
    vr,vl=x[:,[4,7]].mean(1),x[:,[5,6]].mean(1)
    theta=np.deg2rad(bearing)
    x=np.column_stack([x,(right+left)/2,right-left,(vr+vl)/2,vr-vl,
        np.full(len(x),np.sin(theta)),np.full(len(x),np.cos(theta))]).astype(np.float32)
    poses=columns(base,bs,[f'position_m_{a}' for a in 'xyz']+[f'quaternion_{a}' for a in 'xyzw'])
    joints=columns(joint,js,[f'q_rad_j{i:02}' for i in range(1,18)])
    full=dict(wheel_state=state,wheel_commands=commands,park=park,step=step,time_s=time_s)
    sampled=dict(x=x,wheels=commands[ids].astype(np.float32),park=park[ids,None].astype(np.float32),
        step=step[ids],time_s=time_s[ids],base_pose=poses[ids].astype(np.float32),joint_positions=joints[ids].astype(np.float32))
    assert np.isfinite(x).all() and np.isfinite(commands).all()
    hasher=hashlib.sha256()
    # Exact equality of retained signals is required, including the 1 kHz driving
    # observations even for repetitions where that archive is not retained.
    for name,a in sorted({**full,**sampled}.items()):
        hasher.update(name.encode());hasher.update(str(a.shape).encode());hasher.update(a.tobytes())
    # full['park'] and step/time share names; separately include all full-rate fields.
    for name,a in sorted(full.items()):hasher.update(name.encode());hasher.update(a.tobytes())
    final_pose=dict(zip(bs['columns'],base[-1].tolist()))
    return sampled,full,hasher.hexdigest(),final_pose


def collect(native_root,output,repetitions=50,workers=8,limit=None,host="mint",repetition_offset=0):
    native_root,output=Path(native_root).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    for name in ('sampled','full_rate','logs'):(output/name).mkdir(exist_ok=True)
    plan=split_plan();save_json(output/'split_plan.json',plan)
    save_json(output/'targets.json',[r['target'] for r in plan])
    save_json(output/'retention.json',dict(requested_runs=30*repetitions,
        saved_budget_bytes=750_000_000,temporary_budget_bytes=500_000_000,
        all_runs='30 Hz FP32 training and replay; timestamp metadata retained',
        full_rate_runs='First 5 repetitions per bearing on each host (10 total), original float64 1 kHz driving fields',
        excluded=['mass matrices','Jacobians','dynamics diagnostics'],
        deduplication='Identical retained trajectories stored once; every execution has its own audit record'))
    records=load_records(output)
    assert [r['local_index'] for r in records]==list(range(len(records)))
    total=30*repetitions;end=min(total,len(records)+limit) if limit else total
    start_time=time.monotonic();initial_count=len(records)
    known={r['trajectory_hash'] for r in records}
    saved_bytes=sum(f.stat().st_size for f in output.rglob('*') if f.is_file())
    # Count the new environment too; leave 1 GB for preparation and reports.
    environment=output.parent/'.venv'
    environment_bytes=sum(f.stat().st_size for f in environment.rglob('*') if f.is_file()) if environment.exists() else 0
    archive_limit=750_000_000
    env=dict(os.environ,EXPERT_SKIP_VISUALS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    pixi=str(next(p for p in [Path.home()/'.local/bin/pixi',Path.home()/'.pixi/bin/pixi'] if p.exists()))
    peak_scratch=0
    def archive(path,arrays):
        nonlocal saved_bytes
        if path.exists():return
        buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);payload=buffer.getvalue()
        # Reserve 1 GB of the 5 GB saved-data budget for audit, prepared tensors,
        # trained models and notebook outputs. Never silently drop requested data.
        if saved_bytes+len(payload)>archive_limit:raise RuntimeError('Archive quota reached; collection stopped before exceeding the 6 GB total budget.')
        with np.load(io.BytesIO(payload)) as saved:
            assert all(np.array_equal(v,saved[k],equal_nan=True) for k,v in arrays.items())
        tmp=path.with_suffix('.tmp');tmp.write_bytes(payload);tmp.replace(path)
        saved_bytes+=len(payload)
    batch_size=24 if host=="mint" else 16
    for first in range(len(records),end,batch_size):
        count=min(batch_size,end-first)
        active_workers=min(workers,count)
        while count%active_workers:active_workers-=1
        repeats_per_worker=count//active_workers
        scratch_root=Path('/dev/shm') if Path('/dev/shm').is_dir() else output
        scratch=scratch_root/f'driving30-{host}-{os.getpid()}-{first:05}'
        if shutil.disk_usage(scratch_root).free<1_000_000_000:raise RuntimeError('Less than 1 GB scratch space available')
        assert not scratch.exists()
        cmd=[pixi,'run','collect-driving','--workers',str(active_workers),'--repeat',str(repeats_per_worker),'--fork-workers','--binary',
            '--episode-offset',str(first),'--targets',str(output/'targets.json'),'--out',str(scratch)]
        log_path=output/'logs'/f'batch_{first:05}.log'
        with log_path.open('w') as log:
            process=subprocess.Popen(cmd,cwd=native_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while process.poll() is None:
                if scratch.exists():
                    scratch_bytes=sum(f.stat().st_size for f in scratch.rglob('*') if f.is_file())
                    peak_scratch=max(peak_scratch,scratch_bytes)
                    if scratch_bytes>450_000_000:
                        os.killpg(process.pid,signal.SIGTERM);process.wait()
                        raise RuntimeError('Scratch budget reached; stopped workers before 500 MB.')
                time.sleep(.5)
            if process.returncode:raise RuntimeError(f'Collector failed; see {log_path}')
        manifests=list(scratch.rglob('manifest.json'));assert len(manifests)==count
        scratch_bytes=sum(f.stat().st_size for f in scratch.rglob('*') if f.is_file())
        peak_scratch=max(peak_scratch,scratch_bytes)
        assert scratch_bytes<500_000_000
        expected={BEARINGS[i%30]:i for i in range(first,first+count)}
        batch=[]
        for file in manifests:
            m=json.loads(file.read_text());assert m['export_complete'] and m['dt_s']==.001
            target=m['target_position_world_m'];bearing=round(np.rad2deg(np.arctan2(target[1],target[0])))%360 or 360
            index=expected.pop(bearing);rep=index//30+repetition_offset
            sampled,full,fingerprint,final=arrays_from_run(file.parent,bearing)
            if m['termination']=='parked_command':
                assert sampled['park'].sum()==1 and sampled['park'][-1,0]==1
                assert np.all(sampled['wheels'][-1]==0)
            sample_path=Path('sampled')/(fingerprint+'.npz');full_path=Path('full_rate')/(fingerprint+'.npz')
            archive(output/sample_path,sampled)
            if index//30<5:archive(output/full_path,full)
            known.add(fingerprint)
            m['final_pre_step_base_state']=final
            batch.append(dict(local_index=index,global_index=rep*30+BEARINGS.index(bearing),source_host=host,noise_enabled=False,episode_id=f'{rep:03}_{bearing:03}',repetition=rep,
                bearing_deg=bearing,trajectory_hash=fingerprint,archive=str(sample_path),
                full_rate_archive=str(full_path) if index//30<5 else None,manifest=m))
        assert not expected
        batch.sort(key=lambda r:r['local_index'])
        with (output/'episodes.jsonl').open('a') as f:
            for row in batch:f.write(json.dumps(row)+'\n')
            f.flush();os.fsync(f.fileno())
        records.extend(batch)
        with gzip.open(log_path.with_suffix('.log.gz'),'wb') as f:f.write(log_path.read_bytes())
        log_path.unlink()
        shutil.rmtree(scratch)
        elapsed=time.monotonic()-start_time;rate=(len(records)-initial_count)/elapsed
        status=dict(stage='collecting',completed=len(records),requested=total,unique_trajectories=len(known),
            elapsed_this_process_s=elapsed,episodes_per_second=rate,eta_seconds=(total-len(records))/rate,
            archive_bytes=saved_bytes,peak_scratch_bytes=peak_scratch,
            environment_bytes=environment_bytes,
            successful_episodes=sum(r['manifest']['termination']=='parked_command' for r in records),
            updated_utc=datetime.now(timezone.utc).isoformat())
        save_json(output/'status.json',status)
        if first%300<batch_size:print(json.dumps(status),flush=True)
    status=json.loads((output/'status.json').read_text())
    status['stage']='collected' if len(records)==total else 'preflight_complete'
    save_json(output/'status.json',status)


def prepare(output):
    import pandas as pd
    output=Path(output);records=load_records(output);plan={r['bearing_deg']:r for r in split_plan()}
    used=set();inputs=[];wheels=[];parks=[];metadata=[];audit=[]
    for row in records:
        group=row['bearing_deg'];key=(group,row['trajectory_hash'])
        audit.append(dict(episode_id=row['episode_id'],bearing_deg=group,split=plan[group]['split'],
            termination=row['manifest']['termination'],source_host=row['source_host'],trajectory_hash=row['trajectory_hash'],
            full_rate_retained=row['full_rate_archive'] is not None))
        if key in used or row['manifest']['termination']!='parked_command':continue
        used.add(key)
        with np.load(output/row['archive']) as a:
            x,y,park=a['x'],a['wheels'],a['park']
            assert x.shape[1]==14 and x.dtype==np.float32
            assert park.sum()==1 and park[-1,0]==1
            inputs.append(x);wheels.append(y);parks.append(park)
            for i,(step,t) in enumerate(zip(a['step'],a['time_s'])):
                metadata.append(dict(episode_id=row['episode_id'],bearing_deg=group,split=plan[group]['split'],
                    fold=plan[group]['fold'],source_host=row['source_host'],step=int(step),time_s=float(t),row_in_episode=i))
    assert used,'No successful demonstrations; inspect the audit.'
    np.savez_compressed(output/'training_data.npz',x=np.concatenate(inputs),wheels=np.concatenate(wheels),park=np.concatenate(parks))
    pd.DataFrame(metadata).to_csv(output/'rows.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'episode_audit.csv',index=False)
    save_json(output/'features.json',dict(inputs=FEATURES,wheel_outputs=LABELS,dtype='float32',input_scaling='none',
        sample_hz=30,goal_source='Explicit requested initial target bearing; not camera observations',
        timestamps='metadata, not network inputs',split_counts={'train':21,'test':4,'holdout':5},noise_enabled=False,platforms='Mac and Mint; engine identity and source host retained',unique_trajectories_used=len(used),
        terminal_label='First expert navigation_done, inclusive; timeout episodes excluded and reported',
        retention='30 Hz training + poses for all runs; 1 kHz driving for first five per host per bearing',
        omitted='No dynamics diagnostics, mass matrices or Jacobians',
        duplicates='Exact repeated trajectories remain separate in the audit; fit each unique trajectory once'))
    return pd.DataFrame(audit),pd.DataFrame(metadata)
