"""Measurement harness only. Each child generates complete trajectories entirely in C++."""
import argparse,json,os,platform,subprocess,time,hashlib,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--executable',default='build-clang19/expert_compatible');p.add_argument('--repeat',type=int,default=2);p.add_argument('--tag',default='cpu_benchmark');p.add_argument('--workers',nargs='+',type=int,default=[1,2,4,8]);p.add_argument('--diagnostics',action='store_true');p.add_argument('--fork-workers',action='store_true');p.add_argument('--compact',action='store_true',help='Hash all outputs, retain first dataset, discard duplicate generated datasets after timing');p.add_argument('--driving-only',action='store_true');p.add_argument('--output-root',type=Path,help='Optional storage root, e.g. /dev/shm for a RAM filesystem');p.add_argument('--targets',type=Path);p.add_argument('--dynamic',action='store_true');args=p.parse_args()
root=Path(__file__).resolve().parents[1];os.chdir(root)
env={**os.environ,'EXPERT_SKIP_VISUALS':'1','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
(root/'results').mkdir(exist_ok=True)
records=[];expected_by_target={}
for workers in args.workers:
    out=(args.output_root.resolve() if args.output_root else root/'data')/args.tag/f'workers_{workers}'
    cmd=[str((root/args.executable).resolve()),'--workers',str(workers),'--repeat',str(args.repeat),'--binary','--out',str(out)]
    if args.targets:cmd.extend(['--targets',str(args.targets.resolve())])
    if args.dynamic:cmd.append('--dynamic')
    if args.driving_only:cmd.append('--driving-only')
    if args.diagnostics:cmd.append('--diagnostics')
    if args.fork_workers:cmd.append('--fork-workers')
    start=time.perf_counter();r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);wall=time.perf_counter()-start
    (root/'results'/f'{args.tag}_{workers}.log').write_text(r.stdout)
    if r.returncode:raise RuntimeError(r.stdout[-2000:])
    events=[]
    for line in r.stdout.splitlines():
        for prefix in ['RESULT ','BATCH ']:
            if prefix in line:
                try:events.append((prefix.strip(),json.loads(line.split(prefix,1)[1])))
                except json.JSONDecodeError:pass
    batch=[v for k,v in events if k=='BATCH'][-1]
    episodes=[v for k,v in events if k=='RESULT']
    if len(episodes)!=workers*args.repeat or not batch.get('ok'):
        raise RuntimeError('Missing episodes or failed batch')
    record={'workers':workers,'episodes':workers*args.repeat,'wall_seconds_external':wall,'seconds_per_saved_episode':wall/(workers*args.repeat),'episodes_per_second':workers*args.repeat/wall,'result_records':len(episodes),'batch':batch,'runs':episodes}
    records.append(record);print(json.dumps({k:v for k,v in record.items() if k not in ['runs','batch']}),flush=True)
    checks=[]
    for manifest in sorted(out.rglob('manifest.json')):
        folder=manifest.parent;hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(folder.glob('*.npy')) if f.name!='timing.npy'}
        metadata=json.loads(manifest.read_text());target=tuple(metadata['target_position_world_m'])
        first=target not in expected_by_target
        if first:expected_by_target[target]=hashes
        equal=hashes==expected_by_target[target]
        checks.append({'folder':os.path.relpath(folder,root),'numerical_arrays_equal':equal,'hashes':hashes,'manifest':json.loads(manifest.read_text())})
        if args.compact and equal and not first:shutil.rmtree(folder)
    if len(checks)!=workers*args.repeat or not all(c['manifest'].get('export_complete') for c in checks):
        raise RuntimeError('Missing or incomplete exports')
    record['repeat_checks']=checks;record['all_repeat_arrays_equal']=all(x['numerical_arrays_equal'] for x in checks)
    (root/'results'/f'{args.tag}.json').write_text(json.dumps({'machine':platform.platform(),'cpu_count':os.cpu_count(),'records':records},indent=2)+'\n')
