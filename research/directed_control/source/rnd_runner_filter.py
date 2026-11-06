"""Isolated matched EGL development: parameter and activation variants, validation only."""
import argparse,json,os,sys,time,hashlib,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
R=Path(__file__).parent
os.environ.update(NATIVE_ROOT=str(R/'native'),CURRICULUM_BINARY=str(R/'native/sensor_curriculum'),CAMERA_URDF=str(R/'native/assets/summit_xl_arm_description/summit_xl_complete.urdf'),CAMERA_RENDERER='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(R/'runtime/v2_sensors/arm'))
import full_run
from validate_golden import case_for
full_run.PIXI=R/'exec_native'
ORIGINAL=full_run.arm_policy
IDS=[6,16,23,45,55,80]
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,allow_nan=False));t.replace(p)
def run(job):
 name,dot,out,condition=job;meta=json.loads((R/'rnd_candidates'/f'{name}.json').read_text());slope=meta['arm_leaky_slope']
 wa=dict(np.load(R/'rnd_candidates'/f'{name}.npz',allow_pickle=False));wd=dict(np.load(R/'baseline/driving_seed7.npz',allow_pickle=False))
 inputs=[];torques=[];stops=[];filter_state=None;measured=[]
 def policy(w,raw):
  nonlocal filter_state
  measured.append(raw.copy())
  window=float(meta.get('velocity_ema_ms',0))
  if window:
   if filter_state is None:filter_state=raw[7:14].copy()
   else:filter_state+=(raw[7:14]-filter_state)/(1+window)
   raw=raw.copy();raw[7:14]=filter_state
  tau,stop=ORIGINAL(w,raw)
  if slope:
   x=raw
   if 'export_input_mean' in w:x=(x-w['export_input_mean'])/w['export_input_scale']
   for layer in ['shared.0','shared.2']:
    z=x@w[layer+'.weight'].T+w[layer+'.bias'];x=np.where(z>=0,z,slope*z)
   tau=x@w['arm.weight'].T+w['arm.bias']
   if 'export_output_scale' in w:tau=tau*w['export_output_scale']+w['export_output_mean']
  # Explicit hand-specified viscous-feedback architecture, not learned damping.
  gain=float(meta.get('velocity_skip_gain',0))
  if gain:tau=tau-gain*raw[7:14]
  inputs.append(raw.copy());torques.append(tau.copy());stops.append(stop)
  return tau,stop
 full_run.arm_policy=policy
 case,_=case_for(R/'data',dot,'val');start=time.monotonic();folder=Path(out)/name
 try:
  row=full_run.one(wd,wa,dot,True,case=case,condition=condition)
  row.update(name=name,dot=dot,elapsed_s=time.monotonic()-start,renderer='egl',arm_activation_negative_slope=slope)
  folder.mkdir(parents=True,exist_ok=True)
  np.savez_compressed(folder/f'trace_{dot}.npz',x=np.asarray(inputs),x_measured=np.asarray(measured),tau_raw=np.asarray(torques),STOP_distance=np.asarray(stops),step=np.arange(len(inputs)))
 except Exception as e:row=dict(name=name,dot=dot,error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start,success=False)
 save(folder/f'run_{dot}.json',row)
 return {k:v for k,v in row.items() if k not in ('frames','log')}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--names',nargs='+',required=True);p.add_argument('--ids',nargs='+',type=int,default=IDS);p.add_argument('--jobs',type=int,default=2);p.add_argument('--condition');a=p.parse_args()
 valid=set(json.loads((R/'data/split.json').read_text())['val']);assert set(a.ids)<=valid
 condition=json.loads(Path(a.condition).read_text()) if a.condition else {};out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 jobs=[];rows=[]
 for name in a.names:
  for dot in a.ids:
   f=out/name/f'run_{dot}.json'
   if f.exists():rows.append({k:v for k,v in json.loads(f.read_text()).items() if k not in ('frames','log')})
   else:jobs.append((name,dot,str(out),condition))
 cfg=dict(names=a.names,ids=a.ids,condition=condition,renderer='egl',jobs=a.jobs,selection='validation-only exploratory fixed candidates; no optimizer or test data',native_sha=hashlib.sha256((R/'native/sensor_curriculum').read_bytes()).hexdigest(),model_sha={n:hashlib.sha256((R/'rnd_candidates'/f'{n}.npz').read_bytes()).hexdigest() for n in a.names})
 if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==cfg
 else:save(out/'config.json',cfg)
 start=time.monotonic()
 with ProcessPoolExecutor(a.jobs) as pool:
  futures=[pool.submit(run,j) for j in jobs]
  for f in as_completed(futures):
   row=f.result();rows.append(row);save(out/'summary.json',rows)
   save(out/'progress.json',dict(completed=len(rows),total=len(a.names)*len(a.ids),elapsed_s=time.monotonic()-start))
   print(json.dumps({k:row.get(k) for k in ['name','dot','success','min_mm','at_network_stop_mm','end_mm','elapsed_s','error']}),flush=True)
   if 'error' in row:raise RuntimeError('Technical failure; preserve evidence and investigate')
 save(out/'completion.json',dict(attempts=len(rows),elapsed_s=time.monotonic()-start))
if __name__=='__main__':main()
