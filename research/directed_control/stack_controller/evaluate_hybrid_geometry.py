"""Original EGL full-policy validation of learned diagonal damping."""
import argparse,hashlib,json,os,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
os.environ.update(NATIVE_ROOT=str(ROOT/'native'),CURRICULUM_BINARY=str(ROOT/'native/sensor_curriculum'),
 CAMERA_URDF=str(ROOT/'native/assets/summit_xl_arm_description/summit_xl_complete.urdf'),CAMERA_RENDERER='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(ROOT/'runtime/v2_sensors/arm'))
import full_run_stack_ticktruth as full_run
from validate_golden import case_for
from hybrid_geometry_runtime import GeometryPolicy
full_run.PIXI=ROOT/'exec_native';ORIGINAL=full_run.arm_policy
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,allow_nan=False));tmp.replace(p)
def run_one(job):
    dot,models,out=job;models=Path(models);out=Path(out)
    base=dict(np.load(ROOT/'baseline/reaching_seed7.npz'));drive=dict(np.load(ROOT/'baseline/driving_seed7.npz'))
    pw=dict(np.load(ROOT/'models/position.npz'));tw=dict(np.load(ROOT/'models/torque.npz'))
    policy=GeometryPolicy(pw,tw);inputs=[];torques=[];stops=[];states=[]
    base['ttp_thresh']=np.array(.005)
    def recorded(w,x):
        tau,stop=policy(w,x);inputs.append(x.copy());torques.append(tau.copy());stops.append(stop);states.append(policy.last);return tau,stop
    full_run.MOTOR_TRACE=[]
    full_run.arm_policy=recorded;case,_=case_for(ROOT/'data',dot,os.environ.get('EVALUATION_SPLIT','val'));start=time.monotonic()
    try:
        row=full_run.one(drive,base,dot,True,case=case,camera_dir=out/f'camera_{dot}' if os.environ.get('RECORD_CAMERA')=='1' else None,condition=json.loads((ROOT/"condition.json").read_text()))
        row.update(dot=dot,name=os.environ['GEOMETRY_CAMERA']+'_q6half_timestamp_reference_recovery',elapsed_s=time.monotonic()-start,renderer='egl',architecture='engineered learned-geometry servo and identified torque')
    except Exception as e:row=dict(dot=dot,name='learned_diagonal_damping',success=False,status='execution_failure',error=repr(e),traceback=traceback.format_exc(),elapsed_s=time.monotonic()-start)
    np.savez_compressed(out/f'trace_{dot}.npz',x=np.asarray(inputs),tau_raw=np.asarray(torques),STOP_distance=np.asarray(stops),
        cartesian_damping=np.array([s['cartesian_damping'] for s in states]),reference=np.array([s['reference'] for s in states]),velocity=np.array([s['velocity'] for s in states]),position=np.array([s['position'] for s in states]),jacobian_condition=np.array([s['condition'] for s in states]),feedforward=np.array([s['feedforward'] for s in states]),filtered_velocity=np.array([s['filtered_velocity'] for s in states]),blocked=np.array([s['blocked'] for s in states]),step=np.arange(len(inputs)),motor_command_step=np.array([m['command_step'] for m in full_run.MOTOR_TRACE]),motor_returned_step=np.array([m['returned_step'] for m in full_run.MOTOR_TRACE]),motor_requested=np.array([m['requested'] for m in full_run.MOTOR_TRACE]),motor_pre_effect=np.array([m['pre_effect'] for m in full_run.MOTOR_TRACE]),motor_sent=np.array([m['sent'] for m in full_run.MOTOR_TRACE]),motor_extra=np.array([m['extra'] for m in full_run.MOTOR_TRACE]),motor_true_distance_m=np.array([m['true_distance_m'] for m in full_run.MOTOR_TRACE]),motor_returned_time_s=np.array([m['returned_time_s'] for m in full_run.MOTOR_TRACE]))
    save(out/f'run_{dot}.json',row);return {k:v for k,v in row.items() if k not in ['frames','log']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--models',type=Path,default=ROOT/'models');p.add_argument('--camera',choices=['original','calibrated_arm_only'],default='original');p.add_argument('--out',type=Path,default=ROOT/'validation');p.add_argument('--ids',type=int,nargs='+',default=[6,16,23,45,55,80]);p.add_argument('--jobs',type=int,default=2);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--record-camera',action='store_true');a=p.parse_args()
    assert set(a.ids)<=set(json.loads((ROOT/'data/split.json').read_text())[a.split])
    os.environ['GEOMETRY_CAMERA']=a.camera
    os.environ['EVALUATION_SPLIT']=a.split;os.environ['RECORD_CAMERA']='1' if a.record_camera else '0'
    cfg=dict(ids=a.ids,camera=a.camera,renderer='egl',native_sha256=sha(ROOT/'native/sensor_curriculum'),baseline_sha256=sha(ROOT/'baseline/reaching_seed7.npz'),position_sha256=sha(ROOT/'models/position.npz'),torque_sha256=sha(ROOT/'models/torque.npz'),runtime_sha256=sha(ROOT/'hybrid_geometry_runtime.py'),full_run_sha256=sha(Path(full_run.__file__)),envelope_sha256=sha(ROOT/'models/hybrid_bounds.npz'),condition=json.loads((ROOT/"condition.json").read_text()),case_sha256={str(dot):sha(ROOT/"data/expert"/a.split/f"run_{dot}.npz") for dot in a.ids},test=a.split=='test',split=a.split,record_camera=a.record_camera)
    a.out.mkdir(exist_ok=True)
    if (a.out/'config.json').exists():
        old=json.loads((a.out/'config.json').read_text());assert all(old[k]==v for k,v in cfg.items() if k!='ids')
    else:save(a.out/'config.json',cfg)
    jobs=[];rows=[]
    for dot in a.ids:
        path=a.out/f'run_{dot}.json'
        if path.exists():rows.append({k:v for k,v in json.loads(path.read_text()).items() if k not in ['frames','log']})
        else:jobs.append((dot,str(a.models),str(a.out)))
    with ProcessPoolExecutor(a.jobs) as pool:
        for f in as_completed([pool.submit(run_one,j) for j in jobs]):
            row=f.result();rows.append(row);save(a.out/'summary.json',rows)
            print(json.dumps({k:row.get(k) for k in ['dot','success','status','min_mm','at_network_stop_mm','end_mm','elapsed_s']}),flush=True)
    save(a.out/'completion.json',dict(attempted=len(rows),expected=len(a.ids),status='complete exploratory validation'))
if __name__=='__main__':main()
