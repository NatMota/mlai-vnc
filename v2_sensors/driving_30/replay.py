"""Reproduce frozen 30 Hz driving policies in the collection engine and view them."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = DATA / 'replay'


def native_root():
    for path in (Path(os.environ.get('NATIVE_ROOT', ROOT.parents[1]/'collection/native')), ROOT.parents[1]/'collection/native'):
        if (path/'assets').exists():
            return path.resolve()
    raise FileNotFoundError('collection/native not found; set NATIVE_ROOT.')


def rotations(rpy):
    r,p,y=rpy
    cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
        [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])


def geometry(native, joint_names):
    tree=ET.parse(native/'assets/summit_xl_arm_description/summit_xl_complete.urdf').getroot()
    joints=[]
    for j in tree.findall('joint'):
        origin=j.find('origin');a=j.find('axis')
        xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ')
        rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
        joints.append(dict(parent=j.find('parent').get('link'),child=j.find('child').get('link'),
            origin=np.column_stack([rotations(rpy),xyz]).ravel().tolist(),
            axis=np.fromstring(a.get('xyz') if a is not None else '1 0 0',sep=' ').tolist(),
            index=joint_names.index(j.get('name')) if j.get('name') in joint_names else -1))
    children={j['child'] for j in joints}
    root=next(l.get('name') for l in tree.findall('link') if l.get('name') not in children)
    ordered=[];known={root}
    while joints:
        ready=[j for j in joints if j['parent'] in known]
        assert ready, 'Disconnected URDF tree'
        for j in ready:ordered.append(j);known.add(j['child']);joints.remove(j)
    return dict(root=root,joints=ordered,meshes=json.loads((ROOT/'replay_assets/meshes.json').read_text()))


def fk(frame, geom):
    x,y,z,w=np.asarray(frame['quaternion'])/np.linalg.norm(frame['quaternion'])
    base=np.eye(4)
    base[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    base[:3,3]=frame['base'];poses={geom['root']:base}
    for j in geom['joints']:
        origin=np.eye(4);origin[:3,:]=np.array(j['origin']).reshape(3,4)
        motion=np.eye(4)
        if j['index']>=0:
            axis=np.array(j['axis']);axis/=np.linalg.norm(axis)
            x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            q=frame['joints'][j['index']]
            motion[:3,:3]=np.eye(3)+np.sin(q)*K+(1-np.cos(q))*(K@K)
        poses[j['child']]=poses[j['parent']]@origin@motion
    return poses


def validate_and_summarise(payload, geom, *, checkpoint=None, output=None):
    from dataset import load_records
    records=load_records(DATA)
    references={r['bearing_deg']:r for r in records if r['source_host']=='mint' and r['repetition']==0}
    checkpoint=Path(checkpoint) if checkpoint is not None else DATA/'training/final.npz'
    output=Path(output) if output is not None else OUT
    weights=np.load(checkpoint)
    rows=[];pose_error=0.;network_error=0.;kinematic_error=0.
    for run in payload['runs']:
        model,expert=run['model'],run['expert']
        assert expert['park'] is not None and expert['termination']=='parked'
        assert expert['final']['step']==expert['park']['step']+500
        if model['park'] is None:
            assert model['final']['step']==expert['park']['step']+3000
            assert model['termination']=='no_park_within_grace_period'
        else:
            assert model['park']['step']<=expert['park']['step']+3000
            assert model['final']['step']==model['park']['step']+500
        reference=references[run['bearing_deg']]
        assert payload['engine_sha256']==reference['manifest']['default_engine_binary_sha256']
        archived=np.load(DATA/reference['archive'])
        assert abs(expert['park']['time_s']-archived['time_s'][-1])<1e-8
        by_step={f['step']:f for f in expert['frames']}
        for step,base,joints in zip(archived['step'],archived['base_pose'],archived['joint_positions']):
            f=by_step[int(step)]
            pose_error=max(pose_error,float(np.max(np.abs(np.r_[f['base'],f['quaternion']]-base))),float(np.max(np.abs(np.array(f['joints'])-joints))))
        # Independent NumPy inference against the native FP32 rollout at its
        # actual decision states, not only against training examples.
        angle=np.deg2rad(run['bearing_deg'])
        initial=np.array(model['frames'][0]['joints'][:4])
        for f in model['frames']:
            if not f['decision']:continue
            if f['parked'] and model['park'] and f['step']>model['park']['step']:continue
            q=np.array(f['joints'][:4])-initial;v=np.array(f['wheel_velocity'])
            r,l=q[[0,3]].mean(),q[[1,2]].mean();vr,vl=v[[0,3]].mean(),v[[1,2]].mean()
            x=np.r_[q,v,(r+l)/2,r-l,(vr+vl)/2,vr-vl,np.sin(angle),np.cos(angle)].astype(np.float32)
            for layer in ('shared.0','shared.2'):
                x=np.maximum(x@weights[layer+'.weight'].T+weights[layer+'.bias'],0)
            prob=float(1/(1+np.exp(-(x@weights['park.weight'].T+weights['park.bias'])[0])))
            network_error=max(network_error,abs(prob-f['park_probability']))
            if not f['parked']:
                predicted=np.clip(x@weights['wheels.weight'].T+weights['wheels.bias'],-3,3)
                network_error=max(network_error,float(np.max(np.abs(predicted-f['command']))))
        for side in (model,expert):
            for f in side['frames'][::max(1,len(side['frames'])//5)]:
                ee=fk(f,geom)['robot_j2s7s300_end_effector'][:3,3]
                kinematic_error=max(kinematic_error,float(np.max(np.abs(ee-f['ee']))))
        def angle_error(a,b):return float(np.rad2deg(abs((a-b+np.pi)%(2*np.pi)-np.pi)))
        p,e=model['park'],expert['park'];mfinal,efinal=model['final'],expert['final']
        row=dict(split=run['split'],bearing_deg=run['bearing_deg'],parked=p is not None,
            model_park_s=p['time_s'] if p else np.nan,expert_park_s=e['time_s'],
            park_xy_error_m=float(np.linalg.norm(np.array(p['base'][:2])-e['base'][:2])) if p else np.nan,
            park_yaw_error_deg=angle_error(p['yaw'],e['yaw']) if p else np.nan,
            final_xy_error_m=float(np.linalg.norm(np.array(mfinal['base'][:2])-efinal['base'][:2])),
            final_yaw_error_deg=angle_error(mfinal['yaw'],efinal['yaw']),
            model_final_speed_m_s=mfinal['speed_m_s'],expert_final_speed_m_s=efinal['speed_m_s'],
            clipped_decisions=model['clipped_decisions'])
        row['park_time_delta_s']=row['model_park_s']-row['expert_park_s']
        run['metrics']={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
        rows.append(row)
    assert pose_error<1e-5, ('Expert replay differs from archived collection',pose_error)
    assert network_error<2e-5, ('Native policy differs from NumPy inference',network_error)
    assert kinematic_error<1e-7, ('Visual kinematics do not match simulator',kinematic_error)
    checks=dict(max_archived_expert_state_error=pose_error,max_numpy_native_prediction_error=network_error,
        max_visual_fk_error_m=kinematic_error,model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        evaluated_bearings=len(rows),holdout_untouched=not any(r['split']=='holdout' for r in payload['runs']),noise_enabled=False,
        policy='30 Hz FP32; wheel output clipped to expert range [-3,3] rad/s; park >=0.5 latched',
        physics='Original Mint collection engine and physical settings at 1 kHz; arm held with existing controller',
        comparison='Expert ends on PARK plus 0.5 s drift. Model has 3 extra seconds after expert PARK to park; a successful park gets 0.5 s drift. Expert has no fixed-duration cap.',
        reference_tail='Expert archive ends at park command. Reproduced prefix verified against archive, then extended by 0.5 s with zero wheel commands and arm hold.')
    (output/'checks.json').write_text(json.dumps(checks,indent=2))
    return pd.DataFrame(rows)


def reproduce(force=False):
    """Use saved physics rollouts, or regenerate them with force=True. No training."""
    from dataset import split_plan
    native=native_root();OUT.mkdir(exist_ok=True)
    request=dict(termination_rule='expert_park_plus_3s',cases=[p for p in split_plan() if p['split'] in ('train','test')],
        weights={k:v.tolist() for k,v in np.load(DATA/'training/final.npz').items()})
    path=OUT/'request.json'
    changed=path.exists() and json.loads(path.read_text())!=request
    if changed and not force:raise RuntimeError('Checkpoint or cases changed. Use force=True to regenerate.')
    path.write_text(json.dumps(request))
    raw=OUT/'rollouts.json'
    if force or not raw.exists():
        from datetime import datetime
        if raw.exists():raw.rename(OUT/('rollouts-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'.json'))
        pixi=next(p for p in [Path.home()/'.local/bin/pixi',Path.home()/'.pixi/bin/pixi'] if p.exists())
        env=dict(os.environ,EXPERT_SKIP_VISUALS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        with (OUT/'run.log').open('w') as log:
            subprocess.run([str(pixi),'run','build-clang19/policy_replay',str(native/'assets'),str(path),str(raw)],
                cwd=native,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    payload=json.loads(raw.read_text())
    assert payload['termination_rule']=='expert_park_plus_3s', 'Regenerate old fixed-duration rollouts with force=True'
    geom=geometry(native,payload['joint_names'])
    metrics=validate_and_summarise(payload,geom);metrics.to_csv(OUT/'metrics.csv',index=False)
    payload['geometry']=geom
    # Keep trajectories and true times, with lightweight meshes reused from the
    # earlier notebook. The browser animates poses; it never invents motion.
    page=(ROOT/'replay_template.html').read_text().replace('__PLOTLY__',(ROOT/'replay_assets/plotly.min.js').read_text()).replace('__DATA__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
    (OUT/'viewer.html').write_text(page)
    return metrics


def show():
    from IPython.display import IFrame,display
    display(IFrame('/files/driving_30/data/replay/viewer.html',width='100%',height=810))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true')
    args=parser.parse_args();print(reproduce(args.force).to_string(index=False))
