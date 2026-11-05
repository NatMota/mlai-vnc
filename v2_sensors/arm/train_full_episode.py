"""Train sensor-only arm torques from reset, including learned holding before PARK."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from train_golden import load_data, train_one, write_json, sha
import sensors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','out','driving'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=1200)
    p.add_argument('--stop-epochs',type=int,default=600)
    p.add_argument('--arm-hidden',type=int,default=128)
    p.add_argument('--arm-scaling',choices=['physical','standard'],default='physical')
    p.add_argument('--lr',type=float,default=.01)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--chunk',type=int,default=32768)
    p.add_argument('--mask-park-seconds',type=float,default=0.)
    p.add_argument('--allowed-conditions',nargs='+',default=['baseline'])
    p.add_argument('--extra-training',type=Path,nargs='*',default=[])
    p.add_argument('--train-driving',action='store_true')
    p.add_argument('--dagger',type=Path,nargs='*',default=[])
    p.add_argument('--driving-epochs',type=int,default=300)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    a.save_epochs=(300,600,1200)
    started=time.monotonic();records,audit=load_data(a.data,driving_height=True,full_arm=True,allowed_conditions=a.allowed_conditions)
    extra_audits=[]
    for root in a.extra_training:
        additional,extra_audit=load_data(root,driving_height=True,full_arm=True,allowed_conditions=a.allowed_conditions)
        records['train'].extend(additional['train']);extra_audits.append(dict(root=str(root),audit=extra_audit))
    # The existing driving model remains frozen; only 18-input arm networks are trained.
    for r in records['train']+records['val']:
        x,y,d=r['arm'];assert x.shape[1]==18
        flag=x[:,-1];assert flag[0]==0 and flag[-1]==1
        assert np.all(np.diff(flag)>=0)
    if a.mask_park_seconds<0:raise ValueError('Mask duration must be nonnegative')
    if a.mask_park_seconds:
        masked=[]
        for r in records['train']:
            for key,steps in (('drive',r['drive_steps']),('arm',r['arm_steps'])):
                keep=np.abs(steps-r['parking_step'])>a.mask_park_seconds*1000
                if not keep.any():raise ValueError('Mask removed an entire episode')
                masked.append(dict(dot=r['dot'],part=key,removed=int((~keep).sum())))
                r[key]=tuple(v[keep] for v in r[key])
        audit['mask']=dict(half_window_seconds=a.mask_park_seconds,rows=masked,applied_after_causal_features=True,validation_masked=False)
    if a.train_driving:
        for r in records['train']+records['val']:
            x,y,z=r['drive'];r['drive']=(x[:,sensors.DRIVING_NO_SPEED_COLS],y,z)
    if a.dagger:
        if not a.train_driving:raise ValueError('DAgger here updates wheel, arm and STOP networks together')
        train_ids=set(json.loads((a.data/'split.json').read_text())['train']);dagger_audit=[]
        for path in sorted(p for root in a.dagger for p in root.glob('dot*.npz')):
            with np.load(path,allow_pickle=False) as z:
                prov=json.loads(str(z['provenance']));dot=int(prov['dot'])
                assert prov['split']=='train' and dot in train_ids
                dx,dy=z['drive_x'],z['drive_y'];ax,ay=z['arm_x'],z['arm_y']
                assert dx.shape[1]==11 and ax.shape[1]==18 and len(ax)>0 and len(dx)>0
                distance=np.clip(z['arm_distance_m'],0,.25)/.25
                for v in (dx,dy,ax,ay,distance):assert np.isfinite(v).all()
                records['train'].append(dict(dot=dot,dagger=True,drive=(dx,dy,np.zeros(len(dx),np.float32)),
                    drive_aux_mask=np.zeros(len(dx),np.float32),arm_aux_mask=np.zeros(len(ax),np.float32),arm=(ax,ay,distance)))
                dagger_audit.append(dict(dot=dot,path=str(path),sha256=sha(path),arm_rows=len(ax),drive_rows=len(dx),
                    park_supervision=False,stop_supervision=False,oracle_internal_inputs=False))
        if not dagger_audit:raise ValueError('No DAgger records found')
        audit['dagger']=dagger_audit
    cfg=vars(a).copy();cfg={k:str(v) if isinstance(v,Path) else [str(x) if isinstance(x,Path) else x for x in v] if isinstance(v,list) else v for k,v in cfg.items()}
    cfg.update(driving_sha256=sha(a.driving),input_names=[f'arm_q_{j}' for j in range(7)]+[f'arm_qd_{j}' for j in range(7)]+['camera_x','camera_y','camera_z','own_park'],
        execution='learned arm every 1 ms from reset; no PD or gravity term; STOP only after own PARK',
        split_sha256=audit['split_sha256'],trainer_sha256=sha(__file__),test_files_opened=0)
    write_json(a.out/'config.json',cfg);write_json(a.out/'data_audit.json',audit)
    if extra_audits:write_json(a.out/'extra_training_audits.json',extra_audits)
    if a.train_driving:
        arm_epochs=a.epochs;a.epochs=a.driving_epochs
        train_one(records,'driving',a.seed,a);a.epochs=arm_epochs
    else:
        shutil.copy2(a.driving,a.out/f'driving_seed{a.seed}.npz')
        for suffix in ('history','summary'):
            src=a.driving.with_name(f'driving_seed{a.seed}_{suffix}.json')
            shutil.copy2(src,a.out/src.name)
    torque,tm=train_one(records,'torque',a.seed,a)
    stop,sm=train_one(records,'stop',a.seed,a)
    combined=dict(torque)
    combined.update({'stopnet.'+k:v for k,v in stop.items() if k.endswith(('.weight','.bias')) or k.startswith('export_')})
    combined.update(info_stopnet='18->32->16->1 relu',info_stopnet_epochs=np.int64(sm['selected_epoch']))
    np.savez(a.out/f'reaching_seed{a.seed}.npz',**combined)
    write_json(a.out/'summary.json',dict(seed=a.seed,torque=tm,stop=sm,elapsed_s=time.monotonic()-started,status='trained; full simulation pending'))


if __name__=='__main__':main()
