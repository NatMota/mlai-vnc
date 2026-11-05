"""Validation-only STOP recovery pilot; never opens test trajectories."""
import argparse
import fcntl
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
import train_golden as tg


def diagnose(runs, weights):
    rows=[]
    for r in runs:
        x,_,label=r['arm']; truth=r['distance_m']; pred=tg.infer_export(weights,x,'stop')*.25
        parked=x[:,-1]>0.5; near=parked & (truth<=.02)
        def first(mask):
            at=np.flatnonzero(mask)
            return None if not len(at) else int(at[0])
        hit=first(parked & (pred<=.01)); truehit=first(parked & (truth<=.01))
        rows.append(dict(dot=int(r['dot']),source=r['source'],rows=len(x),parked_rows=int(parked.sum()),
            mae_mm=float(np.mean(np.abs(pred-truth))*1000),clipped_label_mae_mm=float(np.mean(np.abs(pred-label*.25))*1000),
            near20_rows=int(near.sum()),near20_mae_mm=float(np.mean(np.abs(pred[near]-truth[near]))*1000) if near.any() else None,
            first_predicted_10mm_step=hit,first_true_10mm_step=truehit,
            true_mm_at_first_prediction=None if hit is None else float(truth[hit]*1000),
            predicted_mm_at_first_truth=None if truehit is None else float(pred[truehit]*1000),
            threshold_confusion={str(mm):dict(tp=int(np.sum(parked&(pred<=mm/1000)&(truth<=.01))),fp=int(np.sum(parked&(pred<=mm/1000)&(truth>.01))),fn=int(np.sum(parked&(pred>mm/1000)&(truth<=.01)))) for mm in (5,10,15,20)}))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    cfg=json.loads((a.reference/'config.json').read_text());data=Path(cfg['data'])
    records,audit=tg.load_data(data,driving_height=True,full_arm=True)
    for split,runs in records.items():
        for r in runs:
            r['source']='original';
            with np.load(data/'expert'/split/f"run_{r['dot']}.npz",allow_pickle=False) as z:r['distance_m']=z['hand_distance_expert'].copy()
    train_ids=set(json.loads((data/'split.json').read_text())['train']);recovery=[];da=[]
    for path in sorted(p for root in cfg['dagger'] for p in Path(root).glob('dot*.npz')):
        with np.load(path,allow_pickle=False) as z:
            prov=json.loads(str(z['provenance']));dot=int(prov['dot']);x=z['arm_x'];y=z['arm_y'];d=z['arm_distance_m'];steps=z['arm_step']
            assert prov['split']=='train' and dot in train_ids
            assert x.shape==(len(d),18) and y.shape==(len(d),7) and len(d)>0
            np.testing.assert_array_equal(steps,np.arange(len(d)))
            assert all(np.isfinite(v).all() for v in (x,y,d)) and np.all(d>=0)
            assert np.isin(x[:,-1],[0,1]).all() and np.all(np.diff(x[:,-1])>=0)
            recovery.append(dict(dot=dot,source=str(path),distance_m=d,arm=(x,y,np.clip(d,0,.25)/.25)))
            da.append(dict(path=str(path),sha256=tg.sha(path),dot=dot,rows=len(d),provenance=prov,step_contiguous_from_reset=True,prepark_rows=int((x[:,-1]==0).sum()),near10_rows=int((d<=.01).sum())))
    assert recovery
    audit.update(recovery=da,timing='teacher queried on same pre-step state and step as arm_x, before call(req); 1ms successive native oracle query enforced',input_names=cfg['input_names'],test_files_opened=0)
    tg.write_json(a.out/'data_audit.json',audit)
    old=dict(np.load(a.reference/'stop_seed7.npz',allow_pickle=False))
    groups=dict(original_train=list(records['train']),recovery_train=recovery,validation=records['val'])
    tg.write_json(a.out/'offline_before.json',{k:diagnose(v,old) for k,v in groups.items()})
    records['train'].extend(recovery) # Deliberately STOP-only records: no DAgger exclusion, no PARK labels.
    args=SimpleNamespace(**{k:cfg[k] for k in ('epochs','stop_epochs','arm_hidden','arm_scaling','lr','device','chunk')},out=a.out,save_epochs=())
    tg.write_json(a.out/'config.json',dict(reference=str(a.reference),reference_config=cfg,stop_sha256=tg.sha(a.reference/'stop_seed7.npz'),pilot_source_sha256=tg.sha(__file__),trainer_sha256=tg.sha(tg.__file__),seed=7,threshold_m=.01,threshold_tuned=False,test_files_opened=0,run_weighting='each original or recovery episode equal',train_runs=len(records['train']),validation_runs=len(records['val'])))
    torch.set_num_threads(4)
    print('Offline audit complete; waiting for GPU lock',flush=True)
    with open('/tmp/comp0245-validation-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        weights,meta=tg.train_one(records,'stop',7,args)
    tg.write_json(a.out/'offline_after.json',{k:diagnose(v,weights) for k,v in groups.items()})
    tg.write_json(a.out/'completion.json',dict(status='STOP training and offline audit complete; cumulative validation pending',elapsed_s=time.monotonic()-started,stop=meta))

if __name__=='__main__':main()
