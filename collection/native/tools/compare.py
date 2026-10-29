"""Audit only: compare native CSV/NPY output with the untouched Python capture."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from dataset import read_table

p=argparse.ArgumentParser()
p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path)
p.add_argument('--report',type=Path);p.add_argument('--prefix',action='store_true')
args=p.parse_args()
report={'reference':str(args.reference),'candidate':str(args.candidate),'tables':{},'passed':True,'exact_recorded_values':True}
for name in ['target','joint_state','base_state','kinematics','base_controller','arm_controller','commands']:
    a=read_table(args.reference,name);b=read_table(args.candidate,name)
    if args.prefix:a=a.head(len(b))
    entry={'rows_reference':len(a),'rows_candidate':len(b),'schema_equal':list(a)==list(b)}
    ok=len(a)==len(b) and entry['schema_equal']
    if ok:
        numeric=a.select_dtypes('number').columns
        strings=[c for c in a if c not in numeric]
        entry['labels_equal']=a[strings].equals(b[strings])
        # Tight floating point bound. Event labels must still match at the same millisecond.
        x=a[numeric].to_numpy();y=b[numeric].to_numpy()
        entry['nan_mask_equal']=bool(np.array_equal(np.isnan(x),np.isnan(y)))
        diff=np.abs(x-y)
        maxima=np.max(np.where(np.isnan(diff),0,diff),axis=0)
        entry['max_absolute_error']=float(np.nanmax(diff))
        entry['worst_columns']=dict(sorted(zip(numeric,maxima.tolist()),key=lambda z:z[1],reverse=True)[:6])
        entry['allclose_rtol_1e-7_atol_1e-8']=bool(np.allclose(x,y,rtol=1e-7,atol=1e-8,equal_nan=True))
        entry['exact_values']=bool(np.array_equal(x,y,equal_nan=True)) and entry['labels_equal']
        ok=entry['labels_equal'] and entry['nan_mask_equal'] and entry['allclose_rtol_1e-7_atol_1e-8']
    entry['passed']=ok;report['tables'][name]=entry;report['passed'] &= ok;report['exact_recorded_values'] &= entry.get('exact_values',False)
a=read_table(args.reference,'timing');b=read_table(args.candidate,'timing')
if args.prefix:a=a.head(len(b))
time_columns=['step','time_s','post_step_time_s']
same_time=list(a)==list(b) and np.array_equal(a[time_columns].to_numpy(),b[time_columns].to_numpy())
report['simulation_timestamps_equal']=bool(same_time)
report['passed'] &= bool(same_time);report['exact_recorded_values'] &= bool(same_time)
if (args.reference/'tensors.npz').exists():
    ref=np.load(args.reference/'tensors.npz');report['tensors']={}
    for name in ref.files:
        f=args.candidate/f'{name}.npy'
        if not f.exists():report['tensors'][name]={'recorded':False};continue
        b=np.load(f);a=ref[name];a=a[:len(b)] if args.prefix else a
        ok=a.shape==b.shape and np.allclose(a,b,rtol=1e-7,atol=1e-8,equal_nan=True)
        entry={'recorded':True,'passed':bool(ok),'reference_shape':list(a.shape),'candidate_shape':list(b.shape)}
        entry['exact_values']=bool(np.array_equal(a,b,equal_nan=True))
        if a.shape==b.shape:entry['max_absolute_error']=float(np.nanmax(np.abs(a-b)))
        report['tensors'][name]=entry;report['passed'] &= bool(ok);report['exact_recorded_values'] &= entry['exact_values']
text=json.dumps(report,indent=2)
if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(text+'\n')
print(text)
raise SystemExit(0 if report['passed'] else 1)
