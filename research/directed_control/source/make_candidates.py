"""Frozen hypothesis-led parameter edits; no fitting and no test-data access."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).parent; out=R/'rnd_candidates';out.mkdir(exist_ok=True)
w=dict(np.load(R/'baseline/reaching_seed7.npz',allow_pickle=False))
mean=np.array(json.loads((R/'baseline/torque_training_summary.json').read_text())['input_mean'])
manifest=[]
def write(name,cols=None,gain=1,leak=0,hypothesis=''):
 v={k:a.copy() for k,a in w.items()}
 if cols is not None:
  old=v['shared.0.weight'][:,cols].copy()
  v['shared.0.bias']+=(1-gain)*(old@mean[cols]).astype(np.float32)
  v['shared.0.weight'][:,cols]*=gain
 p=out/(name+'.npz');np.savez(p,**v)
 delta=[]
 for k in w:
  if not np.array_equal(w[k],v[k]):
   for ix in np.argwhere(w[k]!=v[k]):
    ix=tuple(ix);delta.append(dict(tensor=k,index=list(map(int,ix)),before=float(w[k][ix]),after=float(v[k][ix])))
 row=dict(name=name,arm_leaky_slope=leak,hypothesis=hypothesis,model_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),changes=delta)
 (out/(name+'.json')).write_text(json.dumps(row,indent=2));manifest.append(row)
write('baseline',hypothesis='Unmodified baseline, matched same runtime')
write('q7_gain05',[6],.5,hypothesis='Reduce cross-joint extrapolation from narrow-training-range joint7; may weaken restoring response')
write('velocity_gain075',list(range(7,14)),.75,hypothesis='Reduce amplification of velocity departures and later measurement noise; may weaken damping')
write('velocity_gain125',list(range(7,14)),1.25,hypothesis='Strengthen learned damping response if underdamped; may amplify destabilising cross-joint coupling')
write('camera_gain125',list(range(14,17)),1.25,hypothesis='Strengthen corrective response to tracked target displacement; may overshoot')
write('leaky0001',leak=.001,hypothesis='Small negative activation branch changes piecewise feedback without hard zeroing; may add harmful previously unused responses')
write('leaky001',leak=.01,hypothesis='Stronger controlled activation-function intervention; offline changes up to2.9Nm so test separately')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Created',len(manifest),'frozen candidates including baseline')
