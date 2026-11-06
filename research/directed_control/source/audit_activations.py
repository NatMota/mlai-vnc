"""Read-only neuron audit on training demonstrations; never changes a checkpoint."""
import json,sys
from pathlib import Path
import numpy as np
R=Path(__file__).parent
sys.path.insert(0,str(R/'runtime/v2_sensors/arm'))
from render_golden import camera_inputs
sys.path.insert(0,str(R/'runtime/v2_sensors/driving_v2'))
from sensors import ARM_TICK
w=dict(np.load(R/'baseline/reaching_seed7.npz',allow_pickle=False))
assert 'export_input_mean' not in w,'Use actual normalisation for another export'
ids=json.loads((R/'data/split.json').read_text())['train']
xs=[]
for dot in ids:
 with np.load(R/f'data/expert/train/run_{dot}.npz',allow_pickle=False) as a:
  m=json.loads(str(a['manifest']));park=m['parking_step']
  if park<0 or any(t in m['termination'].lower() for t in ['timeout','cap','limit']):continue
  q=a['q'][:,4:11];angle=np.floor(q.astype(float)/ARM_TICK)*ARM_TICK;speed=np.diff(angle,axis=0,prepend=angle[:1])/.001
  ix=np.arange(0,len(q),33);cam=camera_inputs('train',dot,ix,root=R/'data/camera')
  xs.append(np.c_[angle[ix],speed[ix],cam['arm'],ix>=park].astype(np.float32))
x=np.concatenate(xs);layers={};intermediate={}
for prefix in ['shared','stopnet']:
 h=x
 for layer in [prefix+'.0',prefix+'.2']:
  z=h@w[layer+'.weight'].T+w[layer+'.bias'];h=np.maximum(z,0);intermediate[layer]=(z,h)
  freq=np.mean(z>0,axis=0)
  layers[layer]={'width':len(freq),'inactive_on_all_sampled_rows':np.flatnonzero(freq==0).tolist(),'active_on_all_sampled_rows':np.flatnonzero(freq==1).tolist(),'positive_fraction':freq.tolist(),'preactivation_quantiles':np.quantile(z,[0,.01,.5,.99,1]).tolist(),'activation_p99_per_neuron':np.quantile(h,.99,axis=0).tolist()}
# Analytic input Jacobian for each output at a bounded training-state sample.
std=np.std(x,axis=0);indices=np.linspace(0,len(x)-1,min(512,len(x)),dtype=int)
sens={}
for prefix,out in [('shared','arm'),('stopnet','stopnet.4')]:
 mats=[]
 for i in indices:
  z0=intermediate[prefix+'.0'][0][i];z1=intermediate[prefix+'.2'][0][i]
  J=(w[out+'.weight']*(z1>0)[None,:])@w[prefix+'.2.weight']
  J=(J*(z0>0)[None,:])@w[prefix+'.0.weight']
  if prefix=='stopnet':J*=float(w['ttp_clip'])
  mats.append(abs(J)*std[None,:])
 sens[out]={'median_abs_derivative_times_training_std':np.median(mats,axis=0).tolist(),'units':'Nm' if out=='arm' else 'm'}
# Separate activation-function perturbation diagnostic, not a new rollout/model.
base=intermediate['shared.2'][1]@w['arm.weight'].T+w['arm.bias']
variants={}
for slope in [.001,.01]:
 h=x
 for layer in ['shared.0','shared.2']:
  z=h@w[layer+'.weight'].T+w[layer+'.bias'];h=np.where(z>=0,z,slope*z)
 pred=h@w['arm.weight'].T+w['arm.bias']
 variants[str(slope)]={'rms_torque_change_per_joint_Nm':np.sqrt(np.mean((pred-base)**2,axis=0)).tolist(),'max_abs_torque_change_Nm':float(abs(pred-base).max())}
result={'kind':'training-only activation and sensitivity audit, not evidence of rollout improvement','runs':len(xs),'rows':len(x),'stride_ms':33,'feature_order':['q'+str(i) for i in range(7)]+['qd'+str(i) for i in range(7)]+['camera_x','camera_y','camera_z','park'],'training_feature_std':std.tolist(),'layers':layers,'sensitivity':sens,'leaky_relu_offline_diagnostics':variants}
(R/'activation_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'runs':len(xs),'rows':len(x),'layers':{k:{a:v[a] for a in ['width','inactive_on_all_sampled_rows','active_on_all_sampled_rows','preactivation_quantiles']} for k,v in layers.items()},'leaky_relu':variants},indent=2))
