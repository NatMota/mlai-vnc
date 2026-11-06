"""Identify torque coefficients from original demonstration signals, not runtime expert state."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from train_trig_position import features,write
R=Path(__file__).parent
def main():
 started=time.monotonic();out=R/'identified_torque_training';out.mkdir(exist_ok=False)
 split=json.loads((R/'data/split.json').read_text());manifest=json.loads((R/'data/manifest.json').read_text());hashes={x['path']:x['sha256'] for x in manifest['files']}
 axes=np.arange(6);ticks=2*np.pi/np.array([3686400]*4+[2534400]*3);data={};audit=[]
 for s in ['train','val']:
  qq=[];ee=[];vv=[];tt=[];ii=[]
  for runid,dot in enumerate(split[s]):
   path=R/'data/expert'/s/f'run_{dot}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[str(path.relative_to(R/'data'))]
   with np.load(path,allow_pickle=False) as a:
    count=np.floor(a['q'][:,4:11]/ticks)*ticks;qd=np.diff(count,axis=0,prepend=count[:1])/.001;ix=np.arange(0,len(count),10)
    qq.append(count[ix]);ee.append(a['arm_q_des'][ix]-count[ix]);vv.append(a['arm_qd_des'][ix]-qd[ix]);tt.append(a['arm_torque_command'][ix]);ii.append(np.full(len(ix),runid))
    audit.append(dict(split=s,dot=dot,rows=len(ix),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  q,e,v,t,ids=[np.concatenate(a) for a in [qq,ee,vv,tt,ii]]
  data[s]=dict(q=q,e=e,v=v,t=t,ids=ids,F=features(q,axes),w=np.concatenate([np.full(len(a),1/(len(a)*100)) for a in tt]))
 write(out/'audit.json',audit)
 config=dict(architecture='tau_j = trig(q1..q6) @ learned_coeff_j + learnedKp_j*(own_reference_j-counted_q_j) + learnedKd_j*(own_desired_speed_j-counted_qd_j); clip output to +/-30Nm',training_reference='expert internal q_des and qd_des only during coefficient identification',runtime_reference='student generated, never simulator or expert',ridge_values=[1e-12,1e-10,1e-8,1e-6],fit_rule='linear fit to unsaturated abs(torque)<29.5Nm rows, equal mean per eligible run; validation scores ALL rows after actual output clipping',no_URDF=True,no_analytic_mass_or_gravity=True,no_DART_or_DAgger=True,no_test_opened=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 write(out/'config.json',config);fit=[];coefs=[];kp=[];kd=[];scales=[]
 for j in range(7):
  d=data['train'];es=max(float(d['e'][:,j].std()),1e-4);vs=max(float(d['v'][:,j].std()),1e-4)
  mask=abs(d['t'][:,j])<29.5;weights=np.zeros(len(mask))
  eligible=sum(bool(np.any(mask&(d['ids']==rid))) for rid in range(100))
  for rid in range(100):
   m=mask&(d['ids']==rid);n=int(m.sum())
   if n:weights[m]=1/(n*eligible)
  X=np.c_[d['F'],d['e'][:,j]/es,d['v'][:,j]/vs];G=X.T@(X*weights[:,None]);b=X.T@(weights*d['t'][:,j]);values,vectors=np.linalg.eigh(G);values=np.maximum(values,0);rhs=vectors.T@b
  candidates=[]
  for lam in config['ridge_values']:
   beta=vectors@(rhs/(values+lam));errors={}
   for s,d in data.items():
    pred=np.clip(d['F']@beta[:-2]+beta[-2]*d['e'][:,j]/es+beta[-1]*d['v'][:,j]/vs,-30,30);err=pred-d['t'][:,j]
    errors[s]=dict(mse=float(np.sum(d['w']*err**2)),MAE_Nm=float(np.sum(d['w']*abs(err))),p95_absolute_Nm=float(np.quantile(abs(err),.95)))
   entry=dict(joint=j+1,ridge=lam,Kp=float(beta[-2]/es),Kd=float(beta[-1]/vs),coefficient_norm=float(np.linalg.norm(beta[:-2])),eligible_fit_rows=int(mask.sum()),training_rows=len(mask),errors=errors)
   candidates.append((entry,beta))
  entry,beta=min(candidates,key=lambda z:z[0]['errors']['val']['mse'])
  fit.append(dict(selected=entry,candidates=[a for a,b in candidates]));coefs.append(beta[:-2]);kp.append(beta[-2]/es);kd.append(beta[-1]/vs);scales.append([es,vs])
  write(out/'progress.json',dict(completed_joints=j+1,elapsed_s=time.monotonic()-started,fit=fit));print(json.dumps(entry),flush=True)
 np.savez(out/'identified_torque.npz',axes=axes,coef=np.stack(coefs,axis=1),Kp=np.array(kp),Kd=np.array(kd))
 summary=dict(elapsed_s=time.monotonic()-started,fit=fit,Kp=kp,Kd=kd,positive_gains=bool(min(kp)>0 and min(kd)>0),runtime_expert_inputs=False,sha256=hashlib.sha256((out/'identified_torque.npz').read_bytes()).hexdigest())
 write(out/'summary.json',summary)
if __name__=='__main__':main()
