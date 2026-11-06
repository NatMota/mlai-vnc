"""Closed-form encoder-to-position fitting; train-only Fourier coefficients, no URDF."""
import json,time,os
from pathlib import Path
import numpy as np
R=Path(__file__).parent
def features(q,axes):
 q=np.atleast_2d(q);f=np.ones((len(q),1),np.float64)
 for j in axes:
  v=np.c_[np.ones(len(q)),np.sin(q[:,j]),np.cos(q[:,j])]
  f=(f[:,:,None]*v[:,None,:]).reshape(len(q),-1)
 return f
def infer(w,q):return features(q,w['axes'])@w['coef']
def inverse_rotate(quat,v):
 quat=quat/np.linalg.norm(quat,axis=1,keepdims=True);q=quat[:,:3];w=quat[:,3:4]
 return v-2*w*np.cross(q,v)+2*np.cross(q,np.cross(q,v))
def write(p,a):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(a,indent=2,allow_nan=False));t.replace(p)
def main():
 import hashlib
 start=time.monotonic();out=R/'trig_position_training';out.mkdir(exist_ok=False)
 split=json.loads((R/'data/split.json').read_text());manifest=json.loads((R/'data/manifest.json').read_text());hashes={x['path']:x['sha256'] for x in manifest['files']}
 # Same datasheet count resolution as the original input pipeline.
 ticks=2*np.pi/np.array([3686400]*4+[2534400]*3)
 data={};audit=[]
 for s in ['train','val']:
  xx=[];yy=[];ww=[];ph=[]
  for dot in split[s]:
   p=R/'data/expert'/s/f'run_{dot}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(R/'data'))]
   with np.load(p,allow_pickle=False) as a:
    m=json.loads(str(a['manifest']));ix=np.arange(0,len(a['q']),10)
    q=np.floor(a['q'][ix,4:11]/ticks)*ticks
    y=inverse_rotate(a['base_quaternion'][ix],a['hand_position_expert'][ix]-a['base_position'][ix])
    xx.append(q);yy.append(y);ww.append(np.full(len(q),1/(len(q)*len(split[s]))));ph.append(ix>=m['parking_step'])
    audit.append(dict(split=s,dot=dot,rows=len(q)))
  data[s]=tuple(np.concatenate(v) for v in [xx,yy,ww,ph])
 write(out/'audit.json',audit)
 config=dict(architecture='tensor product of [1,sin(qj),cos(qj)] with learned 3D linear coefficients',axes_variants=[list(range(7)),list(range(6))],ridge_values=[1e-12,1e-10,1e-8,1e-6,1e-4],training_weight='equal mean per episode',selection='minimum weighted validation coordinate MSE',stride_ms=10,no_URDF=True,no_link_lengths=True,no_runtime_truth=True,no_test_data=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 write(out/'config.json',config);candidates=[];best=None
 for axes in config['axes_variants']:
  q,y,w,ph=data['train'];dim=3**len(axes);G=np.zeros((dim,dim));B=np.zeros((dim,3));tick=time.monotonic()
  for i in range(0,len(q),2048):
   f=features(q[i:i+2048],axes);weighted=f*w[i:i+2048,None];G+=f.T@weighted;B+=weighted.T@y[i:i+2048]
  values,vectors=np.linalg.eigh(G);negative=float(values.min());values=np.maximum(values,0);rhs=vectors.T@B
  coefs=[vectors@(rhs/(values[:,None]+lam)) for lam in config['ridge_values']]
  errors={s:[[] for _ in coefs] for s in data};losses={s:np.zeros(len(coefs)) for s in data}
  for s,(q,y,w,ph) in data.items():
   for i in range(0,len(q),2048):
    f=features(q[i:i+2048],axes)
    for k,coef in enumerate(coefs):
     e=f@coef-y[i:i+2048];losses[s][k]+=float(np.sum(w[i:i+2048]*np.mean(e*e,axis=1)));errors[s][k].extend(np.linalg.norm(e,axis=1)*1000)
  for k,(lam,coef) in enumerate(zip(config['ridge_values'],coefs)):
   entry=dict(axes=axes,ridge=lam,parameters=dim*3,training_mse=float(losses['train'][k]),validation_mse=float(losses['val'][k]),coefficient_norm=float(np.linalg.norm(coef)),gram_min_eigenvalue=negative,fit_seconds=time.monotonic()-tick)
   for s in data:
    e=np.array(errors[s][k]);mask=data[s][3]
    entry[s]=dict(median_mm=float(np.median(e)),p95_mm=float(np.quantile(e,.95)),postPARK_median_mm=float(np.median(e[mask])),postPARK_p95_mm=float(np.quantile(e[mask],.95)),max_mm=float(e.max()))
   name=f'axes{len(axes)}_ridge{lam:g}';np.savez(out/f'{name}.npz',axes=np.array(axes),coef=coef,ridge=np.array(lam));entry['file']=name+'.npz';candidates.append(entry)
   if best is None or entry['validation_mse']<best['validation_mse']:best=entry
   print(json.dumps(entry),flush=True)
  write(out/'progress.json',dict(completed_axes=len(axes),elapsed_s=time.monotonic()-start,candidates=candidates))
 src=out/best['file'];(out/'selected.npz').write_bytes(src.read_bytes())
 write(out/'summary.json',dict(candidates=candidates,selected=best,elapsed_s=time.monotonic()-start,selected_sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
if __name__=='__main__':main()
