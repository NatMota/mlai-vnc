"""Direct SVD checks numerical conditioning of the learned geometric feature map."""
import json,time,hashlib
from pathlib import Path
import numpy as np
from train_trig_position import features,inverse_rotate,write
R=Path(__file__).parent
def main():
 start=time.monotonic();out=R/'trig_svd_training';out.mkdir(exist_ok=False)
 split=json.loads((R/'data/split.json').read_text());manifest=json.loads((R/'data/manifest.json').read_text());hashes={x['path']:x['sha256'] for x in manifest['files']}
 tick=2*np.pi/np.array([3686400]*4+[2534400]*3);data={};axes=np.arange(6)
 for s in ['train','val']:
  fq=[];y=[];weights=[];phase=[]
  for dot in split[s]:
   p=R/'data/expert'/s/f'run_{dot}.npz';assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(R/'data'))]
   with np.load(p,allow_pickle=False) as a:
    m=json.loads(str(a['manifest']));ix=np.arange(0,len(a['q']),10)
    q=np.floor(a['q'][ix,4:11]/tick)*tick
    fq.append(features(q,axes));y.append(inverse_rotate(a['base_quaternion'][ix],a['hand_position_expert'][ix]-a['base_position'][ix]));weights.append(np.full(len(ix),1/(100*len(ix))));phase.append(ix>=m['parking_step'])
  data[s]=tuple(np.concatenate(z) for z in [fq,y,weights,phase])
 config=dict(axes=axes.tolist(),rcond=[1e-8,1e-10,1e-12],solver='direct weighted FP64 SVD, no Gram matrix',input='counted encoder angles',labels='same original demonstration base-relative hand positions',selection='minimum weighted validation coordinate MSE',test_files_opened=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 write(out/'config.json',config)
 X,y,w,ph=data['train'];U,s,Vt=np.linalg.svd(X*np.sqrt(w[:,None]),full_matrices=False);rhs=U.T@(y*np.sqrt(w[:,None]));del U
 result=[];best=None
 for cut in config['rcond']:
  active=s>cut*s[0];coef=Vt[active].T@(rhs[active]/s[active,None]);row=dict(rcond=cut,rank=int(active.sum()),coefficient_norm=float(np.linalg.norm(coef)))
  for name,(x,y,w,ph) in data.items():
   diff=x@coef-y;err=np.linalg.norm(diff,axis=1)*1000
   row[name]=dict(mse=float(np.sum(w*np.mean(diff**2,axis=1))),median_mm=float(np.median(err)),postPARK_median_mm=float(np.median(err[ph])),p95_mm=float(np.quantile(err,.95)),postPARK_p95_mm=float(np.quantile(err[ph],.95)),max_mm=float(err.max()))
  filename=f'rcond{cut:g}.npz';np.savez(out/filename,coef=coef,axes=axes,rcond=np.array(cut));row['file']=filename;result.append(row)
  if best is None or row['val']['mse']<best['val']['mse']:best=row
  print(json.dumps(row),flush=True)
 (out/'selected.npz').write_bytes((out/best['file']).read_bytes());np.save(out/'singular_values.npy',s)
 write(out/'summary.json',dict(elapsed_s=time.monotonic()-start,selected=best,candidates=result))
if __name__=='__main__':main()
