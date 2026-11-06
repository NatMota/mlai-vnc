"""Algebraically identical position/Jacobian evaluation; no learned weights change."""
import numpy as np

def position_jacobian(w,q):
    axes=np.asarray(w['axes'],dtype=int)
    angle=np.asarray(q)[axes]
    sine=np.sin(angle);cosine=np.cos(angle)
    rows=np.zeros((len(axes)+1,1));rows[0,0]=1
    for k in range(len(axes)):
        old_position=rows[0].copy()
        rows=(rows[:,:,None]*np.array([1,sine[k],cosine[k]])).reshape(len(axes)+1,-1)
        rows[k+1]=(old_position[:,None]*np.array([0,cosine[k],-sine[k]])).reshape(-1)
    values=rows@w['coef']
    jac=np.zeros((3,7));jac[:,axes]=values[1:].T
    return values[0],jac

if __name__=='__main__':
    import json,time
    from pathlib import Path
    from learned_geometry_branch.geometry_runtime import position_jacobian as reference
    root=Path(__file__).parent
    rng=np.random.default_rng(7)
    q=rng.uniform(-np.pi,np.pi,size=(500,7))
    results=[]
    for path in ['trig_position_training/selected.npz','trig_svd_training/selected.npz']:
        w=dict(np.load(root/path));errors=[];timings={}
        for fn in [reference,position_jacobian]:
            times=[]
            for x in q:
                start=time.perf_counter();fn(w,x);times.append((time.perf_counter()-start)*1e6)
            timings[fn.__module__]=dict(median_us=float(np.median(times)),p95_us=float(np.quantile(times,.95)))
        for x in q:
            p,j=position_jacobian(w,x);p0,j0=reference(w,x)
            errors.append(max(np.max(abs(p-p0)),np.max(abs(j-j0))))
        assert max(errors)<1e-12
        results.append(dict(model=path,max_absolute_difference=max(errors),timing=timings))
    (root/'geometry_fast_verification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
