"""Diagnostic-only internal-motion audit of saved 1kHz controller traces."""
import argparse, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args()
    rows=[]
    for path in sorted(a.folder.glob('run_*.json')):
        r=json.loads(path.read_text());dot=r['dot'];trace=a.folder/f'trace_{dot}.npz'
        if not trace.exists():continue
        with np.load(trace) as z:
            v=z['filtered_velocity'];tau=z['tau_raw'];x=z['x']
        item=dict(dot=dot,status=r['status'],success=r['success'],post_stop_peak_mm=r.get('post_stop_peak_mm'))
        for name,mask in [('prePARK',(x[:,17]<.5)&(np.arange(len(x))>=100)),
                          ('postSTOP',np.arange(len(x))>=round(r['stop_s']*1000) if r.get('stop_s') is not None else np.zeros(len(x),bool))]:
            vv=v[mask];tt=tau[mask]
            if len(vv)<20:continue
            window=np.hanning(len(vv))
            amplitude=2*np.abs(np.fft.rfft((vv-vv.mean(axis=0))*window[:,None],axis=0))/max(window.sum(),1e-12)
            freq=np.fft.rfftfreq(len(vv),.001);band=(freq>=25)&(freq<=45)
            item[name]=dict(samples=len(vv),rms_rad_s=np.sqrt(np.mean(vv**2,axis=0)).tolist(),
                            clip_fraction=np.mean(abs(tt)>=30,axis=0).tolist(),
                            mode25_45Hz_amplitude=amplitude[band].max(axis=0).tolist())
        rows.append(item)
    result=dict(scope='Recorded filtered sensor velocities and requested torque, not a physical safety certificate.',
                complete_cases=len(rows),rows=rows,
                flagged=[r['dot'] for r in rows if r.get('postSTOP') and
                         (max(r['postSTOP']['rms_rad_s'])>.3 or max(r['postSTOP']['clip_fraction'])>.02)])
    path=a.folder/'motion_audit.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));tmp.replace(path)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))

if __name__=='__main__':main()
