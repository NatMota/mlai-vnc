"""Analysis of the frozen 100-target tests. This module never trains or selects models."""
import json
import os
from pathlib import Path
import zipfile
import subprocess
import sys
import time
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from report_golden import read, losses, build_viewer as original_viewer, MODELS

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results/golden_11110'


def preview_server():
    """Serve replay files on loopback; Jupyter sandboxes HTML and blocks its image cookies."""
    if os.environ.get('GOLDEN_REPLAY_BASE_URL'):
        return os.environ['GOLDEN_REPLAY_BASE_URL'].rstrip('/')
    url='http://127.0.0.1:8899'
    expected=(RESULTS/'completion.json').read_bytes()
    def ready():
        try:
            with urllib.request.urlopen(url+'/completion.json',timeout=1) as response:
                if response.read()!=expected: raise RuntimeError('Port 8899 serves a different project')
                return True
        except urllib.error.URLError: return False
    if not ready():
        process=subprocess.Popen([sys.executable,'-m','http.server','8899','--bind','127.0.0.1',
            '--directory',str(RESULTS)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,start_new_session=True)
        for _ in range(30):
            if ready(): break
            if process.poll() is not None: raise RuntimeError('Could not start replay server on port 8899')
            time.sleep(.1)
        else: raise RuntimeError('Replay server did not become ready')
    return url


def table(seed):
    raw=read(RESULTS/f'seed{seed}.json'); refs=read(RESULTS/'expert_test.json')
    rows=[]
    for r in raw:
        target=refs[str(r['dot'])]['target']
        rows.append(dict(dot=r['dot'],status=r['status'],success=r['success'],parked=r['parked'],
            park_s=r.get('park_s'),stop_s=r.get('stop_s'),
            park_error_cm=None if not r['parked'] or r.get('park_error_vs_expert_m') is None else 100*r['park_error_vs_expert_m'],
            park_heading_error_deg=None if r.get('park_heading_error_rad') is None else np.degrees(r['park_heading_error_rad']),
            at_stop_mm=r.get('at_network_stop_mm'),closest_mm=r.get('min_mm'),end_mm=r.get('end_mm'),
            post_stop_drift_mm=r.get('post_stop_drift_mm'),post_park_base_drift_cm=None if r.get('post_park_base_drift_m') is None else 100*r['post_park_base_drift_m'],
            timed_out=r.get('timed_out',False),elapsed_s=r['elapsed_s'],
            camera_visible_fraction=np.mean([f['visible'] for f in r['frames']]) if r['frames'] else None,
            target_bearing_deg=float(np.degrees(np.arctan2(target[1],target[0]))%360),
            target_range_m=float(np.hypot(target[0],target[1])),target_height_m=target[2]))
    df=pd.DataFrame(rows).set_index('dot')
    ids=read(RESULTS/'config.json')['cases']
    assert len(df)==100 and set(df.index)==set(ids), 'Every reserved target must be accounted for'
    assert not (df.success & (~df.parked | df.stop_s.isna() | (df.at_stop_mm>10))).any()
    return df


def wilson(k,n):
    z=1.959963984540054; p=k/n; denom=1+z*z/n
    centre=(p+z*z/(2*n))/denom
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return (0. if k==0 else max(0.,centre-half)),(1. if k==n else min(1.,centre+half))


def score(seed):
    d=table(seed);lo,hi=wilson(int(d.success.sum()),len(d))
    return dict(seed=seed,attempts=len(d),parked=int(d.parked.sum()),stopped=int(d.stop_s.notna().sum()),
        success_10mm=int(d.success.sum()),success_percent=100*d.success.mean(),
        wilson95_low_percent=100*lo,wilson95_high_percent=100*hi,
        median_park_error_cm=d.park_error_cm.median(),median_abs_heading_error_deg=d.park_heading_error_deg.abs().median(),
        median_closest_mm=d.closest_mm.median(),median_stop_mm=d.at_stop_mm.median(),
        median_park_s=d.park_s.median(),median_stop_s=d.stop_s.median(),
        timeouts=int(d.timed_out.sum()),execution_failures=int(d.status.isin(['execution_failure','numerical_failure','early_native_termination']).sum()),
        sum_case_wall_s=d.elapsed_s.sum(),median_case_wall_s=d.elapsed_s.median())


def outcomes(seed):
    d=table(seed);fig,ax=plt.subplots(1,3,figsize=(15,4))
    statuses=d.status.value_counts()
    ax[0].barh(statuses.index,statuses.values);ax[0].set(xlabel='Attempts out of 100',title='Every attempt counted')
    parked=d[d.parked]
    im=ax[1].scatter(parked.target_bearing_deg,parked.park_error_cm,c=parked.target_height_m,cmap='viridis')
    ax[1].set(xlabel='Target bearing (degrees)',ylabel='Error at PARK (cm)',title=f'PARK: {len(parked)}/100')
    fig.colorbar(im,ax=ax[1],label='Target height (m)')
    ax[2].scatter(d.target_height_m,d.closest_mm,label='Closest after PARK',alpha=.7,s=22)
    ax[2].scatter(d.target_height_m,d.at_stop_mm,label='At exact STOP',marker='x',s=22)
    ax[2].axhline(10,color='black',ls=':',label='10 mm success threshold')
    ax[2].set(xlabel='Target height (m)',ylabel='Distance (mm)',title='Unstopped attempts have no STOP distance')
    ax[2].legend(fontsize=8);fig.tight_layout();return fig


def stopping(seed):
    d=table(seed);fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].scatter(d.closest_mm,d.at_stop_mm,c=d.target_bearing_deg,cmap='hsv',s=25)
    ax[0].axhline(10,color='black',ls=':');ax[0].axvline(10,color='black',ls=':')
    ax[0].set(xlabel='Closest after PARK (mm)',ylabel='Distance at STOP (mm)',title='Approach versus stopping')
    stopped=d[d.stop_s.notna()]
    ax[1].scatter(stopped.at_stop_mm,stopped.end_mm,s=25)
    maxval=max(stopped.at_stop_mm.max(),stopped.end_mm.max())
    if np.isfinite(maxval):ax[1].plot([0,maxval],[0,maxval],':',color='grey')
    ax[1].set(xlabel='Distance at STOP (mm)',ylabel='Distance 0.5 s later (mm)',title='Post-STOP drift, stopped attempts only')
    ax[2].scatter(d.camera_visible_fraction,d.at_stop_mm,s=25)
    ax[2].set(xlabel='Visible fraction of recorded camera frames',ylabel='Distance at STOP (mm)',title='Visibility and stopping outcome')
    fig.tight_layout();return fig


def distances(seed):
    # First ten manifest IDs, fixed independently of outcomes. All 100 are in the replay selector/table.
    ids=read(RESULTS/'config.json')['cases'][:10]
    raw={r['dot']:r for r in read(RESULTS/f'seed{seed}.json')}
    fig,axes=plt.subplots(2,5,figsize=(16,5.5))
    for ax,dot in zip(axes.flat,ids):
        r=raw[dot];ax.set_title(f'Target {dot}',fontsize=10)
        parked = r['park_s'] is not None
        origin = r['park_s'] if parked else 0.
        if not r['log']:ax.text(.5,.5,r['status'].replace('_',' '),ha='center',transform=ax.transAxes,fontsize=8)
        else:
            a=pd.DataFrame(r['log']);t=a.t-origin
            ax.plot(t,a.ee_mm,label='Actual (scoring only)');ax.plot(t,1000*a.ttp_s,'--',label='Predicted STOP distance')
            ax.axhline(10,color='grey',ls=':')
            if r['stop_s'] is not None:ax.axvline(r['stop_s']-origin,color='black',lw=.7)
            if not parked:ax.text(.5,.95,'No PARK',ha='center',va='top',transform=ax.transAxes,fontsize=8)
        ax.set(xlabel='Seconds after PARK' if parked else 'Seconds since reset',ylabel='mm')
    h,l=next((a.get_legend_handles_labels() for a in axes.flat if a.lines),([],[]))
    if h:fig.legend(h,l,loc='upper center',ncol=2,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.94));return fig


def reading(seed):
    d=table(seed);s=score(seed)
    closest=int((d.closest_mm<=10).sum())
    return (f"Seed {seed}: {s['parked']}/100 PARK, {s['stopped']}/100 STOP and {s['success_10mm']}/100 successful stops within 10 mm "
        f"(Wilson 95% interval {s['wilson95_low_percent']:.1f}–{s['wilson95_high_percent']:.1f}%). "
        f"{closest}/100 came within 10 mm during reaching; passing through is not a successful stop. "
        f"Median parking error was {s['median_park_error_cm']:.1f} cm among parked attempts; median distance at STOP was "
        f"{s['median_stop_mm']:.1f} mm among stopped attempts. {s['timeouts']} attempts timed out and {s['execution_failures']} had execution/termination failures. "
        "These conditional medians exclude attempts without the relevant event; the success rate keeps all 100. "
        "This is a frozen baseline result, not a basis for selecting checkpoints. Representative seed 7 remains the validation-selected seed.")


def build_viewer(seed):
    return original_viewer(seed,results=RESULTS,split='test',embed_camera=False)


def package_replay(seed,viewer=None):
    viewer=Path(build_viewer(seed) if viewer is None else viewer);root=viewer.parent;out=root/f'replay_test_seed{seed}.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        z.write(viewer,viewer.name)
        for folder in sorted(root.glob(f'seed{seed}_dot*_camera')):
            for image in sorted(folder.glob('*.png')):z.write(image,image.relative_to(root))
        for data in sorted((root/f'viewer_seed{seed}_runs').glob('*.json')):z.write(data,data.relative_to(root))
        z.writestr('READ_ME.txt','Extract the entire archive. Keep the camera and run folders beside '+viewer.name+'. Serve the extracted folder with python -m http.server 8000 and visit http://localhost:8000/'+viewer.name+'. All trajectories are recorded; no simulator is required. Only the selected trajectory is loaded.\n')
    return out
