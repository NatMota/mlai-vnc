"""Read saved golden-data results and build analysis without rerunning training."""
import base64
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODELS = HERE / 'models/golden_11109'
RESULTS = HERE / 'results/golden_11109'
OLD = HERE.parent / 'driving_30'


def read(path):
    return json.loads(Path(path).read_text())


def table(seed):
    rows = []
    for r in read(RESULTS / f'seed{seed}.json'):
        rows.append(dict(dot=r['dot'], parked=r['parked'], park_s=r['park_s'],
            park_error_cm=100*r['park_error_vs_expert_m'] if r['parked'] else np.nan,
            stop_s=r['stop_s'], at_stop_mm=r['at_network_stop_mm'], closest_mm=r['min_mm'],
            end_mm=r['end_mm'], success=bool(r['parked'] and r['stop_s'] is not None and r['at_network_stop_mm'] <= 10),
            camera_visible_fraction=np.mean([f['visible'] for f in r['frames']]),
            elapsed_s=r['elapsed_s']))
    return pd.DataFrame(rows).set_index('dot')


def losses(seed):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.3))
    terms = [('driving','command','Wheel MSE (rad/s)²'), ('driving','aux','Time-to-PARK MSE (scaled)'),
             ('torque','command','Torque MSE (standardised)'), ('torque','aux','Auxiliary distance MSE (scaled)'),
             ('stop','command','Independent STOP distance MSE (scaled)')]
    for ax, (kind, term, title) in zip(axes, terms):
        h = pd.DataFrame(read(MODELS / f'{kind}_seed{seed}_history.json'))
        chosen = read(MODELS / f'{kind}_seed{seed}_summary.json')['selected_epoch']
        ax.plot(h.epoch, h[f'train_{term}'], label='Train')
        ax.plot(h.epoch, h[f'val_{term}'], '--', label='Validation')
        ax.axvline(chosen, color='grey', lw=1, label=f'Selected {chosen}')
        ax.set(title=title, xlabel='Epoch', yscale='log'); ax.legend(fontsize=7)
    fig.suptitle(f'Seed {seed}: every epoch, equal weight per run'); fig.tight_layout()
    return fig


def outcomes(seed):
    d = table(seed); ghosts = read(RESULTS / 'expert_validation.json')
    heights = [ghosts[str(i)]['target'][2] for i in d.index]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    axes[0].bar([str(i) for i in d.index], d.park_error_cm)
    axes[0].set(xlabel='Validation target ID', ylabel='Error at PARK (cm)', title='Blank bars: no PARK')
    axes[1].scatter(heights, d.closest_mm, label='Closest during reaching')
    axes[1].scatter(heights, d.at_stop_mm, marker='x', label='At STOP')
    axes[1].axhline(10, color='grey', ls=':'); axes[1].legend(fontsize=8)
    axes[1].set(xlabel='Target height (m)', ylabel='Gripper distance (mm)', title='Distances only where the arm reached')
    axes[2].bar([str(i) for i in d.index], d.camera_visible_fraction)
    axes[2].set(ylim=(0,1), xlabel='Validation target ID', ylabel='Fraction of camera frames', title='Actual camera visibility')
    for ax in (axes[0],axes[2]): ax.tick_params(axis='x', rotation=60)
    fig.tight_layout(); return fig


def distances(seed):
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.5))
    for ax, r in zip(axes.flat, read(RESULTS / f'seed{seed}.json')):
        ax.set_title(f'Dot {r["dot"]}', fontsize=10)
        if not r['log']:
            ax.text(.5,.5,'No PARK',ha='center',transform=ax.transAxes)
        else:
            log = pd.DataFrame(r['log']); t = log.t-r['park_s']
            ax.plot(t,log.ee_mm,label='True distance (scoring only)')
            ax.plot(t,1000*log.ttp_s,'--',label='Predicted STOP distance')
            ax.axhline(10,color='grey',ls=':')
            if r['stop_s'] is not None: ax.axvline(r['stop_s']-r['park_s'],color='black',lw=.7)
        ax.set(xlabel='Seconds after PARK',ylabel='mm')
    handles, labels = next((ax.get_legend_handles_labels() for ax in axes.flat if ax.lines),([],[]))
    if handles: fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=8)
    fig.tight_layout(rect=(0,0,1,.94)); return fig


def reading(seed):
    d = table(seed); selected = {k:read(MODELS/f'{k}_seed{seed}_summary.json') for k in ('driving','torque','stop')}
    starts = {k:read(MODELS/f'{k}_seed{seed}_history.json')[0]['val_total'] for k in selected}
    changes = '; '.join(f'{k} {starts[k]:.4g} to {v["selected_val_loss"]:.4g} (epoch {v["selected_epoch"]})' for k,v in selected.items())
    stop = d.at_stop_mm.dropna()
    stop_text = f'{stop.median():.0f} mm median distance at STOP across the {len(stop)} runs that stopped' if len(stop) else 'no distance-at-STOP result because none stopped'
    return (f'Validation losses fell: {changes}. In the 10 predetermined simulation checks, '
            f'{int(d.parked.sum())}/10 parked, {int(d.stop_s.notna().sum())}/10 issued STOP, and {int(d.success.sum())}/10 stopped within 10 mm. '
            f'There was {stop_text}. Missing PARK or STOP is a failed full task, not a missing observation to discard. '
            'These are integration checks on validation targets, not the final 100-target test score. '
            'A better offline fit has not established reliable end-to-end reaching. The untouched test evaluation is ticket #11110.')


def build_viewer(seed):
    sys.path.insert(0,str(OLD))
    import replay
    ghosts = read(RESULTS/'expert_validation.json'); runs=[]
    def frame(f):
        out = dict(f, command=f.get('wheel',[0]*4), park_probability=float(f['parked']), speed_m_s=0., tau=f.get('tau',[0]*7))
        if f.get('camera_src'):
            out['camera_src'] = 'data:image/png;base64,' + base64.b64encode((RESULTS/f['camera_src']).read_bytes()).decode('ascii')
        return out
    for r in read(RESULTS/f'seed{seed}.json'):
        e=ghosts[str(r['dot'])]
        runs.append(dict(bearing_deg=r['dot'],split='validation',target=e['target'],
            metrics=dict(model_park_s=r['park_s'],expert_park_s=e['park_s'],park_xy_error_m=r['park_error_vs_expert_m'] if r['parked'] else None,
                         park_yaw_error_deg=None,final_xy_error_m=r['final_base_error_vs_expert_m'],stop_mm=r['at_network_stop_mm'],closest_mm=r['min_mm']),
            model=dict(frames=[frame(f) for f in r['frames']],final=dict(time_s=r['frames'][-1]['time_s'])),
            expert=dict(frames=[frame(f) for f in e['frames']],final=dict(time_s=e['frames'][-1]['time_s']))))
    names=read(HERE.parent/'driving_v2/closed_loop_native/result_wheels_live_dot.json')['joint_names']
    payload=dict(joint_names=names,runs=runs,geometry=replay.geometry(replay.native_root(),names),
                 recording_description=f'Seed {seed}; 10 fixed validation targets. Learned driving and reaching. Ghost: recorded expert. No trajectory truncation. Camera inset: actual recorded frames, 128 px preview of the 256 px sensor.')
    page=(OLD/'replay_template.html').read_text()
    page=page.replace('<aside>','<aside><section><img id="sensorImage" alt="Recorded front camera" style="width:100%;image-rendering:auto"><p class="note">Actual camera at the displayed robot state</p></section>')
    page=page.replace('function text(m,e){','function text(m,e){if(m.camera_src){$("sensorImage").src=m.camera_src;}')
    page=page.replace('${r.bearing_deg}°','target ${r.bearing_deg}')
    page=page.replace('Select train or test run','Select validation run')
    page=page.replace('Recorded robot motion · model + ghost expert',f'Golden data · seed {seed} · model + ghost expert')
    page=page.replace('Every target is 1.1 m from the origin and 0.5 m high.','Targets vary in bearing, radius and height.')
    page=page.replace('Model: 30 Hz. Expert and physics: 1,000 Hz. Park probability ≥ 0.5 latches zero wheel commands; no recovery. The supplied controller holds the arm.',
        'Driving and camera: 30 Hz. Learned arm torques and physics: 1,000 Hz. PARK latches zero wheels. STOP ends the attempt; the network continues producing hold torques.')
    page=page.replace("for(const name of ['Front right','Front left','Back left','Back right'])", "for(const name of ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6','Joint 7'])")
    page=page.replace('m.command[i].toFixed(2)','m.tau[i].toFixed(2)').replace('e.command[i].toFixed(2)','e.tau[i].toFixed(2)')
    page=page.replace('Command · rad/s','Torque · N m')
    page=page.replace('\\nPark probability ${m.park_probability.toFixed(3)} · Speed ${m.speed_m_s.toFixed(2)} / ${e.speed_m_s.toFixed(2)} m/s',
        '\\nGripper distance: model ${m.ee_mm.toFixed(1)} mm / expert ${e.ee_mm.toFixed(1)} mm')
    page=page.replace('Heading difference ${fmt(m.park_yaw_error_deg,1)}°','Distance at STOP ${fmt(m.stop_mm,1)} mm; closest ${fmt(m.closest_mm,1)} mm')
    page=page.replace('__PLOTLY__',(OLD/'replay_assets/plotly.min.js').read_text())
    page=page.replace('__DATA__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
    out=RESULTS/f'viewer_golden_seed{seed}.html';out.write_text(page);return out
