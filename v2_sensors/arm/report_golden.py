"""Read saved golden-data results and build analysis without rerunning training."""
import base64
import json
import gzip
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
    path=Path(path)
    if path.exists():return json.loads(path.read_text())
    with gzip.open(str(path)+'.gz','rt') as f:return json.load(f)


def table(seed):
    rows = []
    for r in read(RESULTS / f'seed{seed}.json'):
        rows.append(dict(dot=r['dot'], parked=r['parked'], park_s=r['park_s'],
            park_error_cm=100*r['park_error_vs_expert_m'] if r['parked'] and r.get('park_error_vs_expert_m') is not None else np.nan,
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
        parked = r['park_s'] is not None
        origin = r['park_s'] if parked else 0.
        if not r['log']:
            ax.text(.5,.5,'No PARK',ha='center',transform=ax.transAxes)
        else:
            log = pd.DataFrame(r['log']); t = log.t-origin
            ax.plot(t,log.ee_mm,label='True distance (scoring only)')
            ax.plot(t,1000*log.ttp_s,'--',label='Predicted STOP distance')
            ax.axhline(10,color='grey',ls=':')
            if r['stop_s'] is not None: ax.axvline(r['stop_s']-origin,color='black',lw=.7)
            if not parked: ax.text(.5,.95,'No PARK',ha='center',va='top',transform=ax.transAxes)
        ax.set(xlabel='Seconds after PARK' if parked else 'Seconds since reset',ylabel='mm')
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


def build_viewer(seed, results=RESULTS, split='validation', embed_camera=True):

    sys.path.insert(0,str(OLD))
    import replay
    ghosts = read(results / f'expert_{split}.json'); runs=[]
    def frame(f):
        out = dict(f, command=f.get('wheel',[0]*4), park_probability=float(f['parked']), speed_m_s=0., tau=f.get('tau',[0]*7))
        if embed_camera and f.get('camera_src'):
            out['camera_src'] = 'data:image/png;base64,' + base64.b64encode((results/f['camera_src']).read_bytes()).decode('ascii')
        return out
    for r in read(results/f'seed{seed}.json'):
        if not r['frames']: continue  # Failed attempts remain in the score table; there is no motion to replay.
        e=ghosts[str(r['dot'])]
        runs.append(dict(bearing_deg=r['dot'],split='test' if split == 'test' else 'validation',target=e['target'],
            metrics=dict(model_park_s=r['park_s'],expert_park_s=e['park_s'],park_xy_error_m=r['park_error_vs_expert_m'] if r['parked'] else None,
                         park_yaw_error_deg=None if r.get('park_heading_error_rad') is None else float(np.degrees(r['park_heading_error_rad'])),final_xy_error_m=r['final_base_error_vs_expert_m'],stop_mm=r['at_network_stop_mm'],closest_mm=r['min_mm']),
            model=dict(frames=[frame(f) for f in r['frames']],final=dict(time_s=r['frames'][-1]['time_s'])),
            expert=dict(frames=[frame(f) for f in e['frames']],final=dict(time_s=e['frames'][-1]['time_s']))))
    names=read(HERE.parent/'driving_v2/closed_loop_native/result_wheels_live_dot.json')['joint_names']
    full_episode=any(f.get('arm_control')=='learned-from-reset' for r in runs for f in r['model']['frames'][:1])
    payload=dict(joint_names=names,runs=runs,geometry=replay.geometry(replay.native_root(),names),
                 recording_description=f'Seed {seed}; {len(runs)} recorded {split} targets. Learned driving and reaching. Ghost: recorded expert. No trajectory truncation. Camera inset: actual recorded frames, 128 px preview of the 256 px sensor.')
    page=(OLD/'replay_template.html').read_text()
    page=page.replace('<aside>','<aside><section><img id="sensorImage" alt="Recorded front camera" style="width:100%;image-rendering:auto"><p class="note">Actual camera at the displayed robot state</p></section>')
    page=page.replace('function text(m,e){','function text(m,e){if(m.camera_src){$("sensorImage").src=m.camera_src;}')
    page=page.replace("${m.parked?'PARK':", "${m.stopped?'STOP':m.parked?'PARK':")
    page=page.replace('${r.bearing_deg}°','target ${r.bearing_deg}')
    page=page.replace('Select train or test run',f'Select {split} run')
    page=page.replace('Recorded robot motion · model + ghost expert',f'Golden data · seed {seed} · model + ghost expert')
    page=page.replace('Every target is 1.1 m from the origin and 0.5 m high.','Targets vary in bearing, radius and height.')
    page=page.replace('Model: 30 Hz. Expert and physics: 1,000 Hz. Park probability ≥ 0.5 latches zero wheel commands; no recovery. The supplied controller holds the arm.',
        'Driving and camera: 30 Hz. Learned arm torques and physics: 1,000 Hz. PARK latches zero wheels. Before PARK, native arm-hold torques are not logged here (n/a). After STOP, the learned network runs for another 0.5 s.')
    page=page.replace("for(const name of ['Front right','Front left','Back left','Back right'])", "for(const name of ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6','Joint 7'])")
    page=page.replace('m.command[i].toFixed(2)',"(m.parked?m.tau[i].toFixed(2):'n/a')").replace('e.command[i].toFixed(2)','e.tau[i].toFixed(2)')
    page=page.replace('Command · rad/s','Torque · N m')
    if full_episode:
        page=page.replace('Actual camera at the displayed robot state','Rendered camera. Delayed policies may use an earlier frame.')
        page=page.replace("(m.parked?m.tau[i].toFixed(2):'n/a')", "((m.sent_arm_torque||m.tau)[i].toFixed(2))")
        page=page.replace('Before PARK, native arm-hold torques are not logged here (n/a).',
                          'Learned arm torque from reset; no added PD hold or gravity compensation. Torque table shows final applied arm torque.')
        page=page.replace('</p></section>','</p></section><section id="policyTelemetry"></section>',1)
        page=page.replace('</style>','.view{grid-template-columns:minmax(0,1fr) 280px}aside{max-height:510px;overflow:auto;font-size:12px}#sensorImage{height:130px;object-fit:contain}#policyTelemetry p{margin:4px 0}#policyTelemetry td,#policyTelemetry th{padding:2px} @media(max-width:900px){.view{grid-template-columns:1fr}aside{max-height:none}} </style>')
        page=page.replace('function text(m,e){', '''function text(m,e){
          const number=(x,d=3)=>x==null?'n/a':Number(x).toFixed(d);
          const ws=m.wheel_speed_input||[], wc=m.wheel||[];
          $('policyTelemetry').innerHTML='<b>Wheel speeds · rad/s</b><table><tr><th>Wheel</th><th>Encoder</th><th>Command</th></tr>'+['FR','FL','BL','BR'].map((n,i)=>'<tr><td>'+n+'</td><td>'+number(ws[i])+'</td><td>'+number(wc[i])+'</td></tr>').join('')+'</table>'+
          '<p>PARK prediction: '+number(m.driving_ttp_s)+' s; threshold '+number(m.park_threshold??0.25)+' s; latched '+Boolean(m.parked)+'</p>'+
          '<p>STOP prediction: '+number(m.ttp_s==null?null:m.ttp_s*1000,1)+' mm; threshold '+number((m.stop_threshold??0.01)*1000,1)+' mm; latched '+Boolean(m.stopped)+'</p>'+
          '<details><summary>Requested arm torques and capture timing</summary><p>Arm command · N m: '+(m.applied_arm_torque||m.tau).map(x=>number(x,2)).join(', ')+'</p>'+
          '<p>Camera capture step: '+number(m.camera_capture_step,0)+'; encoder capture step: '+number(m.arm_capture_step,0)+'</p></details>';
        ''')
    page=page.replace('\\nPark probability ${m.park_probability.toFixed(3)} · Speed ${m.speed_m_s.toFixed(2)} / ${e.speed_m_s.toFixed(2)} m/s',
        '\\nGripper distance: model ${m.ee_mm.toFixed(1)} mm / expert ${e.ee_mm.toFixed(1)} mm')
    page=page.replace('Heading difference ${fmt(m.park_yaw_error_deg,1)}°','Heading difference ${fmt(m.park_yaw_error_deg,1)}°<br>Distance at STOP ${fmt(m.stop_mm,1)} mm; closest ${fmt(m.closest_mm,1)} mm')
    page=page.replace('__PLOTLY__',(OLD/'replay_assets/plotly.min.js').read_text())
    page=page.replace('__DATA__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))
    out=results/f'viewer_golden_seed{seed}.html';out.write_text(page);return out
