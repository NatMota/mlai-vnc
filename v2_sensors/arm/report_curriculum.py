"""Read completed full-episode evaluations; never train or select a checkpoint."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import report_golden as plots
import report_test_golden as testing

ROOT=Path(os.environ.get('CURRICULUM_RESULTS',Path(__file__).parent/'results/overnight_20261105'))


def select(name,split='test'):
    out=ROOT/name/split
    completed=json.loads((out/'completion.json').read_text())
    assert completed['all_accounted'] and completed['attempts']==100
    testing.RESULTS=out
    return testing


def models(name):
    cfg=json.loads((ROOT/name/'test/config.json').read_text())
    own=ROOT/name/'models'
    return own if (own/'summary.json').exists() else Path(cfg['models'])


def scores(name):
    rows=[]
    for split in ('val','test'):
        rows.append(dict(split=split,**select(name,split).score(7)))
    return pd.DataFrame(rows).set_index('split')


def losses(name):
    plots.MODELS=models(name)
    return plots.losses(7)


def hold_table(name,split='test'):
    select(name,split)
    rows=plots.read(ROOT/name/split/'seed7.json');out=[]
    for r in rows:
        pre=[f for f in r['frames'] if not f['parked']]
        if not pre:
            out.append(dict(dot=r['dot'],status=r['status'],prepark_frames=0));continue
        q=np.array([f['joints'][4:11] for f in pre]);tau=np.array([f['applied_arm_torque'] for f in pre])
        out.append(dict(dot=r['dot'],status=r['status'],prepark_frames=len(pre),
                        max_arm_deviation_rad=float(np.abs(q-q[0]).max()),
                        maximum_command_Nm=float(np.abs(tau).max()),
                        torque_clipped_fraction=float((np.abs(tau)>=29.999).mean())))
    return pd.DataFrame(out).set_index('dot')


def holding(name):
    d=hold_table(name);fig,ax=plt.subplots(1,2,figsize=(12,3.8))
    ax[0].scatter(d.index,d.max_arm_deviation_rad)
    ax[0].set(xlabel='Test target ID',ylabel='Largest joint deviation from reset (rad)',title='Arm during driving')
    ax[1].scatter(d.index,100*d.torque_clipped_fraction)
    ax[1].set(xlabel='Test target ID',ylabel='Clipped command components (%)',title='Before PARK, sampled at camera frames')
    fig.tight_layout();return fig


def build(name):
    score=scores(name)
    score.to_csv(ROOT/name/'scores.csv')
    out=ROOT/name/'test';select(name)
    testing.table(7).to_csv(out/'seed7_metrics.csv')
    hold_table(name).to_csv(out/'holding_metrics.csv')
    page=plots.build_viewer(7,out,'test',embed_camera=False)
    testing.package_replay(7,page)
    return score


def player_url(name):
    base=os.environ.get('GOLDEN_REPLAY_BASE_URL','https://nm-linmint-mlflow.taile381c0.ts.net:8443/golden-replays')
    return f'{base}/overnight_20261105/{name}/test/viewer_golden_seed7.html'


def reading(name):
    s=scores(name);test=s.loc['test'];val=s.loc['val'];h=hold_table(name)
    return (f"Both networks controlled the complete run, including learned arm torques before PARK. "
            f"On validation, {int(val.parked)}/100 parked and {int(val.success_10mm)}/100 met the full task criterion. "
            f"On the frozen test comparison, {int(test.parked)}/100 parked, {int(test.stopped)}/100 issued STOP, "
            f"and {int(test.success_10mm)}/100 were within 10 mm at STOP. "
            f"Median parking error among parked attempts was {test.median_park_error_cm:.1f} cm; "
            f"median distance at STOP among stopped attempts was {test.median_stop_mm:.1f} mm. "
            f"Median maximum pre-PARK joint deviation was {h.max_arm_deviation_rad.median():.3f} rad. "
            "The 0.5 s after STOP is observed with learned torques still active, not a frozen simulation. "
            "These are previously viewed test targets and one seed; this is a frozen subsequent comparison, "
            "not an untouched estimate of generalisation. All timeouts and failures remain in the denominator.")
