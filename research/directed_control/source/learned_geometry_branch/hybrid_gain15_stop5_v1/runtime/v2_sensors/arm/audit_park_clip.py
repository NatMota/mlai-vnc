"""Compare PARK first triggers on recorded validation trajectories, never test data."""
import argparse
import json
from pathlib import Path
import numpy as np
from train_golden import load_data, infer_export, sensors
from render_golden import frame_steps, sha
from driving_height import save


def audit(data, root):
    records, provenance = load_data(data, driving_height=True, park_target='distance', park_scale_m=.5)
    names = ['park_distance_four', 'park_weight10', 'park_clip05_e300', 'park_clip05_e1000']
    weights = {n: dict(np.load(root/n/'models/no_speed11/driving_seed7.npz', allow_pickle=False)) for n in names}
    rows = []
    for r in records['val']:
        with np.load(data/'expert/val'/f"run_{r['dot']}.npz", allow_pickle=False) as a:
            m = json.loads(str(a['manifest'])); park = int(m['parking_step'])
            frames = frame_steps(int(m['steps'])); steps = np.r_[frames[frames < park], frames[frames >= park][:100]]
            actual = np.linalg.norm(a['base_position'][steps, :2]-a['base_position'][park, :2], axis=1)
        x = r['drive'][0][:, sensors.DRIVING_NO_SPEED_COLS]
        for name, w in weights.items():
            _, pred = infer_export(w, x, 'driving'); pred *= float(w['park_distance_scale_m'])
            cross = np.flatnonzero(pred <= .05); first = int(cross[0]) if len(cross) else None
            near = (actual <= .05) & (steps <= park)
            rows.append(dict(model=name, dot=r['dot'], actual_at_trigger_m=None if first is None else float(actual[first]),
                             predicted_at_trigger_m=None if first is None else float(pred[first]),
                             first_trigger_s=None if first is None else float(steps[first]/1000),
                             near_mae_m=float(np.abs(pred[near]-actual[near]).mean()) if near.any() else None))
    summaries=[]
    for name in names:
        rr=[r for r in rows if r['model']==name];hit=[r for r in rr if r['actual_at_trigger_m'] is not None]
        summaries.append(dict(model=name,runs=len(rr),ever_trigger=len(hit),
            within_5cm=sum(r['actual_at_trigger_m']<=.05 for r in hit),
            over_10cm=sum(r['actual_at_trigger_m']>.1 for r in hit),
            median_actual_at_trigger_m=float(np.median([r['actual_at_trigger_m'] for r in hit])),
            mean_run_near_mae_m=float(np.mean([r['near_mae_m'] for r in rr if r['near_mae_m'] is not None]))))
    a=json.loads((root/'park_clip05_e300/models/no_speed11/driving_seed7_history.json').read_text())
    b=json.loads((root/'park_clip05_e1000/models/no_speed11/driving_seed7_history.json').read_text())
    fields=['train_command','train_aux','train_total','val_command','val_aux','val_total']
    max_diff=max(abs(x[k]-y[k]) for x,y in zip(a,b) for k in fields)
    assert len(a)==301 and len(b)==1001 and max_diff<1e-6
    baseline_arm=sha(root/'park_distance_four/arms/wider.npz')
    for name in names[2:]:
        assert sha(root/name/'arms/wider.npz')==baseline_arm
        w=weights[name];assert float(w['park_distance_scale_m'])==.5 and float(w['info_auxiliary_loss_weight'])==1
        np.testing.assert_allclose(w['park_distance_threshold_m'],.05)
        h=json.loads((root/name/'models/no_speed11/driving_seed7_history.json').read_text())
        chosen=json.loads((root/name/'models/no_speed11/driving_seed7_summary.json').read_text())
        assert chosen['selected_epoch']==min(h,key=lambda r:r['val_total'])['epoch']
        assert all(abs(r['val_total']-r['val_command']-r['val_aux'])<1e-6 for r in h)
    result=dict(summary=summaries,per_run=rows,checks=dict(first_300_updates_max_difference=max_diff,
                frozen_arm_identical=True,checkpoint_selection_verified=True,test_trajectories_opened=0),
                split_sha256=provenance['split_sha256'],source_sha256=sha(Path(__file__)))
    for name in names[2:]:save(root/name/'offline_audit.json',result)
    print(json.dumps(dict(summary=summaries,checks=result['checks']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();audit(a.data,a.root)
