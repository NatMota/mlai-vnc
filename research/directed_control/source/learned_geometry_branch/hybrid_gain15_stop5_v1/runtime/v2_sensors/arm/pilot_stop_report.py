"""Summarise saved validation-only STOP audits without choosing a threshold."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('models',type=Path);a=p.parse_args();out={}
    for version in ('before','after'):
        groups=json.loads((a.models/f'offline_{version}.json').read_text());summary={}
        for group,rows in groups.items():
            hits=[r['true_mm_at_first_prediction'] for r in rows if r['true_mm_at_first_prediction'] is not None]
            near=[r['near20_mae_mm'] for r in rows if r['near20_mae_mm'] is not None]
            summary[group]=dict(runs=len(rows),rows=sum(r['rows'] for r in rows),
                equal_run_clipped_label_mae_mm=float(np.mean([r['clipped_label_mae_mm'] for r in rows])),
                near20_equal_run_mae_mm=float(np.mean(near)) if near else None,near20_runs=len(near),
                first_crossings=len(hits),first_crossings_within10mm=sum(v<=10 for v in hits),
                median_true_mm_at_first_crossing=float(np.median(hits)) if hits else None,
                threshold_confusion={str(mm):{k:sum(r['threshold_confusion'][str(mm)][k] for r in rows) for k in ('tp','fp','fn')} for mm in (5,10,15,20)})
        out[version]=summary
    alignment=[]
    cfg=json.loads((a.models/'config.json').read_text())
    for root in map(Path,cfg['reference_config']['dagger']):
        for row in json.loads((root/'summary.json').read_text()):
            if row['stop_s'] is None:continue
            step=round(row['stop_s']*1000)
            with np.load(root/f"dot{row['bearing']}.npz",allow_pickle=False) as z:
                error=abs(float(z['arm_distance_m'][step])-row['at_network_stop_mm']/1000)
            alignment.append(dict(source=str(root),dot=row['bearing'],stop_step=step,difference_m=error))
    out.update(distance_alignment=dict(check='saved teacher distance versus separately logged simulator distance at exact STOP',events=len(alignment),max_difference_m=max(x['difference_m'] for x in alignment),rows=alignment),
               fixed_threshold_m=.01,threshold_tuned=False,test_files_opened=0)
    (a.models/'offline_summary.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='distance_alignment'},indent=2))

if __name__=='__main__':main()
