"""Compare the prescribed six saved full validation runs, keeping all failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    ids=[6,16,23,45,55,80];configs=[json.loads((d/'config.json').read_text()) for d in (a.old,a.new)]
    for c in configs:assert c['cases']==ids and c['split']=='val' and c['jobs']==2 and not c['condition']
    for k in ('native_hashes','source_hashes','split_sha256','renderer','arm_control'):
        assert configs[0][k]==configs[1][k],k
    assert configs[0]['model_hashes']['driving_seed7.npz']==configs[1]['model_hashes']['driving_seed7.npz']
    results={};table=[]
    for label,d in (('old_stop',a.old),('recovery_stop',a.new)):
        rows=json.loads((d/'summary.json').read_text());assert len(rows)==6 and {r['dot'] for r in rows}==set(ids)
        stop=[r['at_network_stop_mm'] for r in rows if r['at_network_stop_mm'] is not None]
        results[label]=dict(attempts=6,successes=sum(bool(r['success']) for r in rows),parks=sum(bool(r['parked']) for r in rows),stops=len(stop),median_stop_mm=statistics.median(stop) if stop else None,execution_failures=sum(r['status']=='execution_failure' for r in rows),elapsed_s=json.loads((d/'completion.json').read_text())['elapsed_s'])
        for r in rows:table.append(dict(model=label,**{k:r.get(k) for k in ('dot','status','success','parked','park_s','stop_s','at_network_stop_mm','min_mm','end_mm')}))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'validation_comparison.json').write_text(json.dumps(dict(results=results,rows=table,configs=configs,threshold_m=.01,test_files_opened=0),indent=2)+'\n')
    with (a.out/'validation_comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
