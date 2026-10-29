"""Compare the ten original inspected targets with a native --targets batch."""
import argparse,json,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('original',type=Path,help='Original inspection folder')
p.add_argument('candidate',type=Path,help='Native batch containing run_0 through run_9')
p.add_argument('--tag',default='verified')
args=p.parse_args()
root=Path(__file__).resolve().parents[1]
names=['run_ahead_001','run_001','run_003','run_004','run_005','run_006','run_007','run_008','run_009','run_010']
records=[]
for i,name in enumerate(names):
    report=root/'results'/f'{args.tag}_{i}.json'
    r=subprocess.run([sys.executable,str(root/'tools/compare.py'),str(args.original/name),str(args.candidate/f'run_{i}'),'--report',str(report)],capture_output=True,text=True)
    if not report.exists():raise RuntimeError(r.stdout+r.stderr)
    d=json.loads(report.read_text())
    records.append({'target_index':i,'reference':name,'passed':d['passed'],'exact_recorded_values':d['exact_recorded_values'],'simulation_timestamps_equal':d['simulation_timestamps_equal'],'max_table_error':max(x['max_absolute_error'] for x in d['tables'].values()),'max_tensor_error':max(x.get('max_absolute_error',0) for x in d.get('tensors',{}).values())})
result={'all_passed':all(x['passed'] for x in records),'all_exact':all(x['exact_recorded_values'] for x in records),'runs':records}
(root/'results'/f'{args.tag}_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['all_passed'] and result['all_exact'] else 1)
