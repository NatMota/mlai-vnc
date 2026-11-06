"""Recompute published counts/medians from original records; never run training or overwrite them."""
import csv
import argparse
import json
import re
from pathlib import Path
from statistics import median

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--audit', type=Path, required=True, help='Extracted analysis evidence package')
parser.add_argument('--out', type=Path, required=True, help='New directory for recomputed results')
args = parser.parse_args()
HERE = args.audit.resolve()
REPO = args.repo.resolve()
V2 = REPO / 'v2_sensors'
ART = HERE / 'artifacts'
rows = []

def read(path):
    return json.loads(path.read_text())

def record(owner, claim, value, expected, source, tolerance=0):
    ok = expected is None or abs(value - expected) <= tolerance
    rows.append(dict(owner=owner, claim=claim, recomputed=value, reported=expected,
                     status='MATCH' if expected is not None and ok else 'RECOMPUTED' if ok else 'MISMATCH', source=str(source)))

# Nat's early driving comparisons.
for name, expected in [('wheels',12),('wheels_start_dot',11),('wheels_live_dot',23),('wheels_live_dot_skid',9)]:
    p = V2 / f'driving_v2/closed_loop_native/result_{name}.json'
    if p.exists():
        data = read(p)['runs']
        record('Nat', name+' parked', sum(bool(x['model']['final']['parked']) for x in data), expected, p)
for name, expected in [('camera_flag',17),('camera_flag_ttp',16),('camera_ttp_noflag_0.35s',6),
                       ('camera_flag_ttp_scaled',14),('camera_flag_ttp_scaled_seed1',0),
                       ('camera_flag_ttp_scaled_seed2',4),('camera_flag_ttp_scaled_seed3',21)]:
    p = V2 / 'driving_v2/camera_loop' / name
    data = [read(x) for x in p.glob('run_*.json')]
    if data:
        record('Nat', name+' parked', sum(bool(x['parked']) for x in data), expected,p)
for p in sorted((V2/'driving_v2/camera_loop').glob('camera_flag_ttp_scaled_tail100*')):
    data=[read(x) for x in p.glob('run_*.json')]
    if data: record('Nat',p.name+' parked',sum(bool(x['parked']) for x in data),None,p)

# Early arm comparisons use their historical metrics, not the later strict-success definition.
prelim = {'arm30_sample':189,'arm30_mean':221,'arm1k':18,'arm1k_e77_dist':29,
          'arm1k_e77_dist_camera':105,'arm1k_e77_dist_notime_camera':189,
          'arm1k_e77_dist_notime_raw_camera':186,'arm1k_e77_dist_notime_phys_camera':57,
          'arm1k_e77_phys_imu_camera':116,'arm1k_e77_phys_imu_noenc_camera':61,
          'arm1k_e77_dist_notime_phys_camera_dots':205,'arm1k_e77_phys_stopnet_camera_dots':181,
          'arm1k_e77_phys_camdot_camera':70}
for name, expected in prelim.items():
    p=V2/f'arm/results/{name}.json'; data=read(p)
    record('Nat',name+' median closest mm',median(x['min_mm'] for x in data),expected,p,.6)
p=V2/'arm/results/arm1k_e77_dist.json'
record('Nat','distance STOP median mm',median(x['at_network_stop_mm'] for x in read(p)),35,p,.6)
p=V2/'arm/results/stop_weight_choice_25.json';w=read(p)['weights']
for k,expected in [('3',11),('10',30),('30',60),('100',109)]:
    record('Nat','STOP weight '+k+' torque loss increase %',100*(w[k]['val_torque']/w['1']['val_torque']-1),expected,p,.6)

# Main complete episodes, all 24 saved conditions. Recalculate independently of summary tables.
expected={'basics_standard':(100,91,0,169),'dart':(100,78,1,61),
          'dagger':(84,69,2,37),'dart_dagger':(87,66,4,34)}
for p in sorted((V2/'arm/results/overnight_20261105').glob('*/test/seed7_metrics.csv')):
    data=list(csv.DictReader(p.open())); name=p.parents[1].name
    assert len(data)==100 and len({x['dot'] for x in data})==100, p
    actual=(sum(x['parked']=='True' for x in data),sum(x['stop_s'] not in ('','None','nan') for x in data),
            sum(x['success']=='True' for x in data),median(float(x['at_stop_mm']) for x in data if x['stop_s'] not in ('','None','nan')))
    for i,label in enumerate(['PARK','STOP','success','at STOP mm']):
        exp=expected.get(name,(None,)*4)[i]
        record('Nat',name+' '+label,actual[i],exp,p,.6 if i==3 else 0)

# Haosheng's released original per-target V1 and V2 records.
root=ART/'540bdcf942a5448baefdc2802271bbf3/eval'
for name,expected in [('abl_all_world_val120',117),('final_fused30_allworld_test120',120),
                      ('fresh500_fused30',492),('fresh500_expert',497),('outside400_fused30',279),('outside400_expert',279)]:
    p=root/(name+'.jsonl');data=[json.loads(x) for x in p.read_text().splitlines()]
    record('Haosheng',name+' reached',sum(bool(x['reached']) for x in data),expected,p)
p=ART/'49e4cd1353c046d581c1ff0373d9664f/outputs/golden/G.jsonl';data=[json.loads(x) for x in p.read_text().splitlines()]
for label,value,expected in [('PARK',sum(x['parked'] for x in data),100),('STOP',sum(x['stop_s'] is not None for x in data),68),
                             ('success',sum(x['success'] for x in data),1),('at STOP mm',median(x['at_stop_mm'] for x in data if x['stop_s'] is not None),17)]:
    record('Haosheng','golden '+label,value,expected,p,.1 if label=='at STOP mm' else 0)

# Vincent's logs preserve rounded per-target observations, sufficient for success counts.
key_counts={'saved_new200.log':195,'saved_test120_seed1.log':117,'saved_val120_seed1.log':117,'nm_gap_new200.log':185}
for p in sorted(ART.glob('*/results/*.log')):
    if p.parent.parent.name in ['540bdcf942a5448baefdc2802271bbf3']:continue
    result=[x for x in p.read_text(errors='replace').splitlines() if x.startswith('RESULT ')]
    if result:record('Vincent',p.name+' successes / '+str(len(result)),sum('reached=True' in x for x in result),key_counts.get(p.name),p)
for p in sorted(ART.glob('*/results/my_arm*.json')):
    data=read(p)
    if not isinstance(data,list):continue
    exp=82 if p.name=='my_arm_test100_final.json' else 76 if p.name=='my_arm_val100_wheel.json' else None
    record('Vincent',p.name+' successes / '+str(len(data)),sum(bool(x['success']) for x in data),exp,p)

# Occlusion: independently group all 1,200 episode records; Table II contains eight groups.
p=HERE/'recovered-occlusion/lighthouse-analysis/analysis/episodes.csv'
if p.exists():
    data=list(csv.DictReader(p.open()))
    assert len(data)==1200
    table={('baseline','val','control'):(100,93,0,183.6),('baseline','val','lighthouse'):(100,93,0,239.1),
           ('recovery','val','control'):(93,70,4,27.9),('recovery','val','lighthouse'):(97,71,5,30.3),
           ('disturbed','val','control'):(86,84,0,276.8),('disturbed','val','lighthouse'):(91,88,0,326.0),
           ('recovery','test','control'):(87,65,5,30.9),('recovery','test','lighthouse'):(93,67,2,34.3)}
    for key in sorted({(x['condition'],x['split'],x['variant']) for x in data}):
        group=[x for x in data if (x['condition'],x['split'],x['variant'])==key]
        assert len(group)==100 and len({x['dot'] for x in group})==100
        vals=[sum(x[k]=='True' for x in group) for k in ['parked','stopped','success']]
        vals.append(median(float(x['stop_mm']) for x in group if x['stopped']=='True'))
        for i,label in enumerate(['PARK','STOP','success','STOP mm']):
            record('Nat','occlusion '+str(key)+' '+label,vals[i],table.get(key,(None,)*4)[i],p,.06 if i==3 else 0)
    p=p.with_name('paired.csv')
    pairs=[x for x in csv.DictReader(p.open()) if x['condition']=='recovery' and x['split']=='val' and x['stopped_control']=='True' and x['stopped_lighthouse']=='True']
    record('Nat','occlusion paired stopping targets',len(pairs),66,p)
    record('Nat','occlusion paired STOP change mm',median(float(x['stop_mm_lighthouse'])-float(x['stop_mm_control']) for x in pairs),.43,p,.006)

out=args.out.resolve();out.mkdir(parents=True, exist_ok=False)
with (out/'archived-results-check.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(out/'archived-results-check.json').write_text(json.dumps(rows,indent=2))
print('Checked',len(rows),'metrics;',sum(x['status']=='MATCH' for x in rows),'matched stated values;',sum(x['status']=='MISMATCH' for x in rows),'mismatches')
for x in rows:
    if x['status']=='MISMATCH':print(x)

assert sum(x['status']=='MATCH' for x in rows) >= 95, 'Expected report checks are missing'
assert not any(x['status']=='MISMATCH' for x in rows), 'One or more reported values do not match'
