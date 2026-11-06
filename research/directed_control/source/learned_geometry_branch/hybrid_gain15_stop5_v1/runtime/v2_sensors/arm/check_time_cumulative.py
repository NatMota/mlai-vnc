"""Verify frozen models, complete target coverage and replay image references."""
import json
from pathlib import Path
import numpy as np
from render_golden import sha


def check(root):
    cfg=json.loads((root/'config.json').read_text());ids=set(cfg['cases']);models={};images=0;counts={};by_variant={}
    for name,*_ in cfg['lineup']:
        out=root/name;c=json.loads((out/'config.json').read_text())
        for filename,digest in c['model_hashes'].items():assert sha(out/'models'/filename)==digest
        rows=json.loads((out/'seed7.json').read_text())
        assert len(rows)==100 and {r['dot'] for r in rows}==ids
        by_variant[name]={r['dot']:r for r in rows}
        assert all(r['seed']==7 and r['split']=='test' for r in rows)
        for r in rows:
            if r['success']:assert r['parked'] and r['stop_s'] is not None and r['at_network_stop_mm']<=10
            for f in r['frames']:
                assert f.get('camera_src') and (out/f['camera_src']).is_file()
                images+=1
        counts[name]=dict(attempts=len(rows),replayable=sum(bool(r['frames']) for r in rows))
        models[name]=dict(np.load(out/'models/reaching_seed7.npz',allow_pickle=False))
    first=models['height'];keys=[k for k in first if k.startswith('stopnet.')]
    assert keys
    for w in models.values():
        assert sorted(k for k in w if k.startswith('stopnet.'))==sorted(keys)
        for k in keys:np.testing.assert_array_equal(w[k],first[k])
    drive=[sha(root/n/'models/driving_seed7.npz') for n in ['no_speed','longer_arm','wider_arm']]
    assert len(set(drive))==1
    assert sha(root/'height/models/reaching_seed7.npz')==sha(root/'no_speed/models/reaching_seed7.npz')
    for dot in ids:
        reference=by_variant['no_speed'][dot]
        for name in ('longer_arm','wider_arm'):
            r=by_variant[name][dot]
            assert r['parked']==reference['parked']
            if r['parked']:
                np.testing.assert_allclose([r['park_s'],r['park_error_vs_expert_m']], [reference['park_s'],reference['park_error_vs_expert_m']],atol=1e-9,rtol=0)
    source=Path(__file__).parent
    for name,digest in cfg['source_hashes'].items():
        p=source/name if (source/name).exists() else source.parent/'driving_v2'/name
        assert sha(p)==digest
    result=dict(counts=counts,camera_frames=images,all_camera_files_present=True,
        unchanged_stop_network=True,identical_parking_for_arm_variants=True,unchanged_driving_for_arm_variants=True,
        unchanged_arm_for_driving_variants=True,frozen_hashes_verified=True)
    (root/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':check(Path(__file__).resolve().parent/'results/time_cumulative_test')
