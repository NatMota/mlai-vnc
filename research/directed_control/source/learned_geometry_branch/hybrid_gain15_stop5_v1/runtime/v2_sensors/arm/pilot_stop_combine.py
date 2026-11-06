"""Replace only the independent STOP weights in a frozen torque pilot export."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--torque-models',type=Path,required=True);p.add_argument('--stop-models',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    original=a.torque_models/'reaching_seed7.npz';newstop=a.stop_models/'stop_seed7.npz'
    w=dict(np.load(original,allow_pickle=False));s=dict(np.load(newstop,allow_pickle=False))
    assert s['0.weight'].shape==(32,18) and s['2.weight'].shape==(16,32) and s['4.weight'].shape==(1,16)
    assert float(w['ttp_clip'])==.25 and np.isclose(float(w['ttp_thresh']),.01)
    assert 'export_input_mean' in s and 'export_input_scale' in s
    before={k:v.copy() for k,v in w.items() if not k.startswith('stopnet.') and k!='info_stopnet_epochs'}
    w={k:v for k,v in w.items() if not k.startswith('stopnet.')}
    w.update({'stopnet.'+k:v for k,v in s.items() if k.endswith(('.weight','.bias')) or k.startswith('export_')})
    w['info_stopnet_epochs']=s['info_epochs']
    for k,v in before.items():np.testing.assert_array_equal(v,w[k])
    a.out.mkdir(parents=True,exist_ok=False)
    np.savez(a.out/'reaching_seed7.npz',**w)
    shutil.copy2(a.torque_models/'driving_seed7.npz',a.out/'driving_seed7.npz')
    (a.out/'composition.json').write_text(json.dumps(dict(torque_input=str(original),torque_input_sha256=sha(original),stop_input=str(newstop),stop_input_sha256=sha(newstop),model_hashes={k:sha(a.out/k) for k in ('driving_seed7.npz','reaching_seed7.npz')},preserved_non_stop_arrays=len(before),stop_owns_normalisation=True,threshold_changed=False),indent=2)+'\n')

if __name__=='__main__':main()
