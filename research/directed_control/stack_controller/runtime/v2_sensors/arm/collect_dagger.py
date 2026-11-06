"""One on-policy DAgger collection pass, exclusively on training targets."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
import numpy as np
from full_run import one
from validate_golden import case_for,write_json
from render_golden import sha


def attempt(job):
    dot,case,wd,wa,out,condition=job
    row=one(wd,wa,dot,False,case=case,condition=condition,dagger_out=out/f'dot{dot}.npz')
    row.pop('log',None)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','models','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--ids',nargs='+',type=int)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--condition',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    split=json.loads((a.data/'split.json').read_text());ids=a.ids or split['train']
    assert set(ids)<=set(split['train']) and len(ids)==len(set(ids))
    condition=json.loads(a.condition.read_text()) if a.condition else {}
    wd=dict(np.load(a.models/'driving_seed7.npz'));wa=dict(np.load(a.models/'reaching_seed7.npz'))
    native=Path(os.environ['NATIVE_ROOT']);binary=Path(os.environ.get('CURRICULUM_BINARY',native/'build-clang19/sensor_curriculum'))
    write_json(a.out/'config.json',dict(ids=ids,split='train',split_sha256=sha(a.data/'split.json'),condition=condition,
        model_hashes={k:sha(a.models/k) for k in ('driving_seed7.npz','reaching_seed7.npz')},
        native_sha256=sha(binary),full_run_sha256=sha(Path(__file__).with_name('full_run.py')),
        oracle='stateful expert; arm phase follows student PARK, base phase unchanged',
        park_labels='no counterfactual time labels; original demonstrations retain PARK supervision',
        expert_commands_applied=False,test_and_validation_targets_used=False))
    jobs=[(dot,case_for(a.data,dot,'train')[0],wd,wa,a.out,condition) for dot in ids]
    started=time.monotonic();rows=[]
    with ProcessPoolExecutor(a.jobs) as pool:
        futures=[pool.submit(attempt,j) for j in jobs]
        for f in as_completed(futures):
            rows.append(f.result());write_json(a.out/'summary.json',rows)
            write_json(a.out/'progress.json',dict(completed=len(rows),total=len(ids),elapsed_s=time.monotonic()-started))
    assert len(list(a.out.glob('dot*.npz')))==len(ids)
    write_json(a.out/'completion.json',dict(attempts=len(rows),all_accounted=True,elapsed_s=time.monotonic()-started))


if __name__=='__main__':main()
