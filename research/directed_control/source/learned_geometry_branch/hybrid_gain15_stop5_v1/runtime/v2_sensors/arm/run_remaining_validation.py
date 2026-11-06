"""Complete the untouched validation half with frozen models; no notebook or MLflow writes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from run_curriculum_ladder import save


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'data', 'native'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    all_ids = json.loads((args.data / 'split.json').read_text())['val']
    first = json.loads((root / 'validation_50_plan.json').read_text())['ids']
    remaining = [i for i in all_ids if i not in first]
    assert len(all_ids) == 100 and len(set(all_ids)) == 100
    assert len(first) == len(remaining) == 50 and not set(first) & set(remaining)
    runtime = root / 'validation_runtime'
    native_files = [runtime / 'native' / n for n in ('sensor_curriculum', 'sensor_curriculum.cpp')]
    source = runtime / 'v2_sensors/arm'
    jobs = [(7, 'B'), (1, 'A'), (1, 'B'), (2, 'A'), (2, 'B')]
    plan = dict(ids=remaining, first_half_ids=first, split='val', seeds=[7, 1, 2],
                new_batches=5, reused_reference='A seed 7 already has all 100 validation targets',
                test_used=False, training=False, notebook_writes=False, mlflow_writes=False,
                renderer='egl', workers=2,
                native_hashes={str(p): sha(p) for p in native_files})
    with (root / 'validation_remaining_50_plan.json').open('x') as f:
        json.dump(plan, f, indent=2)
    started = time.monotonic()
    completed = []

    def status(stage, **extra):
        save(root / 'remaining_validation_status.json', dict(
            stage=stage, completed=completed, elapsed_s=time.monotonic() - started,
            notebook_writes=False, mlflow_writes=False, **extra))

    env = dict(os.environ, NATIVE_ROOT=str(args.native),
               CURRICULUM_BINARY=str(native_files[0]), CURRICULUM_SOURCE=str(native_files[1]),
               CAMERA_RENDERER='egl',
               CAMERA_URDF=str(args.native / 'assets/summit_xl_arm_description/summit_xl_complete.urdf'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    try:
        status('waiting_for_gpu_lock')
        with open('/tmp/comp0245-validation-gpu.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for seed, short in jobs:
                old = json.loads((root / f'validation_{short}_seed{seed}_50/config.json').read_text())
                models = Path(old['models'])
                assert sorted(old['native_hashes'].values()) == sorted(plan['native_hashes'].values())
                for name, digest in old['model_hashes'].items():
                    assert sha(models / name) == digest, f'Model changed: {models / name}'
                out = root / f'validation_{short}_seed{seed}_remaining50'
                status('running', seed=seed, architecture=short, out=str(out))
                cmd = [sys.executable, source / 'evaluate_curriculum.py', '--data', args.data,
                       '--models', models, '--out', out, '--split', 'val', '--seed', seed,
                       '--jobs', 2, '--ids', *remaining]
                with (root / f'validation_{short}_seed{seed}_remaining50.log').open('x') as log:
                    subprocess.run([str(v) for v in cmd], env=env, check=True,
                                   stdout=log, stderr=subprocess.STDOUT)
                rows = json.loads((out / 'summary.json').read_text())
                assert len(rows) == 50 and {r['dot'] for r in rows} == set(remaining)
                assert all(r['seed'] == seed for r in rows)
                if any(r['status'] in ('execution_failure', 'numerical_failure') for r in rows):
                    raise RuntimeError(f'Execution failure retained in {out}')
                completed.append(dict(seed=seed, architecture=short, out=str(out)))
        records = []
        for seed in (7, 1, 2):
            for short in ('A', 'B'):
                paths = ([root.parent / 'overnight_20261105/basics_standard/val/summary.json']
                         if (seed, short) == (7, 'A') else
                         [root / f'validation_{short}_seed{seed}_{suffix}/summary.json'
                          for suffix in ('50', 'remaining50')])
                rows = [r for p in paths for r in json.loads(p.read_text())]
                assert len(rows) == 100 and {r['dot'] for r in rows} == set(all_ids)
                stop = [r['at_network_stop_mm'] for r in rows if r.get('at_network_stop_mm') is not None]
                closest = [r['min_mm'] for r in rows if r.get('min_mm') is not None]
                records.append(dict(seed=seed, architecture=short, attempts=100,
                    parked=sum(r['parked'] for r in rows), stopped=len(stop),
                    success_10mm=sum(r['success'] for r in rows), closest_recorded=len(closest),
                    median_closest_mm=statistics.median(closest) if closest else None,
                    median_stop_mm=statistics.median(stop) if stop else None,
                    sources=[str(p) for p in paths]))
        save(root / 'paired_validation_100_comparison.json', records)
        status('complete', models_compared=6, targets_per_model=100)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
