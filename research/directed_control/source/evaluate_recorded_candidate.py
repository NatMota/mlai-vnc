"""Isolated frozen-candidate evaluation with diagnostic-only 1kHz truth and optional camera previews."""
import argparse, hashlib, importlib.util, json, os, sys, types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

R = Path(__file__).parent
F = R / 'learned_geometry_branch/hybrid_gain15_stop5_v1'
E = None
CAMERA = False

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(p)

def instrumented_source():
    path = F / 'runtime/v2_sensors/arm/full_run_gated.py'
    source = path.read_text()
    marker = '    import pybullet as p\n'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '    global RESEARCH_TRUTH\n    RESEARCH_TRUTH = []\n')
    marker = "            f = state['frame']; t = f['time_s']; step = f['step']\n"
    assert source.count(marker) == 1
    source = source.replace(marker, marker + "            RESEARCH_TRUTH.append((t, state['ee_distance_m']))\n")
    marker = "        f = state['frame']\n        park_frame = "
    assert source.count(marker) == 1
    source = source.replace(marker, "        f = state['frame']\n        RESEARCH_TRUTH.append((f['time_s'], state['ee_distance_m']))\n        park_frame = ")
    return path, source

def setup(runtime, data, split, camera):
    global E, CAMERA
    CAMERA = camera
    sys.path.insert(0, str(F))
    os.environ['GEOMETRY_CAMERA'] = 'calibrated_arm_only'
    spec = importlib.util.spec_from_file_location('frozen_evaluation', F / 'evaluate_hybrid_geometry.py')
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    spec = importlib.util.spec_from_file_location('candidate_runtime', runtime)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    E.GeometryPolicy = policy.GeometryPolicy
    original_case = E.case_for
    E.case_for = lambda ignored, dot, ignored_split: original_case(Path(data), dot, split)
    original_path, source = instrumented_source()
    module = types.ModuleType('research_recorded_full_run')
    module.__file__ = str(original_path)
    exec(compile(source, str(original_path), 'exec'), module.__dict__)
    module.PIXI = F / 'exec_native'
    E.full_run = module

def run(job):
    import numpy as np
    dot, out, name = job
    out = Path(out)
    original_one = E.full_run.one
    def recorded_one(*args, **kwargs):
        if CAMERA:
            kwargs['camera_dir'] = out / f'camera_{dot}'
        row = original_one(*args, **kwargs)
        truth = np.asarray(E.full_run.RESEARCH_TRUTH, dtype=np.float64)
        np.savez_compressed(out / f'truth_{dot}.npz', time_s=truth[:, 0], distance_m=truth[:, 1])
        stop = row.get('stop_s')
        post = truth[truth[:, 0] >= stop] if stop is not None else truth[:0]
        row['post_stop_peak_mm'] = float(post[:, 1].max() * 1000) if len(post) else None
        row['within_10mm_through_post_stop'] = bool(row['success'] and len(post) and row['post_stop_peak_mm'] <= 10)
        row['diagnostic_truth_rate_hz'] = 1000
        return row
    E.full_run.one = recorded_one
    try:
        row = E.run_one((dot, str(F / 'models'), str(out)))
    finally:
        E.full_run.one = original_one
    row['name'] = name
    path = out / f'run_{dot}.json'
    full = json.loads(path.read_text())
    full['name'] = name
    save(path, full)
    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=F / 'data')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--ids', nargs='+', type=int)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--camera', action='store_true')
    a = parser.parse_args()
    split = json.loads((a.data / 'split.json').read_text())
    ids = a.ids or split[a.split]
    assert len(ids) == len(set(ids)) and set(ids) <= set(split[a.split])
    a.out.mkdir(exist_ok=False, parents=True)
    _, source = instrumented_source()
    (a.out / 'full_run_instrumented_source.py').write_text(source)
    config = dict(name=a.out.name, ids=ids, split=a.split, camera=a.camera,
                  runtime=str(a.runtime.resolve()), runtime_sha256=sha(a.runtime),
                  baseline_manifest_sha256=sha(F / 'frozen_manifest.json'), runner_sha256=sha(__file__),
                  evaluation_sha256=sha(F / 'evaluate_hybrid_geometry.py'),
                  instrumented_source_sha256=sha(a.out / 'full_run_instrumented_source.py'),
                  data_sha256={str(i): sha(a.data / 'expert' / a.split / f'run_{i}.npz') for i in ids},
                  models={p.name: sha(p) for p in (F / 'models').glob('*.npz')},
                  native_sha256=sha(F / 'native/sensor_curriculum'),
                  truth_usage='Diagnostic-only recording and scoring; never supplied to policy.')
    save(a.out / 'config.json', config)
    rows = []
    with ProcessPoolExecutor(a.jobs, initializer=setup,
                             initargs=(str(a.runtime.resolve()), str(a.data.resolve()), a.split, a.camera)) as pool:
        futures = [pool.submit(run, (dot, str(a.out.resolve()), a.out.name)) for dot in ids]
        for f in as_completed(futures):
            row = f.result()
            rows.append(row)
            save(a.out / 'summary.json', rows)
            print(json.dumps({k: row.get(k) for k in ['dot', 'status', 'at_network_stop_mm', 'end_mm', 'post_stop_peak_mm', 'elapsed_s']}), flush=True)
            if row.get('error'):
                raise RuntimeError(row['error'])
    assert sha(a.runtime) == config['runtime_sha256']
    save(a.out / 'completion.json', dict(attempts=len(rows), expected=len(ids)))

if __name__ == '__main__':
    main()
