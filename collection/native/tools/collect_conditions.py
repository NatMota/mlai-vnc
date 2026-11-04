"""Collect expert runs for every condition in effects/, on the same dots, unattended.

For each condition and each chunk of dots: the collector writes raw runs to a local scratch folder (fast disk),
pack_runs.py turns each run into one compressed .npz on the output disk (named by its dot index), and the raw
runs are deleted. A chunk whose packed files all exist is skipped, so a restart carries on where it stopped.
A chunk that fails is logged and skipped. At the end, a summary per condition is written to OUT/summary.json.

usage: python tools/collect_conditions.py OUT SCRATCH [--conditions a b ...] [--targets effects/targets_1000.json]
                                                     [--chunk 100] [--workers 6] [--max-steps 20000] [--limit N]
  run from collection/native, with build-clang19/expert_compatible built.
"""
import argparse, json, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

NATIVE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NATIVE / 'tools'))
import pack_runs

CONDITIONS = ['baseline', 'self_collision', 'noise', 'delay', 'damping', 'elasticity', 'inertia']


def log(out, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(out / 'collect.log', 'a') as f:
        f.write(line + '\n')


def pack_one(args):
    run, dest, condition, urdf = args
    pack_runs.pack(run, dest, pack_runs.TruePose(urdf), condition)
    shutil.rmtree(run)
    return dest.name


def summary(folder):
    rows = []
    for f in sorted(folder.glob('run_*.npz')):
        d = np.load(f); m = json.loads(str(d['manifest']))
        rows.append(dict(dot=int(f.stem.split('_')[1]), termination=m['termination'], parked=m['parking_step'] >= 0, steps=m['steps'],
                         final_true_mm=1000 * float(d['hand_distance'][-1]), closest_true_mm=1000 * float(d['hand_distance'].min())))
    if not rows:
        return {}
    term = {}
    for r in rows:
        term[r['termination']] = term.get(r['termination'], 0) + 1
    return dict(runs=len(rows), terminations=term, parked=sum(r['parked'] for r in rows),
                median_final_true_mm=float(np.median([r['final_true_mm'] for r in rows])),
                median_closest_true_mm=float(np.median([r['closest_true_mm'] for r in rows])),
                within_10mm_at_end=sum(r['final_true_mm'] <= 10 for r in rows), median_steps=float(np.median([r['steps'] for r in rows])))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('scratch')
    ap.add_argument('--conditions', nargs='*', default=CONDITIONS); ap.add_argument('--targets', default=str(NATIVE / 'effects/targets_1000.json'))
    ap.add_argument('--chunk', type=int, default=100); ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--max-steps', type=int, default=20000); ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    out, scratch = Path(a.out), Path(a.scratch); out.mkdir(parents=True, exist_ok=True); scratch.mkdir(parents=True, exist_ok=True)
    targets = json.loads(Path(a.targets).read_text())[:a.limit]
    urdf = NATIVE / 'assets/summit_xl_arm_description/summit_xl_complete.urdf'
    binary = os.environ.get('EXPERT_BINARY', str(NATIVE / 'build-clang19/expert_compatible'))
    log(out, f'start: {len(targets)} dots, conditions {a.conditions}, chunk {a.chunk}, workers {a.workers}, max steps {a.max_steps}')
    for cond in a.conditions:
        dest = out / cond; dest.mkdir(exist_ok=True)
        for start in range(0, len(targets), a.chunk):
            dots = list(range(start, min(start + a.chunk, len(targets))))
            if all((dest / f'run_{d}.npz').exists() for d in dots):
                continue
            free = shutil.disk_usage(scratch).free / 1e9
            if free < 5:
                log(out, f'STOP: only {free:.1f} GB free on scratch'); sys.exit(1)
            chunk_targets = scratch / 'chunk_targets.json'; chunk_targets.write_text(json.dumps([targets[d] for d in dots]))
            raw = scratch / f'{cond}_{start}'; shutil.rmtree(raw, ignore_errors=True)
            n = len(dots); workers = min(a.workers, n); repeat = -(-n // workers)
            t0 = time.time()
            r = subprocess.run([binary, '--assets', str(NATIVE / 'assets'), '--effects', str(NATIVE / f'effects/{cond}.json'),
                                '--targets', str(chunk_targets), '--workers', str(workers), '--repeat', str(repeat), '--dynamic',
                                '--max-steps', str(a.max_steps), '--binary', '--out', str(raw)],
                               cwd=NATIVE, env=dict(os.environ, EXPERT_SKIP_VISUALS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                               capture_output=True, text=True)
            if r.returncode != 0:
                log(out, f'{cond} dots {dots[0]}-{dots[-1]}: collector failed ({r.returncode}): {r.stderr[-300:]}')
            jobs = []
            for mf in raw.rglob('manifest.json'):
                k = int(mf.parent.name.split('_')[1])               # episode index within the chunk
                if k < n:
                    jobs.append((mf.parent, dest / f'run_{dots[k]}.npz', cond, urdf))
            with ProcessPoolExecutor(a.workers) as pool:
                list(pool.map(pack_one, jobs))
            shutil.rmtree(raw, ignore_errors=True)
            log(out, f'{cond} dots {dots[0]}-{dots[-1]}: {len(jobs)} runs packed in {time.time() - t0:.0f} s')
        s = summary(dest); (dest / 'summary.json').write_text(json.dumps(s, indent=1))
        log(out, f'{cond} done: {json.dumps(s)}')
    (out / 'summary.json').write_text(json.dumps({c: json.loads((out / c / 'summary.json').read_text()) for c in a.conditions if (out / c / 'summary.json').exists()}, indent=1))
    log(out, 'all done')
