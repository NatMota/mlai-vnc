"""Choose training dots spread evenly in bearing, distance and height at the same time (a Latin hypercube
selection from the training split), nested so that the 25 are inside the 50, which are inside the 100.

For N dots, each axis is cut into N equal bands. Going through the height bands in order, the dot taken from each
band is the one whose distance band and bearing band are still empty (or, failing that, the one that fills the
emptier of the two). Dots already chosen for a smaller N are kept and count as filling their bands.

usage: python tools/select_dots.py [--sizes 25 50 100] [--out effects/dots_spread.json]
"""
import argparse, json
from pathlib import Path
import numpy as np

NATIVE = Path(__file__).resolve().parents[1]
RANGES = dict(bearing=(-np.pi, np.pi), distance=(1.0, 2.0), height=(0.1, 1.25))


def coords(targets):
    t = np.array(targets)
    return dict(bearing=np.arctan2(t[:, 1], t[:, 0]), distance=np.hypot(t[:, 0], t[:, 1]), height=t[:, 2])


def band(v, axis, n):
    lo, hi = RANGES[axis]
    return np.clip(((v - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def select(c, pool, n, chosen):
    """Extend `chosen` (a list of dot indices) to n dots from `pool`, one per height band where possible."""
    chosen = list(chosen)
    b = {a: band(c[a], a, n) for a in RANGES}
    for h in range(n):
        if any(b['height'][i] == h for i in chosen):
            continue
        cands = [i for i in pool if b['height'][i] == h and i not in chosen]
        if not cands:
            continue
        used_d = {b['distance'][i] for i in chosen}; used_b = {b['bearing'][i] for i in chosen}
        score = [(b['distance'][i] in used_d) + (b['bearing'][i] in used_b) for i in cands]
        chosen.append(cands[int(np.argmin(score))])
    return chosen


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--sizes', type=int, nargs='+', default=[25, 50, 100])
    ap.add_argument('--out', default=str(NATIVE / 'effects/dots_spread.json'))
    a = ap.parse_args()
    targets = json.loads((NATIVE / 'effects/targets_1000.json').read_text())
    pool = json.loads((NATIVE / 'effects/split_1000.json').read_text())['train']
    c = coords(targets); out = {}; chosen = []
    for n in sorted(a.sizes):
        chosen = select(c, pool, n, chosen)
        out[str(n)] = sorted(chosen)
        b = {ax: len({int(x) for x in band(c[ax][chosen], ax, n)}) for ax in RANGES}
        print(f'{n}: {len(chosen)} dots; distinct bands filled: {b}')
    out['rule'] = 'Latin hypercube selection from the training split: one dot per height band, preferring empty distance and bearing bands; nested'
    Path(a.out).write_text(json.dumps(out, indent=1))
