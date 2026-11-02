"""Load and summarise test results for the v2 iterations (used by the notebook and by mlflow_runs.py)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REF = json.loads((HERE / 'expert_reference.json').read_text())['bearings']


def seen_at_start():
    return {int(b): json.loads((HERE / f'camera_loop/camera_flag_ttp/run_{b}.json').read_text())['log'][0]['visible'] for b in REF}


def tests(folder, label=None):
    """One row per bearing for a camera-loop result folder."""
    seen, rows = seen_at_start(), []
    for b, x in REF.items():
        r = json.loads((HERE / f'camera_loop/{folder}/run_{b}.json').read_text()); L = r['log']
        pt = next((l['t'] for l in L if l['park_prob'] >= 0.5), None)
        rows.append(dict(model=label or folder, bearing=int(b), split=x['split'], seen_at_start=seen[int(b)], parked=bool(r['parked']),
                         final_error_cm=100 * float(np.hypot(*np.subtract(r['final_base'], x['final_base_xy']))),
                         heading_error_deg=float(np.rad2deg(abs((r['final_yaw'] - x['final_yaw'] + np.pi) % (2 * np.pi) - np.pi))),
                         park_time_vs_expert_s=None if pt is None else pt - x['park_time_s'],
                         camera_distance_error_cm=float(np.mean([100 * abs(l['est_dist'] - l['true_dist']) for l in L if l['visible']] or [np.nan]))))
    return pd.DataFrame(rows)


def summary(df):
    def one(g):
        return pd.Series({'parked (of 25)': int(g.parked.sum()),
                          'dot in view at start (of 11)': int(g[g.seen_at_start].parked.sum()),
                          'dot out of view at start (of 14)': int(g[~g.seen_at_start].parked.sum()),
                          'test runs (of 4)': int(g[g.split == 'test'].parked.sum()),
                          'median final error (cm)': round(g.final_error_cm.median(), 1),
                          'median heading error (°)': round(g.heading_error_deg.median(), 1)})
    out = df.groupby('model', sort=False).apply(one)
    counts = [c for c in out.columns if 'of ' in c]
    out[counts] = out[counts].astype(int)
    return out


def history(name):
    return pd.DataFrame(json.loads((HERE / f'models/{name}_history.json').read_text()))
