"""Shared notebook helpers for the camera checks: load a saved check, summarise it, show image grids."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load(path):
    cam = np.load(path)
    f = pd.DataFrame(json.loads(str(cam['rows'])))
    f['dir_err_deg'] = np.abs((f.est_dir_deg - f.true_dir_deg + 180) % 360 - 180)
    f['dist_err_cm'] = 100 * (f.est_dist_m - f.true_dist_m).abs()
    return cam, f


def summary(f):
    print(f'{len(f)} frames from {f.bearing.nunique()} bearings')
    print('Detector vs ground truth:', f"missed {int(((~f.found) & (f.true_visible_px > 0)).sum())},",
          f"false detections {int((f.found & (f.true_visible_px == 0)).sum())},",
          f"red pixels that were not the dot {int(f.false_red_px.sum())}")
    per = f.groupby('bearing').agg(seen_at_start=('found', 'first'), seen_at_park=('found', 'last'),
        fraction_seen=('found', 'mean'), dot_pixels=('dot_px', lambda s: s[s > 0].median()),
        direction_error_deg=('dir_err_deg', 'median'), distance_error_cm=('dist_err_cm', 'median'))
    print(f"Seen at start: {int(per.seen_at_start.sum())} of {len(per)} | seen at park: {int(per.seen_at_park.sum())} of {len(per)}")
    print(f"When seen: median direction error {f.dir_err_deg.median():.2f}°, median distance error {f.dist_err_cm.median():.2f} cm")
    return per.round(2)


def grid(cam, f, moment, label):
    fig, axes = plt.subplots(5, 6, figsize=(15, 13))
    for ax, im, b in zip(axes.flat, cam[f'img_{moment}'], cam['bearings']):
        rows = f[f.bearing == b]
        found = rows.iloc[0].found if moment == 'start' else rows.iloc[-1].found
        ax.imshow(im); ax.axis('off')
        ax.set_title(f"{b}°  {'dot found' if found else 'no dot'}", color='green' if found else 'red', fontsize=10)
    fig.suptitle(f'{label} at {moment}', fontsize=14); plt.tight_layout(); plt.show()
