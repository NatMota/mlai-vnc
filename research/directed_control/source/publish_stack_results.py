"""Publish completed reserved disturbance evidence; never run or tune a policy."""
from pathlib import Path
import json
import nbformat

R = Path(__file__).parent
B = R / 'learned_geometry_branch'
summary = json.loads((B / 'stack_reserved_test_summary300.json').read_text())
assert summary['complete'] and summary['completed_episodes'] == 300
assert summary['source_hashes_verified']
nb = nbformat.read(R / 'research_notebook.ipynb', as_version=4)
cells = []

def add(kind, ident, source):
    cell = getattr(nbformat.v4, 'new_' + kind + '_cell')(source)
    cell['id'] = ident
    cells.append(cell)

add('markdown', 'rnd-stack-reserved-scope', '''## Frozen disturbance evaluation

This is a separately tuned controller using the same learned coefficients and frozen driving network as the clean research model. It reduces reference speed and feedback gains, compensates the arm camera estimate for capture age using delivered wheel measurements, and recovers its own reference toward the existing training-derived posture bounds. All settings were selected on validation. The candidate then passed 98 of 100 validation episodes, including the exact half-second settling check, before its source hashes and test plan were frozen.

The enabled disturbances are encoder noise (0.0001 rad standard deviation), motor damping (1.0), elasticity (1.0), added motor inertia (0.001 kg m²), nominal 3 ms measurement delay and random starting poses. Reset x and y are each sampled uniformly within ±0.15 m and yaw within ±30°. The same recorded starts are used across sensor-noise seeds 7, 1 and 2. These are **three noise realisations of one frozen model on the same 100 targets**, not three independently trained models or 300 independent targets. Occlusion and bursty recording loss are excluded. Neither DART nor DAgger is used.

The predeclared prediction was at least 95 canonical successes per seed. Every failed attempt remains in the denominator; no test result changes the candidate or triggers a selective rerun. These targets have historical project exposure. The original runner's completion receipt inherits the words “exploratory validation”; its saved configuration and per-episode split explicitly identify these as reserved test evaluations, and the derived audit records that metadata defect.

Canonical success is PARK, STOP and true distance within 10 mm at STOP. Settling additionally checks every 1 kHz sample for the following 0.5 s. STOP-distance medians exclude attempts that never issue STOP, so they must be read with the success and STOP counts.''')
add('code', 'rnd-stack-reserved-table', '''import json, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path.cwd()
B = ROOT/'learned_geometry_branch'
stack = json.loads((B/'stack_reserved_test_summary300.json').read_text())
assert stack['complete'] and stack['completed_episodes'] == 300
stack_rows = []
stack_audits = {}
for seed in [7, 1, 2]:
    folder = 'stack_recovery_reserved_recording_adapter_v1' if seed == 7 else f'stack_reserved_seed{seed}_v1'
    audit = json.loads((B/folder/f'test_seed{seed}'/'reserved_audit.json').read_text())
    assert audit['complete'] and audit['completed'] == 100
    stack_audits[seed] = audit
    stopped = [r for r in audit['rows'] if r['STOP_mm'] is not None]
    stack_rows.append(dict(noise_seed=seed, attempts=100, PARK=audit['PARK'], STOP=audit['STOP'],
        success=audit['canonical_success'], settled=audit['canonical_success_and_settled'],
        median_STOP_mm=audit['median_STOP_mm'], p95_STOP_mm=np.quantile([r['STOP_mm'] for r in stopped], .95),
        median_final_mm=audit['median_final_mm'], internal_flags=len(audit['internal_diagnostic_flags'])))
display(pd.DataFrame(stack_rows).round(3))
print('Predeclared ≥95/100 prediction met per seed:', {s: a['canonical_success'] >= 95 for s, a in stack_audits.items()})
print('Total attempts:', stack['completed_episodes'], '; unique targets:', stack['unique_target_count'])
by_target = {}
for seed, audit in stack_audits.items():
    for row in audit['rows']:
        by_target.setdefault(row['dot'], []).append(int(row['success']))
assert len(by_target) == 100 and all(len(v) == 3 for v in by_target.values())
print('Targets succeeding under all three noise seeds:', sum(all(v) for v in by_target.values()), '/100')
print('Targets failing under all three noise seeds:', sum(not any(v) for v in by_target.values()), '/100')
''')
add('code', 'rnd-stack-reserved-distributions', '''import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for seed, audit in stack_audits.items():
    stop = np.sort([r['STOP_mm'] for r in audit['rows'] if r['STOP_mm'] is not None])
    final = np.sort([r['final_mm'] for r in audit['rows'] if r['final_mm'] is not None])
    axes[0].plot(stop, np.arange(1, len(stop)+1)/100, label=f'Noise seed {seed}')
    axes[1].plot(final, np.arange(1, len(final)+1)/100, label=f'Noise seed {seed}')
for ax, title in zip(axes, ['Distance at STOP', 'Distance at episode end, including failures']):
    ax.axvline(10, color='grey', ls='--'); ax.set(xlabel='True distance (mm)', ylabel='Fraction of all 100 attempts', title=title)
    ax.set_xscale('symlog', linthresh=10); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(ROOT/'frozen_stack_distances.png', dpi=150); plt.show()
''')
add('code', 'rnd-stack-reserved-failures', '''failure_rows = []
for seed, audit in stack_audits.items():
    for row in audit['failures']:
        failure_rows.append(dict(noise_seed=seed, target=row['dot'], status=row['status'], PARK=row['PARK'],
            park_s=row['park_s'], deadline_s=row['limit_s'], STOP_s=row['stop_s'],
            STOP_mm=row['STOP_mm'], final_mm=row['final_mm']))
display(pd.DataFrame(failure_rows).round(3))
print('Successful attempts with settling exceptions:', {s: [r['dot'] for r in a['settling_exceptions']] for s, a in stack_audits.items()})
print('Internal-motion diagnostic flags:', {s: [r['dot'] for r in a['internal_diagnostic_flags']] for s, a in stack_audits.items()})
''')
add('markdown', 'rnd-stack-reserved-reading', '''### Reading these results

The success counts, distance distributions and complete failure list above are loaded from the frozen evaluation records. Noise seeds reuse targets, so pooling them describes repeated trials rather than increasing the number of independent target locations. The clean and disturbed candidates also have different controller settings; this is a robustness extension, not an isolated measurement of the disturbances' causal effect.

Camera previews show the image captured at the displayed frame. With delay, the controller may use an earlier delivered image; capture and delivery timing are retained in the logs. The player displays final sent arm torque, including motor corrections, separately from the underlying requested torque. Requested torque clipping does not bound the final motor-corrected torque.

The unchanged coefficient-fitting curves appear earlier in this notebook. No additional neural training epochs or recovery-data training were performed for these three evaluations. Source, data and native executable hashes are preserved with the freeze and each evaluation.''')
links = []
for seed in [7, 1, 2]:
    folder = 'stack_recovery_reserved_recording_adapter_v1' if seed == 7 else f'stack_reserved_seed{seed}_v1'
    url = f'https://nm-linmint-mlflow.taile381c0.ts.net:8443/golden-replays/directed-control-rnd/learned_geometry_branch/{folder}/test_seed{seed}/viewer_test_seed{seed}.html'
    links.append(f'- [Disturbance stack, noise seed {seed}: 100 recorded test replays]({url})')
add('markdown', 'rnd-stack-reserved-players', '### Recorded disturbance replays\n\n' + '\n'.join(links))
ids = {c['id'] for c in cells}
nb.cells = [c for c in nb.cells if c.get('id') not in ids] + cells
nbformat.write(nb, R/'research_notebook.ipynb')
