# report_statistics: saved notebook reading view

[V2 overview](../README.md) · [Iteration index](../ITERATIONS.md) · [Original notebook](../driving_v2/report_statistics.ipynb)

Generated from saved outputs without running cells. Code is collapsed. Interactive players remain in the original notebook; plots and tables appear below. Original notebook wording is preserved, including historical hypotheses and status notes.

Source SHA-256: `b276ff1cd563ba1b2d61d5b77075c0ffd137917f8e2154951d14cc09f6943858`

## Contents
- [Report statistics: frozen analysis notebook](#cell-0)
- [Part 2: verifying the experimental setup](#cell-2)
- [2.1 The 1 kg end-effector link](#cell-3)
- [2.2 C++ data collection](#cell-5)
- [2.3 The expert issues torque at 1 kHz, and the real arm cannot](#cell-6)
- [2.4 Other checks](#cell-9)
- [Part 3: statistics](#cell-11)
- [3.1 How well each model copies the expert](#cell-12)
- [3.2 Per joint: where the torque error sits (Lab 1)](#cell-15)
- [3.3 Do we need a network? Linear regression, trees, forest (Lab 2)](#cell-17)
- [3.4 Lowest error any single-instant model could reach](#cell-20)
- [3.5 Speed inputs: fit against the robot](#cell-23)
- [3.6 Settings check: depth, learning rate, batch size, data amount (Lab 4, Lecture 4.1)](#cell-26)
- [3.7 PARK and STOP decisions, and where the error sits (Lab 4, Lecture 1.2, preprocessing notes, Choudhury L05)](#cell-29)
- [3.8 Covariate shift, measured directly (Laskey, Ross, Choudhury L01, L03)](#cell-32)
- [3.9 The STOP network on the learner's states](#cell-35)
- [3.10 Is the network copying? Action predictability (Wen 2020)](#cell-38)
- [3.11 The task in one dimension](#cell-41)
- [3.12 Driving in 1D: how strongly the command reacts](#cell-44)
- [3.13 Arm gains: did recovery data teach feedback?](#cell-47)
- [3.14 Is each network a stable controller? (Zhang 2025)](#cell-50)
- [3.15 How error grows after a mistake (Ross, Choudhury L03, L04, L09)](#cell-53)
- [3.16 How far is fixed noise from Laskey's rule?](#cell-56)
- [3.17 Do the heads compete for the shared layers? (GradNorm, PCGrad, Kirchdorfer, UDL 9.3.6)](#cell-59)
- [3.18 Full runs on the test targets (Lecture 1.2, Lab 3, Lab 4)](#cell-62)
- [3.19 Where runs end, and what predicts failure (Lab 4)](#cell-66)
- [3.20 Robustness: effects added in turn (Lab 3, Lecture 4.2)](#cell-69)
- [3.21 Clean and disturbed together, every run counted](#cell-72)
- [3.22 Paired changes with intervals](#cell-75)
- [3.23 Training curves of the models tested on the robot](#cell-77)
- [3.24 Target uncertainty is separate from seed variation](#cell-80)
- [3.25 Model comparison ledger](#cell-82)
- [3.26 On the way to the target: arrive, pass through, or stay?](#cell-83)
- [3.27 After STOP](#cell-88)

<a id="cell-0"></a>

# Report statistics: frozen analysis notebook

This copy contains the setup and Parts 2–3 of the report workspace, preserving the saved analysis and cell IDs. It excludes working notes and report prose. The original workspace is unchanged.

Set `COMP0245_REPORT_DATA` to the extracted `report-analysis/notebook/data` directory before running. See [reproduction instructions](https://github.com/NatMota/mlai-vnc/blob/v2-sensor/docs/REPRODUCE.md).

Source SHA-256: `d7e4327a33a395252e4d35330649bf44e0d8697714408cda1660d0308db445cd`. Only the setup data path was made configurable in this copy.


<a id="cell-1"></a>

<details><summary>Code · original cell 1</summary>

```python
import json, csv, math, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
DATA = Path(os.environ['COMP0245_REPORT_DATA']).expanduser()
assert DATA.is_dir(), DATA
BLUE, ORANGE, AQUA, GREY, INK, MUTED, LIGHT = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#0b0b0b', '#8a8984', '#e8e7e3'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 10, 'axes.titlesize': 11, 'axes.titleweight': 'regular', 'axes.titlelocation': 'left', 'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                     'axes.linewidth': .6, 'axes.edgecolor': MUTED, 'xtick.color': GREY, 'ytick.color': GREY, 'axes.labelcolor': GREY, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': LIGHT, 'grid.linewidth': .6, 'axes.axisbelow': True, 'lines.linewidth': 2, 'legend.frameon': False, 'legend.fontsize': 9, 'figure.dpi': 110})
def J(name):
    p = DATA / name
    return json.load(open(p)) if p.exists() else None
OFF, DEEP, EXTRA, STAB, COMP, HEAD, SWEEP, MAC, ARM, CL, PROBE = (J(n) for n in ('offline_stats.json', 'deep_stats.json', 'extra_stats.json', 'stability.json', 'compounding.json', 'head_balance.json', 'sweep.json', 'mac_stats.json', 'arm_dynamics.json', 'closed_loop_stats.json', 'copycat_probe.json'))
def per_target(name):
    t = pd.read_csv(DATA / f'per-target/{name}.csv'); t['stopped'] = t.stop_s.notna(); return t.set_index('dot')
NAMES = {'golden_seed7': '34 seed 7', 'golden_seed1': '35 seed 1', 'golden_seed2': '36 seed 2', 'height': '37 height', 'no_speed': '38 no speeds', 'longer_arm': '39 longer', 'wider_arm': '40 wider', 'baseline': '41 learned arm', 'basics_standard': '43 standardised', 'dart': '52 fixed-noise DART', 'dagger': '54 DAgger', 'dart_dagger': '56 both'}
MODELS = ['baseline', 'basics_standard', 'dart', 'dagger', 'dart_dagger']
SHORT = {'baseline': '41 learned arm', 'basics_standard': '43 clean', 'dart': '52 fixed DART', 'dagger': '54 DAgger', 'dart_dagger': '56 both'}
COL = {'baseline': MUTED, 'basics_standard': BLUE, 'dart': AQUA, 'dagger': ORANGE, 'dart_dagger': GREY, 'expert': INK, 'height': BLUE, 'no_speed': ORANGE}
print('loaded:', {k: v is not None for k, v in dict(offline=OFF, deep=DEEP, extra=EXTRA, stability=STAB, compounding=COMP, heads=HEAD, sweep=SWEEP, mac=MAC, arm=ARM, full_runs=CL).items()})
```

</details>

```text
loaded: {'offline': True, 'deep': True, 'extra': True, 'stability': True, 'compounding': True, 'heads': True, 'sweep': True, 'mac': True, 'arm': True, 'full_runs': True}

```


<a id="cell-2"></a>

# Part 2: verifying the experimental setup

The checks made before trusting any result. In the report these get one short section.

<a id="cell-3"></a>

## 2.1 The 1 kg end-effector link

With the simulator's URDF flags, Bullet loads the virtual end-effector link at 1 kg and 1 kg m². That sets the arm's joint inertia to about 1 kg m² on every joint and opens a gap between the gravity the simulator applies and the gravity the supplied controller's Pinocchio model cancels. Raised with the lecturers. Francisco replied: "That's well spotted. I'd rather you do not change the simulation." So it stays. Under the position gain alone the gripper sags by 33 to 47 mm.

The `changeDynamics` massless test had no effect, so those fields in the results file are invalid and are not used here.

<a id="cell-4"></a>

<details><summary>Code · original cell 4</summary>

```python
g = ARM['pinocchio']['gravity_gap_median_nm']; rms = ARM['expert_torque_rms_nm']; I_ = ARM['joint_inertia_default_median']; x = np.arange(1, 8); w = .38
fig, ax = plt.subplots(1, 2, figsize=(10, 2.9))
ax[0].bar(x - w / 2, g, w, color=ORANGE, label='gravity gap against the controller model'); ax[0].bar(x + w / 2, rms, w, color=BLUE, label='expert torque RMS'); ax[0].set_xlabel('joint'); ax[0].set_ylabel('N m'); ax[0].set_title('The gap, against the size of the torques'); ax[0].legend(fontsize=8)
ax[1].bar(x, I_, color=BLUE, width=.55); ax[1].set_xlabel('joint'); ax[1].set_ylabel('kg m²'); ax[1].set_title('Joint inertia, median over expert states')
for a in ax: a.grid(axis='x', visible=False)
plt.tight_layout(); plt.show(); print('End-effector link:', ARM['end_effector_link'])
```

</details>

![Saved plot from cell 4](assets/2a5f575624703f6ec87ac77dde5d51367175dc3e539025611b0883f380c832a0.png)

```text
End-effector link: {'mass': 1.0, 'local_inertia': [1.0, 1.0, 1.0]}

```


<a id="cell-5"></a>

## 2.2 C++ data collection

The expert was ported to C++ on the same Bullet build. About 9 times faster on the M1 and 12 times on Mint, with no numerical difference across ten audited runs (80,010 samples). Options that changed results were rejected: Eigen-only maths and a Bullet built from source.

| Machine | C++, seconds per saved run | Python, seconds per saved run | Speed-up |
|---|---|---|---|
| M1 laptop | 0.640 | 5.46 to 5.77 | about 9 times |
| Mint | 0.882 | 10.48 to 10.88 | about 12 times |

<a id="cell-6"></a>

## 2.3 The expert issues torque at 1 kHz, and the real arm cannot

The Kinova Gen2 datasheet gives a low-level API limit of 500 Hz. So how does the expert behave if each torque is held for longer than one 1 ms simulator step? Tested on 30 targets. Other limits differ too: 0.20 m/s against the simulator's 0.25, joint 6 at 8 rad/s against a declared 48°/s, 30 against 20 N m. So this behaviour is simulator only.

The earlier one-joint estimate, kd·dt/I of about 0.06 at 500 Hz, predicts no trouble, so I had left the cause open. The right panel redoes the check with the full 7 by 7 mass matrix. **This check was not on the approved list.** It is a CPU check of a few seconds on the same saved states.

<a id="cell-7"></a>

<details><summary>Code · original cell 7</summary>

```python
RATE = J('stability_rate.json'); H = MAC['hold_rate']; hz = ['1000', '500', '250', '100', '30']; lab = [f'{h} Hz' for h in hz]
fig, ax = plt.subplots(1, 3, figsize=(14, 3)); x = np.arange(5)
ax[0].bar(x, [H[h]['median_mm'] for h in hz], color=BLUE, width=.55); ax[0].axhline(10, color=MUTED, ls='--', lw=1); ax[0].text(4.4, 11.5, '10 mm', ha='right', fontsize=9); ax[0].set_yscale('log'); ax[0].set_ylabel('median distance from expert run (mm)'); ax[0].set_title('Measured: torque held across steps')
ax[1].bar(x, [H[h]['matched_of_30'] for h in hz], color=AQUA, width=.55); ax[1].set_ylim(0, 32); ax[1].set_title('Measured: runs matched, of 30')
ax[2].bar(x, [RATE['rates'][l]['unstable_share'] * 100 for l in lab], color=ORANGE, width=.55); ax[2].set_ylim(0, 105); ax[2].set_ylabel('% of states'); ax[2].set_title("Predicted: expert's PD unstable, full mass matrix")
for a in ax: a.set_xticks(x); a.set_xticklabels(lab); a.grid(axis='x', visible=False)
plt.tight_layout(); plt.show()
display(pd.DataFrame([dict(rate=k, radius_p10=v['radius'][0], radius_p50=v['radius'][1], radius_p90=v['radius'][2], unstable_pct=v['unstable_share'] * 100) for k, v in RATE['rates'].items()]).round(3))
print('Mass matrix eigenvalues, median over states (kg m²):', np.round(RATE['mass_matrix_eigenvalues_median'], 3))
```

</details>

![Saved plot from cell 7](assets/6d1bf58a4f2fcc2645b63bd9ac58988f2ac499c6006ae90fa0dfaf176e1c506d.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rate</th>
      <th>radius_p10</th>
      <th>radius_p50</th>
      <th>radius_p90</th>
      <th>unstable_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000 Hz</td>
      <td>0.997</td>
      <td>0.997</td>
      <td>1.135</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>500 Hz</td>
      <td>1.006</td>
      <td>1.699</td>
      <td>3.277</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333 Hz</td>
      <td>2.015</td>
      <td>3.055</td>
      <td>5.425</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>250 Hz</td>
      <td>3.026</td>
      <td>4.415</td>
      <td>7.580</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100 Hz</td>
      <td>9.162</td>
      <td>12.667</td>
      <td>20.656</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>30 Hz</td>
      <td>33.763</td>
      <td>45.743</td>
      <td>73.056</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>

```text
Mass matrix eigenvalues, median over states (kg m²): [0.022 0.127 0.215 0.337 1.583 3.256 3.909]

```


<a id="cell-8"></a>

**Reading.** The joint diagonals are about 1 kg m², because the heavy link dominates them, but the lightest direction of the whole matrix is 0.022 kg m². Along that direction kd·dt/I is 1.35 at 1 kHz and 2.69 at 500 Hz, and the limit for this kind of integrator is about 2. So 1 kHz sits near the edge and 500 Hz is over it, which matches the measurement. This is a linear model with no solver damping, so it shows the failure is expected and does not prove the cause.

<a id="cell-9"></a>

## 2.4 Other checks

- **Renderer.** The GPU renderer applies a debug colour palette, the CPU one gives white. Training and testing use one renderer. My earlier explanation of the red marking was wrong twice, and was corrected by checking the files.
- **Expert runs differ between machines.** Rebuilt runs do not match the supplied package: PARK moves by up to 0.21 s on one target. So scores are only compared on one machine. Table below.
- **Tracking.** MLflow experiment `v2-sensors`, two experiment notebooks of 361 and 398 cells, a written hypothesis before each training, 66 commits on the branch.
- **Input features.** Long feature analysis of the simulator with ChatGPT, then 12 wheel features (mean, right minus left), inputs of 16, 17 and finally 11. Camera placement study: hand, nose and corner saw the target at PARK on 0, 12 and 19 of 30. IMU in iterations 28 and 29: loss down 23%, reach worse. Weight sweep in iteration 25. Separate STOP network in iteration 27.

<a id="cell-10"></a>

<details><summary>Code · original cell 10</summary>

```python
P = J('expert_rebuild_parity_mac_vs_package.json')
if P: display(pd.DataFrame([dict(target=r['dot'], steps_a=r['steps'][0], steps_b=r['steps'][1], park_step_a=r['park'][0], park_step_b=r['park'][1], gripper_distance_mm=r['hand_distance_expert'] * 1000) for r in P]).round(2))
```

</details>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>steps_a</th>
      <th>steps_b</th>
      <th>park_step_a</th>
      <th>park_step_b</th>
      <th>gripper_distance_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>9549</td>
      <td>9552</td>
      <td>6230</td>
      <td>6233</td>
      <td>1.86</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10507</td>
      <td>10461</td>
      <td>6754</td>
      <td>6731</td>
      <td>4.44</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8001</td>
      <td>8001</td>
      <td>4075</td>
      <td>4070</td>
      <td>1.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>8001</td>
      <td>8001</td>
      <td>6130</td>
      <td>6342</td>
      <td>50.21</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-11"></a>

# Part 3: statistics

Tables and figures below are read from the saved results. Numbers in the readings were filled in from the same files when this notebook was built. The fit statistics use 100 validation runs at the expert's own states. Full-run scores use the 100 test targets. All of it is seed 7 unless a row says otherwise.

<a id="cell-12"></a>

## 3.1 How well each model copies the expert

What: R² and adjusted R² per output, RMSE in physical units, F, and F corrected for correlated rows. Source: Lab 1, Lecture 1.2. RSE, RSS and intervals are in the table below it.

The rows are 1 ms apart, so they are far from independent. Lecture 1.2 warns about both: F at large n ("even an F>1 close to one can be an indication that H0 is false") and correlated noise ("reduced confidence margin"). So I also report effective n = n(1-ρ)/(1+ρ) with ρ the lag-1 residual correlation. Treat F and F_eff as descriptive only. Putting a network's parameter count and an effective sample size into a regression test does not give a valid F distribution, so neither column supports a significance claim. They are here because Lab 1 asks for F. R² and RMSE are what to read.

<a id="cell-13"></a>

<details><summary>Code · original cell 13</summary>

```python
def block(b):
    f = [x['F_eff'] for x in b if x['F_eff'] is not None]
    return dict(R2=np.mean([x['r2'] for x in b]), adjR2=np.mean([x['adj_r2'] for x in b]), RMSE=np.mean([x['rmse'] for x in b]), RSE=np.mean([x['rse'] for x in b]), RSS=np.sum([x['rss'] for x in b]), F=np.median([x['F'] for x in b]), lag1=np.median([x['residual_lag1'] for x in b]), n=b[0]['n'], n_eff=np.median([x['n_eff'] for x in b]), F_eff=np.median(f) if f else np.nan)
rows = []
for m in MODELS:
    for out, lab in (('wheels', 'wheels (rad/s)'), ('time_to_park', 'time to PARK (s)'), ('torque', 'torques (N m)'), ('stop_distance', 'STOP distance (mm)')):
        rows.append(dict(model=SHORT[m], output=lab, **block(OFF['models'][m][out])))
fit = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
for a, out, title in ((ax[0], 'wheels (rad/s)', 'Wheels: adjusted R² at expert states'), (ax[1], 'torques (N m)', 'Torques: adjusted R² at expert states')):
    d = fit[fit.output == out]; y = np.arange(len(d))[::-1]
    a.barh(y, d.adjR2, color=[COL[m] for m in MODELS], height=.62); a.set_yticks(y); a.set_yticklabels(d.model); a.set_xlim(0, 1.08); a.set_title(title); a.grid(axis='y', visible=False)
    for yi, v in zip(y, d.adjR2): a.text(v + .012, yi, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout(); plt.show()
display(fit.round({'R2': 3, 'adjR2': 3, 'RMSE': 3, 'RSE': 3, 'RSS': 0, 'F': 0, 'lag1': 3, 'n_eff': 0, 'F_eff': 1}))
```

</details>

![Saved plot from cell 13](assets/9148bc8a53c6ef77b06da1daea35f05fd54d27880abdd130ec778c86db729ff2.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>output</th>
      <th>R2</th>
      <th>adjR2</th>
      <th>RMSE</th>
      <th>RSE</th>
      <th>RSS</th>
      <th>F</th>
      <th>lag1</th>
      <th>n</th>
      <th>n_eff</th>
      <th>F_eff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>wheels (rad/s)</td>
      <td>0.869</td>
      <td>0.866</td>
      <td>0.607</td>
      <td>0.614</td>
      <td>39005.0</td>
      <td>314.0</td>
      <td>0.909</td>
      <td>26428</td>
      <td>1266.0</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41 learned arm</td>
      <td>time to PARK (s)</td>
      <td>0.960</td>
      <td>0.959</td>
      <td>0.263</td>
      <td>0.265</td>
      <td>1824.0</td>
      <td>1138.0</td>
      <td>0.988</td>
      <td>26428</td>
      <td>154.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41 learned arm</td>
      <td>torques (N m)</td>
      <td>0.979</td>
      <td>0.978</td>
      <td>0.378</td>
      <td>0.383</td>
      <td>990908.0</td>
      <td>2265.0</td>
      <td>0.614</td>
      <td>897417</td>
      <td>214838.0</td>
      <td>1168.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41 learned arm</td>
      <td>STOP distance (mm)</td>
      <td>0.977</td>
      <td>0.977</td>
      <td>13.720</td>
      <td>13.729</td>
      <td>168923104.0</td>
      <td>32718.0</td>
      <td>0.983</td>
      <td>897417</td>
      <td>7674.0</td>
      <td>238.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43 clean</td>
      <td>wheels (rad/s)</td>
      <td>0.869</td>
      <td>0.866</td>
      <td>0.607</td>
      <td>0.614</td>
      <td>39005.0</td>
      <td>314.0</td>
      <td>0.909</td>
      <td>26428</td>
      <td>1266.0</td>
      <td>8.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>43 clean</td>
      <td>time to PARK (s)</td>
      <td>0.960</td>
      <td>0.959</td>
      <td>0.263</td>
      <td>0.265</td>
      <td>1824.0</td>
      <td>1138.0</td>
      <td>0.988</td>
      <td>26428</td>
      <td>154.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>43 clean</td>
      <td>torques (N m)</td>
      <td>0.988</td>
      <td>0.988</td>
      <td>0.296</td>
      <td>0.300</td>
      <td>655209.0</td>
      <td>2783.0</td>
      <td>0.469</td>
      <td>897417</td>
      <td>324620.0</td>
      <td>1970.7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>43 clean</td>
      <td>STOP distance (mm)</td>
      <td>0.993</td>
      <td>0.993</td>
      <td>7.446</td>
      <td>7.451</td>
      <td>49761148.0</td>
      <td>112927.0</td>
      <td>0.983</td>
      <td>897417</td>
      <td>7570.0</td>
      <td>808.4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>52 fixed DART</td>
      <td>wheels (rad/s)</td>
      <td>0.867</td>
      <td>0.864</td>
      <td>0.610</td>
      <td>0.616</td>
      <td>39376.0</td>
      <td>311.0</td>
      <td>0.909</td>
      <td>26428</td>
      <td>1259.0</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>52 fixed DART</td>
      <td>time to PARK (s)</td>
      <td>0.960</td>
      <td>0.959</td>
      <td>0.264</td>
      <td>0.267</td>
      <td>1844.0</td>
      <td>1125.0</td>
      <td>0.988</td>
      <td>26428</td>
      <td>158.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>52 fixed DART</td>
      <td>torques (N m)</td>
      <td>0.988</td>
      <td>0.988</td>
      <td>0.303</td>
      <td>0.307</td>
      <td>687990.0</td>
      <td>2619.0</td>
      <td>0.505</td>
      <td>897417</td>
      <td>294990.0</td>
      <td>1771.9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>52 fixed DART</td>
      <td>STOP distance (mm)</td>
      <td>0.993</td>
      <td>0.993</td>
      <td>7.458</td>
      <td>7.463</td>
      <td>49918136.0</td>
      <td>112570.0</td>
      <td>0.958</td>
      <td>897417</td>
      <td>19217.0</td>
      <td>2268.7</td>
    </tr>
    <tr>
      <th>12</th>
      <td>54 DAgger</td>
      <td>wheels (rad/s)</td>
      <td>0.757</td>
      <td>0.752</td>
      <td>0.827</td>
      <td>0.835</td>
      <td>72259.0</td>
      <td>146.0</td>
      <td>0.953</td>
      <td>26428</td>
      <td>638.0</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>13</th>
      <td>54 DAgger</td>
      <td>time to PARK (s)</td>
      <td>0.923</td>
      <td>0.921</td>
      <td>0.366</td>
      <td>0.370</td>
      <td>3541.0</td>
      <td>563.0</td>
      <td>0.983</td>
      <td>26428</td>
      <td>233.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>54 DAgger</td>
      <td>torques (N m)</td>
      <td>0.780</td>
      <td>0.775</td>
      <td>1.189</td>
      <td>1.202</td>
      <td>10063587.0</td>
      <td>202.0</td>
      <td>0.871</td>
      <td>897417</td>
      <td>61755.0</td>
      <td>16.7</td>
    </tr>
    <tr>
      <th>15</th>
      <td>54 DAgger</td>
      <td>STOP distance (mm)</td>
      <td>0.993</td>
      <td>0.993</td>
      <td>7.446</td>
      <td>7.451</td>
      <td>49761148.0</td>
      <td>112927.0</td>
      <td>0.983</td>
      <td>897417</td>
      <td>7570.0</td>
      <td>808.4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>56 both</td>
      <td>wheels (rad/s)</td>
      <td>0.831</td>
      <td>0.827</td>
      <td>0.690</td>
      <td>0.697</td>
      <td>50273.0</td>
      <td>232.0</td>
      <td>0.934</td>
      <td>26428</td>
      <td>899.0</td>
      <td>3.2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>56 both</td>
      <td>time to PARK (s)</td>
      <td>0.942</td>
      <td>0.941</td>
      <td>0.317</td>
      <td>0.321</td>
      <td>2662.0</td>
      <td>765.0</td>
      <td>0.979</td>
      <td>26428</td>
      <td>284.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>56 both</td>
      <td>torques (N m)</td>
      <td>0.878</td>
      <td>0.875</td>
      <td>0.915</td>
      <td>0.925</td>
      <td>6099086.0</td>
      <td>282.0</td>
      <td>0.764</td>
      <td>897417</td>
      <td>120327.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>56 both</td>
      <td>STOP distance (mm)</td>
      <td>0.993</td>
      <td>0.993</td>
      <td>7.458</td>
      <td>7.463</td>
      <td>49918140.0</td>
      <td>112570.0</td>
      <td>0.958</td>
      <td>897417</td>
      <td>19217.0</td>
      <td>2268.7</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-14"></a>

**Reading.** The clean arm explains 98.8% of torque variance on rows it has never seen, and the DAgger arm only 77.5%, yet the DAgger arm does far better on the robot. Wheels sit near 0.87. The blind start accounts for part of that, 22% of the squared wheel error from 14% of rows, and most of the error is on rows where the target is seen. F and F_eff are descriptive only, as noted above; the drop from raw to corrected values shows how correlated the rows are and nothing more. Both STOP rows repeat because DAgger gives the STOP network no new rows, so it is the same network.

<a id="cell-15"></a>

## 3.2 Per joint: where the torque error sits (Lab 1)

Lab 1 asks for error per joint, confidence intervals and sources of error. The interval on RMSE is a bootstrap over whole runs, because rows within a run are related. The residual interval plays the role of the lecture's prediction interval: the range of error on a new row.

<a id="cell-16"></a>

<details><summary>Code · original cell 16</summary>

```python
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for m, off in (('basics_standard', -.18), ('dagger', .18)):
    b = OFF['models'][m]['torque']; x = np.arange(1, 8) + off; v = [q['rmse'] for q in b]; lo = [q['rmse'] - q['rmse_ci'][0] for q in b]; hi = [q['rmse_ci'][1] - q['rmse'] for q in b]
    ax[0].bar(x, v, width=.34, color=COL[m], label=SHORT[m]); ax[0].errorbar(x, v, yerr=[lo, hi], fmt='none', ecolor=INK, elinewidth=1, capsize=2)
ax[0].set_xlabel('joint'); ax[0].set_ylabel('RMSE (N m)'); ax[0].set_title('Torque RMSE per joint, 95% bootstrap over runs'); ax[0].legend(); ax[0].grid(axis='x', visible=False)
b = OFF['models']['basics_standard']['torque']; x = np.arange(1, 8)
ax[1].vlines(x, [q['residual_pi95'][0] for q in b], [q['residual_pi95'][1] for q in b], color=BLUE, lw=5); ax[1].axhline(0, color=MUTED, lw=.8); ax[1].set_xlabel('joint'); ax[1].set_ylabel('residual (N m)'); ax[1].set_title('Clean model: 95% of residuals on a new row'); ax[1].grid(axis='x', visible=False)
plt.tight_layout(); plt.show()
display(pd.DataFrame([dict(joint=q['output'] + 1, adjR2=q['adj_r2'], RMSE=q['rmse'], RMSE_low=q['rmse_ci'][0], RMSE_high=q['rmse_ci'][1], R2_low=q['r2_ci'][0], R2_high=q['r2_ci'][1], resid_low=q['residual_pi95'][0], resid_high=q['residual_pi95'][1], lag1=q['residual_lag1'], F_eff=q['F_eff']) for q in b]).round(3))
```

</details>

![Saved plot from cell 16](assets/65acb23eb43258621163002013ead2d4e966c23262eb787e308674314397c9b4.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>joint</th>
      <th>adjR2</th>
      <th>RMSE</th>
      <th>RMSE_low</th>
      <th>RMSE_high</th>
      <th>R2_low</th>
      <th>R2_high</th>
      <th>resid_low</th>
      <th>resid_high</th>
      <th>lag1</th>
      <th>F_eff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.984</td>
      <td>0.397</td>
      <td>0.333</td>
      <td>0.461</td>
      <td>0.976</td>
      <td>0.990</td>
      <td>-0.287</td>
      <td>0.334</td>
      <td>0.130</td>
      <td>2128.553</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0.982</td>
      <td>0.269</td>
      <td>0.241</td>
      <td>0.296</td>
      <td>0.978</td>
      <td>0.986</td>
      <td>-0.444</td>
      <td>0.373</td>
      <td>0.559</td>
      <td>651.410</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0.983</td>
      <td>0.479</td>
      <td>0.398</td>
      <td>0.549</td>
      <td>0.968</td>
      <td>0.989</td>
      <td>-0.286</td>
      <td>0.331</td>
      <td>0.124</td>
      <td>1970.651</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0.993</td>
      <td>0.225</td>
      <td>0.202</td>
      <td>0.250</td>
      <td>0.991</td>
      <td>0.994</td>
      <td>-0.352</td>
      <td>0.298</td>
      <td>0.773</td>
      <td>651.118</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.984</td>
      <td>0.415</td>
      <td>0.342</td>
      <td>0.488</td>
      <td>0.967</td>
      <td>0.991</td>
      <td>-0.278</td>
      <td>0.256</td>
      <td>0.150</td>
      <td>1996.529</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0.992</td>
      <td>0.197</td>
      <td>0.176</td>
      <td>0.221</td>
      <td>0.990</td>
      <td>0.993</td>
      <td>-0.283</td>
      <td>0.272</td>
      <td>0.705</td>
      <td>822.806</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0.999</td>
      <td>0.093</td>
      <td>0.082</td>
      <td>0.107</td>
      <td>0.998</td>
      <td>0.999</td>
      <td>-0.163</td>
      <td>0.171</td>
      <td>0.469</td>
      <td>17809.327</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-17"></a>

## 3.3 Do we need a network? Linear regression, trees, forest (Lab 2)

Same training runs as the networks. Trees use depth 4 to 20, chosen on validation. A random forest answers Lab 2's extra question. Trees and forest were scored offline only and never drove the robot, so Lab 2's real-world question can only be answered in part.

<a id="cell-18"></a>

<details><summary>Code · original cell 18</summary>

```python
rows = []
for key, lab, net in (('wheels', 'wheels', 'wheels'), ('torque', 'torques', 'torque')):
    e = OFF['baselines'][key]; n = block(OFF['models']['basics_standard'][net])
    for name, b, params in (('linear', e['linear']['val'], e['linear']['params']), (f"tree, depth {e['tree']['best_depth']}", e['tree']['val'], e['tree']['params']), (f"forest, 30 trees", e['forest']['val'], e['forest']['val'][0]['p'])):
        s = block(b); rows.append(dict(output=lab, model=name, R2=s['R2'], adjR2=s['adjR2'], RMSE=s['RMSE'], params=params))
    rows.append(dict(output=lab, model='network (43)', R2=n['R2'], adjR2=n['adjR2'], RMSE=n['RMSE'], params=OFF['models']['basics_standard'][net][0]['p']))
base = pd.DataFrame(rows)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
for a, out in zip(ax[:2], ('wheels', 'torques')):
    d = base[base.output == out]; y = np.arange(len(d))[::-1]; a.barh(y, d.R2, color=[MUTED, MUTED, MUTED, BLUE], height=.6); a.set_yticks(y); a.set_yticklabels(d.model); a.set_xlim(0, 1.1); a.set_title(f'{out.capitalize()}: R² on validation'); a.grid(axis='y', visible=False)
    for yi, v in zip(y, d.R2): a.text(v + .012, yi, f'{v:.3f}', va='center', fontsize=9)
for key, col, lab in (('wheels', BLUE, 'wheels'), ('torque', ORANGE, 'torques')):
    g = OFF['baselines'][key]['tree']['grid']; x = [q['depth'] for q in g]; ax[2].plot(x, [q['val_mse'] for q in g], color=col, marker='o', label=f'{lab}, validation'); ax[2].plot(x, [q['train_mse'] for q in g], color=col, ls='--', marker='o', ms=3, label=f'{lab}, training')
ax[2].set_yscale('log'); ax[2].set_xlabel('tree depth'); ax[2].set_ylabel('standardised MSE'); ax[2].set_title('Effect of depth'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show(); display(base.round(3))
```

</details>

![Saved plot from cell 18](assets/5b7683843e43b75988e53ce2a861c9ae686761f5cd18372f6082c554f9717721.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>output</th>
      <th>model</th>
      <th>R2</th>
      <th>adjR2</th>
      <th>RMSE</th>
      <th>params</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wheels</td>
      <td>linear</td>
      <td>0.429</td>
      <td>0.428</td>
      <td>1.266</td>
      <td>48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wheels</td>
      <td>tree, depth 12</td>
      <td>0.876</td>
      <td>0.852</td>
      <td>0.588</td>
      <td>4300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>wheels</td>
      <td>forest, 30 trees</td>
      <td>0.935</td>
      <td>-1708.992</td>
      <td>0.426</td>
      <td>122620</td>
    </tr>
    <tr>
      <th>3</th>
      <td>wheels</td>
      <td>network (43)</td>
      <td>0.869</td>
      <td>0.866</td>
      <td>0.607</td>
      <td>549</td>
    </tr>
    <tr>
      <th>4</th>
      <td>torques</td>
      <td>linear</td>
      <td>0.809</td>
      <td>0.809</td>
      <td>1.036</td>
      <td>133</td>
    </tr>
    <tr>
      <th>5</th>
      <td>torques</td>
      <td>tree, depth 20</td>
      <td>0.964</td>
      <td>0.951</td>
      <td>0.543</td>
      <td>243901</td>
    </tr>
    <tr>
      <th>6</th>
      <td>torques</td>
      <td>forest, 30 trees</td>
      <td>0.976</td>
      <td>0.942</td>
      <td>0.426</td>
      <td>529165</td>
    </tr>
    <tr>
      <th>7</th>
      <td>torques</td>
      <td>network (43)</td>
      <td>0.988</td>
      <td>0.988</td>
      <td>0.296</td>
      <td>19847</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-19"></a>

**Reading.** Linear regression fits torques to R² 0.81 and wheels to only 0.43, so the mapping is strongly non-linear, as Lab 2 expects. Depth overfits wheels past depth 12 (training keeps falling, validation does not). The surprise is that a forest fits wheels better than our network does offline (R² 0.935 against 0.869, RMSE 0.43 against 0.61 rad/s). Given every other result here, that does not tell us it would drive better, and a forest cannot be read as a smooth controller: its output is piecewise constant. The adjusted R² of a forest is meaningless because its leaf count is close to the row count, so the table shows R² for it. The network uses far fewer parameters.

<a id="cell-20"></a>

## 3.4 Lowest error any single-instant model could reach

Find the 5 nearest training rows to each validation row and compare their labels. If near-identical inputs carry different labels, no function of those inputs can do better than that spread. This measures irreducible error (Lecture 1.1), the "input data does not contain sufficient information" case (Lecture 4.1), and the ε limit in Spencer. Method not from the course.

<a id="cell-21"></a>

<details><summary>Code · original cell 21</summary>

```python
if EXTRA and 'collision' in EXTRA:
    k = EXTRA['collision']; fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    b = OFF['models']['basics_standard']
    for a, key, net, title, unit in ((ax[0], 'drive', 'wheels', 'Wheels', 'rad/s'), (ax[1], 'arm', 'torque', 'Torques', 'N m')):
        n = len(k[key]['knn_rmse']); x = np.arange(1, n + 1); w = .26
        a.bar(x - w, k[key]['neighbour_label_sd'], w, color=MUTED, label='spread among nearest rows'); a.bar(x, k[key]['knn_rmse'], w, color=AQUA, label='5-neighbour RMSE'); a.bar(x + w, [q['rmse'] for q in b[net]], w, color=BLUE, label='network RMSE'); a.set_xticks(x); a.set_ylim(0, 1.35 * max(max(k[key]['knn_rmse']), max(q['rmse'] for q in b[net])))
        a.set_xlabel('output'); a.set_ylabel(unit); a.set_title(title); a.legend(fontsize=8); a.grid(axis='x', visible=False)
    plt.tight_layout(); plt.show(); print('median distance to the 5th neighbour, standardised units:', {q: round(k[q]['neighbour_distance'], 3) for q in k})
```

</details>

![Saved plot from cell 21](assets/39c70e356db4c5f6a7c3427128a2a605dc08fb445a9152721adabae3ccbe7475.png)

```text
median distance to the 5th neighbour, standardised units: {'drive': 0.502, 'arm': 0.224}

```


<a id="cell-22"></a>

**Reading.** Exploratory, and it did not give what I wanted. The spread between neighbours is tiny (0.002 rad/s on the wheels, 0.030 N m on the arm), probably because a row's nearest neighbours are adjacent rows of one training run, which I did not check. So this is no estimate of irreducible error, and it should carry no conclusion. Its own error (0.97 rad/s on the wheels) is worse than the network's (0.61), and on the arm (0.39 N m) about level.

The collision that does exist is at the blind start: identical inputs with opposite expert commands (28 turn left, 27 turn right, of 55 blind training runs), so a deterministic network cannot reproduce both. That rules out imitating the exact command. It does not rule out the task, since either consistent turn eventually brings the target into view. Wheel RMSE is 0.77 rad/s on rows with the target out of view and 0.58 on the rest.

<a id="cell-23"></a>

## 3.5 Speed inputs: fit against the robot

RSS with and without the six wheel speed inputs (iteration 37 against 38) on the validation rows, beside the paired robot results. An earlier version of this section called this a partial F test. It is removed as a test: the rows are strongly correlated and a network is not a nested regression, so an F distribution does not apply. The sevenfold RSS difference and the paired robot outcomes stand on their own.

<a id="cell-24"></a>

<details><summary>Code · original cell 24</summary>

```python
if EXTRA and 'partial_f_speeds' in EXTRA:
    p = EXTRA['partial_f_speeds']; sa = per_target('height'); sb = per_target('no_speed')
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.9))
    ax[0].bar(['with speeds\n(37)', 'without\n(38)'], [p['rss_with'], p['rss_without']], color=[BLUE, ORANGE], width=.5); ax[0].set_title('Offline: RSS on validation'); ax[0].grid(axis='x', visible=False)
    ax[1].bar(['with speeds\n(37)', 'without\n(38)'], [sa.park_error_cm.median(), sb.park_error_cm.median()], color=[BLUE, ORANGE], width=.5); ax[1].set_title('On the robot: median park error (cm)'); ax[1].grid(axis='x', visible=False)
    ax[2].bar(['with speeds\n(37)', 'without\n(38)'], [int(sa.parked.sum()), int(sb.parked.sum())], color=[BLUE, ORANGE], width=.5); ax[2].set_title('On the robot: PARK of 100'); ax[2].grid(axis='x', visible=False); ax[2].set_ylim(0, 110)
    plt.tight_layout(); plt.show(); display(pd.DataFrame([dict(rss_with_speeds=p['rss_with'], rss_without_speeds=p['rss_without'], ratio=p['rss_without'] / p['rss_with'], rows=p['n'])]).round(1))
```

</details>

![Saved plot from cell 24](assets/d1e8d76993debfc14a1621096c31c4313d9b27a9e8e04990b93fb9265134b69e.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rss_with_speeds</th>
      <th>rss_without_speeds</th>
      <th>ratio</th>
      <th>rows</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5611.7</td>
      <td>39005.4</td>
      <td>7.0</td>
      <td>105712</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-25"></a>

**Reading.** RSS is 39,005 without the speeds and 5,612 with them, 7.0 times worse. On the robot, removing them took PARK from 87 to 100 and median park error from 18.7 to 5.6 cm. RSS measures prediction of the expert's next command at the expert's states. It does not measure what happens when the network's output feeds back into those inputs. The mechanism is in Part 1.6.

<a id="cell-26"></a>

## 3.6 Settings check: depth, learning rate, batch size, data amount (Lab 4, Lecture 4.1)

Validation loss only. The arm runs for 100 epochs against 1,200 for the final model, and these networks train without the extra output heads, so this compares settings and does not reproduce final scores. Lab 4 asks for one larger and one smaller batch size; our base is the full batch, so only smaller ones exist.

<a id="cell-27"></a>

<details><summary>Code · original cell 27</summary>

```python
if SWEEP:
    S = pd.DataFrame([{k: r.get(k) for k in ('network', 'tag', 'seed', 'params', 'best_val', 'best_epoch', 'last_train', 'last_val', 'seconds', 'runs')} for r in SWEEP])
    S = S.groupby(['network', 'tag'], as_index=False).agg(best_val=('best_val', 'mean'), last_train=('last_train', 'mean'), params=('params', 'first'), seconds=('seconds', 'mean'), runs=('runs', 'first'))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
    order = ['base', 'shallow', 'deep', 'lr0.001', 'lr0.003', 'lr0.03']; lab = ['two layers (ours)', 'one wide layer', 'three narrow', 'lr 0.001', 'lr 0.003', 'lr 0.03']
    for a, net, title in ((ax[0], 'drive', 'Driving, 300 epochs'), (ax[1], 'arm', 'Arm torques, 100 epochs')):
        d = S[S.network == net].set_index('tag'); extra = [t for t in d.index if t.startswith('batch')]; tags = order + extra; v = [d.best_val.get(t, np.nan) for t in tags]; y = np.arange(len(tags))[::-1]
        a.barh(y, v, color=[BLUE] + [MUTED] * (len(tags) - 1), height=.62); a.set_yticks(y); a.set_yticklabels(lab + [t.replace('batch', 'batch ') for t in extra]); a.set_title(title + ': best validation MSE'); a.grid(axis='y', visible=False)
        for yi, q in zip(y, v): a.text(q, yi, f' {q:.3f}', va='center', fontsize=9)
    plt.tight_layout(); plt.show()
    fig, ax = plt.subplots(figsize=(6, 2.9))
    for net, col, lab_ in (('drive', BLUE, 'driving'), ('arm', ORANGE, 'arm')):
        d = S[S.network == net].set_index('tag'); x = [10, 25, 50, 100]; y = [d.best_val.get(f'data{n}', np.nan) for n in (10, 25, 50)] + [d.best_val['base']]; y0 = y[-1]; ax.plot(x, np.array(y) / y0, color=col, marker='o', label=lab_)
    ax.set_xlabel('training runs'); ax.set_ylabel('validation MSE, relative to 100 runs'); ax.set_title('How much data'); ax.legend(); plt.tight_layout(); plt.show(); display(S.round(4))
```

</details>

![Saved plot from cell 27](assets/08f3a632c487b0efbbabcf7f301e366e500d8ff953e76a41c8927e1749e6f055.png)

![Saved plot from cell 27](assets/156ef032e120b34d142e64790ba3e5feeb215dcdf865202612758817dbf5cd6b.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>network</th>
      <th>tag</th>
      <th>best_val</th>
      <th>last_train</th>
      <th>params</th>
      <th>seconds</th>
      <th>runs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>arm</td>
      <td>base</td>
      <td>0.0295</td>
      <td>0.0278</td>
      <td>19847</td>
      <td>12.2875</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arm</td>
      <td>batch16384</td>
      <td>0.0088</td>
      <td>0.0069</td>
      <td>19847</td>
      <td>16.8250</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arm</td>
      <td>batch65536</td>
      <td>0.0110</td>
      <td>0.0085</td>
      <td>19847</td>
      <td>15.1721</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arm</td>
      <td>data10</td>
      <td>0.2103</td>
      <td>0.0112</td>
      <td>19847</td>
      <td>1.5597</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>arm</td>
      <td>data25</td>
      <td>0.0395</td>
      <td>0.0229</td>
      <td>19847</td>
      <td>3.3576</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>arm</td>
      <td>data50</td>
      <td>0.0311</td>
      <td>0.0252</td>
      <td>19847</td>
      <td>6.3515</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>arm</td>
      <td>deep</td>
      <td>0.0353</td>
      <td>0.0340</td>
      <td>19511</td>
      <td>14.6443</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>arm</td>
      <td>lr0.001</td>
      <td>0.1232</td>
      <td>0.1210</td>
      <td>19847</td>
      <td>12.5944</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>arm</td>
      <td>lr0.003</td>
      <td>0.0549</td>
      <td>0.0531</td>
      <td>19847</td>
      <td>14.4578</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>arm</td>
      <td>lr0.03</td>
      <td>0.0265</td>
      <td>0.0245</td>
      <td>19847</td>
      <td>14.0442</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>arm</td>
      <td>shallow</td>
      <td>0.0347</td>
      <td>0.0330</td>
      <td>19845</td>
      <td>28.9590</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>drive</td>
      <td>base</td>
      <td>0.3109</td>
      <td>0.2884</td>
      <td>532</td>
      <td>0.4076</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>drive</td>
      <td>batch1024</td>
      <td>0.2257</td>
      <td>0.2084</td>
      <td>532</td>
      <td>10.8674</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>drive</td>
      <td>batch4096</td>
      <td>0.2258</td>
      <td>0.1979</td>
      <td>532</td>
      <td>3.0266</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>drive</td>
      <td>data10</td>
      <td>1.1530</td>
      <td>0.1830</td>
      <td>532</td>
      <td>0.5593</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>drive</td>
      <td>data25</td>
      <td>0.3674</td>
      <td>0.2853</td>
      <td>532</td>
      <td>0.3918</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>drive</td>
      <td>data50</td>
      <td>0.3295</td>
      <td>0.2777</td>
      <td>532</td>
      <td>0.3959</td>
      <td>50.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>drive</td>
      <td>deep</td>
      <td>0.2844</td>
      <td>0.2666</td>
      <td>508</td>
      <td>0.5091</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>drive</td>
      <td>lr0.001</td>
      <td>1.0298</td>
      <td>0.9713</td>
      <td>532</td>
      <td>0.3982</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>drive</td>
      <td>lr0.003</td>
      <td>0.4770</td>
      <td>0.4520</td>
      <td>532</td>
      <td>0.4199</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>drive</td>
      <td>lr0.03</td>
      <td>0.2644</td>
      <td>0.2494</td>
      <td>532</td>
      <td>0.4226</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>drive</td>
      <td>shallow</td>
      <td>0.4942</td>
      <td>0.4480</td>
      <td>532</td>
      <td>0.3652</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-28"></a>

**Reading.** Validation loss, not robot scores. Shape first: at about the same parameter count, three layers beat one on driving (0.284 against 0.494), and on the arm two layers (0.0295) beat both one (0.0347) and three (0.0353). Learning rate: 0.03 ends slightly lower than 0.01 on both, and 0.001 is far behind after these epochs. Batch size is the surprise. Mini-batches of 16,384 reach 0.0088 on the arm in 17 s, against 0.0295 for full batches in 12 s at the same 100 epochs. That is about the level of the final 1,200-epoch model (0.011), so full batches cost a lot of training time, though nothing here tests the mini-batch model on the robot. Data amount, Lecture 4.1's learning curve: with 10 runs training loss is far below validation (0.0112 against 0.2103 on the arm, 0.183 against 1.153 on driving), which is overfitting. By 25 runs the gap is mostly gone, and 25, 50 and 100 runs differ by little (0.0395, 0.0311, 0.0295). These runs all take the same number of steps, so this undersells more data.

<a id="cell-29"></a>

## 3.7 PARK and STOP decisions, and where the error sits (Lab 4, Lecture 1.2, preprocessing notes, Choudhury L05)

The two yes/no decisions at the expert's states. The preprocessing notes warn about rare events and that "errors are not weighted equally in robotics". Choudhury's cliff example: a mistake on "a tiny fraction of the overall data set" can decide the run. So this splits error by phase and shows the share of rows in each phase.

<a id="cell-30"></a>

<details><summary>Code · original cell 30</summary>

```python
if DEEP:
    m = DEEP['models']; rows = []
    for k in MODELS:
        for dec, key in (('PARK', 'park_decision_rows'), ('STOP', 'stop_decision_rows')):
            q = m[k][key]; f1 = 2 * q['precision'] * q['recall'] / max(q['precision'] + q['recall'], 1e-9); rows.append(dict(model=SHORT[k], decision=dec, TP=q['tp'], FP=q['fp'], FN=q['fn'], TN=q['tn'], precision=q['precision'], recall=q['recall'], F1=f1))
    display(pd.DataFrame(rows).round(3))
    ph = m['basics_standard']['torque_by_phase']; L = list(ph); fig, ax = plt.subplots(1, 3, figsize=(13, 3.2)); x = np.arange(len(L)); w = .36
    ax[0].bar(x - w / 2, [ph[k]['row_share'] * 100 for k in L], w, color=MUTED, label='share of rows'); ax[0].bar(x + w / 2, [ph[k]['error_share'] * 100 for k in L], w, color=BLUE, label='share of squared error'); ax[0].set_xticks(x); ax[0].set_xticklabels(L, rotation=30, ha='right'); ax[0].set_ylabel('%'); ax[0].set_title('Clean arm: where rows and error sit'); ax[0].legend(fontsize=8); ax[0].grid(axis='x', visible=False)
    for k, off in (('basics_standard', -w / 2), ('dagger', w / 2)):
        p2 = m[k]['torque_by_phase']; ax[1].bar(x + off, [p2[q]['rmse_nm'] for q in L], w, color=COL[k], label=SHORT[k])
    ax[1].set_xticks(x); ax[1].set_xticklabels(L, rotation=30, ha='right'); ax[1].set_ylabel('N m'); ax[1].set_title('Torque RMSE by phase'); ax[1].legend(fontsize=8); ax[1].grid(axis='x', visible=False)
    L2 = L[1:]; ax[2].bar(np.arange(len(L2)), [ph[k]['stop_mae_mm'] for k in L2], color=BLUE, width=.55); ax[2].axhline(10, color=ORANGE, lw=1.2, ls='--'); ax[2].text(len(L2) - .5, 10.5, '10 mm', color=INK, fontsize=9, ha='right'); ax[2].set_xticks(np.arange(len(L2))); ax[2].set_xticklabels(L2, rotation=30, ha='right'); ax[2].set_ylabel('mm'); ax[2].set_title('STOP network: mean absolute error'); ax[2].grid(axis='x', visible=False)
    plt.tight_layout(); plt.show()
    display(pd.DataFrame([dict(model=SHORT[k], stop_never=m[k]['stop_runs']['never'], stop_fires_at_park=m[k]['stop_runs']['fires_within_50ms_of_park'], stop_true_mm_at_first_fire=m[k]['stop_runs']['true_distance_at_first_fire_mm'][0], stop_minus_crossing_s=m[k]['stop_runs']['first_fire_minus_true_crossing_s'][0], park_never=m[k]['park_runs']['never'], park_minus_expert_s=m[k]['park_runs']['first_fire_minus_expert_park_s'][0]) for k in MODELS]).round(3))
```

</details>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>decision</th>
      <th>TP</th>
      <th>FP</th>
      <th>FN</th>
      <th>TN</th>
      <th>precision</th>
      <th>recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>PARK</td>
      <td>7772</td>
      <td>703</td>
      <td>1100</td>
      <td>16853</td>
      <td>0.917</td>
      <td>0.876</td>
      <td>0.896</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41 learned arm</td>
      <td>STOP</td>
      <td>20111</td>
      <td>7892</td>
      <td>12176</td>
      <td>248639</td>
      <td>0.718</td>
      <td>0.623</td>
      <td>0.667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43 clean</td>
      <td>PARK</td>
      <td>7772</td>
      <td>703</td>
      <td>1100</td>
      <td>16853</td>
      <td>0.917</td>
      <td>0.876</td>
      <td>0.896</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43 clean</td>
      <td>STOP</td>
      <td>24166</td>
      <td>4362</td>
      <td>8121</td>
      <td>252169</td>
      <td>0.847</td>
      <td>0.748</td>
      <td>0.795</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52 fixed DART</td>
      <td>PARK</td>
      <td>7659</td>
      <td>619</td>
      <td>1213</td>
      <td>16937</td>
      <td>0.925</td>
      <td>0.863</td>
      <td>0.893</td>
    </tr>
    <tr>
      <th>5</th>
      <td>52 fixed DART</td>
      <td>STOP</td>
      <td>22057</td>
      <td>4451</td>
      <td>10230</td>
      <td>252080</td>
      <td>0.832</td>
      <td>0.683</td>
      <td>0.750</td>
    </tr>
    <tr>
      <th>6</th>
      <td>54 DAgger</td>
      <td>PARK</td>
      <td>6011</td>
      <td>505</td>
      <td>2861</td>
      <td>17051</td>
      <td>0.922</td>
      <td>0.678</td>
      <td>0.781</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54 DAgger</td>
      <td>STOP</td>
      <td>24166</td>
      <td>4362</td>
      <td>8121</td>
      <td>252169</td>
      <td>0.847</td>
      <td>0.748</td>
      <td>0.795</td>
    </tr>
    <tr>
      <th>8</th>
      <td>56 both</td>
      <td>PARK</td>
      <td>7777</td>
      <td>274</td>
      <td>1095</td>
      <td>17282</td>
      <td>0.966</td>
      <td>0.877</td>
      <td>0.919</td>
    </tr>
    <tr>
      <th>9</th>
      <td>56 both</td>
      <td>STOP</td>
      <td>22057</td>
      <td>4451</td>
      <td>10230</td>
      <td>252080</td>
      <td>0.832</td>
      <td>0.683</td>
      <td>0.750</td>
    </tr>
  </tbody>
</table>
</div>

![Saved plot from cell 30](assets/f3a0bae45a2d78bba711f0c8db259b78e5d8eb6c78dace86cfd317e200cc7c8a.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>stop_never</th>
      <th>stop_fires_at_park</th>
      <th>stop_true_mm_at_first_fire</th>
      <th>stop_minus_crossing_s</th>
      <th>park_never</th>
      <th>park_minus_expert_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>50</td>
      <td>4</td>
      <td>13.307</td>
      <td>-0.166</td>
      <td>7</td>
      <td>-0.392</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43 clean</td>
      <td>51</td>
      <td>0</td>
      <td>10.532</td>
      <td>-0.036</td>
      <td>7</td>
      <td>-0.392</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52 fixed DART</td>
      <td>54</td>
      <td>1</td>
      <td>10.832</td>
      <td>-0.060</td>
      <td>8</td>
      <td>-0.340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54 DAgger</td>
      <td>51</td>
      <td>0</td>
      <td>10.532</td>
      <td>-0.036</td>
      <td>21</td>
      <td>-0.411</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56 both</td>
      <td>54</td>
      <td>1</td>
      <td>10.832</td>
      <td>-0.060</td>
      <td>5</td>
      <td>-0.224</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-31"></a>

**Reading.** These rows are the expert's states, so they show the decision at its best. STOP fires on none of the runs within 50 ms of PARK. Precision and recall at the row level are 0.85 and 0.75, and it never fires on 51 of 100 expert runs. Its first firing is at 10.5 mm, so its threshold sits close to the 10 mm line. For torque, 68% of rows are before PARK and hold only 4% of the squared error, so most of the average loss comes from rows where nothing happens. Section 3.26 shows what happens on the network's own states.

<a id="cell-32"></a>

## 3.8 Covariate shift, measured directly (Laskey, Ross, Choudhury L01, L03)

Laskey splits a policy's loss on its own states into a shift term and the standard loss on the expert's states. DAgger collection saved exactly what that needs: the learner's states with the expert's labels. So this is the same model on the same training targets, scored two ways. The final DAgger model has no labelled runs of its own, because that would need a fourth round, so the trend stops at round 2.

Note on the expert's label here: it is the stateful copy of the expert, so far from its path it asks for large correcting torques. That is the point of DAgger, and also part of why the number is large.

<a id="cell-33"></a>

<details><summary>Code · original cell 33</summary>

```python
if DEEP:
    s = DEEP['shift']; L = ['basics_standard', 'dart', 'dagger round 1', 'dagger round 2']; lab = ['43 clean', '52 fixed DART', 'DAgger round 1', 'DAgger round 2']; x = np.arange(4); w = .36
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.2))
    ax[0].bar(x - w / 2, [s[k]['torque_rmse_nm_expert'][0] for k in L], w, color=BLUE, label='at expert states'); ax[0].bar(x + w / 2, [s[k]['torque_rmse_nm_own'][0] for k in L], w, color=ORANGE, label='at its own states'); ax[0].set_ylabel('N m'); ax[0].set_title('Torque RMSE, after PARK'); ax[0].legend(fontsize=8)
    ax[1].bar(x - w / 2, [s[k]['wheel_mse_at_expert_states'][0] for k in L], w, color=BLUE, label='at expert states'); ax[1].bar(x + w / 2, [s[k]['wheel_mse_on_own_states'][0] for k in L], w, color=ORANGE, label='at its own states'); ax[1].set_ylabel('MSE (rad/s)²'); ax[1].set_title('Wheel commands'); ax[1].legend(fontsize=8)
    ax[2].bar(x - w / 2, [s[k]['arm_shift']['auc'] for k in L], w, color=BLUE, label='arm rows'); ax[2].bar(x + w / 2, [s[k]['drive_shift']['auc'] for k in L], w, color=AQUA, label='driving rows'); ax[2].axhline(.5, color=MUTED, lw=1, ls='--'); ax[2].text(3.45, .52, 'no shift', fontsize=9, ha='right'); ax[2].set_ylim(.4, 1.02); ax[2].set_title('Can a classifier tell learner from expert? (AUC)'); ax[2].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.28), ncol=2)
    for a in ax: a.set_xticks(x); a.set_xticklabels(lab, rotation=20, ha='right'); a.grid(axis='x', visible=False)
    plt.tight_layout(); plt.show()
    display(pd.DataFrame([dict(model=l, runs=s[k]['runs'], torque_rmse_expert=s[k]['torque_rmse_nm_expert'][0], torque_rmse_own=s[k]['torque_rmse_nm_own'][0], wheel_ratio_own_over_expert=s[k]['wheel_ratio_per_run'][0], arm_auc=s[k]['arm_shift']['auc'], arm_rows_clearly_unlike_expert=s[k]['arm_shift']['learner_rows_clearly_unlike_expert'], drive_auc=s[k]['drive_shift']['auc'], first_stop_true_mm_own=s[k]['true_distance_at_first_stop_own_states_mm'][0], never_stop_share=s[k]['never_stop_share']) for k, l in zip(L, lab)]).round(3))
```

</details>

![Saved plot from cell 33](assets/41ba062ca9d2c4545d5694754460fc8e219be36316ca336539462c45ec3f94b8.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>runs</th>
      <th>torque_rmse_expert</th>
      <th>torque_rmse_own</th>
      <th>wheel_ratio_own_over_expert</th>
      <th>arm_auc</th>
      <th>arm_rows_clearly_unlike_expert</th>
      <th>drive_auc</th>
      <th>first_stop_true_mm_own</th>
      <th>never_stop_share</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43 clean</td>
      <td>100</td>
      <td>0.235</td>
      <td>16.020</td>
      <td>5.438</td>
      <td>0.990</td>
      <td>0.899</td>
      <td>0.948</td>
      <td>153.060</td>
      <td>0.010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52 fixed DART</td>
      <td>100</td>
      <td>0.253</td>
      <td>10.939</td>
      <td>5.710</td>
      <td>0.982</td>
      <td>0.880</td>
      <td>0.954</td>
      <td>59.478</td>
      <td>0.170</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DAgger round 1</td>
      <td>100</td>
      <td>0.866</td>
      <td>7.501</td>
      <td>5.261</td>
      <td>0.969</td>
      <td>0.774</td>
      <td>0.968</td>
      <td>40.985</td>
      <td>0.190</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DAgger round 2</td>
      <td>93</td>
      <td>1.124</td>
      <td>7.640</td>
      <td>1.329</td>
      <td>0.965</td>
      <td>0.758</td>
      <td>0.955</td>
      <td>51.638</td>
      <td>0.312</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-34"></a>

**Reading.** The clean arm's torque error is 0.24 N m at the expert's states and 16.0 N m on its own, 68 times larger. Round 1 of DAgger gives 0.87 and 7.5, 9 times. So DAgger worsens the fit at the expert's states and reduces the error on its own. The ratios are large partly because the expert's label far from its path is a large correction.

AUC stays high in every row (0.990 for the clean arm, 0.965 for round 2), and the classifier's folds split whole runs, so rows from one run never sit on both sides. AUC measures how easy the states are to tell apart. It says nothing about the cost of the difference. Own-state torque error falls while AUC stays high, so the learner's states remain unlike the expert's and acting in them became less costly. The wheels' median per-run ratio improves from 5.4 to 1.3 by round 2.

<a id="cell-35"></a>

## 3.9 The STOP network on the learner's states

The STOP network never gets DAgger rows. So how far from the target is the gripper, on the learner's own states, the first time the network says stop? Compare with the expert's states, where it is nearly exact.

<a id="cell-36"></a>

<details><summary>Code · original cell 36</summary>

```python
if DEEP:
    s = DEEP['shift']; bands = ['over 100 mm', '50 to 100 mm', '25 to 50 mm', '10 to 25 mm', '10 mm or less']; x = np.arange(5); w = .36
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.1)); e = s['basics_standard']
    ax[0].bar(x - w / 2, [e['stop_abs_err_mm_expert'][b]['median'] for b in bands], w, color=BLUE, label='at expert states'); ax[0].bar(x + w / 2, [e['stop_abs_err_mm_own'][b]['median'] for b in bands], w, color=ORANGE, label='at its own states'); ax[0].set_xticks(x); ax[0].set_xticklabels(bands, rotation=20, ha='right'); ax[0].set_ylabel('median absolute error (mm)'); ax[0].set_title('Clean model: STOP distance error by true distance'); ax[0].legend(fontsize=8); ax[0].grid(axis='x', visible=False)
    L = ['basics_standard', 'dart', 'dagger round 1', 'dagger round 2']; lab = ['43 clean', '52 fixed DART', 'DAgger round 1', 'DAgger round 2']; v = [s[k]['true_distance_at_first_stop_own_states_mm'] for k in L]
    ax[1].bar(lab, [q[0] for q in v], color=[BLUE, AQUA, ORANGE, ORANGE], width=.55, yerr=[[q[0] - q[1] for q in v], [q[2] - q[0] for q in v]], ecolor=INK, capsize=3); ax[1].axhline(DEEP['models']['basics_standard']['stop_runs']['true_distance_at_first_fire_mm'][0], color=MUTED, ls='--', lw=1); ax[1].text(3.45, 14, 'at expert states', fontsize=9, ha='right', bbox=dict(fc='white', ec='none', pad=1)); ax[1].set_ylabel('mm'); ax[1].set_title('True distance when STOP first fires, own states'); ax[1].grid(axis='x', visible=False); ax[1].tick_params(axis='x', rotation=20)
    plt.tight_layout(); plt.show()
```

</details>

![Saved plot from cell 36](assets/18adbedf3b5547b8ccc88d50edd3fb2a7426b21ad528690a36cc26298d2304f6.png)


<a id="cell-37"></a>

**Reading.** At the expert's states STOP error is at most a few mm in every band. On the clean model's own states it is 64 mm far from the target and 33 mm inside 10 mm. First firing moves from 153 mm to 41 mm with DAgger's round 1. The STOP network's weights are identical in those two models, so the change comes only from the inputs the improved arm now gives it. This is a change in what the network is given, with no learning in the STOP network itself.

<a id="cell-38"></a>

## 3.10 Is the network copying? Action predictability (Wen 2020)

Wen's test for copying: train a small network to predict the wheel command from the previous 9 commands. If it is easier to predict on a learner's runs than on the expert's, the learner is repeating itself. A positive log ratio means more predictable than the expert. Iterations 37 and 38 differ only by the six speed inputs.

<a id="cell-39"></a>

<details><summary>Code · original cell 39</summary>

```python
if EXTRA and 'action_predictability' in EXTRA:
    a = EXTRA['action_predictability']; L = [k for k in ('height', 'no_speed', 'baseline', 'basics_standard', 'dart', 'dagger', 'dart_dagger') if a.get(k)]; lab = {'height': '37 speeds kept', 'no_speed': '38 speeds removed', **SHORT}
    fig, ax = plt.subplots(figsize=(7, 2.9)); y = np.arange(len(L))[::-1]; v = [a[k]['log_ratio'] for k in L]; ax.barh(y, v, color=[BLUE if q > 0 else ORANGE for q in v], height=.6); ax.axvline(0, color=INK, lw=.8); ax.set_yticks(y); ax.set_yticklabels([lab[k] for k in L]); ax.set_xlabel('log(expert MSE / learner MSE): above 0 means more predictable than the expert'); ax.set_title('Action predictability of wheel commands'); ax.grid(axis='y', visible=False); plt.tight_layout(); plt.show()
    display(pd.DataFrame([dict(model='expert', mse=a['expert_mse'], log_ratio=0)] + [dict(model=lab[k], **a[k]) for k in L]).round(4))
```

</details>

![Saved plot from cell 39](assets/7f1b09292726c4bad715388e502e1e3827afb1e864eb66ef7f94fb13865466a2.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>mse</th>
      <th>log_ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>expert</td>
      <td>0.0732</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37 speeds kept</td>
      <td>0.0027</td>
      <td>3.3007</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38 speeds removed</td>
      <td>0.0014</td>
      <td>3.9658</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41 learned arm</td>
      <td>0.0019</td>
      <td>3.6738</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43 clean</td>
      <td>0.0030</td>
      <td>3.1855</td>
    </tr>
    <tr>
      <th>5</th>
      <td>52 fixed DART</td>
      <td>0.0018</td>
      <td>3.6923</td>
    </tr>
    <tr>
      <th>6</th>
      <td>54 DAgger</td>
      <td>0.0050</td>
      <td>2.6901</td>
    </tr>
    <tr>
      <th>7</th>
      <td>56 both</td>
      <td>0.0042</td>
      <td>2.8519</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-40"></a>

**Reading.** A negative result to keep. Every learner is far more predictable than the expert, and iteration 38 without speeds (3.97) is more predictable than 37 with them (3.30), the opposite ordering to what the copycat diagnostic expects. Predictable commands can come from unhelpful repetition or from smooth successful control, and this test cannot separate them here. This does not refute Wen et al.: the task differs and the diagnostic is adapted. It only says action predictability did not discriminate harmful copying on this task. The range gain in Section 3.12 did.

<a id="cell-41"></a>

## 3.11 The task in one dimension

The framing: driving is a turn, then one straight approach; reaching is one progress variable per target. PCA is from Lecture 2.2, and Liu 2025 validates its theory with a one-dimensional simulation. The 1D reduction itself is ours. First, is it true of the expert?

<a id="cell-42"></a>

<details><summary>Code · original cell 42</summary>

```python
e = MAC['expert_1d']
display(pd.DataFrame([('Driving', 'path length / straight line (10th, 50th, 90th percentile)', e['path_over_straight']), ('Driving', 'share of drive within 10 degrees of target', e['share_of_drive_within_10deg']), ('Driving', 'base speed, 95th percentile (m/s)', e['base_speed_95pct_m_s']), ('Driving', 'time to PARK from distance to go, R²', e['ttp_from_distance_to_go_r2']), ('Driving', 'speed from distance to go, R²', e['speed_from_distance_to_go_r2']), ('Driving', 'range at PARK minus standoff table: median, SD (m)', [e['minus_standoff_table_median_m'], e['minus_standoff_table_sd_m']]), ('Reaching', 'gripper path / straight line', e['gripper_path_over_straight']), ('Reaching', 'share of frames where gripper distance does not grow', e['gripper_distance_not_growing']), ('Reaching', 'first principal direction of 7-joint path, per run', e['joint_path_pc1_per_run']), ('Reaching', 'first three directions, pooled over runs', e['joint_path_pc123_pooled']), ('Reaching', 'normalised distance curve, variance explained by one mean curve', e['reach_curve_variance_explained'])], columns=['phase', 'measure', 'value']))
if COMP:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.3)); G = np.array(COMP['expert']['grid'])
    ax[0].plot(COMP['expert']['range_to_go_grid'], COMP['expert']['range_to_go_m']['median'], color=INK, label='expert')
    for k in ('height', 'no_speed'):
        if k in COMP: ax[0].plot(COMP['expert']['range_to_go_grid'], COMP[k]['range_to_go_m']['median'], color=COL[k], label={'height': '37 speeds kept', 'no_speed': '38 speeds removed'}[k])
    ax[0].set_xlabel('seconds before PARK'); ax[0].set_ylabel('distance to its own park point (m)'); ax[0].set_title('1D driving: median over runs'); ax[0].invert_xaxis(); ax[0].legend(fontsize=8)
    ax[1].plot(G, COMP['expert']['gripper_mm']['median'], color=INK, label='expert')
    for k in ('basics_standard', 'dart', 'dagger'):
        ax[1].plot(G, COMP[k]['gripper_mm']['median'], color=COL[k], label=SHORT[k]); ax[1].fill_between(G, COMP[k]['gripper_mm']['p25'], COMP[k]['gripper_mm']['p75'], color=COL[k], alpha=.12, lw=0)
    ax[1].axhline(10, color=MUTED, ls='--', lw=1); ax[1].set_yscale('log'); ax[1].set_xlabel('seconds after PARK'); ax[1].set_ylabel('gripper to target (mm, log)'); ax[1].set_title('1D reaching: median and middle half'); ax[1].legend(fontsize=8); plt.tight_layout(); plt.show()
```

</details>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>phase</th>
      <th>measure</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Driving</td>
      <td>path length / straight line (10th, 50th, 90th ...</td>
      <td>[1.0672453457925688, 1.2204257842358728, 1.602...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Driving</td>
      <td>share of drive within 10 degrees of target</td>
      <td>[0.5267262969588552, 0.7374887398658794, 0.889...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Driving</td>
      <td>base speed, 95th percentile (m/s)</td>
      <td>[0.25129967128603037, 0.25156130108924024, 0.2...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Driving</td>
      <td>time to PARK from distance to go, R²</td>
      <td>0.988642</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Driving</td>
      <td>speed from distance to go, R²</td>
      <td>0.649371</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Driving</td>
      <td>range at PARK minus standoff table: median, SD...</td>
      <td>[0.05758769210232001, 0.01403798224707332]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Reaching</td>
      <td>gripper path / straight line</td>
      <td>[1.0424975438731432, 1.0929924120024304, 1.508...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Reaching</td>
      <td>share of frames where gripper distance does no...</td>
      <td>[0.9829600484261501, 1.0, 1.0]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Reaching</td>
      <td>first principal direction of 7-joint path, per...</td>
      <td>[0.8192198118532168, 0.936617101350537, 0.9817...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Reaching</td>
      <td>first three directions, pooled over runs</td>
      <td>[0.4386665842222059, 0.377791579965176, 0.1175...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Reaching</td>
      <td>normalised distance curve, variance explained ...</td>
      <td>0.997573</td>
    </tr>
  </tbody>
</table>
</div>

![Saved plot from cell 42](assets/172181f899c5e256bb9b8ce45a70292a436390a218410d0a1dab7ac163c2f41d.png)


<a id="cell-43"></a>

**Reading.** The expert's range at PARK is the standoff table plus 5.8 cm with an SD of 1.4 cm, so parking is close to a function of height alone. In each run the 7-joint path is about one direction (94%), and it takes three directions across runs, one per coordinate of the target. Two things do not reduce: the blind turn, which is its own problem with no feedback on bearing, and torque, which acts through inertia, so the arm is a harder kind of 1D problem than the wheels.

<a id="cell-44"></a>

## 3.12 Driving in 1D: how strongly the command reacts

In one dimension the loop is `δ(t+1) = δ(t) - dt·ρ·δu(t)`, `δu(t) = a·δu(t-1) + k·δ(t)`, with `k` the gain on range, `a` the gain on the network's own measured speed, and ρ the wheel radius, 0.11 m. These are read exactly from the network: a ReLU network is piecewise linear (Lecture 3.2), so its local slope comes from the active units. Reference: the expert's law `v = clip(1.0·d, 0.05, 0.25)` gives k = 9.1 rad/s per m and a 1 s time constant in the last 25 cm. Method not from the course.

<a id="cell-45"></a>

<details><summary>Code · original cell 45</summary>

```python
if EXTRA and 'driving_gains' in EXTRA:
    g = EXTRA['driving_gains']; P = 'last 1 s before PARK'; keys = [k for k in g if k != 'expert']; show = {'34 speeds kept, seed 7': '34 speeds kept, seed 7', '35 speeds kept, seed 1': '35 speeds kept, seed 1', '36 speeds kept, seed 2': '36 speeds kept, seed 2', '37 speeds kept, height added': '37 speeds kept, height added', '38 speeds removed': '38 speeds removed', '43 basics_standard': '43 clean (network of 38)', '52 dart': '52 fixed DART', '54 dagger': '54 DAgger', '56 dart_dagger': '56 both'}; rows = [dict(model=show[k], range_gain=g[k][P]['range_gain_rad_s_per_m'][0], self_gain=g[k][P]['self_gain'][0], steering_gain=g[k][P]['steering_gain'][0], ttp_s_per_m=g[k][P]['ttp_s_per_m'][0], time_constant_s=g[k][P]['time_constant_s'][0], share_not_decaying=g[k][P]['share_not_decaying']) for k in keys]
    T = pd.DataFrame(rows); fig, ax = plt.subplots(1, 3, figsize=(13, 3.2)); y = np.arange(len(T))[::-1]
    ax[0].barh(y, T.range_gain, color=BLUE, height=.6); ax[0].axvline(1 / .11, color=ORANGE, lw=1.3, ls='--'); ax[0].set_yticks(y); ax[0].set_yticklabels(T.model); ax[0].set_title('Gain on range (rad/s per m)'); ax[0].text(1 / .11, y[-1] - .3, ' expert', color=INK, fontsize=9)
    ax[1].barh(y, T.self_gain, color=MUTED, height=.6); ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_title('Gain on its own measured speed')
    ax[2].barh(y, T.time_constant_s, color=AQUA, height=.6); ax[2].axvline(1, color=ORANGE, lw=1.3, ls='--'); ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_title('1D loop time constant (s)'); ax[2].set_xscale('log')
    for a in ax: a.grid(axis='y', visible=False)
    plt.tight_layout(); plt.show(); display(T.round(3))
```

</details>

![Saved plot from cell 45](assets/af764260ac4ecec6ee7c74abc6216186466f0ec65597caee53946dce6f5e6605.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>range_gain</th>
      <th>self_gain</th>
      <th>steering_gain</th>
      <th>ttp_s_per_m</th>
      <th>time_constant_s</th>
      <th>share_not_decaying</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34 speeds kept, seed 7</td>
      <td>0.108</td>
      <td>0.853</td>
      <td>1.320</td>
      <td>0.373</td>
      <td>15.406</td>
      <td>0.132</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35 speeds kept, seed 1</td>
      <td>0.092</td>
      <td>1.009</td>
      <td>1.635</td>
      <td>0.880</td>
      <td>268.821</td>
      <td>0.688</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36 speeds kept, seed 2</td>
      <td>0.050</td>
      <td>0.952</td>
      <td>1.780</td>
      <td>0.176</td>
      <td>11.602</td>
      <td>0.249</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37 speeds kept, height added</td>
      <td>0.096</td>
      <td>0.889</td>
      <td>1.299</td>
      <td>0.249</td>
      <td>24.377</td>
      <td>0.294</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38 speeds removed</td>
      <td>9.573</td>
      <td>0.000</td>
      <td>2.325</td>
      <td>11.019</td>
      <td>0.933</td>
      <td>0.035</td>
    </tr>
    <tr>
      <th>5</th>
      <td>43 clean (network of 38)</td>
      <td>9.573</td>
      <td>0.000</td>
      <td>2.325</td>
      <td>11.019</td>
      <td>0.933</td>
      <td>0.035</td>
    </tr>
    <tr>
      <th>6</th>
      <td>52 fixed DART</td>
      <td>9.396</td>
      <td>0.000</td>
      <td>2.244</td>
      <td>10.660</td>
      <td>0.915</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54 DAgger</td>
      <td>9.258</td>
      <td>0.000</td>
      <td>10.832</td>
      <td>9.822</td>
      <td>0.965</td>
      <td>0.005</td>
    </tr>
    <tr>
      <th>8</th>
      <td>56 both</td>
      <td>5.814</td>
      <td>0.000</td>
      <td>9.409</td>
      <td>8.145</td>
      <td>1.547</td>
      <td>0.001</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-46"></a>

**Reading.** Prediction held, more cleanly than I expected. With speed inputs the range gain is 0.05 to 0.11 rad/s per metre, about 100 times below the expert's 9.1, and the gain on measured speed is 0.85 to 1.01, close to 1. That is a network repeating its last command with almost no regard for range. Without them, the gain is 9.6 and the time constant 0.93 s, against the expert's 9.1 and 1 s. Nobody told it that law: it came out of the data once the shortcut was gone. Iterations 38 and 43 share one driving network, so they have one row each with the same values. DAgger also raises the steering gain from 2.3 to 10.8. The expert's law has its own steering, and I did not compare that.

<a id="cell-47"></a>

## 3.13 Arm gains: did recovery data teach feedback?

From the same exact local slopes: K is how strongly torque reacts to joint angle, D to joint speed, and the target gain is N m per cm of target movement. The expert's own PD law is K = 100 and D = 30. Noise line: encoder noise of 0.14 rad/s on speed, through D, as torque noise.

<a id="cell-48"></a>

<details><summary>Code · original cell 48</summary>

```python
if DEEP:
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.2)); x = np.arange(1, 8); w = .26; L = ['basics_standard', 'dart', 'dagger']
    for i, k in enumerate(L):
        g = DEEP['models'][k]['arm_gains']; off = (i - 1) * w
        ax[0].bar(x + off, g['k_diag_median'], w, color=COL[k], label=SHORT[k]); ax[1].bar(x + off, g['d_diag_median'], w, color=COL[k]); ax[2].bar(x + off, g['target_gain_nm_per_cm'], w, color=COL[k]); ax[3].bar(x + off, g['torque_noise_sd_nm'], w, color=COL[k])
    ax[0].axhline(100, color=INK, ls='--', lw=1); ax[0].text(7.4, 103, 'expert', ha='right', fontsize=9); ax[1].axhline(30, color=INK, ls='--', lw=1); ax[1].text(7.4, 31, 'expert', ha='right', fontsize=9)
    for a, t, u in zip(ax, ('K: gain on joint angle', 'D: gain on joint speed', 'Gain on target position', 'Torque noise from encoders'), ('N m/rad', 'N m/(rad/s)', 'N m per cm', 'N m, SD')): a.set_title(t); a.set_ylabel(u); a.set_xlabel('joint'); a.set_xticks(x); a.grid(axis='x', visible=False)
    ax[0].set_ylim(0, 135); ax[0].legend(fontsize=8, loc='upper left', ncol=3, columnspacing=.8, handlelength=1.2); plt.tight_layout(); plt.show()
    display(pd.DataFrame([dict(model=SHORT[k], K_median=np.median(DEEP['models'][k]['arm_gains']['k_diag_median']), D_median=np.median(DEEP['models'][k]['arm_gains']['d_diag_median']), target_gain_median=np.median(DEEP['models'][k]['arm_gains']['target_gain_nm_per_cm']), noise_torque_median=np.median(DEEP['models'][k]['arm_gains']['torque_noise_sd_nm']), torque_rmse=np.mean([q['rmse'] for q in OFF['models'][k]['torque']])) for k in MODELS]).round(3))
```

</details>

![Saved plot from cell 48](assets/04368d85b3094864d511b29a424e2e568ab1ddf5742766d273789c2750a9483f.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>K_median</th>
      <th>D_median</th>
      <th>target_gain_median</th>
      <th>noise_torque_median</th>
      <th>torque_rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>4.712</td>
      <td>8.045</td>
      <td>0.055</td>
      <td>2.685</td>
      <td>0.378</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43 clean</td>
      <td>7.488</td>
      <td>6.237</td>
      <td>0.024</td>
      <td>1.791</td>
      <td>0.296</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52 fixed DART</td>
      <td>8.666</td>
      <td>15.104</td>
      <td>0.031</td>
      <td>2.758</td>
      <td>0.303</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54 DAgger</td>
      <td>68.963</td>
      <td>30.837</td>
      <td>0.740</td>
      <td>6.360</td>
      <td>1.189</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56 both</td>
      <td>80.962</td>
      <td>27.225</td>
      <td>0.836</td>
      <td>5.287</td>
      <td>0.915</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-49"></a>

**Reading.** Median K goes from 7 to 69 N m/rad and D from 6 to 31, and the expert's are 100 and 30. Fixed noise sits between (9, 15). A supported explanation: each expert reach is close to one direction in joint space (Section 3.11), so a demonstration says little about how to respond to a displacement in other directions. Several networks can then match the torques along the path while having very different slopes away from it. DAgger rows supply those slopes. PCA alone cannot establish that the demonstrations caused the low gains, so this is an explanation and not a proof.

The cost is measurement sensitivity. Encoder noise passes through D, so torque noise is 1.6 to 2.9 N m for the clean model and 4.0 to 8.4 N m for DAgger. Stronger gains correct a real displacement and also amplify a false one reported by a noisy sensor. The training noise and the sensor noise also differ in kind: action noise held for 33 ms, against encoder noise entering a 1 ms backward difference.

<a id="cell-50"></a>

## 3.14 Is each network a stable controller? (Zhang 2025)

Zhang names control-theoretic stability as the mechanism behind compounding, and calls it the continuous form of recoverability. Linearise robot plus network around saved states: `M(q) δq̈ = (∂τ/∂q − ∂G/∂q) δq + (∂τ/∂q̇) δq̇`, with M and G from the simulator's own model, 1 kg link included. A positive growth rate means small errors grow on their own. Also shown: the discrete loop at 1 ms using the backward-difference speed, with and without the 3 ms delay. The simulator is not stepped. Method not from the course. Local linear checks on a piecewise linear network do not prove behaviour over a whole run.

<a id="cell-51"></a>

<details><summary>Code · original cell 51</summary>

```python
if STAB:
    L = list(STAB); nice = lambda k: SHORT.get(k, k.replace('basics_standard', '43 clean')); fig, ax = plt.subplots(1, 3, figsize=(13, 3.2)); y = np.arange(len(L))[::-1]
    ax[0].barh(y, [STAB[k]['unstable_share'] * 100 for k in L], color=BLUE, height=.6); ax[0].set_yticks(y); ax[0].set_yticklabels([nice(k) for k in L]); ax[0].set_xlabel('% of states locally unstable'); ax[0].set_title('Continuous check')
    ax[1].barh(y, [STAB[k]['growth_rate_per_s'][1] for k in L], color=ORANGE, height=.6); ax[1].axvline(0, color=INK, lw=.8); ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlabel('median fastest growth rate (per s)'); ax[1].set_title('Growth rate')
    h = .36; ax[2].barh(y + h / 2, [STAB[k]['unstable_share_1ms'] * 100 for k in L], h, color=AQUA, label='1 ms loop'); ax[2].barh(y - h / 2, [STAB[k]['unstable_share_3ms_delay'] * 100 for k in L], h, color=MUTED, label='with 3 ms delay'); ax[2].set_yticks(y); ax[2].set_yticklabels([]); ax[2].set_xlabel('% of states unstable'); ax[2].set_title('Discrete loop'); ax[2].legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
    for a in ax: a.grid(axis='y', visible=False)
    plt.tight_layout(); plt.show()
    display(pd.DataFrame([dict(model=nice(k), rows=STAB[k]['rows'], unstable_pct=STAB[k]['unstable_share'] * 100, growth_p10=STAB[k]['growth_rate_per_s'][0], growth_p50=STAB[k]['growth_rate_per_s'][1], growth_p90=STAB[k]['growth_rate_per_s'][2], e_fold_s=STAB[k]['e_fold_time_s_if_unstable'], radius_1ms=STAB[k]['spectral_radius_1ms'][0], unstable_1ms_pct=STAB[k]['unstable_share_1ms'] * 100, unstable_3ms_pct=STAB[k]['unstable_share_3ms_delay'] * 100) for k in L]).round(3))
```

</details>

![Saved plot from cell 51](assets/47f103f30f7bed8de62eccca9a8192a04378489001b11d3885f46d32f5106f21.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rows</th>
      <th>unstable_pct</th>
      <th>growth_p10</th>
      <th>growth_p50</th>
      <th>growth_p90</th>
      <th>e_fold_s</th>
      <th>radius_1ms</th>
      <th>unstable_1ms_pct</th>
      <th>unstable_3ms_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>expert gains (kp 100, kd 30)</td>
      <td>300</td>
      <td>0.000</td>
      <td>-3.183</td>
      <td>-3.175</td>
      <td>-3.165</td>
      <td>NaN</td>
      <td>0.997</td>
      <td>12.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41 learned arm</td>
      <td>1209</td>
      <td>100.000</td>
      <td>1.806</td>
      <td>12.267</td>
      <td>47.284</td>
      <td>0.082</td>
      <td>1.013</td>
      <td>100.000</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43 clean</td>
      <td>1209</td>
      <td>99.835</td>
      <td>0.890</td>
      <td>3.265</td>
      <td>22.599</td>
      <td>0.305</td>
      <td>1.003</td>
      <td>99.835</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>52 fixed DART</td>
      <td>1209</td>
      <td>97.270</td>
      <td>0.152</td>
      <td>0.681</td>
      <td>2.785</td>
      <td>1.429</td>
      <td>1.001</td>
      <td>97.270</td>
      <td>100.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>54 DAgger</td>
      <td>1209</td>
      <td>49.049</td>
      <td>-0.861</td>
      <td>-0.020</td>
      <td>4.360</td>
      <td>1.027</td>
      <td>1.000</td>
      <td>53.184</td>
      <td>98.428</td>
    </tr>
    <tr>
      <th>5</th>
      <td>56 both</td>
      <td>1209</td>
      <td>52.605</td>
      <td>-0.899</td>
      <td>0.077</td>
      <td>3.408</td>
      <td>1.251</td>
      <td>1.000</td>
      <td>53.515</td>
      <td>99.586</td>
    </tr>
    <tr>
      <th>6</th>
      <td>43 clean at its own states</td>
      <td>819</td>
      <td>99.756</td>
      <td>0.997</td>
      <td>3.034</td>
      <td>21.191</td>
      <td>0.329</td>
      <td>1.003</td>
      <td>99.756</td>
      <td>100.000</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-52"></a>

**Reading.** Exploratory: local linear checks on a piecewise linear network, on saved states. They do not prove behaviour over a run and should not carry the central conclusion. The expert's gains are stable everywhere (-3.17 per second). The clean arm is locally unstable at 99.8% of expert states, growth rate 3.26 per second. Learning to hold without standardising gives 12.27 per second, fixed DART 0.68, DAgger -0.02 with 49% of states unstable. The check at the clean model's own states gives about the same result as at the expert's. The 3 ms delay column marks every model as unstable, the expert's gains included, so it cannot separate them and I would not use it. Over a whole run errors do not grow by the factor these rates imply, because the state keeps moving through different linear regions.

<a id="cell-53"></a>

## 3.15 How error grows after a mistake (Ross, Choudhury L03, L04, L09)

BC can compound with T², on-policy data gives T. For each test run, distance in joint space from the expert's path, by time since PARK. Growth exponent is the log-log slope per run between 0.1 and 2 s. 1 means linear growth, 2 quadratic. Bounds in the papers are worst cases on a different cost, so this is an illustration and does not test the bound.

<a id="cell-54"></a>

<details><summary>Code · original cell 54</summary>

```python
if COMP:
    G = np.array(COMP['expert']['grid']); fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    for k in ('basics_standard', 'dart', 'dagger', 'dart_dagger'):
        v = COMP[k]['joint_dev_rad']; ax[0].plot(G, v['median'], color=COL[k], label=SHORT[k]); ax[0].fill_between(G, v['p25'], v['p75'], color=COL[k], alpha=.1, lw=0)
    ax[0].set_xlabel('seconds after PARK'); ax[0].set_ylabel('joint distance from expert path (rad)'); ax[0].set_title('Arm: distance from the expert path'); ax[0].legend(fontsize=8)
    Gb = np.arange(0, 12.01, .25)
    for k, col, lab in (('height', BLUE, '37 speeds kept'), ('no_speed', ORANGE, '38 speeds removed'), ('dagger', AQUA, '54 DAgger')):
        if k in COMP: ax[1].plot(Gb, COMP[k]['base_path_dev_m']['median'], color=col, label=lab)
    ax[1].set_xlim(0, 9); ax[1].set_xlabel('seconds since start'); ax[1].set_ylabel('base distance from expert path (m)'); ax[1].set_title('Base: distance from the expert path'); ax[1].legend(fontsize=8); plt.tight_layout(); plt.show()
    display(pd.DataFrame([dict(model={'height': '37 speeds kept', 'no_speed': '38 speeds removed', **SHORT}[k], growth_exponent_arm=COMP[k]['growth_exponent_joint'][0], growth_exponent_base=COMP[k]['growth_exponent_base'][0], park_along_range_cm=COMP[k]['park_radial_abs_cm'][0], park_sideways_cm=COMP[k]['park_tangential_cm'][0], park_heading_deg=COMP[k]['park_heading_deg'][0], tracker_range_err_at_park_mm=COMP[k]['tracker_range_err_at_park_mm'][0]) for k in COMP if k != 'expert']).round(3))
```

</details>

![Saved plot from cell 54](assets/be9798d1e8d992890bbbc1d5802aa177df04a9b70eba8ff83c8de49ae1890b01.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>growth_exponent_arm</th>
      <th>growth_exponent_base</th>
      <th>park_along_range_cm</th>
      <th>park_sideways_cm</th>
      <th>park_heading_deg</th>
      <th>tracker_range_err_at_park_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37 speeds kept</td>
      <td>1.291</td>
      <td>0.874</td>
      <td>17.686</td>
      <td>6.660</td>
      <td>10.378</td>
      <td>7.746</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38 speeds removed</td>
      <td>1.516</td>
      <td>0.803</td>
      <td>3.732</td>
      <td>3.691</td>
      <td>2.826</td>
      <td>8.597</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41 learned arm</td>
      <td>0.530</td>
      <td>0.793</td>
      <td>3.317</td>
      <td>3.491</td>
      <td>2.990</td>
      <td>8.464</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43 clean</td>
      <td>0.911</td>
      <td>0.800</td>
      <td>3.376</td>
      <td>3.778</td>
      <td>2.936</td>
      <td>9.044</td>
    </tr>
    <tr>
      <th>4</th>
      <td>52 fixed DART</td>
      <td>0.601</td>
      <td>0.749</td>
      <td>2.955</td>
      <td>3.854</td>
      <td>2.934</td>
      <td>9.243</td>
    </tr>
    <tr>
      <th>5</th>
      <td>54 DAgger</td>
      <td>0.771</td>
      <td>0.843</td>
      <td>3.269</td>
      <td>2.567</td>
      <td>2.989</td>
      <td>8.984</td>
    </tr>
    <tr>
      <th>6</th>
      <td>56 both</td>
      <td>0.801</td>
      <td>0.900</td>
      <td>2.704</td>
      <td>3.386</td>
      <td>1.766</td>
      <td>9.691</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-55"></a>

**Reading.** No T² against T here. Exponents are 0.91 for the clean arm ([0.73, 1.13]), 0.77 for DAgger ([0.69, 0.88]) and 0.60 for fixed DART, overlapping and all at or below linear. The bounds are worst cases, on a different cost, so this does not contradict them, but the report should not claim to show them. What separates the models is the level of the curve: the size of the error early on, rather than its growth rate.

<a id="cell-56"></a>

## 3.16 How far is fixed noise from Laskey's rule?

True DART sets the noise covariance from the model's own residuals. So: residual SD of the model on the expert's training states, against what we injected, 0.025 rad/s on the wheels and 0.5 N m on the arm. An initial 0.1 rad/s caused a navigation timeout in a pilot, which is why the wheel noise is so small. Zhang finds larger noise helps within tolerance, and that clean labels and a clean plus noisy mix are what matter.

<a id="cell-57"></a>

<details><summary>Code · original cell 57</summary>

```python
if EXTRA and 'dart_noise' in EXTRA:
    d = EXTRA['dart_noise']['basics_standard']; fig, ax = plt.subplots(1, 2, figsize=(10, 2.9))
    ax[0].bar(range(1, 5), d['wheel_residual_sd'], color=BLUE, width=.55, label='model residual SD'); ax[0].axhline(.025, color=ORANGE, lw=1.5, label='injected, 0.025'); ax[0].set_xticks(range(1, 5)); ax[0].set_ylim(0, .85); ax[0].set_xlabel('wheel'); ax[0].set_ylabel('rad/s'); ax[0].set_title('Wheels'); ax[0].legend(fontsize=8, loc='upper right', ncol=2)
    ax[1].bar(range(1, 8), d['torque_residual_sd'], color=BLUE, width=.55, label='model residual SD'); ax[1].axhline(.5, color=ORANGE, lw=1.5, label='injected, 0.5'); ax[1].set_ylim(0, .7); ax[1].set_xlabel('joint'); ax[1].set_ylabel('N m'); ax[1].set_title('Arm'); ax[1].legend(fontsize=8, loc='upper right', ncol=2)
    for a in ax: a.grid(axis='x', visible=False)
    plt.tight_layout(); plt.show(); display(pd.DataFrame(EXTRA['dart_noise']).T[['ratio_wheel', 'ratio_torque', 'wheel_cov_trace', 'torque_cov_trace']].astype(float).round(3))
```

</details>

![Saved plot from cell 57](assets/5a68259e5bffa315f4802919d7c6df1ba173770a146cc8a42cd00f7e7d4a3acd.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ratio_wheel</th>
      <th>ratio_torque</th>
      <th>wheel_cov_trace</th>
      <th>torque_cov_trace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>basics_standard</th>
      <td>23.705</td>
      <td>0.603</td>
      <td>1.405</td>
      <td>0.634</td>
    </tr>
    <tr>
      <th>dart</th>
      <td>23.802</td>
      <td>0.628</td>
      <td>1.416</td>
      <td>0.688</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-58"></a>

**Reading.** Model error on the expert's states is 0.6 times the injected noise on the arm and 24 times on the wheels. This shows a scale mismatch worth investigating, and it does not show that the arm noise was right or that an adaptive version would do better. Part of the wheel residual is the blind start, which noise cannot fix. It does fit one observation: the driving gains of 52 match the clean model while its arm feedback changes (Sections 3.12 and 3.13).

<a id="cell-59"></a>

## 3.17 Do the heads compete for the shared layers? (GradNorm, PCGrad, Kirchdorfer, UDL 9.3.6)

Gradient size of each head's loss on the shared layers, and the angle between them. GradNorm says the larger gradient dominates; PCGrad counts cosine below 0 as conflict, and magnitude similarity near 0 as domination. The last column is the weight an inverse-loss rule (Kirchdorfer) would give the extra head.

<a id="cell-60"></a>

<details><summary>Code · original cell 60</summary>

```python
if HEAD:
    d = pd.DataFrame(HEAD['driving_retrained_300_epochs']); a = pd.DataFrame(HEAD['arm_saved_checkpoints']); fig, ax = plt.subplots(1, 3, figsize=(13, 3.1))
    ax[0].plot(d.epoch, d.grad_norm_command, color=BLUE, label='wheels'); ax[0].plot(d.epoch, d.grad_norm_aux, color=ORANGE, label='time to PARK'); ax[0].set_yscale('log'); ax[0].set_xlabel('epoch'); ax[0].set_title('Driving: gradient size on shared layers'); ax[0].legend(fontsize=8)
    ax[1].plot(d.epoch, d.cosine, color=GREY); ax[1].axhline(0, color=INK, lw=.8); ax[1].set_xlabel('epoch'); ax[1].set_title('Driving: cosine between the two gradients'); ax[1].set_ylim(-1, 1)
    x = np.arange(len(a)); w = .36; ax[2].bar(x - w / 2, a.grad_norm_command, w, color=BLUE, label='torques'); ax[2].bar(x + w / 2, a.grad_norm_aux, w, color=ORANGE, label='extra distance output'); ax[2].set_yscale('log'); ax[2].set_xticks(x); ax[2].set_xticklabels(a.checkpoint); ax[2].set_title('Arm: gradient size at saved checkpoints'); ax[2].legend(fontsize=8); ax[2].grid(axis='x', visible=False)
    plt.tight_layout(); plt.show()
    display(d.iloc[[0, 1, 5, -1]][['epoch', 'loss_command', 'loss_aux', 'grad_norm_command', 'grad_norm_aux', 'norm_ratio', 'cosine', 'inverse_loss_weights_aux_over_command']].round(4)); display(a[['checkpoint', 'loss_command', 'loss_aux', 'grad_norm_command', 'grad_norm_aux', 'norm_ratio', 'cosine', 'inverse_loss_weights_aux_over_command']].round(4))
```

</details>

![Saved plot from cell 60](assets/05159a9cb0e122467ab49f51723b9a06bd9934114104aa0531cea7f44f7f508a.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>loss_command</th>
      <th>loss_aux</th>
      <th>grad_norm_command</th>
      <th>grad_norm_aux</th>
      <th>norm_ratio</th>
      <th>cosine</th>
      <th>inverse_loss_weights_aux_over_command</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3.0314</td>
      <td>0.2626</td>
      <td>0.4113</td>
      <td>0.4906</td>
      <td>0.8383</td>
      <td>0.1993</td>
      <td>11.5427</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>1.3652</td>
      <td>0.0441</td>
      <td>0.3344</td>
      <td>0.0850</td>
      <td>3.9334</td>
      <td>0.0773</td>
      <td>30.9494</td>
    </tr>
    <tr>
      <th>5</th>
      <td>150</td>
      <td>0.3881</td>
      <td>0.0108</td>
      <td>0.0842</td>
      <td>0.0432</td>
      <td>1.9462</td>
      <td>-0.3177</td>
      <td>35.8384</td>
    </tr>
    <tr>
      <th>10</th>
      <td>300</td>
      <td>0.2888</td>
      <td>0.0082</td>
      <td>0.1793</td>
      <td>0.0282</td>
      <td>6.3690</td>
      <td>0.6005</td>
      <td>35.2165</td>
    </tr>
  </tbody>
</table>
</div>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>checkpoint</th>
      <th>loss_command</th>
      <th>loss_aux</th>
      <th>grad_norm_command</th>
      <th>grad_norm_aux</th>
      <th>norm_ratio</th>
      <th>cosine</th>
      <th>inverse_loss_weights_aux_over_command</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>init</td>
      <td>0.9848</td>
      <td>0.7189</td>
      <td>0.3462</td>
      <td>1.0671</td>
      <td>0.3245</td>
      <td>0.0164</td>
      <td>1.3699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>epoch 300</td>
      <td>0.0200</td>
      <td>0.0023</td>
      <td>0.2681</td>
      <td>0.0118</td>
      <td>22.6646</td>
      <td>-0.0012</td>
      <td>8.7890</td>
    </tr>
    <tr>
      <th>2</th>
      <td>epoch 600</td>
      <td>0.0150</td>
      <td>0.0039</td>
      <td>0.2012</td>
      <td>0.1377</td>
      <td>1.4618</td>
      <td>-0.0353</td>
      <td>3.7972</td>
    </tr>
    <tr>
      <th>3</th>
      <td>epoch 1200</td>
      <td>0.0086</td>
      <td>0.0007</td>
      <td>0.0264</td>
      <td>0.0152</td>
      <td>1.7334</td>
      <td>0.0273</td>
      <td>11.5744</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-61"></a>

**Reading.** By PCGrad's definition the driving heads do conflict for part of training: cosine is negative at 3 of 11 saved points, as low as -0.47, and ends at 0.60. The wheel gradient is 1.8 to 6.4 times the other after the first epoch. Losses end 35 to 1, so an inverse loss weight would give time to PARK about 35 times more. On the arm, cosine is between -0.035 and 0.027, close to orthogonal, and the ratio is 1.5 to 23. So domination on both networks, some conflict on driving, little on the arm. A hand-set weight of 10 had made parking worse, so the sizes alone do not decide this, and these methods remain untested here. This replaces the draft's single checkpoint figure.

<a id="cell-62"></a>

## 3.18 Full runs on the test targets (Lecture 1.2, Lab 3, Lab 4)

Success counts with Wilson intervals, medians with bootstrap intervals over targets. Comparisons are target by target: an exact sign test on distances, which is the lecture's fair-coin test, and McNemar on events. Before looking at any result, the alpha for this list is 0.05, and Holm's correction is applied across the list because many comparisons on one reused test set inflate false positives. Lecture 1.2 also says p is not the probability that H0 is true.

<a id="cell-63"></a>

<details><summary>Code · original cell 63</summary>

```python
S = CL['summary']; order = [('seed7', '34 seed 7'), ('seed1', '35 seed 1'), ('seed2', '36 seed 2'), ('height', '37 height'), ('no_speed', '38 no speeds'), ('longer_arm', '39 longer'), ('wider_arm', '40 wider'), ('baseline', '41 learned arm'), ('basics_standard', '43 standardised'), ('dart', '52 fixed DART'), ('dagger', '54 DAgger'), ('dart_dagger', '56 both')]
T = pd.DataFrame([dict(model=l, PARK=S[k]['parked'], STOP=S[k]['stopped'], success=S[k]['success'], success_low=S[k]['success_ci'][0], success_high=S[k]['success_ci'][1], park_error_cm=S[k]['park_error_cm'][0], closest_mm=S[k]['closest_mm'][0], closest_low=S[k]['closest_mm'][1], closest_high=S[k]['closest_mm'][2], at_stop_mm=S[k]['at_stop_mm'][0], at_stop_low=S[k]['at_stop_mm'][1], at_stop_high=S[k]['at_stop_mm'][2]) for k, l in order])
fig, ax = plt.subplots(1, 3, figsize=(14, 3.5)); x = np.arange(len(T))
ax[0].plot(x, T.PARK, color=BLUE, marker='o', label='PARK'); ax[0].plot(x, T.STOP, color=AQUA, marker='o', label='STOP'); ax[0].set_title('Events, of 100'); ax[0].set_ylim(0, 105); ax[0].legend(fontsize=8)
ax[1].plot(x, T.park_error_cm, color=BLUE, marker='o'); ax[1].set_title('Median park error (cm)')
ax[2].errorbar(x, T.closest_mm, yerr=[T.closest_mm - T.closest_low, T.closest_high - T.closest_mm], color=BLUE, marker='o', capsize=2, label='closest'); ax[2].errorbar(x, T.at_stop_mm, yerr=[T.at_stop_mm - T.at_stop_low, T.at_stop_high - T.at_stop_mm], color=ORANGE, marker='o', capsize=2, label='at STOP'); ax[2].axhline(10, color=MUTED, ls='--', lw=1); ax[2].set_yscale('log'); ax[2].set_title('Gripper to target (mm, log)'); ax[2].legend(fontsize=8)
for a in ax: a.set_xticks(x); a.set_xticklabels(T.model, rotation=45, ha='right')
plt.tight_layout(); plt.show(); display(T.round(1))
```

</details>

![Saved plot from cell 63](assets/9ed26865848db37cc2f4d260748389dfd8965bdbf2a27980ccc178ca2501ba63.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>PARK</th>
      <th>STOP</th>
      <th>success</th>
      <th>success_low</th>
      <th>success_high</th>
      <th>park_error_cm</th>
      <th>closest_mm</th>
      <th>closest_low</th>
      <th>closest_high</th>
      <th>at_stop_mm</th>
      <th>at_stop_low</th>
      <th>at_stop_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34 seed 7</td>
      <td>47</td>
      <td>47</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>27.3</td>
      <td>295.8</td>
      <td>217.3</td>
      <td>357.7</td>
      <td>400.4</td>
      <td>318.6</td>
      <td>532.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35 seed 1</td>
      <td>98</td>
      <td>60</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>22.1</td>
      <td>266.9</td>
      <td>224.6</td>
      <td>308.7</td>
      <td>558.8</td>
      <td>407.5</td>
      <td>923.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36 seed 2</td>
      <td>96</td>
      <td>81</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>28.6</td>
      <td>302.0</td>
      <td>259.1</td>
      <td>356.6</td>
      <td>490.3</td>
      <td>400.8</td>
      <td>720.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37 height</td>
      <td>87</td>
      <td>80</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>18.7</td>
      <td>293.5</td>
      <td>253.1</td>
      <td>351.1</td>
      <td>387.5</td>
      <td>324.5</td>
      <td>419.6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>38 no speeds</td>
      <td>100</td>
      <td>92</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>5.6</td>
      <td>52.5</td>
      <td>45.4</td>
      <td>64.7</td>
      <td>298.5</td>
      <td>253.2</td>
      <td>357.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>39 longer</td>
      <td>100</td>
      <td>98</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>5.6</td>
      <td>98.3</td>
      <td>76.5</td>
      <td>116.4</td>
      <td>193.7</td>
      <td>158.6</td>
      <td>255.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>40 wider</td>
      <td>100</td>
      <td>96</td>
      <td>2</td>
      <td>0.6</td>
      <td>7.0</td>
      <td>5.6</td>
      <td>59.1</td>
      <td>49.4</td>
      <td>70.3</td>
      <td>85.2</td>
      <td>61.8</td>
      <td>134.8</td>
    </tr>
    <tr>
      <th>7</th>
      <td>41 learned arm</td>
      <td>100</td>
      <td>84</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>5.5</td>
      <td>184.5</td>
      <td>143.9</td>
      <td>216.2</td>
      <td>272.7</td>
      <td>226.3</td>
      <td>383.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>43 standardised</td>
      <td>100</td>
      <td>91</td>
      <td>0</td>
      <td>0.0</td>
      <td>3.7</td>
      <td>5.3</td>
      <td>75.1</td>
      <td>69.3</td>
      <td>88.4</td>
      <td>168.8</td>
      <td>136.5</td>
      <td>201.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>52 fixed DART</td>
      <td>100</td>
      <td>78</td>
      <td>1</td>
      <td>0.2</td>
      <td>5.4</td>
      <td>5.5</td>
      <td>45.0</td>
      <td>39.5</td>
      <td>52.3</td>
      <td>60.9</td>
      <td>47.4</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>54 DAgger</td>
      <td>84</td>
      <td>69</td>
      <td>2</td>
      <td>0.6</td>
      <td>7.0</td>
      <td>4.2</td>
      <td>27.3</td>
      <td>19.8</td>
      <td>32.8</td>
      <td>37.0</td>
      <td>29.9</td>
      <td>47.6</td>
    </tr>
    <tr>
      <th>11</th>
      <td>56 both</td>
      <td>87</td>
      <td>66</td>
      <td>4</td>
      <td>1.6</td>
      <td>9.8</td>
      <td>4.9</td>
      <td>30.0</td>
      <td>24.7</td>
      <td>37.6</td>
      <td>34.5</td>
      <td>26.2</td>
      <td>39.3</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-64"></a>

<details><summary>Code · original cell 64</summary>

```python
def binom_two_sided(k, n):
    if n == 0: return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n)
def compare(a, b, col):
    A, B = per_target(a), per_target(b); d = (B[col] - A[col]).dropna(); pos, neg = int((d > 0).sum()), int((d < 0).sum()); return dict(n=pos + neg, better=neg, worse=pos, median_change=d.median(), p=binom_two_sided(pos, pos + neg))
plan = [('37 to 38: remove speeds', 'height', 'no_speed', 'park_error_cm'), ('37 to 38: remove speeds', 'height', 'no_speed', 'closest_mm'), ('39 to 40: wider arm', 'longer_arm', 'wider_arm', 'closest_mm'), ('40 to 41: learn hold', 'wider_arm', 'baseline', 'closest_mm'), ('41 to 43: standardise', 'baseline', 'basics_standard', 'closest_mm'), ('43 to 52: fixed DART', 'basics_standard', 'dart', 'closest_mm'), ('43 to 52: fixed DART', 'basics_standard', 'dart', 'at_stop_mm'), ('43 to 54: DAgger', 'basics_standard', 'dagger', 'closest_mm'), ('43 to 54: DAgger', 'basics_standard', 'dagger', 'at_stop_mm'), ('52 to 56: add DAgger', 'dart', 'dart_dagger', 'at_stop_mm'), ('54 to 56: add DART', 'dagger', 'dart_dagger', 'closest_mm'), ('54 to 56: add DART', 'dagger', 'dart_dagger', 'at_stop_mm')]
R = pd.DataFrame([dict(comparison=l, measure=c_, **compare(a, b, c_)) for l, a, b, c_ in plan]); o = np.argsort(R.p.values); m = len(R); adj = np.zeros(m); run = 0
for rank, i in enumerate(o): run = max(run, (m - rank) * R.p.values[i]); adj[i] = min(1, run)
R['holm_p'] = adj; R['verdict_at_0.05'] = np.where(R.holm_p <= .05, 'reject H0', 'do not reject'); display(R.round(4))
```

</details>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comparison</th>
      <th>measure</th>
      <th>n</th>
      <th>better</th>
      <th>worse</th>
      <th>median_change</th>
      <th>p</th>
      <th>holm_p</th>
      <th>verdict_at_0.05</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37 to 38: remove speeds</td>
      <td>park_error_cm</td>
      <td>87</td>
      <td>76</td>
      <td>11</td>
      <td>-14.2056</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37 to 38: remove speeds</td>
      <td>closest_mm</td>
      <td>87</td>
      <td>82</td>
      <td>5</td>
      <td>-221.1896</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>39 to 40: wider arm</td>
      <td>closest_mm</td>
      <td>100</td>
      <td>60</td>
      <td>40</td>
      <td>-16.5596</td>
      <td>0.0569</td>
      <td>0.1707</td>
      <td>do not reject</td>
    </tr>
    <tr>
      <th>3</th>
      <td>40 to 41: learn hold</td>
      <td>closest_mm</td>
      <td>100</td>
      <td>14</td>
      <td>86</td>
      <td>103.2428</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>41 to 43: standardise</td>
      <td>closest_mm</td>
      <td>100</td>
      <td>87</td>
      <td>13</td>
      <td>-100.0026</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>43 to 52: fixed DART</td>
      <td>closest_mm</td>
      <td>100</td>
      <td>76</td>
      <td>24</td>
      <td>-27.8436</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>43 to 52: fixed DART</td>
      <td>at_stop_mm</td>
      <td>71</td>
      <td>42</td>
      <td>29</td>
      <td>-20.7494</td>
      <td>0.1539</td>
      <td>0.3078</td>
      <td>do not reject</td>
    </tr>
    <tr>
      <th>7</th>
      <td>43 to 54: DAgger</td>
      <td>closest_mm</td>
      <td>84</td>
      <td>75</td>
      <td>9</td>
      <td>-48.0961</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>43 to 54: DAgger</td>
      <td>at_stop_mm</td>
      <td>66</td>
      <td>65</td>
      <td>1</td>
      <td>-129.9490</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>52 to 56: add DAgger</td>
      <td>at_stop_mm</td>
      <td>59</td>
      <td>52</td>
      <td>7</td>
      <td>-41.3395</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>reject H0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>54 to 56: add DART</td>
      <td>closest_mm</td>
      <td>81</td>
      <td>40</td>
      <td>41</td>
      <td>0.0237</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>do not reject</td>
    </tr>
    <tr>
      <th>11</th>
      <td>54 to 56: add DART</td>
      <td>at_stop_mm</td>
      <td>57</td>
      <td>37</td>
      <td>20</td>
      <td>-5.2177</td>
      <td>0.0331</td>
      <td>0.1326</td>
      <td>do not reject</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-65"></a>

**Reading.** 8 of 12 comparisons survive Holm at 0.05. Holm handles the size of this list only. It does not undo the repeated use of the test set during development, so these are descriptive comparisons on reused targets. I wrote the list after seeing summary scores, so it is not a pre-registered plan. Removing speeds, learning to hold (worse), standardising, DAgger, and adding DAgger to fixed DART all stand. Fixed DART's gain on closest distance stands, its gain at STOP does not. Adding DART on top of DAgger does not stand on either measure, and neither does the wider arm. Comparisons at STOP use only runs where both models stopped, so they say nothing about the runs that never stopped. Magnitudes with intervals are in Section 3.22.

<a id="cell-66"></a>

## 3.19 Where runs end, and what predicts failure (Lab 4)

Lab 4 asks for a confusion matrix and examples of failure. The closest thing here is each run's outcome.

<a id="cell-67"></a>

<details><summary>Code · original cell 67</summary>

```python
F = MAC['failure_breakdown']; L = ['no_speed', 'wider_arm', 'baseline', 'basics_standard', 'dart', 'dagger', 'dart_dagger']; parts = [('no_park', 'no PARK', MUTED), ('park_no_stop', 'PARK, no STOP', '#c9c8c3'), ('stop_within_500ms', 'STOP within 0.5 s', ORANGE), ('stop_later_outside', 'later STOP, outside 10 mm', BLUE), ('success', 'success', AQUA)]
fig, ax = plt.subplots(1, 2, figsize=(13, 3.3)); y = np.arange(len(L))[::-1]; left = np.zeros(len(L))
for k, lab, col in parts:
    v = np.array([F[m_][k] + (F[m_]['stop_at_park_within_50ms'] if k == 'stop_within_500ms' else 0) for m_ in L], float); ax[0].barh(y, v, left=left, color=col, label=lab, height=.62, edgecolor='white', lw=1); left += v
ax[0].set_yticks(y); ax[0].set_yticklabels([NAMES[m_] for m_ in L]); ax[0].set_xlim(0, 100); ax[0].set_title('How each test run ended'); ax[0].legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.1)); ax[0].grid(axis='y', visible=False)
w = .36; x = np.arange(len(L)); ax[1].bar(x - w / 2, [F[m_]['got_within_10mm'] for m_ in L], w, color=AQUA, label='gripper passed within 10 mm'); ax[1].bar(x + w / 2, [F[m_]['success'] for m_ in L], w, color=BLUE, label='STOP within 10 mm'); ax[1].set_xticks(x); ax[1].set_xticklabels([NAMES[m_] for m_ in L], rotation=30, ha='right'); ax[1].set_title('Getting there against stopping there'); ax[1].legend(fontsize=8); ax[1].grid(axis='x', visible=False)
plt.tight_layout(); plt.show()
C = MAC['chain']; display(pd.DataFrame([dict(model=NAMES[m_], rho_park_error_vs_closest=C[m_]['park_error_vs_closest'][0], low=C[m_]['park_error_vs_closest'][1], high=C[m_]['park_error_vs_closest'][2], rho_heading_vs_closest=C[m_]['heading_vs_closest'][0], rho_height_vs_closest=C[m_]['height_vs_closest'][0], rho_height_vs_park=C[m_]['height_vs_park_error'][0], closest_if_park_within_5cm=C[m_]['closest_if_park_error_5cm_or_less'][0], closest_if_park_over_5cm=C[m_]['closest_if_park_error_over_5cm'][0]) for m_ in ('basics_standard', 'dart', 'dagger', 'dart_dagger')]).round(2))
B = MAC['bands']['dagger']; display(pd.DataFrame([dict(band=k, n=v['n'], PARK=v['parked'], STOP=v['stopped'], success=v['success'], park_cm=v['park_error_cm'][0], closest_mm=v['closest_mm'][0], at_stop_mm=v['at_stop_mm'][0]) for k, v in B.items() if v['n']]).round(1))
V = MAC['val_vs_test']; display(pd.DataFrame([dict(model=SHORT[k], val_PARK=V[k]['val']['parked'], test_PARK=V[k]['test']['parked'], val_success=V[k]['val']['success'], test_success=V[k]['test']['success'], val_closest=V[k]['val']['closest_mm'][0], test_closest=V[k]['test']['closest_mm'][0], val_at_stop=V[k]['val']['at_stop_mm'][0], test_at_stop=V[k]['test']['at_stop_mm'][0]) for k in MODELS]).round(1))
```

</details>

![Saved plot from cell 67](assets/a1b80c53c540fd19cbb597ab8941d9dfbd8ace27cbae557c3a5af596390a6699.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>rho_park_error_vs_closest</th>
      <th>low</th>
      <th>high</th>
      <th>rho_heading_vs_closest</th>
      <th>rho_height_vs_closest</th>
      <th>rho_height_vs_park</th>
      <th>closest_if_park_within_5cm</th>
      <th>closest_if_park_over_5cm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43 standardised</td>
      <td>0.07</td>
      <td>-0.12</td>
      <td>0.27</td>
      <td>-0.12</td>
      <td>-0.20</td>
      <td>0.21</td>
      <td>74.89</td>
      <td>75.33</td>
    </tr>
    <tr>
      <th>1</th>
      <td>52 fixed-noise DART</td>
      <td>-0.02</td>
      <td>-0.21</td>
      <td>0.18</td>
      <td>0.02</td>
      <td>-0.36</td>
      <td>0.25</td>
      <td>39.77</td>
      <td>47.61</td>
    </tr>
    <tr>
      <th>2</th>
      <td>54 DAgger</td>
      <td>0.23</td>
      <td>0.04</td>
      <td>0.41</td>
      <td>0.36</td>
      <td>-0.21</td>
      <td>0.29</td>
      <td>26.20</td>
      <td>29.04</td>
    </tr>
    <tr>
      <th>3</th>
      <td>56 both</td>
      <td>-0.10</td>
      <td>-0.32</td>
      <td>0.14</td>
      <td>0.27</td>
      <td>-0.37</td>
      <td>0.38</td>
      <td>43.04</td>
      <td>27.35</td>
    </tr>
  </tbody>
</table>
</div>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band</th>
      <th>n</th>
      <th>PARK</th>
      <th>STOP</th>
      <th>success</th>
      <th>park_cm</th>
      <th>closest_mm</th>
      <th>at_stop_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>target_range_m 1 to 1.33</td>
      <td>39</td>
      <td>34</td>
      <td>25</td>
      <td>1</td>
      <td>4.9</td>
      <td>29.8</td>
      <td>48.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>target_range_m 1.33 to 1.66</td>
      <td>30</td>
      <td>25</td>
      <td>22</td>
      <td>0</td>
      <td>4.0</td>
      <td>19.3</td>
      <td>27.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>target_range_m 1.66 to 2.01</td>
      <td>31</td>
      <td>25</td>
      <td>22</td>
      <td>1</td>
      <td>4.1</td>
      <td>29.8</td>
      <td>36.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>target_height_m 0 to 0.5</td>
      <td>34</td>
      <td>32</td>
      <td>26</td>
      <td>2</td>
      <td>3.9</td>
      <td>31.1</td>
      <td>47.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>target_height_m 0.5 to 1.1</td>
      <td>56</td>
      <td>42</td>
      <td>34</td>
      <td>0</td>
      <td>4.1</td>
      <td>23.8</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>target_height_m 1.1 to 1.3</td>
      <td>10</td>
      <td>10</td>
      <td>9</td>
      <td>0</td>
      <td>16.2</td>
      <td>28.7</td>
      <td>35.4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bearing -180 to -90</td>
      <td>20</td>
      <td>20</td>
      <td>18</td>
      <td>0</td>
      <td>3.4</td>
      <td>29.8</td>
      <td>48.7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bearing -90 to -30</td>
      <td>27</td>
      <td>27</td>
      <td>26</td>
      <td>1</td>
      <td>5.1</td>
      <td>17.5</td>
      <td>30.8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bearing -30 to 30</td>
      <td>12</td>
      <td>12</td>
      <td>10</td>
      <td>0</td>
      <td>3.5</td>
      <td>19.8</td>
      <td>20.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bearing 30 to 90</td>
      <td>13</td>
      <td>11</td>
      <td>11</td>
      <td>1</td>
      <td>4.1</td>
      <td>27.4</td>
      <td>47.3</td>
    </tr>
    <tr>
      <th>10</th>
      <td>bearing 90 to 181</td>
      <td>28</td>
      <td>14</td>
      <td>4</td>
      <td>0</td>
      <td>8.0</td>
      <td>57.0</td>
      <td>56.0</td>
    </tr>
  </tbody>
</table>
</div>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>val_PARK</th>
      <th>test_PARK</th>
      <th>val_success</th>
      <th>test_success</th>
      <th>val_closest</th>
      <th>test_closest</th>
      <th>val_at_stop</th>
      <th>test_at_stop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>100</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>211.0</td>
      <td>184.5</td>
      <td>363.3</td>
      <td>272.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43 clean</td>
      <td>100</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>92.0</td>
      <td>75.1</td>
      <td>183.0</td>
      <td>168.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52 fixed DART</td>
      <td>100</td>
      <td>100</td>
      <td>2</td>
      <td>1</td>
      <td>54.3</td>
      <td>45.0</td>
      <td>81.4</td>
      <td>60.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54 DAgger</td>
      <td>89</td>
      <td>84</td>
      <td>4</td>
      <td>2</td>
      <td>25.3</td>
      <td>27.3</td>
      <td>35.6</td>
      <td>37.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56 both</td>
      <td>93</td>
      <td>87</td>
      <td>1</td>
      <td>4</td>
      <td>28.6</td>
      <td>30.0</td>
      <td>27.8</td>
      <td>34.5</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-68"></a>

**Reading.** Most runs now end with a STOP outside 10 mm, rather than a missing event. The gripper comes within 10 mm at any time on 11 DAgger runs and 10 combined runs, and STOP lands inside on 2 and 4. Those counts include the 0.5 s after STOP, and Section 3.26 separates them. For the clean model, parking accuracy says nothing about the reach (ρ = 0.07 [-0.12, 0.27]). For DAgger it starts to (ρ = 0.23 [0.04, 0.41]), and heading more so (ρ = 0.36 [0.15, 0.55]). Validation and test agree on the DAgger model, with closest 25 mm and 27 mm.

<a id="cell-69"></a>

## 3.20 Robustness: effects added in turn (Lab 3, Lecture 4.2)

Lab 3 asks about damping and delay. Each row adds one simulator effect to a frozen network, with no retraining. The lower rows apply all effects at once to each trained model.

<a id="cell-70"></a>

<details><summary>Code · original cell 70</summary>

```python
S = CL['summary']; steps = [('baseline', '41 clean'), ('encoder_noise', '+ noise'), ('damping', '+ damping'), ('elasticity', '+ elasticity'), ('inertia', '+ inertia'), ('delay', '+ 3 ms delay'), ('random_start_fixed', '+ random start')]; stack = [('standard_stack', '43 clean'), ('dart_stack', '52 fixed DART'), ('dagger_stack', '54 DAgger'), ('dart_dagger_stack', '56 both')]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.1)); x = np.arange(len(steps)); v = [S[k]['closest_mm'] for k, _ in steps]
ax[0].errorbar(x, [q[0] for q in v], yerr=[[q[0] - q[1] for q in v], [q[2] - q[0] for q in v]], color=BLUE, marker='o', capsize=3); ax[0].set_xticks(x); ax[0].set_xticklabels([l for _, l in steps], rotation=25, ha='right'); ax[0].set_ylim(0, None); ax[0].set_ylabel('median closest (mm)'); ax[0].set_title('Effects added in turn to one frozen network')
x = np.arange(4); w = .36; ax[1].bar(x - w / 2, [S[k]['parked'] for k, _ in stack], w, color=BLUE, label='PARK, of 100'); ax[1].bar(x + w / 2, [S[k]['stopped'] for k, _ in stack], w, color=AQUA, label='STOP, of 100'); ax[1].set_xticks(x); ax[1].set_xticklabels([l for _, l in stack]); ax[1].set_title('All effects at once'); ax[1].legend(fontsize=8); ax[1].grid(axis='x', visible=False); plt.tight_layout(); plt.show()
display(pd.DataFrame([dict(condition=l, iteration=S[k]['iteration'], PARK=S[k]['parked'], STOP=S[k]['stopped'], park_error_cm=S[k]['park_error_cm'][0], closest_mm=S[k]['closest_mm'][0], closest_low=S[k]['closest_mm'][1], closest_high=S[k]['closest_mm'][2], at_stop_mm=S[k]['at_stop_mm'][0]) for k, l in steps + stack]).round(1))
```

</details>

![Saved plot from cell 70](assets/1c391ae95ac813f72f2e75d7a1fdec6bb0bd0ef7e319db3812c1efdbc1fcc90f.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>condition</th>
      <th>iteration</th>
      <th>PARK</th>
      <th>STOP</th>
      <th>park_error_cm</th>
      <th>closest_mm</th>
      <th>closest_low</th>
      <th>closest_high</th>
      <th>at_stop_mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 clean</td>
      <td>41</td>
      <td>100</td>
      <td>84</td>
      <td>5.5</td>
      <td>184.5</td>
      <td>143.9</td>
      <td>216.2</td>
      <td>272.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>+ noise</td>
      <td>42</td>
      <td>99</td>
      <td>91</td>
      <td>6.2</td>
      <td>573.9</td>
      <td>493.8</td>
      <td>728.6</td>
      <td>755.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>+ damping</td>
      <td>44</td>
      <td>100</td>
      <td>88</td>
      <td>6.3</td>
      <td>556.2</td>
      <td>483.6</td>
      <td>657.8</td>
      <td>707.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>+ elasticity</td>
      <td>45</td>
      <td>99</td>
      <td>93</td>
      <td>6.0</td>
      <td>662.0</td>
      <td>582.4</td>
      <td>748.6</td>
      <td>790.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>+ inertia</td>
      <td>46</td>
      <td>100</td>
      <td>94</td>
      <td>7.2</td>
      <td>640.6</td>
      <td>546.4</td>
      <td>703.1</td>
      <td>779.3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>+ 3 ms delay</td>
      <td>47</td>
      <td>91</td>
      <td>91</td>
      <td>7.2</td>
      <td>764.1</td>
      <td>680.0</td>
      <td>817.6</td>
      <td>825.9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>+ random start</td>
      <td>48</td>
      <td>89</td>
      <td>89</td>
      <td>7.8</td>
      <td>860.1</td>
      <td>768.0</td>
      <td>919.0</td>
      <td>899.3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>43 clean</td>
      <td>51</td>
      <td>48</td>
      <td>19</td>
      <td>34.0</td>
      <td>386.7</td>
      <td>261.1</td>
      <td>635.8</td>
      <td>1143.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>52 fixed DART</td>
      <td>53</td>
      <td>65</td>
      <td>64</td>
      <td>19.1</td>
      <td>488.6</td>
      <td>364.8</td>
      <td>574.8</td>
      <td>655.6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>54 DAgger</td>
      <td>55</td>
      <td>64</td>
      <td>61</td>
      <td>19.6</td>
      <td>279.6</td>
      <td>225.5</td>
      <td>393.1</td>
      <td>495.6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>56 both</td>
      <td>57</td>
      <td>87</td>
      <td>86</td>
      <td>19.6</td>
      <td>240.9</td>
      <td>211.3</td>
      <td>262.9</td>
      <td>321.9</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-71"></a>

**Reading.** Noise gives the largest first increase in this ordering: closest distance goes from 184 to 574 mm. That does not show motor effects matter less than sensor noise. The order is fixed, and later effects may interact with, or be hidden by, the damage already there, so each step only measures its increment after those before it. The roadmap expected no detectable change from encoder noise, and it was wrong. With all effects at once, the combined model parks on 87 and stops on 86, against 48 and 19 for the clean one, at 322 mm against 1143 mm. Those are single-seed descriptive results. They do not make the recovery models robust in general: no run in the stack is reliably precise.

<a id="cell-72"></a>

## 3.21 Clean and disturbed together, every run counted

The evidence on the four main models in one table: completion, accuracy and robustness. Medians marked "all 100" count a run with no event as a failure ranked last, so they keep the denominator at 100. If more than half the runs have no event, that median does not exist and shows as "none". "Clean" means no added disturbance. "Disturbed" is encoder noise, damping, elasticity, inertia, 3 ms delay and random starts together.

<a id="cell-73"></a>

<details><summary>Code · original cell 73</summary>

```python
M4 = [('basics_standard', 'standard_stack', '43 clean model'), ('dart', 'dart_stack', '52 fixed DART'), ('dagger', 'dagger_stack', '54 DAgger'), ('dart_dagger', 'dart_dagger_stack', '56 both')]
def med_all(v, n=100):
    v = np.sort(np.asarray(v, float)); v = v[np.isfinite(v)]
    if len(v) < n / 2: return None
    full = np.r_[v, np.full(n - len(v), np.inf)]; return float(np.median(full))
fmt = lambda x: 'none' if x is None or not np.isfinite(x) else f'{x:.0f}'
def row(k, cond, lab):
    t = per_target(k); s = t.at_stop_mm.where(t.stopped); return {'model': lab, 'condition': cond, 'PARK /100': int(t.parked.sum()), 'STOP /100': int(t.stopped.sum()), 'success /100': int(t.success.sum()), 'closest mm, all 100': fmt(med_all(t.closest_mm.where(t.parked))), 'at STOP mm, all 100': fmt(med_all(s)), 'at STOP mm, runs with STOP': fmt(np.nanmedian(s))}
T = pd.DataFrame([row(a, 'clean', l) for a, _, l in M4] + [row(b, 'disturbed', l) for _, b, l in M4]); display(T.set_index(['condition', 'model']))
fig, ax = plt.subplots(1, 3, figsize=(13, 3)); x = np.arange(4); w = .36; D = T[T.condition == 'clean']; S_ = T[T.condition == 'disturbed']; names = [l for _, _, l in M4]
for a, col, title in ((ax[0], 'PARK /100', 'PARK, of 100'), (ax[1], 'STOP /100', 'STOP, of 100'), (ax[2], 'success /100', 'Success, of 100')):
    a.bar(x - w / 2, D[col], w, color=BLUE, label='clean'); a.bar(x + w / 2, S_[col], w, color=ORANGE, label='disturbed'); a.set_xticks(x); a.set_xticklabels(names, rotation=15, ha='right'); a.set_title(title); a.grid(axis='x', visible=False); a.set_ylim(0, 105 if col != 'success /100' else 10)
ax[2].legend(fontsize=8)
plt.tight_layout(); plt.show()
```

</details>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>PARK /100</th>
      <th>STOP /100</th>
      <th>success /100</th>
      <th>closest mm, all 100</th>
      <th>at STOP mm, all 100</th>
      <th>at STOP mm, runs with STOP</th>
    </tr>
    <tr>
      <th>condition</th>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">clean</th>
      <th>43 clean model</th>
      <td>100</td>
      <td>91</td>
      <td>0</td>
      <td>75</td>
      <td>181</td>
      <td>169</td>
    </tr>
    <tr>
      <th>52 fixed DART</th>
      <td>100</td>
      <td>78</td>
      <td>1</td>
      <td>45</td>
      <td>180</td>
      <td>61</td>
    </tr>
    <tr>
      <th>54 DAgger</th>
      <td>84</td>
      <td>69</td>
      <td>2</td>
      <td>31</td>
      <td>52</td>
      <td>37</td>
    </tr>
    <tr>
      <th>56 both</th>
      <td>87</td>
      <td>66</td>
      <td>4</td>
      <td>34</td>
      <td>53</td>
      <td>34</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">disturbed</th>
      <th>43 clean model</th>
      <td>48</td>
      <td>19</td>
      <td>0</td>
      <td>none</td>
      <td>none</td>
      <td>1143</td>
    </tr>
    <tr>
      <th>52 fixed DART</th>
      <td>65</td>
      <td>64</td>
      <td>0</td>
      <td>731</td>
      <td>1015</td>
      <td>656</td>
    </tr>
    <tr>
      <th>54 DAgger</th>
      <td>64</td>
      <td>61</td>
      <td>0</td>
      <td>596</td>
      <td>938</td>
      <td>496</td>
    </tr>
    <tr>
      <th>56 both</th>
      <td>87</td>
      <td>86</td>
      <td>0</td>
      <td>257</td>
      <td>364</td>
      <td>322</td>
    </tr>
  </tbody>
</table>
</div>

![Saved plot from cell 73](assets/d0b41057d198ccd7e30587f2eecfda57fe0b3327393ef8fa56c63712e3751ebd.png)


<a id="cell-74"></a>

**Reading.** Better reaching comes with worse completion. From the clean model to DAgger, median distance at STOP falls from 169 to 37 mm, while PARK falls from 100 to 84 and STOP from 91 to 69. Keep the 100-run columns first. The conditional medians exclude failures, so read them second.

<a id="cell-75"></a>

## 3.22 Paired changes with intervals

The same comparisons as Section 3.18, now with the size of the change. Each is a median of paired differences on targets where both models have the event, with a 95% bootstrap interval over targets. Negative means closer to the target. This covers target uncertainty for these two trained models and says nothing about training seeds (Section 3.24). McNemar counts below use all 100 targets, so they show what the distance comparisons leave out.

<a id="cell-76"></a>

<details><summary>Code · original cell 76</summary>

```python
def paired(a, b, col, B=5000):
    rng = np.random.default_rng(7); d = (per_target(b)[col] - per_target(a)[col]).dropna().values; s = np.median(d[rng.integers(0, len(d), (B, len(d)))], 1); return dict(pairs=len(d), median_change=np.median(d), low=np.percentile(s, 2.5), high=np.percentile(s, 97.5), better=int((d < 0).sum()), worse=int((d > 0).sum()))
PL = [('37 to 38: remove speeds', 'height', 'no_speed', 'park_error_cm'), ('37 to 38: remove speeds', 'height', 'no_speed', 'closest_mm'), ('40 to 41: learn hold', 'wider_arm', 'baseline', 'closest_mm'), ('41 to 43: standardise', 'baseline', 'basics_standard', 'closest_mm'), ('43 to 52: fixed DART', 'basics_standard', 'dart', 'closest_mm'), ('43 to 52: fixed DART', 'basics_standard', 'dart', 'at_stop_mm'), ('43 to 54: DAgger', 'basics_standard', 'dagger', 'closest_mm'), ('43 to 54: DAgger', 'basics_standard', 'dagger', 'at_stop_mm'), ('54 to 56: add DART', 'dagger', 'dart_dagger', 'closest_mm'), ('54 to 56: add DART', 'dagger', 'dart_dagger', 'at_stop_mm')]
P_ = pd.DataFrame([dict(comparison=l, measure=m_, **paired(a, b, m_)) for l, a, b, m_ in PL]); display(P_.round(1))
fig, ax = plt.subplots(figsize=(9, 3.2)); Q = P_[P_.measure != 'park_error_cm'].reset_index(drop=True); y = np.arange(len(Q))[::-1]; ax.errorbar(Q.median_change, y, xerr=[Q.median_change - Q.low, Q.high - Q.median_change], fmt='o', color=BLUE, capsize=3); ax.axvline(0, color=INK, lw=.8); ax.set_yticks(y); ax.set_yticklabels([f"{a}, {b.replace('_mm', '').replace('_', ' ')}" for a, b in zip(Q.comparison, Q.measure)]); ax.set_xlabel('median paired change (mm): left of 0 is closer to the target'); ax.set_title('Paired change, 95% bootstrap over targets'); ax.grid(axis='y', visible=False); plt.tight_layout(); plt.show()
def mcn(a, b, col):
    A, B = per_target(a)[col].astype(bool), per_target(b)[col].astype(bool); oa, ob = int((A & ~B).sum()), int((~A & B).sum()); n = oa + ob; return dict(only_first=oa, only_second=ob, p=min(1., 2 * sum(math.comb(n, i) for i in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.)
display(pd.DataFrame([dict(comparison=l, event=e, **mcn(a, b, e)) for l, a, b in (('43 to 54: DAgger', 'basics_standard', 'dagger'), ('43 to 52: fixed DART', 'basics_standard', 'dart'), ('54 to 56: add DART', 'dagger', 'dart_dagger')) for e in ('parked', 'stopped', 'success')]).round(4))
```

</details>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comparison</th>
      <th>measure</th>
      <th>pairs</th>
      <th>median_change</th>
      <th>low</th>
      <th>high</th>
      <th>better</th>
      <th>worse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37 to 38: remove speeds</td>
      <td>park_error_cm</td>
      <td>87</td>
      <td>-14.2</td>
      <td>-18.8</td>
      <td>-9.1</td>
      <td>76</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37 to 38: remove speeds</td>
      <td>closest_mm</td>
      <td>87</td>
      <td>-221.2</td>
      <td>-274.1</td>
      <td>-195.8</td>
      <td>82</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>40 to 41: learn hold</td>
      <td>closest_mm</td>
      <td>100</td>
      <td>103.2</td>
      <td>67.4</td>
      <td>131.1</td>
      <td>14</td>
      <td>86</td>
    </tr>
    <tr>
      <th>3</th>
      <td>41 to 43: standardise</td>
      <td>closest_mm</td>
      <td>100</td>
      <td>-100.0</td>
      <td>-129.4</td>
      <td>-64.8</td>
      <td>87</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43 to 52: fixed DART</td>
      <td>closest_mm</td>
      <td>100</td>
      <td>-27.8</td>
      <td>-32.9</td>
      <td>-20.3</td>
      <td>76</td>
      <td>24</td>
    </tr>
    <tr>
      <th>5</th>
      <td>43 to 52: fixed DART</td>
      <td>at_stop_mm</td>
      <td>71</td>
      <td>-20.7</td>
      <td>-72.4</td>
      <td>14.2</td>
      <td>42</td>
      <td>29</td>
    </tr>
    <tr>
      <th>6</th>
      <td>43 to 54: DAgger</td>
      <td>closest_mm</td>
      <td>84</td>
      <td>-48.1</td>
      <td>-61.4</td>
      <td>-36.6</td>
      <td>75</td>
      <td>9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>43 to 54: DAgger</td>
      <td>at_stop_mm</td>
      <td>66</td>
      <td>-129.9</td>
      <td>-175.5</td>
      <td>-97.6</td>
      <td>65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>54 to 56: add DART</td>
      <td>closest_mm</td>
      <td>81</td>
      <td>0.0</td>
      <td>-5.6</td>
      <td>6.7</td>
      <td>40</td>
      <td>41</td>
    </tr>
    <tr>
      <th>9</th>
      <td>54 to 56: add DART</td>
      <td>at_stop_mm</td>
      <td>57</td>
      <td>-5.2</td>
      <td>-10.9</td>
      <td>-2.3</td>
      <td>37</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>

![Saved plot from cell 76](assets/f482102dac6e03d5fc9c4921bb5e4645c9a9d7563b2e714a0b2345a234a2d25b.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>comparison</th>
      <th>event</th>
      <th>only_first</th>
      <th>only_second</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43 to 54: DAgger</td>
      <td>parked</td>
      <td>16</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43 to 54: DAgger</td>
      <td>stopped</td>
      <td>25</td>
      <td>3</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>43 to 54: DAgger</td>
      <td>success</td>
      <td>0</td>
      <td>2</td>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43 to 52: fixed DART</td>
      <td>parked</td>
      <td>0</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43 to 52: fixed DART</td>
      <td>stopped</td>
      <td>20</td>
      <td>7</td>
      <td>0.0192</td>
    </tr>
    <tr>
      <th>5</th>
      <td>43 to 52: fixed DART</td>
      <td>success</td>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>54 to 56: add DART</td>
      <td>parked</td>
      <td>3</td>
      <td>6</td>
      <td>0.5078</td>
    </tr>
    <tr>
      <th>7</th>
      <td>54 to 56: add DART</td>
      <td>stopped</td>
      <td>12</td>
      <td>9</td>
      <td>0.6636</td>
    </tr>
    <tr>
      <th>8</th>
      <td>54 to 56: add DART</td>
      <td>success</td>
      <td>2</td>
      <td>4</td>
      <td>0.6875</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-77"></a>

## 3.23 Training curves of the models tested on the robot

Train (solid) and validation (dashed) loss per epoch for each model's three networks, with the selected checkpoint marked. This complements Section 3.6, which compares settings on separate runs. The DAgger curves are the final round's.

<a id="cell-78"></a>

<details><summary>Code · original cell 78</summary>

```python
fig, ax = plt.subplots(1, 3, figsize=(14, 3.3))
for a, net, title, tr, va in ((ax[0], 'driving', 'Driving: wheel command loss', 'train_command', 'val_command'), (ax[1], 'torque', 'Arm: torque loss', 'train_command', 'val_command'), (ax[2], 'stop', 'STOP network: distance loss', None, None)):
    for m in MODELS:
        f = DATA / f'histories/{m}/{net}_seed7_history.json'
        if not f.exists(): continue
        h = pd.DataFrame(json.load(open(f))); kt = tr or next(k for k in h if k.startswith('train') and 'total' not in k and k != 'train_aux'); kv = va or next(k for k in h if k.startswith('val') and 'total' not in k and k != 'val_aux')
        a.plot(h.epoch, h[kt], color=COL[m], lw=1.2, label=SHORT[m]); a.plot(h.epoch, h[kv], color=COL[m], lw=1.2, ls='--'); e_ = json.load(open(DATA / f'histories/{m}/{net}_seed7_summary.json')).get('selected_epoch')
        if e_ is not None and e_ in set(h.epoch): a.scatter([e_], [h.set_index('epoch').loc[e_, kv]], color=COL[m], s=24, zorder=4, edgecolor='white')
    a.set_yscale('log'); a.set_xlabel('epoch'); a.set_title(title)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()
display(pd.DataFrame([dict(model=SHORT[m], network=net, selected_epoch=json.load(open(DATA / f'histories/{m}/{net}_seed7_summary.json')).get('selected_epoch'), selected_val_loss=json.load(open(DATA / f'histories/{m}/{net}_seed7_summary.json')).get('selected_val_loss')) for m in MODELS for net in ('driving', 'torque', 'stop')]).round(4))
```

</details>

![Saved plot from cell 78](assets/21d162fff1b9cfdacf4095f40a5d64b301c7468fcfb8a945f572bb2d513d7cf4.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>network</th>
      <th>selected_epoch</th>
      <th>selected_val_loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>driving</td>
      <td>299</td>
      <td>0.3842</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41 learned arm</td>
      <td>torque</td>
      <td>1198</td>
      <td>0.0217</td>
    </tr>
    <tr>
      <th>2</th>
      <td>41 learned arm</td>
      <td>stop</td>
      <td>593</td>
      <td>0.0030</td>
    </tr>
    <tr>
      <th>3</th>
      <td>43 clean</td>
      <td>driving</td>
      <td>299</td>
      <td>0.3842</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43 clean</td>
      <td>torque</td>
      <td>1200</td>
      <td>0.0121</td>
    </tr>
    <tr>
      <th>5</th>
      <td>43 clean</td>
      <td>stop</td>
      <td>600</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <th>6</th>
      <td>52 fixed DART</td>
      <td>driving</td>
      <td>300</td>
      <td>0.3879</td>
    </tr>
    <tr>
      <th>7</th>
      <td>52 fixed DART</td>
      <td>torque</td>
      <td>1168</td>
      <td>0.0121</td>
    </tr>
    <tr>
      <th>8</th>
      <td>52 fixed DART</td>
      <td>stop</td>
      <td>582</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <th>9</th>
      <td>54 DAgger</td>
      <td>driving</td>
      <td>292</td>
      <td>0.6960</td>
    </tr>
    <tr>
      <th>10</th>
      <td>54 DAgger</td>
      <td>torque</td>
      <td>1186</td>
      <td>0.0419</td>
    </tr>
    <tr>
      <th>11</th>
      <td>54 DAgger</td>
      <td>stop</td>
      <td>600</td>
      <td>0.0009</td>
    </tr>
    <tr>
      <th>12</th>
      <td>56 both</td>
      <td>driving</td>
      <td>300</td>
      <td>0.4867</td>
    </tr>
    <tr>
      <th>13</th>
      <td>56 both</td>
      <td>torque</td>
      <td>1187</td>
      <td>0.0325</td>
    </tr>
    <tr>
      <th>14</th>
      <td>56 both</td>
      <td>stop</td>
      <td>582</td>
      <td>0.0009</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-79"></a>

**Reading.** Each model's loss is in its own standardised units, so compare shapes across panels and not heights between recovery and clean models. Training and validation stay close on clean data and losses are still falling slowly when training ends. Recovery rows raise training loss above validation because those rows are harder.

<a id="cell-80"></a>

## 3.24 Target uncertainty is separate from seed variation

Bootstrapping 100 targets measures uncertainty for one trained model. It does not cover what changes if the same setup is trained again. Iterations 34 to 36 differ only by seed. Their intervals are the thin marks, and the gap between seeds is often larger than any one interval. From iteration 37 onward every result is seed 7, so no later comparison can be checked against seed spread. The roadmap's own rule was that a difference smaller than the seed spread is not a finding.

<a id="cell-81"></a>

<details><summary>Code · original cell 81</summary>

```python
S = CL['summary']; K = [('seed7', '34 seed 7'), ('seed1', '35 seed 1'), ('seed2', '36 seed 2')]; fig, ax = plt.subplots(1, 3, figsize=(12, 2.8)); x = np.arange(3)
for a, f_, ci, title in ((ax[0], 'parked', 'parked_ci', 'PARK, of 100 (Wilson 95%)'), (ax[1], 'stopped', 'stopped_ci', 'STOP, of 100 (Wilson 95%)')):
    v = [S[k][f_] for k, _ in K]; lo = [S[k][ci][0] for k, _ in K]; hi = [S[k][ci][1] for k, _ in K]; a.errorbar(x, v, yerr=[np.array(v) - lo, np.array(hi) - v], fmt='o', color=BLUE, capsize=4); a.set_ylim(0, 105); a.set_title(title)
v = [S[k]['park_error_cm'] for k, _ in K]; ax[2].errorbar(x, [q[0] for q in v], yerr=[[q[0] - q[1] for q in v], [q[2] - q[0] for q in v]], fmt='o', color=BLUE, capsize=4); ax[2].set_ylim(0, None); ax[2].set_title('Median park error, cm (bootstrap 95%)')
for a in ax: a.set_xticks(x); a.set_xticklabels([l for _, l in K]); a.grid(axis='x', visible=False)
plt.tight_layout(); plt.show()
```

</details>

![Saved plot from cell 81](assets/7b89bacd51635188906d220e9edcafa016bd6bd306ab6f67d76666d65e64ef41.png)


<a id="cell-82"></a>

## 3.25 Model comparison ledger

What differs in each contrast, so unlike experiments are not read as interchangeable. All tests use the same 100 test targets and one machine.

| Contrast | What changed | Training data | Network | Arm before PARK | Seed | Test condition |
|---|---|---|---|---|---|---|
| 34, 35, 36 | Seed only | 100 expert runs | Driving 16 inputs (with speeds), arm 17 inputs, 64 units | Supplied PD and gravity controller | 7, 1, 2 | Clean |
| 34 to 37 | Target height added to driving | Same | Driving 17 inputs | Same | 7 | Clean |
| 37 to 38 | Six wheel speed inputs removed | Same | Driving 11 inputs | Same | 7 | Clean |
| 38 to 39 | 1,200 torque epochs | Same | Arm 64 units | Same | 7 | Clean |
| 39 to 40 | Wider arm | Same | Arm 128 units | Same | 7 | Clean |
| 40 to 41 | Network holds the arm from reset | Arm rows from reset | Arm 18 inputs (PARK flag added), physical scales | Learned | 7 | Clean |
| 41 to 43 | Arm inputs standardised | Same as 41 | Same shape | Learned | 7 | Clean |
| 41 to 42, 44 to 48 | One effect added at each step | No retraining | Frozen 41 | Learned | 7 | Cumulative disturbances |
| 43 to 52 | Fixed noise recovery rows | + 199 noisy expert runs | Same shape, retrained | Learned | 7 | Clean |
| 43 to 54 | DAgger rows (commands only) | + 3 rounds of 100 network runs | Same shape, retrained | Learned | 7 | Clean |
| 52 to 56 | DAgger on top of fixed noise | + both | Same shape, retrained | Learned | 7 | Clean |
| 51, 53, 55, 57 | Same four models | No retraining | Frozen 43, 52, 54, 56 | Learned | 7 | All disturbances, random starts |

Note that the ladder uses iteration 41 and the all-effects rows use iteration 43 and later, so the ladder cannot say how standardisation changes noise sensitivity.

<a id="cell-83"></a>

## 3.26 On the way to the target: arrive, pass through, or stay?

Distance alone mixes three failures. This uses the saved 30 Hz records of each test run, from PARK to STOP. No run is excluded: every one of the 100 runs is in exactly one category.

- **No PARK**: the arm phase never started.
- **Never within 10 mm**: failure to arrive, the gripper was never inside the tolerance.
- **Within 10 mm only after STOP**: STOP fired early, and the arm carried on into the tolerance during the following 0.5 s.
- **Passed through**: inside for under 0.1 s before STOP.
- **Stayed**: inside for 0.1 s or more before STOP, with STOP firing elsewhere or not at all.
- **Success**: STOP fired inside 10 mm.

"Perfect stop" is what a perfect stopping rule could achieve on these same paths, stopping at the best instant before the network's own STOP. Records at 30 Hz can miss a pass shorter than 33 ms. The largest gap between the 30 Hz and 1 kHz closest distance is a few mm.

<a id="cell-84"></a>

<details><summary>Code · original cell 84</summary>

```python
AP = J('approach.json'); L = ['baseline', 'basics_standard', 'dart', 'dagger', 'dart_dagger']; parts = [('no PARK', MUTED), ('never within 10 mm', '#c9c8c3'), ('within 10 mm only after STOP', ORANGE), ('passed through, under 0.1 s inside', GREY), ('stayed 0.1 s or more, no STOP inside', BLUE), ('success', AQUA)]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.5)); y = np.arange(5)[::-1]; left = np.zeros(5)
for k, col in parts:
    v = np.array([AP[m]['categories'].get(k, 0) for m in L], float); ax[0].barh(y, v, left=left, color=col, label=k, height=.62, edgecolor='white', lw=1); left += v
ax[0].set_yticks(y); ax[0].set_yticklabels([SHORT[m] for m in L]); ax[0].set_xlim(0, 100); ax[0].set_title('Where each of the 100 runs ended up'); ax[0].legend(fontsize=7.5, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.1)); ax[0].grid(axis='y', visible=False)
for name, col, lab in (('expert', INK, 'expert'), ('basics_standard', BLUE, '43 clean'), ('dagger', ORANGE, '54 DAgger'), ('dart_dagger', AQUA, '56 both')):
    b = [q for q in AP[name]['bins'] if q['closing_mm_s'] and q['runs'] >= 3]; xs = [np.sqrt(q['lo'] * q['hi']) for q in b]; ax[1].plot(xs, [q['closing_mm_s'][1] for q in b], color=col, marker='o', ms=4, label=lab)
ax[1].axvline(10, color=MUTED, ls='--', lw=1); ax[1].axhline(0, color=INK, lw=.7); ax[1].set_xscale('log'); ax[1].set_xlabel('gripper distance (mm, log)'); ax[1].set_ylabel('closing speed (mm/s)'); ax[1].set_title('Closing speed against distance, median of runs'); ax[1].legend(fontsize=8)
w = .36; x = np.arange(4); M_ = L[1:]; ax[2].bar(x - w / 2, [AP[m]['closest_before_stop'] for m in M_], w, color=BLUE, label='closest point before STOP'); ax[2].bar(x + w / 2, [AP[m]['closest_after_stop'] for m in M_], w, color=ORANGE, label='closest point after STOP'); ax[2].set_xticks(x); ax[2].set_xticklabels([SHORT[m] for m in M_]); ax[2].set_ylabel('runs that issued STOP'); ax[2].set_title('When the closest point falls'); ax[2].legend(fontsize=8); ax[2].grid(axis='x', visible=False)
plt.tight_layout(); plt.show()
display(pd.DataFrame([dict(model=SHORT[m], no_PARK=AP[m]['categories'].get('no PARK', 0), never_within_10mm=AP[m]['categories'].get('never within 10 mm', 0), only_after_STOP=AP[m]['categories'].get('within 10 mm only after STOP', 0), passed_through=AP[m]['categories'].get('passed through, under 0.1 s inside', 0), stayed=AP[m]['categories'].get('stayed 0.1 s or more, no STOP inside', 0), success=AP[m]['categories'].get('success', 0), perfect_stop=AP[m]['oracle_stop_successes'], within_25mm=AP[m]['within_25_before_stop'], within_50mm=AP[m]['within_50_before_stop'], STOP_minus_best_mm=AP[m]['stop_gap_over_best_mm'][1], time_inside_s=(AP[m]['time_inside_when_any_s'] or [0, np.nan])[1], closing_at_25mm=(AP[m]['closing_at_25mm_mm_s'] or [0, np.nan])[1]) for m in L]).round(2))
```

</details>

![Saved plot from cell 84](assets/26706d0862a21ec1940af70717aa1c26e71c27d6b9149b96e4d0591393f672e2.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>no_PARK</th>
      <th>never_within_10mm</th>
      <th>only_after_STOP</th>
      <th>passed_through</th>
      <th>stayed</th>
      <th>success</th>
      <th>perfect_stop</th>
      <th>within_25mm</th>
      <th>within_50mm</th>
      <th>STOP_minus_best_mm</th>
      <th>time_inside_s</th>
      <th>closing_at_25mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>18.60</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43 clean</td>
      <td>0</td>
      <td>100</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>22</td>
      <td>71.86</td>
      <td>NaN</td>
      <td>51.69</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52 fixed DART</td>
      <td>0</td>
      <td>95</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>13</td>
      <td>54</td>
      <td>5.77</td>
      <td>0.46</td>
      <td>22.95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54 DAgger</td>
      <td>16</td>
      <td>73</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>7</td>
      <td>31</td>
      <td>64</td>
      <td>0.65</td>
      <td>0.20</td>
      <td>53.76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56 both</td>
      <td>13</td>
      <td>77</td>
      <td>4</td>
      <td>0</td>
      <td>2</td>
      <td>4</td>
      <td>6</td>
      <td>34</td>
      <td>65</td>
      <td>1.33</td>
      <td>0.23</td>
      <td>34.52</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-85"></a>

The next table takes only the runs that never came within 10 mm and asks what the gripper was doing when the run's approach ended, in the last 0.27 s before STOP or the deadline. Still closing means at over 10 mm/s towards the target, moving away means over 10 mm/s the other way, and nearly still is anything between.

<a id="cell-86"></a>

<details><summary>Code · original cell 86</summary>

```python
def final_state(m):
    x = AP[m]; rows = [r for r in x['per_run'] if r['category'] != 'no PARK']; out = []
    for r, (d, v) in zip(rows, x['traces']):
        if r['category'] != 'never within 10 mm': continue
        v = np.array([np.nan if q is None else q for q in v], float); out.append(dict(speed=np.nanmedian(v[-min(len(v), 4):]), dist=d[-1], stop=r['stop_issued']))
    t = pd.DataFrame(out); return dict(model=SHORT[m], never_within_10mm=len(t), of_which_issued_STOP=int(t.stop.sum()), still_closing=int((t.speed >= 10).sum()), nearly_still=int((t.speed.abs() < 10).sum()), moving_away=int((t.speed <= -10).sum()), median_distance_mm=t.dist.median(), median_closing_mm_s=t.speed.median())
display(pd.DataFrame([final_state(m) for m in L]).round(1))
```

</details>

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>never_within_10mm</th>
      <th>of_which_issued_STOP</th>
      <th>still_closing</th>
      <th>nearly_still</th>
      <th>moving_away</th>
      <th>median_distance_mm</th>
      <th>median_closing_mm_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>100</td>
      <td>84</td>
      <td>39</td>
      <td>8</td>
      <td>53</td>
      <td>345.0</td>
      <td>-28.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43 clean</td>
      <td>100</td>
      <td>91</td>
      <td>14</td>
      <td>8</td>
      <td>78</td>
      <td>159.0</td>
      <td>-102.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52 fixed DART</td>
      <td>95</td>
      <td>73</td>
      <td>18</td>
      <td>28</td>
      <td>49</td>
      <td>62.1</td>
      <td>-12.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54 DAgger</td>
      <td>73</td>
      <td>58</td>
      <td>43</td>
      <td>14</td>
      <td>16</td>
      <td>49.0</td>
      <td>20.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56 both</td>
      <td>77</td>
      <td>56</td>
      <td>32</td>
      <td>32</td>
      <td>13</td>
      <td>43.9</td>
      <td>3.8</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-87"></a>

**Reading.** Most runs never reach the tolerance. 73 DAgger runs and 77 combined runs are never within 10 mm, and 16 and 13 more never park. Stopping at the best instant before the network's own STOP would give 7 and 6 successes, against the actual 2 and 4.

That count is a lower bound for a better STOP, because STOP also ends the path. Of the 73 DAgger runs that never arrive, 43 were still closing at over 10 mm/s when STOP or the deadline came, 14 were nearly still and 16 were moving away, with a median distance of 49 mm. For the combined model the counts are 32, 32 and 13. For the clean model 78 of 100 were moving away, which is a different failure. So "failure to arrive" mixes an early STOP, a stalled reach and a reach that turns away, and the saved runs cannot say how many of the still-closing runs would have arrived, since each run ends 0.5 s after STOP. Running the DAgger model with STOP ignored would settle it, and needs new simulation.

Nothing passes straight through: every pre-STOP visit inside 10 mm lasts at least 0.1 s, a median 0.20 s for DAgger. Across runs that issue STOP the closest point falls about evenly either side of it (37 before, 32 after, for DAgger), so STOP is neither consistently early nor consistently late. Closing speed at each distance is close to the expert's for both recovery models, so at a given distance they approach about as fast as the expert does.

<a id="cell-88"></a>

## 3.27 After STOP

The run continues for 0.5 s after STOP with the networks still in control. How far does the gripper move? This uses the saved distance at STOP and at the end of the run. Every point is a run that issued STOP.

<a id="cell-89"></a>

<details><summary>Code · original cell 89</summary>

```python
fig, ax = plt.subplots(1, 2, figsize=(11, 3.2)); L = ['baseline', 'basics_standard', 'dart', 'dagger', 'dart_dagger']; rows = []
for m in L:
    t = per_target(m); t = t[t.stopped]; d = t.end_mm - t.at_stop_mm; rows.append(dict(model=SHORT[m], runs=len(t), at_STOP_mm=t.at_stop_mm.median(), after_half_second_mm=t.end_mm.median(), median_change_mm=d.median(), p25=d.quantile(.25), p75=d.quantile(.75), moved_away_over_10mm=int((d > 10).sum()), moved_closer_over_10mm=int((d < -10).sum())))
    ax[0].scatter(t.at_stop_mm, t.end_mm, s=14, color=COL[m], alpha=.7, label=SHORT[m])
lim = [2, 2000]; ax[0].plot(lim, lim, color=INK, lw=.7); ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel('distance at STOP (mm)'); ax[0].set_ylabel('distance 0.5 s later (mm)'); ax[0].set_title('On the line means it held still'); ax[0].legend(fontsize=8)
R_ = pd.DataFrame(rows); y = np.arange(5)[::-1]; ax[1].barh(y, R_.median_change_mm, color=[COL[m] for m in L], height=.6, xerr=[R_.median_change_mm - R_.p25, R_.p75 - R_.median_change_mm], ecolor=INK, capsize=3); ax[1].axvline(0, color=INK, lw=.8); ax[1].set_yticks(y); ax[1].set_yticklabels(R_.model); ax[1].set_xscale('symlog', linthresh=10); ax[1].set_xlabel('change in distance over 0.5 s (mm), middle half'); ax[1].set_title('Movement after STOP'); ax[1].grid(axis='y', visible=False); plt.tight_layout(); plt.show(); display(R_.round(1))
```

</details>

![Saved plot from cell 89](assets/631f19ccbe3848c23edf68e1566eb767ca11d065f5a395166e7830480c7828c3.png)

<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>runs</th>
      <th>at_STOP_mm</th>
      <th>after_half_second_mm</th>
      <th>median_change_mm</th>
      <th>p25</th>
      <th>p75</th>
      <th>moved_away_over_10mm</th>
      <th>moved_closer_over_10mm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>41 learned arm</td>
      <td>84</td>
      <td>272.7</td>
      <td>340.4</td>
      <td>21.0</td>
      <td>-25.8</td>
      <td>90.5</td>
      <td>47</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43 clean</td>
      <td>91</td>
      <td>168.8</td>
      <td>306.2</td>
      <td>119.0</td>
      <td>31.8</td>
      <td>236.5</td>
      <td>74</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>52 fixed DART</td>
      <td>78</td>
      <td>60.9</td>
      <td>63.5</td>
      <td>0.8</td>
      <td>-6.2</td>
      <td>10.7</td>
      <td>20</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54 DAgger</td>
      <td>69</td>
      <td>37.0</td>
      <td>35.6</td>
      <td>-0.5</td>
      <td>-7.0</td>
      <td>8.0</td>
      <td>16</td>
      <td>12</td>
    </tr>
    <tr>
      <th>4</th>
      <td>56 both</td>
      <td>66</td>
      <td>34.5</td>
      <td>36.0</td>
      <td>0.3</td>
      <td>-5.7</td>
      <td>7.7</td>
      <td>12</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>


<a id="cell-90"></a>

**Reading.** The clean models keep moving away after STOP (median change 103.1 mm for 43), which is the arm failing to hold. The recovery models hold close to where they stopped (median -0.6 mm for DAgger, middle half -6.3 to 7.0 mm). This is one of the clearest gains from recovery data, and one that distance at STOP does not show.
