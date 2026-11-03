# Driving v2 · Rung 1: does the network work when it can see the dot?

**Question:** if the network is told where the dot is *relative to the robot*, can it copy the expert's wheel commands?

This is an **oracle** test. The dot's position comes from the simulator, not from a sensor. Later, a camera replaces it. If this rung fails, a camera will not help.

**One change at a time.** Three models, same network, same data, same training. Only the inputs differ:

| Model | Inputs |
|---|---|
| `wheels` | 12 wheel features (angles, speeds, summaries) |
| `wheels+start_dot` | + sin/cos of the dot's direction **at the start** (the old setup) |
| `wheels+live_dot` | + distance and sin/cos of the dot's direction **relative to the robot, every step** |

**Outputs (the expert's):** four wheel-velocity commands + PARK.
Arm torques are not in this dataset (it was collected driving-only), so they come later.


```python
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

DATA = Path('../driving_30/data/expert_30hz')
records = [json.loads(line) for line in open(DATA / 'episodes.jsonl')]
split_of = {p['bearing_deg']: p['split'] for p in json.loads((DATA / 'split_plan.json').read_text())}

# Repeated runs are byte-identical: keep one trajectory per unique hash.
unique = {}
for r in records:
    unique.setdefault(r['trajectory_hash'], r)
episodes = sorted(unique.values(), key=lambda r: r['bearing_deg'])
print(f'{len(records)} executions -> {len(episodes)} unique trajectories')
print(pd.Series([split_of[r["bearing_deg"]] for r in episodes]).value_counts().to_dict())
```

    1500 executions -> 30 unique trajectories
    {'train': 21, 'holdout': 5, 'test': 4}


### Build the inputs

`base_pose` is the simulator's true robot pose: x, y, z, then a quaternion (x, y, z, w). It is used **only** to compute the oracle dot features. It is not given to the network.

Live dot features at each step:
- `dot_distance_m`: horizontal distance from the robot to the dot
- `dot_sin`, `dot_cos`: direction of the dot, measured from where the robot is facing


```python
WHEEL = list(range(12))      # first 12 columns of the saved x: wheel features
START_DOT = [12, 13]         # saved sin/cos of the initial bearing

def yaw(q):                  # quaternion (x, y, z, w) -> heading in radians
    x, y, z, w = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

def live_dot(pose, target):
    dx, dy = target[0] - pose[:, 0], target[1] - pose[:, 1]
    angle = np.arctan2(dy, dx) - yaw(pose[:, 3:7])
    return np.c_[np.hypot(dx, dy), np.sin(angle), np.cos(angle)].astype(np.float32)

rows = []
for r in episodes:
    with np.load(DATA / r['archive']) as a:
        target = np.array(r['manifest']['target_position_world_m'])
        rows.append(dict(bearing=r['bearing_deg'], split=split_of[r['bearing_deg']],
                         x=a['x'], live=live_dot(a['base_pose'], target),
                         wheels=a['wheels'], park=a['park'].ravel()))

INPUTS = {
    'wheels':           lambda e: e['x'][:, WHEEL],
    'wheels+start_dot': lambda e: e['x'][:, WHEEL + START_DOT],
    'wheels+live_dot':  lambda e: np.c_[e['x'][:, WHEEL], e['live']],
}
e = rows[0]
print('Bearing', e['bearing'], '| rows', len(e['x']))
pd.DataFrame(e['live'], columns=['dot_distance_m', 'dot_sin', 'dot_cos']).iloc[[0, 10, 30, -1]].round(3)
```

    Bearing 12 | rows 96





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dot_distance_m</th>
      <th>dot_sin</th>
      <th>dot_cos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.100</td>
      <td>0.208</td>
      <td>0.978</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.102</td>
      <td>0.083</td>
      <td>0.997</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1.003</td>
      <td>0.014</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>95</th>
      <td>0.707</td>
      <td>0.014</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>



### Collision check at t = 0

Every run starts at rest. Do the inputs at t = 0 differ between bearings? If they don't, the network can't tell the runs apart, but the expert's first command still differs.


```python
for name, f in INPUTS.items():
    first = np.array([f(e)[0] for e in rows]).round(6)
    print(f'{name:18s} distinct t=0 inputs: {len(np.unique(first, axis=0)):2d} of {len(rows)} bearings')
first_cmd = np.array([e['wheels'][0] for e in rows]).round(3)
print(f'{"expert":18s} distinct t=0 commands: {len(np.unique(first_cmd, axis=0)):2d} of {len(rows)} bearings')
```

    wheels             distinct t=0 inputs:  1 of 30 bearings
    wheels+start_dot   distinct t=0 inputs: 30 of 30 bearings
    wheels+live_dot    distinct t=0 inputs: 30 of 30 bearings
    expert             distinct t=0 commands:  5 of 30 bearings


### Prediction, written before training

- `wheels`: poor everywhere. It can't tell the bearings apart.
- `wheels+start_dot`: much better than `wheels`.
- `wheels+live_dot`: best, especially on test bearings, because the dot direction stays current after the robot turns.

These are **offline** scores on the expert's own states. They don't yet show the robot driving.


```python
SEED, EPOCHS, LR, PARK_WEIGHT = 7, 300, 0.01, 0.1
torch.set_num_threads(1)

class Policy(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.wheels = nn.Linear(16, 4)
        self.park = nn.Linear(16, 1)
    def forward(self, x):
        h = self.shared(x)
        return self.wheels(h), self.park(h).squeeze(1)

def stack(eps, f):
    X = torch.from_numpy(np.concatenate([f(e) for e in eps]).astype(np.float32))
    W = torch.from_numpy(np.concatenate([e['wheels'] for e in eps]).astype(np.float32))
    P = torch.from_numpy(np.concatenate([e['park'] for e in eps]).astype(np.float32))
    # each trajectory counts equally, whatever its length
    w = torch.from_numpy(np.concatenate([np.full(len(e['x']), 1 / (len(e['x']) * len(eps))) for e in eps]).astype(np.float32))
    return X, W, P, w

def loss_fn(model, X, W, P, w, pos_weight):
    pw, pl = model(X)
    wheel_mse = (w * ((pw - W) ** 2).mean(1)).sum()
    park_bce = F.binary_cross_entropy_with_logits(pl, P, pos_weight=pos_weight)
    return wheel_mse + PARK_WEIGHT * park_bce, wheel_mse, park_bce

train = [e for e in rows if e['split'] == 'train']
test = [e for e in rows if e['split'] == 'test']      # holdout stays untouched
models, history = {}, []
for name, f in INPUTS.items():
    torch.manual_seed(SEED)
    Xtr, Wtr, Ptr, wtr = stack(train, f)
    Xte, Wte, Pte, wte = stack(test, f)
    pos_weight = (len(Ptr) - Ptr.sum()) / Ptr.sum()     # from training data only
    model = Policy(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS + 1):
        model.train()
        total, wm, pb = loss_fn(model, Xtr, Wtr, Ptr, wtr, pos_weight)
        with torch.no_grad():
            _, te_wm, _ = loss_fn(model, Xte, Wte, Pte, wte, pos_weight)
        history.append(dict(model=name, epoch=epoch, train_wheel_mse=wm.item(), test_wheel_mse=te_wm.item(), train_park_bce=pb.item()))
        if epoch < EPOCHS:
            opt.zero_grad(); total.backward(); opt.step()
    models[name] = model
    print(f'{name:18s} inputs={Xtr.shape[1]:2d}  train rows={len(Xtr)}  final train wheel MSE={wm.item():.4f}  test={te_wm.item():.4f}')
history = pd.DataFrame(history)
Path('models').mkdir(exist_ok=True)
for name, m in models.items():
    torch.save(m.state_dict(), f'models/rung1_{name.replace("+", "_")}.pt')
```

    wheels             inputs=12  train rows=2965  final train wheel MSE=0.0704  test=0.0773


    wheels+start_dot   inputs=14  train rows=2965  final train wheel MSE=0.0460  test=0.0416


    wheels+live_dot    inputs=15  train rows=2965  final train wheel MSE=0.0342  test=0.0326



```python
fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for name, h in history.groupby('model'):
    ax[0].plot(h.epoch, h.train_wheel_mse, label=name)
    ax[1].plot(h.epoch, h.test_wheel_mse, label=name)
for a, t in zip(ax, ['Train bearings', 'Test bearings (not trained on)']):
    a.set(title=t, xlabel='epoch', yscale='log'); a.legend()
ax[0].set_ylabel('wheel-command MSE (rad/s)²')
plt.tight_layout(); plt.show()
# Test is plotted for inspection only; it does not choose the epoch (fixed at 300).
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_8_0.png)
    


### Offline results

Per trajectory, then averaged. `n` is the number of bearings. Baseline: always command zero.

PARK: **hit** = predicted PARK on the expert's PARK row; **early rows** = rows predicted PARK before the expert parked.


```python
def evaluate(model, f, eps):
    out = []
    for e in eps:
        with torch.no_grad():
            pw, pl = model(torch.from_numpy(f(e).astype(np.float32)))
        pw, prob = pw.numpy(), torch.sigmoid(pl).numpy()
        out.append(dict(bearing=e['bearing'], wheel_mae=np.abs(pw - e['wheels']).mean(),
                        zero_baseline_mae=np.abs(e['wheels']).mean(),
                        park_hit=bool(prob[-1] >= 0.5), early_park_rows=int((prob[:-1] >= 0.5).sum())))
    return pd.DataFrame(out)

results = []
for name, f in INPUTS.items():
    for split, eps in [('train', train), ('test', test)]:
        r = evaluate(models[name], f, eps).assign(model=name, split=split)
        results.append(r)
results = pd.concat(results)
summary = results.groupby(['split', 'model'], sort=False).agg(
    n=('bearing', 'size'), wheel_MAE=('wheel_mae', 'mean'), zero_baseline_MAE=('zero_baseline_mae', 'mean'),
    park_hits=('park_hit', 'sum'), early_park_rows=('early_park_rows', 'sum'))
summary.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>n</th>
      <th>wheel_MAE</th>
      <th>zero_baseline_MAE</th>
      <th>park_hits</th>
      <th>early_park_rows</th>
    </tr>
    <tr>
      <th>split</th>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <th>wheels</th>
      <td>21</td>
      <td>0.087</td>
      <td>1.995</td>
      <td>21</td>
      <td>100</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels</th>
      <td>4</td>
      <td>0.086</td>
      <td>2.040</td>
      <td>4</td>
      <td>19</td>
    </tr>
    <tr>
      <th>train</th>
      <th>wheels+start_dot</th>
      <td>21</td>
      <td>0.096</td>
      <td>1.995</td>
      <td>21</td>
      <td>101</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+start_dot</th>
      <td>4</td>
      <td>0.095</td>
      <td>2.040</td>
      <td>4</td>
      <td>21</td>
    </tr>
    <tr>
      <th>train</th>
      <th>wheels+live_dot</th>
      <td>21</td>
      <td>0.095</td>
      <td>1.995</td>
      <td>21</td>
      <td>96</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+live_dot</th>
      <td>4</td>
      <td>0.097</td>
      <td>2.040</td>
      <td>4</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



### Test bearings, worst first


```python
results[results.split == 'test'].pivot(index='bearing', columns='model', values='wheel_mae').round(3).sort_values('wheels+live_dot', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>model</th>
      <th>wheels</th>
      <th>wheels+live_dot</th>
      <th>wheels+start_dot</th>
    </tr>
    <tr>
      <th>bearing</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24</th>
      <td>0.103</td>
      <td>0.106</td>
      <td>0.111</td>
    </tr>
    <tr>
      <th>132</th>
      <td>0.083</td>
      <td>0.102</td>
      <td>0.097</td>
    </tr>
    <tr>
      <th>108</th>
      <td>0.084</td>
      <td>0.095</td>
      <td>0.088</td>
    </tr>
    <tr>
      <th>204</th>
      <td>0.075</td>
      <td>0.088</td>
      <td>0.085</td>
    </tr>
  </tbody>
</table>
</div>



### What this does **not** show

- **Offline only.** Scores use the expert's own states. The real test is closed-loop: the network drives, and its errors change what it sees next.
- **Oracle input.** The live dot comes from the simulator's true pose. A camera will replace it in rung 2.
- **No arm torques.** This dataset is driving-only.
- **Small n.** 21 training and 4 test bearings, one seed. The 4 test bearings were seen in earlier experiments. The 5 holdout bearings are not used here.
- **One copy of each trajectory.** The dataset repeats each of the 30 trajectories; repeats are identical and are dropped.

## Closed loop: the network drives the robot

Each model now drives the robot in the fast native PyBullet build (`comp0245-native-expert/build-clang19/policy_replay`), the same engine that collected the data. Headless, 1 kHz physics, 30 Hz decisions. The arm is held in its start pose.

For each of the 25 train and test bearings, the expert runs once and each model runs once from the same start:
- the expert runs until its PARK, then 0.5 s of drift;
- the model has until **expert PARK + 3 s** to issue its own PARK (probability ≥ 0.5, latched), then 0.5 s of drift.

The 5 holdout bearings are not run.

**Errors compare the model's base with the expert's base**: position (m) and heading (degrees), at PARK and at the end.

This cell reads saved rollouts. It doesn't run physics.


```python
import replay_v2
closed = []
for name in ['wheels', 'wheels+start_dot', 'wheels+live_dot']:
    _, m, diff = replay_v2.load(name)
    closed.append(m)
    print(f'{name:18s} max NumPy vs native prediction difference: {diff:.1e}')
closed = pd.concat(closed)
closed.groupby(['split', 'model'], sort=False).agg(
    n=('bearing_deg', 'size'), parked=('parked', 'sum'),
    park_xy_error_m=('park_xy_error_m', 'mean'), park_yaw_error_deg=('park_yaw_error_deg', 'mean'),
    final_xy_error_m=('final_xy_error_m', 'mean'), final_yaw_error_deg=('final_yaw_error_deg', 'mean')).round(3)
```

    wheels             max NumPy vs native prediction difference: 7.2e-07


    wheels+start_dot   max NumPy vs native prediction difference: 1.8e-06


    wheels+live_dot    max NumPy vs native prediction difference: 1.2e-06





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>n</th>
      <th>parked</th>
      <th>park_xy_error_m</th>
      <th>park_yaw_error_deg</th>
      <th>final_xy_error_m</th>
      <th>final_yaw_error_deg</th>
    </tr>
    <tr>
      <th>split</th>
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
      <th>train</th>
      <th>wheels</th>
      <td>21</td>
      <td>9</td>
      <td>0.287</td>
      <td>38.143</td>
      <td>0.447</td>
      <td>89.919</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels</th>
      <td>4</td>
      <td>3</td>
      <td>0.421</td>
      <td>40.923</td>
      <td>0.473</td>
      <td>71.182</td>
    </tr>
    <tr>
      <th>train</th>
      <th>wheels+start_dot</th>
      <td>21</td>
      <td>10</td>
      <td>0.275</td>
      <td>43.575</td>
      <td>0.361</td>
      <td>74.568</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+start_dot</th>
      <td>4</td>
      <td>1</td>
      <td>0.037</td>
      <td>4.078</td>
      <td>0.246</td>
      <td>117.384</td>
    </tr>
    <tr>
      <th>train</th>
      <th>wheels+live_dot</th>
      <td>21</td>
      <td>19</td>
      <td>0.153</td>
      <td>14.917</td>
      <td>0.182</td>
      <td>18.577</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+live_dot</th>
      <td>4</td>
      <td>4</td>
      <td>0.121</td>
      <td>12.204</td>
      <td>0.127</td>
      <td>12.484</td>
    </tr>
  </tbody>
</table>
</div>



### Every bearing: final position error (m), worst first for the live-dot model


```python
closed.pivot_table(index=['split', 'bearing_deg'], columns='model', values='final_xy_error_m').round(2)\
      .sort_values('wheels+live_dot', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>wheels</th>
      <th>wheels+live_dot</th>
      <th>wheels+start_dot</th>
    </tr>
    <tr>
      <th>split</th>
      <th>bearing_deg</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">train</th>
      <th>168</th>
      <td>0.37</td>
      <td>0.43</td>
      <td>0.48</td>
    </tr>
    <tr>
      <th>156</th>
      <td>0.45</td>
      <td>0.30</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.17</td>
      <td>0.26</td>
      <td>0.53</td>
    </tr>
    <tr>
      <th>192</th>
      <td>0.06</td>
      <td>0.23</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>test</th>
      <th>204</th>
      <td>0.03</td>
      <td>0.20</td>
      <td>0.39</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">train</th>
      <th>216</th>
      <td>0.12</td>
      <td>0.18</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.57</td>
      <td>0.18</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>288</th>
      <td>0.53</td>
      <td>0.18</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>228</th>
      <td>0.22</td>
      <td>0.18</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>312</th>
      <td>0.58</td>
      <td>0.17</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.52</td>
      <td>0.17</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>264</th>
      <td>0.41</td>
      <td>0.17</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>240</th>
      <td>0.29</td>
      <td>0.17</td>
      <td>0.51</td>
    </tr>
    <tr>
      <th>252</th>
      <td>0.36</td>
      <td>0.16</td>
      <td>0.52</td>
    </tr>
    <tr>
      <th>test</th>
      <th>24</th>
      <td>0.58</td>
      <td>0.16</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th rowspan="7" valign="top">train</th>
      <th>84</th>
      <td>0.69</td>
      <td>0.16</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.73</td>
      <td>0.16</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.65</td>
      <td>0.16</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.74</td>
      <td>0.14</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>336</th>
      <td>0.53</td>
      <td>0.14</td>
      <td>0.09</td>
    </tr>
    <tr>
      <th>360</th>
      <td>0.24</td>
      <td>0.11</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>348</th>
      <td>0.51</td>
      <td>0.10</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>test</th>
      <th>108</th>
      <td>0.68</td>
      <td>0.08</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>train</th>
      <th>120</th>
      <td>0.65</td>
      <td>0.07</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>test</th>
      <th>132</th>
      <td>0.59</td>
      <td>0.07</td>
      <td>0.28</td>
    </tr>
  </tbody>
</table>
</div>



## 3D replay: model vs ghost expert

Solid robot: the model. Cyan ghost: the expert. Use **Previous / Next** or the dropdown to change bearing, and Play, Pause, Restart or the timeline to move through the run. One player per model.

### `wheels+live_dot` (oracle dot, every step)

<a href="/files/driving_v2/closed_loop_native/viewer_wheels_live_dot.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
from IPython.display import IFrame, HTML
import uuid

def lazy_player(url, height=810):
    """A 3D player that loads only when its button is pressed. Browsers keep only about 16 3D views (WebGL
    contexts) at once, so loading every player in the notebook together leaves some of them blank."""
    k = uuid.uuid4().hex[:8]
    return HTML(f"""<div><button onclick="var f=document.getElementById('player{k}');f.src='{url}';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player{k}" style="display:none;width:100%;height:{height}px;border:0"></iframe></div>""")

def player(name):
    path = replay_v2.build_viewer(name)
    return lazy_player('/files/driving_v2/closed_loop_native/' + path.name)
player('wheels+live_dot')
```




<div><button onclick="var f=document.getElementById('playerd6f4d2b8');f.src='/files/driving_v2/closed_loop_native/viewer_wheels_live_dot.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerd6f4d2b8" style="display:none;width:100%;height:810px;border:0"></iframe></div>



### `wheels+start_dot` (dot direction at the start only)

<a href="/files/driving_v2/closed_loop_native/viewer_wheels_start_dot.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
player('wheels+start_dot')
```




<div><button onclick="var f=document.getElementById('playerd12adf93');f.src='/files/driving_v2/closed_loop_native/viewer_wheels_start_dot.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerd12adf93" style="display:none;width:100%;height:810px;border:0"></iframe></div>



### `wheels` (no dot)

<a href="/files/driving_v2/closed_loop_native/viewer_wheels.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
player('wheels')
```




<div><button onclick="var f=document.getElementById('player48c7aaf0');f.src='/files/driving_v2/closed_loop_native/viewer_wheels.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player48c7aaf0" style="display:none;width:100%;height:810px;border:0"></iframe></div>



## Change 1: skid-steer output

**One change only.** The inputs (15, including the accumulated wheel angles and the live dot), data, loss, seed, epochs and PARK head are the same as `wheels+live_dot`.

The wheel head now predicts **2 numbers**: right-side and left-side speed. A fixed step copies them into the four wheels: `[front_right, front_left, back_left, back_right] = [right, left, left, right]`. The output is still four wheel-velocity commands, like the expert's.

**Why:** the expert always gives both wheels on a side the same command (checked below). With four free outputs, the network can make small left/right mismatches the expert never makes, and that wastes effort fighting the skid steering.

**Prediction:** small offline change. In closed loop, a modest drop in heading error. Not a big jump, because the main problem is probably recovery from drift (change 2).


```python
labels = np.concatenate([e['wheels'] for e in rows])
print('Expert |front_right - back_right| max:', np.abs(labels[:, 0] - labels[:, 3]).max())
print('Expert |front_left  - back_left|  max:', np.abs(labels[:, 1] - labels[:, 2]).max())
```

    Expert |front_right - back_right| max: 0.0
    Expert |front_left  - back_left|  max: 0.0



```python
# [right, left] -> [front_right, front_left, back_left, back_right]
SKID = torch.tensor([[1., 0.], [0., 1.], [0., 1.], [1., 0.]])

class SkidPolicy(Policy):
    def __init__(self, n_in):
        super().__init__(n_in)
        self.wheels = nn.Linear(16, 2)          # right, left
    def forward(self, x):
        h = self.shared(x)
        return self.wheels(h) @ SKID.T, self.park(h).squeeze(1)

name, f = 'wheels+live_dot+skid', INPUTS['wheels+live_dot']
torch.manual_seed(SEED)
Xtr, Wtr, Ptr, wtr = stack(train, f); Xte, Wte, Pte, wte = stack(test, f)
pos_weight = (len(Ptr) - Ptr.sum()) / Ptr.sum()
model = SkidPolicy(Xtr.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
for epoch in range(EPOCHS + 1):
    total, wm, pb = loss_fn(model, Xtr, Wtr, Ptr, wtr, pos_weight)
    with torch.no_grad():
        _, te_wm, _ = loss_fn(model, Xte, Wte, Pte, wte, pos_weight)
    history = pd.concat([history, pd.DataFrame([dict(model=name, epoch=epoch, train_wheel_mse=wm.item(),
                         test_wheel_mse=te_wm.item(), train_park_bce=pb.item())])], ignore_index=True)
    if epoch < EPOCHS:
        opt.zero_grad(); total.backward(); opt.step()
models[name] = model
print(f'{name}: final train wheel MSE={wm.item():.4f}  test={te_wm.item():.4f}')
print('Parameters:', sum(p.numel() for p in model.parameters()), 'vs', sum(p.numel() for p in models['wheels+live_dot'].parameters()))

# Fold the copy step into an ordinary 4-output layer: identical outputs, so the native replay runs it unchanged.
sd = {k: v.detach() for k, v in model.state_dict().items()}
sd['wheels.weight'], sd['wheels.bias'] = SKID @ sd['wheels.weight'], SKID @ sd['wheels.bias']
folded = Policy(Xtr.shape[1]); folded.load_state_dict(sd)
with torch.no_grad():
    print('Folded vs skid model, max output difference:', (folded(Xte)[0] - model(Xte)[0]).abs().max().item())
torch.save(model.state_dict(), 'models/rung1_wheels_live_dot_skid.pt')
np.savez('models/rung1_wheels_live_dot_skid.npz', **{k: v.numpy() for k, v in sd.items()})
```

    wheels+live_dot+skid: final train wheel MSE=0.0341  test=0.0416
    Parameters: 579 vs 613
    Folded vs skid model, max output difference: 0.0



```python
r = pd.concat([evaluate(models[m], INPUTS['wheels+live_dot'], eps).assign(model=m, split=s)
               for m in ['wheels+live_dot', name] for s, eps in [('train', train), ('test', test)]])
r.groupby(['split', 'model'], sort=False).agg(n=('bearing', 'size'), wheel_MAE=('wheel_mae', 'mean'),
    park_hits=('park_hit', 'sum'), early_park_rows=('early_park_rows', 'sum')).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>n</th>
      <th>wheel_MAE</th>
      <th>park_hits</th>
      <th>early_park_rows</th>
    </tr>
    <tr>
      <th>split</th>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>train</th>
      <th>wheels+live_dot</th>
      <td>21</td>
      <td>0.095</td>
      <td>21</td>
      <td>96</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+live_dot</th>
      <td>4</td>
      <td>0.097</td>
      <td>4</td>
      <td>19</td>
    </tr>
    <tr>
      <th>train</th>
      <th>wheels+live_dot+skid</th>
      <td>21</td>
      <td>0.085</td>
      <td>21</td>
      <td>101</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+live_dot+skid</th>
      <td>4</td>
      <td>0.095</td>
      <td>4</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



### Closed loop: skid output vs four free outputs

Same 25 bearings, same fast native engine, same rules as above. The first run takes about two minutes; after that the saved rollout is reused.


```python
plan = json.loads((DATA / 'split_plan.json').read_text())
cases = [dict(target=p['target'], bearing_deg=p['bearing_deg'], split=p['split']) for p in plan if p['split'] in ('train', 'test')]
replay_v2.run_native(name, cases)
_, skid_closed, diff = replay_v2.load(name)
print(f'max NumPy vs native prediction difference: {diff:.1e}')
both = pd.concat([closed[closed.model == 'wheels+live_dot'], skid_closed])
both.groupby(['split', 'model'], sort=False).agg(
    n=('bearing_deg', 'size'), parked=('parked', 'sum'),
    park_xy_error_m=('park_xy_error_m', 'mean'), park_yaw_error_deg=('park_yaw_error_deg', 'mean'),
    final_xy_error_m=('final_xy_error_m', 'mean'), final_yaw_error_deg=('final_yaw_error_deg', 'mean')).round(3)
```

    max NumPy vs native prediction difference: 1.2e-06





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>n</th>
      <th>parked</th>
      <th>park_xy_error_m</th>
      <th>park_yaw_error_deg</th>
      <th>final_xy_error_m</th>
      <th>final_yaw_error_deg</th>
    </tr>
    <tr>
      <th>split</th>
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
      <th>train</th>
      <th>wheels+live_dot</th>
      <td>21</td>
      <td>19</td>
      <td>0.153</td>
      <td>14.917</td>
      <td>0.182</td>
      <td>18.577</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+live_dot</th>
      <td>4</td>
      <td>4</td>
      <td>0.121</td>
      <td>12.204</td>
      <td>0.127</td>
      <td>12.484</td>
    </tr>
    <tr>
      <th>train</th>
      <th>wheels+live_dot+skid</th>
      <td>21</td>
      <td>8</td>
      <td>0.093</td>
      <td>12.947</td>
      <td>0.158</td>
      <td>60.261</td>
    </tr>
    <tr>
      <th>test</th>
      <th>wheels+live_dot+skid</th>
      <td>4</td>
      <td>1</td>
      <td>0.097</td>
      <td>8.486</td>
      <td>0.247</td>
      <td>15.516</td>
    </tr>
  </tbody>
</table>
</div>




```python
both.pivot_table(index=['split', 'bearing_deg'], columns='model', values='final_xy_error_m').round(2)\
    .sort_values('wheels+live_dot+skid', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>wheels+live_dot</th>
      <th>wheels+live_dot+skid</th>
    </tr>
    <tr>
      <th>split</th>
      <th>bearing_deg</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">train</th>
      <th>156</th>
      <td>0.30</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0.07</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">test</th>
      <th>132</th>
      <td>0.07</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>108</th>
      <td>0.08</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>72</th>
      <td>0.16</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>84</th>
      <td>0.16</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.16</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.14</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.26</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>test</th>
      <th>24</th>
      <td>0.16</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">train</th>
      <th>216</th>
      <td>0.18</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>228</th>
      <td>0.18</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>240</th>
      <td>0.17</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>288</th>
      <td>0.18</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>252</th>
      <td>0.16</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>test</th>
      <th>204</th>
      <td>0.20</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th rowspan="9" valign="top">train</th>
      <th>192</th>
      <td>0.23</td>
      <td>0.11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.17</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>168</th>
      <td>0.43</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>348</th>
      <td>0.10</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>300</th>
      <td>0.18</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>312</th>
      <td>0.17</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>336</th>
      <td>0.14</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>264</th>
      <td>0.17</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>360</th>
      <td>0.11</td>
      <td>0.02</td>
    </tr>
  </tbody>
</table>
</div>



### 3D replay: `wheels+live_dot+skid`

<a href="/files/driving_v2/closed_loop_native/viewer_wheels_live_dot_skid.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
player('wheels+live_dot+skid')
```




<div><button onclick="var f=document.getElementById('player83d0af33');f.src='/files/driving_v2/closed_loop_native/viewer_wheels_live_dot_skid.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player83d0af33" style="display:none;width:100%;height:810px;border:0"></iframe></div>



## Camera check: 180° fisheye RGB-D on top of the hand

**Goal:** can one wrist camera replace the oracle dot? No training here: only rendering and a colour threshold.

**Camera** (code: `wrist_camera.py`):
- mounted on top of the hand near the finger roots: `(0, 0.045, -0.07)` m in the hand frame `robot_j2s7s300_end_effector`, about 0.76 m off the ground in the hold pose;
- faces along the fingers, which is robot-forward while the arm is held;
- 180° equidistant fisheye, 256 × 256, RGB + depth. It is built from five 90° cube-face renders.

**Robot and dot are unchanged** from `robotnik_arm_task.py`: no repainting; the dot is a red sphere with a 3 cm radius.

**Frames:** saved expert poses from the 30 runs, every 6th row (5 Hz), plus start, middle and park. No physics is stepped.

**Detection:**
1. Red pixels: `r > 100` and `r > 1.7 g` and `r > 1.7 b`.
2. Their centre gives the ray direction.
3. The median depth of those pixels, plus the 3 cm radius, gives the distance.
4. The camera's pose comes from the arm joint angles. That converts the estimate into distance and direction relative to the base, the same numbers as the oracle.

**Ground truth, for checking only:** PyBullet's segmentation mask says whether the dot is truly in view.


```python
import os, subprocess, sys
CAM = Path('camera/wrist_fisheye_check.npz')
if not CAM.exists():   # renders with the RoboEnv Python (has PyBullet); about 1-2 minutes
    subprocess.run([os.environ.get('PYBULLET_PYTHON', sys.executable), 'wrist_camera.py', str(CAM)], check=True)
cam = np.load(CAM)
frames = pd.DataFrame(json.loads(str(cam['rows'])))
frames['dir_err_deg'] = np.abs((frames.est_dir_deg - frames.true_dir_deg + 180) % 360 - 180)
frames['dist_err_cm'] = 100 * (frames.est_dist_m - frames.true_dist_m).abs()
print(f'{len(frames)} frames from {frames.bearing.nunique()} bearings')
print('Detector vs ground truth:',
      f"missed {int(((~frames.found) & (frames.true_visible_px > 0)).sum())},",
      f"false detections {int((frames.found & (frames.true_visible_px == 0)).sum())},",
      f"red pixels that were not the dot {int(frames.false_red_px.sum())}")
plt.figure(figsize=(6, 4.5)); plt.imshow(cam['outside']); plt.axis('off')
plt.title('Camera position (green) in the hold pose'); plt.show()
```

    769 frames from 30 bearings
    Detector vs ground truth: missed 0, false detections 0, red pixels that were not the dot 0



    
![png](driving_v2_sensors_files/driving_v2_sensors_35_1.png)
    


### What the camera sees at the start, for every bearing

Title: bearing, and whether the dot was found. The robot hasn't moved yet.


```python
def grid(images, title):
    fig, axes = plt.subplots(5, 6, figsize=(15, 13))
    for ax, im, b in zip(axes.flat, images, cam['bearings']):
        f = frames[(frames.bearing == b)]
        row = f.iloc[0] if title == 'start' else f.iloc[-1]
        ax.imshow(im); ax.axis('off')
        ax.set_title(f"{b}°  {'dot found' if row.found else 'no dot'}", color='green' if row.found else 'red', fontsize=10)
    fig.suptitle(f'Wrist fisheye at {title}', fontsize=14); plt.tight_layout(); plt.show()
grid(cam['img_start'], 'start')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_37_0.png)
    



```python
grid(cam['img_park'], 'park')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_38_0.png)
    


### Visibility and accuracy per bearing

`seen_at_start`: dot found in the first frame. `fraction_seen`: share of the drive's frames in which it was found. Errors are medians over frames where it was found, against the oracle.


```python
per = frames.groupby('bearing').agg(seen_at_start=('found', 'first'), fraction_seen=('found', 'mean'),
    first_seen_s=('time_s', lambda t: t[frames.loc[t.index, 'found']].min()),
    dot_pixels=('dot_px', lambda s: s[s > 0].median()),
    direction_error_deg=('dir_err_deg', 'median'), distance_error_cm=('dist_err_cm', 'median'))
print(f"Seen at start: {int(per.seen_at_start.sum())} of {len(per)} bearings")
print(f"When seen: median direction error {frames.dir_err_deg.median():.2f}°, median distance error {frames.dist_err_cm.median():.2f} cm")
per.round(2)
```

    Seen at start: 5 of 30 bearings
    When seen: median direction error 0.14°, median distance error 0.85 cm





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seen_at_start</th>
      <th>fraction_seen</th>
      <th>first_seen_s</th>
      <th>dot_pixels</th>
      <th>direction_error_deg</th>
      <th>distance_error_cm</th>
    </tr>
    <tr>
      <th>bearing</th>
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
      <th>12</th>
      <td>True</td>
      <td>0.53</td>
      <td>0.0</td>
      <td>64.0</td>
      <td>0.10</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>24</th>
      <td>False</td>
      <td>0.47</td>
      <td>0.2</td>
      <td>63.0</td>
      <td>0.13</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>36</th>
      <td>True</td>
      <td>0.57</td>
      <td>0.0</td>
      <td>51.5</td>
      <td>0.16</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>48</th>
      <td>True</td>
      <td>0.62</td>
      <td>0.0</td>
      <td>59.0</td>
      <td>0.14</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>60</th>
      <td>False</td>
      <td>0.54</td>
      <td>0.4</td>
      <td>60.0</td>
      <td>0.18</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>72</th>
      <td>False</td>
      <td>0.52</td>
      <td>0.6</td>
      <td>61.5</td>
      <td>0.16</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>84</th>
      <td>False</td>
      <td>0.52</td>
      <td>0.6</td>
      <td>54.0</td>
      <td>0.22</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>96</th>
      <td>False</td>
      <td>0.48</td>
      <td>0.8</td>
      <td>55.0</td>
      <td>0.17</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>108</th>
      <td>False</td>
      <td>0.50</td>
      <td>1.0</td>
      <td>56.0</td>
      <td>0.19</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>120</th>
      <td>False</td>
      <td>0.50</td>
      <td>1.2</td>
      <td>54.0</td>
      <td>0.17</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>132</th>
      <td>False</td>
      <td>0.48</td>
      <td>1.4</td>
      <td>56.0</td>
      <td>0.18</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>144</th>
      <td>False</td>
      <td>0.52</td>
      <td>1.6</td>
      <td>46.5</td>
      <td>0.15</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>156</th>
      <td>False</td>
      <td>0.48</td>
      <td>1.8</td>
      <td>50.0</td>
      <td>0.11</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>168</th>
      <td>False</td>
      <td>0.48</td>
      <td>1.8</td>
      <td>45.5</td>
      <td>0.10</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>180</th>
      <td>False</td>
      <td>0.33</td>
      <td>2.6</td>
      <td>55.0</td>
      <td>0.11</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>192</th>
      <td>False</td>
      <td>0.33</td>
      <td>2.6</td>
      <td>56.0</td>
      <td>0.10</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>204</th>
      <td>False</td>
      <td>0.33</td>
      <td>2.4</td>
      <td>53.0</td>
      <td>0.10</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>216</th>
      <td>False</td>
      <td>0.34</td>
      <td>2.2</td>
      <td>61.0</td>
      <td>0.14</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>228</th>
      <td>False</td>
      <td>0.32</td>
      <td>2.0</td>
      <td>57.5</td>
      <td>0.10</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>240</th>
      <td>False</td>
      <td>0.33</td>
      <td>1.8</td>
      <td>58.5</td>
      <td>0.10</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>252</th>
      <td>False</td>
      <td>0.32</td>
      <td>1.6</td>
      <td>65.0</td>
      <td>0.06</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>264</th>
      <td>False</td>
      <td>0.32</td>
      <td>1.4</td>
      <td>58.0</td>
      <td>0.10</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>276</th>
      <td>False</td>
      <td>0.36</td>
      <td>1.2</td>
      <td>64.0</td>
      <td>0.05</td>
      <td>0.85</td>
    </tr>
    <tr>
      <th>288</th>
      <td>False</td>
      <td>0.36</td>
      <td>1.0</td>
      <td>64.0</td>
      <td>0.09</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>300</th>
      <td>False</td>
      <td>0.38</td>
      <td>0.8</td>
      <td>62.0</td>
      <td>0.12</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>312</th>
      <td>False</td>
      <td>0.33</td>
      <td>0.8</td>
      <td>67.0</td>
      <td>0.11</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>324</th>
      <td>False</td>
      <td>0.43</td>
      <td>0.6</td>
      <td>66.0</td>
      <td>0.14</td>
      <td>0.83</td>
    </tr>
    <tr>
      <th>336</th>
      <td>False</td>
      <td>0.47</td>
      <td>0.4</td>
      <td>68.0</td>
      <td>0.14</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>348</th>
      <td>True</td>
      <td>0.53</td>
      <td>0.0</td>
      <td>71.0</td>
      <td>0.10</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>360</th>
      <td>True</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>84.0</td>
      <td>0.11</td>
      <td>0.88</td>
    </tr>
  </tbody>
</table>
</div>



### Reading the check

- **Accuracy when the dot is seen:** see the medians above. This is the camera + colour threshold + depth, with no training.
- **Start-of-run visibility:** with the arm in its hold pose, the hand and fingers sit below and around the camera. Bearings where the dot isn't found at the start were checked against the segmentation mask: the dot is genuinely out of view or hidden, not missed by the detector.
- **Consequence for the policy:** when the dot isn't visible at the start, the camera can't say which way to turn. That's the t = 0 problem again. It needs an explainable rule, for example "turn on the spot until the dot is seen", or memory (the occlusion experiment).

## Camera check: 180° fisheye RGB-D on the chassis nose (prow)

Same fisheye model, detector and checks as the hand camera (code: `prow_camera.py`). Only the mount changes:
- **on the chassis surface**, on the sloping nose in front of and slightly right of the arm base: `(0.29, -0.08, 0.36)` m in the base frame (surface at 0.33 m, plus 3 cm for the camera body);
- facing robot-forward, level;
- placed from a measured map of the chassis top (0.38 m behind the arm, sloping to 0.32 m at the front edge, x ≈ 0.32 m). The arm base occupies |y| < 0.045 m, so the camera sits just to its right.


```python
import camera_check
if not Path('camera/prow_fisheye_check.npz').exists():
    subprocess.run([os.environ.get('PYBULLET_PYTHON', sys.executable), 'prow_camera.py',
                    'camera/prow_fisheye_check.npz'], check=True)
prow, prow_frames = camera_check.load('camera/prow_fisheye_check.npz')
plt.figure(figsize=(6, 4.5)); plt.imshow(prow['outside']); plt.axis('off'); plt.title('Prow camera position (green), at park'); plt.show()
prow_per = camera_check.summary(prow_frames)
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_43_0.png)
    


    769 frames from 30 bearings
    Detector vs ground truth: missed 0, false detections 0, red pixels that were not the dot 0
    Seen at start: 13 of 30 | seen at park: 12 of 30
    When seen: median direction error 0.17°, median distance error 0.97 cm



```python
camera_check.grid(prow, prow_frames, 'start', 'Prow fisheye')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_44_0.png)
    



```python
camera_check.grid(prow, prow_frames, 'park', 'Prow fisheye')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_45_0.png)
    



```python
prow_per
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>seen_at_start</th>
      <th>seen_at_park</th>
      <th>fraction_seen</th>
      <th>dot_pixels</th>
      <th>direction_error_deg</th>
      <th>distance_error_cm</th>
    </tr>
    <tr>
      <th>bearing</th>
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
      <th>12</th>
      <td>True</td>
      <td>False</td>
      <td>0.94</td>
      <td>32.0</td>
      <td>0.22</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>24</th>
      <td>True</td>
      <td>False</td>
      <td>0.95</td>
      <td>43.0</td>
      <td>0.14</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>36</th>
      <td>True</td>
      <td>False</td>
      <td>0.90</td>
      <td>42.0</td>
      <td>0.14</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>48</th>
      <td>True</td>
      <td>False</td>
      <td>0.95</td>
      <td>42.5</td>
      <td>0.09</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>60</th>
      <td>True</td>
      <td>False</td>
      <td>0.92</td>
      <td>40.0</td>
      <td>0.17</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>72</th>
      <td>True</td>
      <td>False</td>
      <td>0.96</td>
      <td>36.0</td>
      <td>0.19</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>84</th>
      <td>False</td>
      <td>False</td>
      <td>0.88</td>
      <td>34.5</td>
      <td>0.14</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>96</th>
      <td>False</td>
      <td>False</td>
      <td>0.84</td>
      <td>36.0</td>
      <td>0.14</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>108</th>
      <td>False</td>
      <td>False</td>
      <td>0.79</td>
      <td>34.5</td>
      <td>0.15</td>
      <td>1.02</td>
    </tr>
    <tr>
      <th>120</th>
      <td>False</td>
      <td>False</td>
      <td>0.80</td>
      <td>31.0</td>
      <td>0.15</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>132</th>
      <td>False</td>
      <td>False</td>
      <td>0.81</td>
      <td>28.0</td>
      <td>0.16</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>144</th>
      <td>False</td>
      <td>False</td>
      <td>0.77</td>
      <td>35.0</td>
      <td>0.13</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>156</th>
      <td>False</td>
      <td>False</td>
      <td>0.73</td>
      <td>30.5</td>
      <td>0.18</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>168</th>
      <td>False</td>
      <td>False</td>
      <td>0.73</td>
      <td>31.5</td>
      <td>0.16</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>180</th>
      <td>False</td>
      <td>False</td>
      <td>0.52</td>
      <td>23.0</td>
      <td>0.29</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>192</th>
      <td>False</td>
      <td>True</td>
      <td>0.70</td>
      <td>23.0</td>
      <td>0.18</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>204</th>
      <td>False</td>
      <td>True</td>
      <td>0.73</td>
      <td>23.0</td>
      <td>0.21</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>216</th>
      <td>False</td>
      <td>True</td>
      <td>0.78</td>
      <td>23.0</td>
      <td>0.19</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>228</th>
      <td>False</td>
      <td>True</td>
      <td>0.81</td>
      <td>23.0</td>
      <td>0.19</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>240</th>
      <td>False</td>
      <td>True</td>
      <td>0.83</td>
      <td>24.0</td>
      <td>0.18</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>252</th>
      <td>False</td>
      <td>True</td>
      <td>0.82</td>
      <td>24.0</td>
      <td>0.17</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>264</th>
      <td>False</td>
      <td>True</td>
      <td>0.84</td>
      <td>24.0</td>
      <td>0.19</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>276</th>
      <td>False</td>
      <td>False</td>
      <td>0.84</td>
      <td>25.0</td>
      <td>0.16</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>288</th>
      <td>True</td>
      <td>True</td>
      <td>0.96</td>
      <td>25.0</td>
      <td>0.25</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>300</th>
      <td>True</td>
      <td>True</td>
      <td>0.96</td>
      <td>26.0</td>
      <td>0.22</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>312</th>
      <td>True</td>
      <td>True</td>
      <td>0.95</td>
      <td>26.5</td>
      <td>0.16</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>324</th>
      <td>True</td>
      <td>True</td>
      <td>0.86</td>
      <td>26.0</td>
      <td>0.21</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>336</th>
      <td>True</td>
      <td>False</td>
      <td>0.74</td>
      <td>27.0</td>
      <td>0.32</td>
      <td>0.99</td>
    </tr>
    <tr>
      <th>348</th>
      <td>True</td>
      <td>True</td>
      <td>0.76</td>
      <td>26.0</td>
      <td>0.21</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>360</th>
      <td>True</td>
      <td>False</td>
      <td>0.91</td>
      <td>22.0</td>
      <td>0.63</td>
      <td>1.05</td>
    </tr>
  </tbody>
</table>
</div>



### Reading the prow check

| | Hand fisheye | Prow fisheye |
|---|---|---|
| Seen at start | 5 / 30 | 13 / 30 |
| Seen during the drive | about 30–60% of frames | **about 70–96% of frames** |
| Seen at park | 0 / 30 | 12 / 30 |
| Accuracy when seen | 0.14°, 0.9 cm | 0.17°, 1.0 cm |

- **During the drive, the prow camera sees the dot most of the time.** The arm is above it, so it rarely gets in the way.
- **Start:** bearings beyond about ±80° are outside the 180° view, so the robot can't tell which way to turn. It needs a rule or the tracker.
- **Park:** the dot finishes just behind the arm's forearm, at the edge of being hidden. Whether it's seen depends on a few degrees of final heading, which is why some bearings show it and others don't. A tracker carrying the last estimate covers this.
- **Detector:** no misses and no false detections against the segmentation ground truth.

## Camera check: 180° fisheye RGB-D on the front-right corner of the chassis

Same camera, detector and checks. Mount moved to the **front-right corner of the chassis top**: `(0.29, -0.13, 0.36)` m in the base frame, still facing forward and level. The chassis top reaches out to about |y| = 0.12 m at the front edge; beyond that, the front wheel is highest.

**Why:** the further the camera sits from the arm's centre line, the less the forearm should stand between it and the dot at park.


```python
import os
if not Path('camera/corner_fisheye_check.npz').exists():
    subprocess.run([os.environ.get('PYBULLET_PYTHON', sys.executable), 'prow_camera.py',
                    'camera/corner_fisheye_check.npz'], check=True, env=dict(os.environ, PROW_MOUNT='0.29,-0.13,0.36'))
corner, corner_frames = camera_check.load('camera/corner_fisheye_check.npz')
plt.figure(figsize=(6, 4.5)); plt.imshow(corner['outside']); plt.axis('off'); plt.title('Corner camera position (green), at park'); plt.show()
corner_per = camera_check.summary(corner_frames)
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_49_0.png)
    


    769 frames from 30 bearings
    Detector vs ground truth: missed 0, false detections 0, red pixels that were not the dot 0
    Seen at start: 13 of 30 | seen at park: 19 of 30
    When seen: median direction error 0.13°, median distance error 0.96 cm



```python
camera_check.grid(corner, corner_frames, 'start', 'Corner fisheye')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_50_0.png)
    



```python
camera_check.grid(corner, corner_frames, 'park', 'Corner fisheye')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_51_0.png)
    


### Nose vs corner, per bearing


```python
pd.concat({'nose': prow_per[['seen_at_start', 'seen_at_park', 'fraction_seen']],
           'corner': corner_per[['seen_at_start', 'seen_at_park', 'fraction_seen']]}, axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">nose</th>
      <th colspan="3" halign="left">corner</th>
    </tr>
    <tr>
      <th></th>
      <th>seen_at_start</th>
      <th>seen_at_park</th>
      <th>fraction_seen</th>
      <th>seen_at_start</th>
      <th>seen_at_park</th>
      <th>fraction_seen</th>
    </tr>
    <tr>
      <th>bearing</th>
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
      <th>12</th>
      <td>True</td>
      <td>False</td>
      <td>0.94</td>
      <td>True</td>
      <td>False</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>24</th>
      <td>True</td>
      <td>False</td>
      <td>0.95</td>
      <td>True</td>
      <td>True</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>36</th>
      <td>True</td>
      <td>False</td>
      <td>0.90</td>
      <td>True</td>
      <td>False</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>48</th>
      <td>True</td>
      <td>False</td>
      <td>0.95</td>
      <td>True</td>
      <td>False</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>60</th>
      <td>True</td>
      <td>False</td>
      <td>0.92</td>
      <td>True</td>
      <td>False</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>72</th>
      <td>True</td>
      <td>False</td>
      <td>0.96</td>
      <td>True</td>
      <td>False</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>84</th>
      <td>False</td>
      <td>False</td>
      <td>0.88</td>
      <td>False</td>
      <td>True</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>96</th>
      <td>False</td>
      <td>False</td>
      <td>0.84</td>
      <td>False</td>
      <td>True</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>108</th>
      <td>False</td>
      <td>False</td>
      <td>0.79</td>
      <td>False</td>
      <td>False</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>120</th>
      <td>False</td>
      <td>False</td>
      <td>0.80</td>
      <td>False</td>
      <td>False</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>132</th>
      <td>False</td>
      <td>False</td>
      <td>0.81</td>
      <td>False</td>
      <td>False</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>144</th>
      <td>False</td>
      <td>False</td>
      <td>0.77</td>
      <td>False</td>
      <td>False</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>156</th>
      <td>False</td>
      <td>False</td>
      <td>0.73</td>
      <td>False</td>
      <td>False</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>168</th>
      <td>False</td>
      <td>False</td>
      <td>0.73</td>
      <td>False</td>
      <td>False</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>180</th>
      <td>False</td>
      <td>False</td>
      <td>0.52</td>
      <td>False</td>
      <td>True</td>
      <td>0.58</td>
    </tr>
    <tr>
      <th>192</th>
      <td>False</td>
      <td>True</td>
      <td>0.70</td>
      <td>False</td>
      <td>True</td>
      <td>0.67</td>
    </tr>
    <tr>
      <th>204</th>
      <td>False</td>
      <td>True</td>
      <td>0.73</td>
      <td>False</td>
      <td>True</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>216</th>
      <td>False</td>
      <td>True</td>
      <td>0.78</td>
      <td>False</td>
      <td>True</td>
      <td>0.78</td>
    </tr>
    <tr>
      <th>228</th>
      <td>False</td>
      <td>True</td>
      <td>0.81</td>
      <td>False</td>
      <td>True</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>240</th>
      <td>False</td>
      <td>True</td>
      <td>0.83</td>
      <td>False</td>
      <td>True</td>
      <td>0.77</td>
    </tr>
    <tr>
      <th>252</th>
      <td>False</td>
      <td>True</td>
      <td>0.82</td>
      <td>False</td>
      <td>True</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>264</th>
      <td>False</td>
      <td>True</td>
      <td>0.84</td>
      <td>False</td>
      <td>True</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>276</th>
      <td>False</td>
      <td>False</td>
      <td>0.84</td>
      <td>False</td>
      <td>True</td>
      <td>0.80</td>
    </tr>
    <tr>
      <th>288</th>
      <td>True</td>
      <td>True</td>
      <td>0.96</td>
      <td>True</td>
      <td>True</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>300</th>
      <td>True</td>
      <td>True</td>
      <td>0.96</td>
      <td>True</td>
      <td>True</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>312</th>
      <td>True</td>
      <td>True</td>
      <td>0.95</td>
      <td>True</td>
      <td>True</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>324</th>
      <td>True</td>
      <td>True</td>
      <td>0.86</td>
      <td>True</td>
      <td>True</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>336</th>
      <td>True</td>
      <td>False</td>
      <td>0.74</td>
      <td>True</td>
      <td>True</td>
      <td>0.74</td>
    </tr>
    <tr>
      <th>348</th>
      <td>True</td>
      <td>True</td>
      <td>0.76</td>
      <td>True</td>
      <td>True</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>360</th>
      <td>True</td>
      <td>False</td>
      <td>0.91</td>
      <td>True</td>
      <td>True</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



### Reading the corner check

| | Hand | Nose | **Corner** |
|---|---|---|---|
| Seen at start | 5 / 30 | 13 / 30 | 13 / 30 |
| Seen during the drive | about 30–60% | about 70–96% | about 60–100% |
| Seen at park | 0 / 30 | 12 / 30 | **19 / 30** |
| Accuracy when seen | 0.14°, 0.9 cm | 0.17°, 1.0 cm | 0.13°, 1.0 cm |

- Moving the camera 5 cm further out, from y = -0.08 to y = -0.13, raises park visibility from 12 to 19 of 30 bearings. The forearm blocks the dot less.
- The start is unchanged: dots beyond about ±80° are outside any forward 180° view.
- The front-right wheel now shows at the bottom edge of the image. It doesn't affect the dot, since it's below the horizon.
- The dot is still hidden at park for 11 bearings, 12°–72° and 108°–168°: the bearings where the robot turned left to reach the dot, so the tracker is still needed for the final moments.

## Camera models: the dot comes from the corner camera

`wheels+live_dot` was given the true dot position at every step. A real robot does not have that. These three models get the dot from the corner fisheye camera instead, and keep the same 12 wheel inputs.

At every step:

1. Render the corner fisheye from the robot's current pose.
2. Find the red pixels. Their centre gives a ray; the depth at those pixels gives a distance.
3. Use the fixed camera mount to turn that into the dot's position relative to the robot.
4. A tracker holds that position. When the dot is seen, it is replaced. When the dot is hidden, it is moved using the wheel encoders.
5. The network gets the tracker's distance and the sin and cos of its direction. Until the dot is first seen, these are 0, 0, 0.

| Model | Dot inputs | PARK output |
|---|---|---|
| `camera+flag` | distance, sin, cos, visible flag (1 when the camera sees the dot in this frame) | PARK yes/no |
| `camera+flag+time_to_park` | distance, sin, cos, visible flag | seconds until the expert would park; PARK at 0.25 s or less |
| `camera+time_to_park` | distance, sin, cos | seconds until the expert would park; PARK at 0.35 s or less |

The network and training are the same as before: 2×16 ReLU, 4 wheel outputs, 21 training runs, Adam, lr 0.01, 300 epochs, seed 7. `camera_features.py` computes the camera inputs for the training runs from the expert's recorded poses and saves them in `camera_features/`.


```python
CAM = Path('camera_features')
for e in rows:
    if e['split'] != 'holdout':
        e['camera'] = np.load(CAM / f"camera_{e['bearing']}.npy")      # distance, sin, cos, visible
        assert len(e['camera']) == len(e['x'])

CLIP, THRESH = 3.0, 0.25          # time to park is clipped at 3 s and scaled to 0..1
CAMERA_MODELS = {                  # name: (dot input columns, PARK output)
    'camera+flag':              ([0, 1, 2, 3], 'park'),
    'camera+flag+time_to_park': ([0, 1, 2, 3], 'time_to_park'),
    'camera+time_to_park':      ([0, 1, 2],    'time_to_park'),
}

def camera_inputs(e, cols):
    return np.c_[e['x'][:, WHEEL], e['camera'][:, cols]].astype(np.float32)

def time_to_park(e):              # one PARK row per run, the last one
    t = np.load(DATA / next(r for r in episodes if r['bearing_deg'] == e['bearing'])['archive'])['time_s']
    return (np.minimum(t[-1] - t, CLIP) / CLIP).astype(np.float32)

class CameraPolicy(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.wheels = nn.Linear(16, 4)
        self.park = nn.Linear(16, 1)          # PARK logit, or scaled time to park
    def forward(self, x):
        h = self.shared(x)
        return self.wheels(h), self.park(h).squeeze(1)

def camera_data(eps, cols, head):
    X = torch.from_numpy(np.concatenate([camera_inputs(e, cols) for e in eps]))
    W = torch.from_numpy(np.concatenate([e['wheels'] for e in eps]).astype(np.float32))
    w = torch.from_numpy(np.concatenate([np.full(len(e['x']), 1 / (len(e['x']) * len(eps))) for e in eps]).astype(np.float32))
    if head == 'park':
        P = torch.from_numpy(np.concatenate([e['park'] for e in eps]).astype(np.float32))
    else:
        P = torch.from_numpy(np.concatenate([time_to_park(e) for e in eps]))
    return X, W, P, w

def camera_losses(model, data, head, pos_weight):
    X, W, P, w = data
    pw, pp = model(X)
    wheel_mse = (w * ((pw - W) ** 2).mean(1)).sum()
    if head == 'park':
        park_loss = F.binary_cross_entropy_with_logits(pp, P, pos_weight=pos_weight)       # weighted by PARK_WEIGHT in the total
    else:
        park_loss = (w * (pp - P) ** 2).sum()                                               # each run counts equally, weight 1
    return wheel_mse, park_loss

camera_history = []
def train_camera_model(name, cols, head):
    torch.manual_seed(SEED)
    tr, te = camera_data(train, cols, head), camera_data(test, cols, head)
    pos_weight = (len(tr[2]) - tr[2].sum()) / tr[2].sum() if head == 'park' else None    # from training data only
    model = CameraPolicy(tr[0].shape[1]); opt = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS + 1):
        wheel_mse, park_loss = camera_losses(model, tr, head, pos_weight)
        with torch.no_grad():
            te_wheel, te_park = camera_losses(model, te, head, pos_weight)                   # test is only watched, never trained on
        camera_history.append(dict(model=name, epoch=epoch, train_wheel_mse=wheel_mse.item(), test_wheel_mse=te_wheel.item(),
                                   train_park_loss=park_loss.item(), test_park_loss=te_park.item()))
        if epoch < EPOCHS:
            total = wheel_mse + (PARK_WEIGHT * park_loss if head == 'park' else park_loss)
            opt.zero_grad(); total.backward(); opt.step()
    return model

camera_models = {}
for name, (cols, head) in CAMERA_MODELS.items():
    camera_models[name] = m = train_camera_model(name, cols, head)
    sd = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    if head == 'time_to_park':
        sd['ttp_clip'], sd['ttp_thresh'] = np.float32(CLIP), np.float32(THRESH)
    # Compare with the saved network that was tested.
    tested = np.load({'camera+flag': 'models/camera_flag.npz', 'camera+flag+time_to_park': 'models/camera_flag_ttp.npz',
                      'camera+time_to_park': 'models/camera_ttp_noflag.npz'}[name])
    diff = max(float(np.abs(sd[k] - tested[k]).max()) for k in tested.files)
    print(f'{name:26s} inputs={len(WHEEL) + len(cols)}  largest weight difference vs the tested model: {diff:.1e}')
# Small differences are floating-point rounding. The tests below use the saved networks in models/.
camera_history = pd.DataFrame(camera_history)
```

    camera+flag                inputs=16  largest weight difference vs the tested model: 7.2e-07


    camera+flag+time_to_park   inputs=16  largest weight difference vs the tested model: 6.7e-04


    camera+time_to_park        inputs=15  largest weight difference vs the tested model: 2.7e-04


### Training curves


```python
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for name, h in camera_history.groupby('model', sort=False):
    line, = ax[0].plot(h.epoch, h.train_wheel_mse, label=f'{name} train')
    ax[0].plot(h.epoch, h.test_wheel_mse, '--', color=line.get_color(), label=f'{name} test')
    a = ax[1] if CAMERA_MODELS[name][1] == 'park' else ax[2]
    a.plot(h.epoch, h.train_park_loss, color=line.get_color(), label=f'{name} train')
    a.plot(h.epoch, h.test_park_loss, '--', color=line.get_color(), label=f'{name} test')
ax[0].set(title='Wheel commands', ylabel='MSE (rad/s)²')
ax[1].set(title='PARK yes/no (camera+flag)', ylabel='class-weighted BCE')
ax[2].set(title='Time to park', ylabel='MSE of (seconds left / 3 s)')
for a in ax:
    a.set(xlabel='epoch', yscale='log'); a.legend(fontsize=7)
plt.tight_layout(); plt.show()
# Solid: 21 training runs. Dashed: 4 test runs, watched only. Training always stops at 300 epochs.
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_58_0.png)
    


**Model to build from:** `camera+flag+time_to_park` (`models/camera_flag_ttp.npz`). PARK at 0.25 s or less. Parked 16 of 25.

### Testing

Each model drives the robot in the fast simulator (`sensor_parking`), with the camera steps above running every 1/30 s. There are 25 bearings: the 21 training runs and the 4 test runs. The 5 holdout bearings are not used. Each run is allowed the expert's park time plus 3 s, and its final position is compared with the expert's. `wheels+live_dot`, which sees the true dot, is shown for comparison.


```python
ref = json.loads(Path('expert_reference.json').read_text())['bearings']
RESULTS = {'camera+flag': 'camera_flag', 'camera+flag+time_to_park': 'camera_flag_ttp',
           'camera+time_to_park': 'camera_ttp_noflag_0.35s'}   # PARK at 0.35 s or less

def angle(a):
    return float(np.rad2deg(abs((a + np.pi) % (2 * np.pi) - np.pi)))

tests = []
for name, folder in RESULTS.items():
    for b, x in ref.items():
        run = json.loads(Path(f'camera_loop/{folder}/run_{b}.json').read_text()); log = run['log']
        park_t = next((l['t'] for l in log if l['park_prob'] >= 0.5), None)
        tests.append(dict(model=name, bearing=int(b), split=x['split'], seen_at_start=log[0]['visible'], parked=run['parked'],
                          final_error_cm=100 * np.hypot(*np.subtract(run['final_base'], x['final_base_xy'])),
                          final_heading_error_deg=angle(run['final_yaw'] - x['final_yaw']),
                          park_time_vs_expert_s=None if park_t is None else park_t - x['park_time_s'],
                          dot_distance_error_cm=100 * np.mean([abs(l['est_dist'] - l['true_dist']) for l in log if l['visible']] or [np.nan])))
tests = pd.DataFrame(tests)
true_dot = closed[closed.model == 'wheels+live_dot'].set_index('bearing_deg')
tests['true_dot_parked'] = tests.bearing.map(true_dot.parked)

def summary(g):
    return pd.Series({'runs': len(g), 'parked': int(g.parked.sum()), 'true-dot model parked': int(g.true_dot_parked.sum()),
                      'median final error (cm)': round(g.final_error_cm.median(), 1)})
by = {'all': tests, 'dot seen at start': tests[tests.seen_at_start], 'dot not seen at start': tests[~tests.seen_at_start]}
pd.concat({k: v.groupby('model', sort=False).apply(summary) for k, v in by.items()}).unstack(0).swaplevel(0, 1, axis=1).sort_index(axis=1, level=0, sort_remaining=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="4" halign="left">all</th>
      <th colspan="4" halign="left">dot not seen at start</th>
      <th colspan="4" halign="left">dot seen at start</th>
    </tr>
    <tr>
      <th></th>
      <th>runs</th>
      <th>parked</th>
      <th>true-dot model parked</th>
      <th>median final error (cm)</th>
      <th>runs</th>
      <th>parked</th>
      <th>true-dot model parked</th>
      <th>median final error (cm)</th>
      <th>runs</th>
      <th>parked</th>
      <th>true-dot model parked</th>
      <th>median final error (cm)</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>camera+flag</th>
      <td>25.0</td>
      <td>17.0</td>
      <td>23.0</td>
      <td>16.7</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>12.0</td>
      <td>15.1</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>22.4</td>
    </tr>
    <tr>
      <th>camera+flag+time_to_park</th>
      <td>25.0</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>14.5</td>
      <td>14.0</td>
      <td>9.0</td>
      <td>12.0</td>
      <td>14.9</td>
      <td>11.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>14.5</td>
    </tr>
    <tr>
      <th>camera+time_to_park</th>
      <td>25.0</td>
      <td>6.0</td>
      <td>23.0</td>
      <td>44.8</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>50.6</td>
      <td>11.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>19.9</td>
    </tr>
  </tbody>
</table>
</div>



### Every bearing: parked or not


```python
tests.pivot_table(index=['split', 'bearing', 'seen_at_start'], columns='model', values='parked', aggfunc='first', sort=False)\
     .reindex(columns=list(RESULTS)).join(true_dot.parked.rename('wheels+live_dot (true dot)'), on='bearing')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th></th>
      <th>camera+flag</th>
      <th>camera+flag+time_to_park</th>
      <th>camera+time_to_park</th>
      <th>wheels+live_dot (true dot)</th>
    </tr>
    <tr>
      <th>split</th>
      <th>bearing</th>
      <th>seen_at_start</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="21" valign="top">train</th>
      <th>12</th>
      <th>True</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36</th>
      <th>True</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>60</th>
      <th>True</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>72</th>
      <th>True</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>84</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>120</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>156</th>
      <th>False</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>168</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>180</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>192</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>216</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>228</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>240</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>252</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>264</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>288</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>300</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>312</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>336</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>348</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>360</th>
      <th>True</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">test</th>
      <th>24</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>108</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>132</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>204</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



### Every bearing: final position error (cm) and camera distance error when the dot is seen (cm)


```python
pd.concat({'final error (cm)': tests.pivot_table(index=['split', 'bearing'], columns='model', values='final_error_cm', sort=False),
           'camera distance error (cm)': tests.pivot_table(index=['split', 'bearing'], columns='model', values='dot_distance_error_cm', sort=False)},
          axis=1).round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="3" halign="left">final error (cm)</th>
      <th colspan="3" halign="left">camera distance error (cm)</th>
    </tr>
    <tr>
      <th></th>
      <th>model</th>
      <th>camera+flag</th>
      <th>camera+flag+time_to_park</th>
      <th>camera+time_to_park</th>
      <th>camera+flag</th>
      <th>camera+flag+time_to_park</th>
      <th>camera+time_to_park</th>
    </tr>
    <tr>
      <th>split</th>
      <th>bearing</th>
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
      <th rowspan="21" valign="top">train</th>
      <th>12</th>
      <td>9.9</td>
      <td>2.9</td>
      <td>6.7</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>36</th>
      <td>18.3</td>
      <td>21.7</td>
      <td>23.4</td>
      <td>0.8</td>
      <td>0.9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>34.8</td>
      <td>4.4</td>
      <td>24.2</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>36.2</td>
      <td>30.9</td>
      <td>23.5</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>24.7</td>
      <td>42.2</td>
      <td>49.1</td>
      <td>1.0</td>
      <td>1.1</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>120</th>
      <td>24.1</td>
      <td>22.3</td>
      <td>49.2</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>156</th>
      <td>28.5</td>
      <td>25.1</td>
      <td>55.2</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>168</th>
      <td>15.5</td>
      <td>18.3</td>
      <td>53.5</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>180</th>
      <td>12.4</td>
      <td>9.9</td>
      <td>49.6</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>192</th>
      <td>9.4</td>
      <td>10.8</td>
      <td>53.5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>216</th>
      <td>12.2</td>
      <td>12.3</td>
      <td>53.1</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>228</th>
      <td>14.9</td>
      <td>12.3</td>
      <td>50.1</td>
      <td>0.8</td>
      <td>0.9</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>240</th>
      <td>12.7</td>
      <td>16.2</td>
      <td>48.5</td>
      <td>0.8</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>252</th>
      <td>14.2</td>
      <td>13.7</td>
      <td>44.8</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>264</th>
      <td>16.7</td>
      <td>12.2</td>
      <td>40.7</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>288</th>
      <td>22.4</td>
      <td>14.7</td>
      <td>36.7</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>300</th>
      <td>29.0</td>
      <td>14.5</td>
      <td>18.0</td>
      <td>1.1</td>
      <td>0.9</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>312</th>
      <td>24.0</td>
      <td>14.9</td>
      <td>31.8</td>
      <td>1.0</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>336</th>
      <td>22.4</td>
      <td>11.9</td>
      <td>19.2</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>348</th>
      <td>19.1</td>
      <td>12.0</td>
      <td>19.9</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>360</th>
      <td>2.5</td>
      <td>20.1</td>
      <td>9.7</td>
      <td>0.9</td>
      <td>0.8</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">test</th>
      <th>24</th>
      <td>13.9</td>
      <td>3.1</td>
      <td>8.1</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>108</th>
      <td>15.3</td>
      <td>20.8</td>
      <td>51.4</td>
      <td>0.9</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>132</th>
      <td>27.5</td>
      <td>22.4</td>
      <td>51.1</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>204</th>
      <td>11.3</td>
      <td>11.2</td>
      <td>54.2</td>
      <td>0.9</td>
      <td>0.9</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>



## Test metrics, plotted

Every count is out of 25 bearings: 21 training runs and 4 test runs. Worst cases come first. `camera+flag+time_to_park` is the model to build from.


```python
runs = {}                         # (model, bearing) -> saved test run
for name, folder in RESULTS.items():
    for b in ref:
        runs[name, int(b)] = json.loads(Path(f'camera_loop/{folder}/run_{b}.json').read_text())
# Expert paths, for drawing only (from the verified expert replay).
expert_path = {r['bearing_deg']: np.array([f['base'][:2] for f in r['expert']['frames']])
               for r in json.loads(Path('closed_loop_native/result_wheels_live_dot.json').read_text())['runs']}
MAIN = 'camera+flag+time_to_park'
COLORS = dict(zip(list(RESULTS) + ['wheels+live_dot (true dot)'], ['tab:blue', 'tab:orange', 'tab:green', 'grey']))
bearings = sorted(int(b) for b in ref)
```

### 1. Which bearings park? (green parked, red not, grey background: dot not visible at the start)


```python
fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(projection='polar')
names = list(RESULTS) + ['wheels+live_dot (true dot)']
unseen = [b for b in bearings if not runs[MAIN, b]['log'][0]['visible']]
for b in unseen:
    ax.bar(np.deg2rad(b), len(names) + 1, width=np.deg2rad(12), color='0.9', zorder=0)
for k, name in enumerate(names):
    parked = [true_dot.parked[b] if 'true dot' in name else runs[name, b]['parked'] for b in bearings]
    ax.scatter(np.deg2rad(bearings), np.full(len(bearings), k + 1), c=['tab:green' if p else 'tab:red' for p in parked], s=60, zorder=2)
    ax.text(np.deg2rad(90), k + 1.25, f'{name}: {sum(parked)}/25', fontsize=8, ha='center')
ax.set_yticks([]); ax.set_theta_zero_location('E'); ax.set_title('Parked, by dot bearing (0°: straight ahead, 90°: left)')
for b in [b for b in bearings if ref[str(b)]['split'] == 'test']:
    ax.annotate('test', (np.deg2rad(b), len(names) + 0.6), fontsize=7, ha='center')
plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_69_0.png)
    


### 2. Final position error, worst first (bars: model to build from; markers: the others)


```python
order = tests[tests.model == MAIN].sort_values('final_error_cm', ascending=False)
fig, ax = plt.subplots(figsize=(15, 4))
x = np.arange(len(order))
ax.bar(x, order.final_error_cm, color=['tab:green' if p else 'tab:red' for p in order.parked], alpha=0.7, label=f'{MAIN} (green parked, red not)')
for name in RESULTS:
    if name != MAIN:
        e = tests[tests.model == name].set_index('bearing').loc[order.bearing, 'final_error_cm']
        ax.plot(x, e.values, 'o', color=COLORS[name], label=name)
ax.plot(x, true_dot.loc[order.bearing, 'final_xy_error_m'].values * 100, 'k_', ms=14, mew=2, label='wheels+live_dot (true dot)')
ax.set_xticks(x, [f"{b}°{' T' if s == 'test' else ''}" for b, s in zip(order.bearing, order.split)])
ax.set(ylabel='final distance from expert (cm)', xlabel='bearing (T = test run)',
       title=f'{MAIN}: median {order.final_error_cm.median():.0f} cm, parked {int(order.parked.sum())}/25')
ax.legend(fontsize=8); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_71_0.png)
    


### 3. Heading error and park timing, per bearing


```python
fig, ax = plt.subplots(1, 2, figsize=(15, 4))
for name in RESULTS:
    t = tests[tests.model == name].sort_values('bearing')
    ax[0].plot(t.bearing, t.final_heading_error_deg, 'o-', color=COLORS[name], label=name)
    p = t.dropna(subset=['park_time_vs_expert_s'])
    ax[1].plot(p.bearing, p.park_time_vs_expert_s, 'o', color=COLORS[name], label=f'{name} ({len(p)}/25 parked)')
ax[0].set(xlabel='bearing (°)', ylabel='final heading error vs expert (°)', title='Heading error at the end')
ax[1].axhline(0, color='k', lw=0.8)
ax[1].set(xlabel='bearing (°)', ylabel='model PARK time minus expert PARK time (s)', title='When it parks (only runs that parked)')
for a in ax: a.legend(fontsize=8)
plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_73_0.png)
    


### 4. Paths from above: the 4 worst and 2 best bearings for the model to build from


```python
pick = list(order.bearing[:4]) + list(order.bearing[-2:])
fig, ax = plt.subplots(2, 3, figsize=(15, 9))
for a, b in zip(ax.ravel(), pick):
    e = expert_path[b]; a.plot(e[:, 0], e[:, 1], 'k--', lw=2, label='expert')
    for name in RESULTS:
        m = np.array([l['base'] for l in runs[name, b]['log']])
        a.plot(m[:, 0], m[:, 1], color=COLORS[name], lw=2 if name == MAIN else 1, label=f"{name} ({'parked' if runs[name, b]['parked'] else 'no park'})")
    d = np.deg2rad(b); a.plot(1.1 * np.cos(d), 1.1 * np.sin(d), 'o', color='red', ms=9, label='dot')
    a.plot(0, 0, 'ks', ms=6); a.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2)); a.set_aspect('equal'); a.grid(alpha=0.3)   # same scale in every panel; black square: start
    row = order.set_index('bearing').loc[b]
    a.set_title(f"{b}° ({ref[str(b)]['split']}): {row.final_error_cm:.0f} cm from expert")
ax[0, 0].legend(fontsize=7); plt.tight_layout(); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_75_0.png)
    


### 5. Camera: how often it sees the dot, when it first sees it, and how accurate it is


```python
log = {b: runs[MAIN, b]['log'] for b in bearings}
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
vis = [np.mean([l['visible'] for l in log[b]]) for b in bearings]
first = [next((l['t'] for l in log[b] if l['visible']), np.nan) for b in bearings]
ax[0].bar(bearings, vis, width=8, color='tab:orange'); ax[0].set(xlabel='bearing (°)', ylabel='fraction of decisions with dot seen', ylim=(0, 1.05),
                                                      title=f'Dot seen, mean {np.mean(vis):.0%} of decisions')
ax[1].bar(bearings, first, width=8, color='tab:orange')
ax[1].set(xlabel='bearing (°)', ylabel='time of first sighting (s)', title=f'First sighting: {sum(f == 0 for f in first)}/25 at the start')
dist_err = [100 * (l['est_dist'] - l['true_dist']) for b in bearings for l in log[b] if l['visible']]
dir_err = [(l['est_dir_deg'] - l['true_dir_deg'] + 180) % 360 - 180 for b in bearings for l in log[b] if l['visible']]
ax[2].hist(dist_err, bins=40, color='tab:orange'); ax2 = ax[2].twiny()
ax[2].set(xlabel='distance error when seen (cm)', ylabel='decisions',
          title=f'n={len(dist_err)} sightings: distance {np.mean(np.abs(dist_err)):.2f} cm, direction {np.mean(np.abs(dir_err)):.2f}° mean abs error')
ax2.set_xticks([]); plt.tight_layout(); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_77_0.png)
    


### 6. Time to park: what the network predicts while driving vs the expert's real time left


```python
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
for b in bearings:
    L = log[b]; t = np.array([l['t'] for l in L]); pred = np.array([l['ttp_s'] for l in L])
    ax[0].plot(ref[str(b)]['park_time_s'] - t, pred, color='tab:green' if runs[MAIN, b]['parked'] else 'tab:red', alpha=0.6, lw=1)
ax[0].plot([-3, 3], [-3, 3], 'k:', lw=1); ax[0].axhline(THRESH, color='k', lw=1, ls='--')
ax[0].set(xlim=(3, -3.2), ylim=(-0.6, 3.2), xlabel="expert's time left to park (s); negative means the expert has already parked",
          ylabel='predicted time to park (s)', title=f'{MAIN} while testing (green parked, red not; dashed line: 0.25 s)')
with torch.no_grad():
    for e in test:
        pred = camera_models[MAIN](torch.from_numpy(camera_inputs(e, CAMERA_MODELS[MAIN][0])))[1].numpy() * CLIP
        ax[1].plot(time_to_park(e) * CLIP, pred, '.', ms=4, label=f"test run {e['bearing']}°")
ax[1].plot([0, 3], [0, 3], 'k:', lw=1); ax[1].axhline(THRESH, color='k', lw=1, ls='--')
ax[1].set(xlabel='true time to park (s, clipped at 3)', ylabel='predicted (s)', title='Offline, on the 4 test runs (expert states)')
ax[1].legend(fontsize=8); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_79_0.png)
    


## Iteration 8: scaled wheel angles

### Why: the inputs are on very different scales

The network takes its inputs as they come. The wheel angles keep adding up over a run, while the dot inputs stay between -1 and 1 and the wheel speeds between -3 and 3. The tables below show this on the training data and on two test runs.


```python
import v2_results
INPUT_NAMES = ['angle FR', 'angle FL', 'angle BL', 'angle BR', 'speed FR', 'speed FL', 'speed BL', 'speed BR',
               'mean angle', 'R-L angle', 'mean speed', 'R-L speed', 'dot distance', 'dot sin', 'dot cos', 'visible']
Xtrain = np.concatenate([camera_inputs(e, [0, 1, 2, 3]) for e in train])
pd.DataFrame({'min': Xtrain.min(0), 'max': Xtrain.max(0), 'standard deviation': Xtrain.std(0)}, index=INPUT_NAMES).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>standard deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>angle FR</th>
      <td>-8.110000</td>
      <td>13.36</td>
      <td>4.90</td>
    </tr>
    <tr>
      <th>angle FL</th>
      <td>-7.640000</td>
      <td>13.30</td>
      <td>5.04</td>
    </tr>
    <tr>
      <th>angle BL</th>
      <td>-7.730000</td>
      <td>13.37</td>
      <td>5.08</td>
    </tr>
    <tr>
      <th>angle BR</th>
      <td>-8.190000</td>
      <td>13.41</td>
      <td>4.94</td>
    </tr>
    <tr>
      <th>speed FR</th>
      <td>-3.170000</td>
      <td>3.18</td>
      <td>2.02</td>
    </tr>
    <tr>
      <th>speed FL</th>
      <td>-3.140000</td>
      <td>3.22</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>speed BL</th>
      <td>-3.190000</td>
      <td>3.21</td>
      <td>1.90</td>
    </tr>
    <tr>
      <th>speed BR</th>
      <td>-3.190000</td>
      <td>3.21</td>
      <td>2.04</td>
    </tr>
    <tr>
      <th>mean angle</th>
      <td>-0.000000</td>
      <td>4.76</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>R-L angle</th>
      <td>-17.389999</td>
      <td>17.27</td>
      <td>9.35</td>
    </tr>
    <tr>
      <th>mean speed</th>
      <td>-0.280000</td>
      <td>2.33</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>R-L speed</th>
      <td>-6.090000</td>
      <td>6.07</td>
      <td>3.41</td>
    </tr>
    <tr>
      <th>dot distance</th>
      <td>0.000000</td>
      <td>1.24</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>dot sin</th>
      <td>-0.970000</td>
      <td>0.97</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>dot cos</th>
      <td>0.000000</td>
      <td>1.00</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>visible</th>
      <td>0.000000</td>
      <td>1.00</td>
      <td>0.40</td>
    </tr>
  </tbody>
</table>
</div>



### First and last row of two test runs, with what the expert and `camera+flag+time_to_park` output


```python
w6 = np.load('models/camera_flag_ttp.npz')
def net(w, x):
    h = x
    for layer in ('shared.0', 'shared.2'):
        h = np.maximum(h @ w[layer + '.weight'].T + w[layer + '.bias'], 0)
    return h @ w['wheels.weight'].T + w['wheels.bias'], float((h @ w['park.weight'].T + w['park.bias'])[0]) * float(w['ttp_clip'])

tables = {}
for b in (24, 204):
    e = next(e for e in test if e['bearing'] == b)
    t = np.load(DATA / next(r for r in episodes if r['bearing_deg'] == b)['archive'])['time_s']
    X = camera_inputs(e, [0, 1, 2, 3])
    cols = {}
    for i, label in ((0, 'start'), (len(X) - 1, 'end (expert parks)')):
        wheels, ttp = net(w6, X[i])
        cols[f'{label}, t = {t[i]:.2f} s'] = list(np.round(X[i], 2)) + [
            ', '.join(f'{v:.2f}' for v in e['wheels'][i]), f'{t[-1] - t[i]:.2f}',
            ', '.join(f'{v:.2f}' for v in wheels), f"{ttp:.2f} ({'PARK' if ttp <= THRESH else 'drive'})"]
    tables[f'{b}° test run'] = pd.DataFrame(cols, index=INPUT_NAMES + ['expert wheels FR, FL, BL, BR', 'expert time left (s)',
                                                                  'model wheels FR, FL, BL, BR', 'model time to park (s)'])
pd.concat(tables, axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">24° test run</th>
      <th colspan="2" halign="left">204° test run</th>
    </tr>
    <tr>
      <th></th>
      <th>start, t = 0.00 s</th>
      <th>end (expert parks), t = 3.35 s</th>
      <th>start, t = 0.00 s</th>
      <th>end (expert parks), t = 6.11 s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>angle FR</th>
      <td>0.0</td>
      <td>4.77</td>
      <td>0.0</td>
      <td>-2.25</td>
    </tr>
    <tr>
      <th>angle FL</th>
      <td>0.0</td>
      <td>2.29</td>
      <td>0.0</td>
      <td>11.69</td>
    </tr>
    <tr>
      <th>angle BL</th>
      <td>0.0</td>
      <td>2.29</td>
      <td>0.0</td>
      <td>11.67</td>
    </tr>
    <tr>
      <th>angle BR</th>
      <td>0.0</td>
      <td>4.73</td>
      <td>0.0</td>
      <td>-2.29</td>
    </tr>
    <tr>
      <th>speed FR</th>
      <td>0.0</td>
      <td>-0.39</td>
      <td>0.0</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>speed FL</th>
      <td>0.0</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>-0.44</td>
    </tr>
    <tr>
      <th>speed BL</th>
      <td>0.0</td>
      <td>0.41</td>
      <td>0.0</td>
      <td>-0.43</td>
    </tr>
    <tr>
      <th>speed BR</th>
      <td>0.0</td>
      <td>-0.39</td>
      <td>0.0</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th>mean angle</th>
      <td>0.0</td>
      <td>3.52</td>
      <td>0.0</td>
      <td>4.71</td>
    </tr>
    <tr>
      <th>R-L angle</th>
      <td>0.0</td>
      <td>2.46</td>
      <td>0.0</td>
      <td>-13.95</td>
    </tr>
    <tr>
      <th>mean speed</th>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>R-L speed</th>
      <td>0.0</td>
      <td>-0.8</td>
      <td>0.0</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>dot distance</th>
      <td>1.11</td>
      <td>0.73</td>
      <td>0.0</td>
      <td>0.69</td>
    </tr>
    <tr>
      <th>dot sin</th>
      <td>0.41</td>
      <td>-0.03</td>
      <td>0.0</td>
      <td>-0.05</td>
    </tr>
    <tr>
      <th>dot cos</th>
      <td>0.91</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>visible</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>expert wheels FR, FL, BL, BR</th>
      <td>3.00, -3.00, -3.00, 3.00</td>
      <td>0.00, 0.00, 0.00, 0.00</td>
      <td>-3.00, 3.00, 3.00, -3.00</td>
      <td>0.00, 0.00, 0.00, 0.00</td>
    </tr>
    <tr>
      <th>expert time left (s)</th>
      <td>3.35</td>
      <td>0.00</td>
      <td>6.11</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>model wheels FR, FL, BL, BR</th>
      <td>0.74, -0.44, -0.50, 0.81</td>
      <td>-0.46, 0.38, 0.35, -0.43</td>
      <td>-0.52, 0.63, 0.52, -0.44</td>
      <td>0.22, -0.28, -0.32, 0.29</td>
    </tr>
    <tr>
      <th>model time to park (s)</th>
      <td>2.36 (drive)</td>
      <td>0.02 (PARK)</td>
      <td>2.44 (drive)</td>
      <td>0.09 (PARK)</td>
    </tr>
  </tbody>
</table>
</div>



Reading the tables:

- The wheel angles reach about 14 at 204°, and their standard deviation over the training data is about 5. The dot inputs never leave -1 to 1 (distance up to about 1.2 m).
- At the start the model turns the right way but gives about 0.6 rad/s where the expert gives 3.
- At the end the model parks on the expert's states (time to park 0.02 s and 0.09 s).

### What changed

The six angle inputs (4 wheel angles, mean angle, right minus left angle) are standardised: the training-set mean is subtracted and the result divided by the training-set standard deviation. Nothing else changes: same inputs, outputs, loss, seed 7 and 300 epochs. The scaling is folded into the first layer when the model is saved, so the saved network takes raw inputs and runs in the same test loop. `train_v2.py` trains it; without `--scale-angles` it reproduces `camera+flag+time_to_park` exactly.


```python
h6 = camera_history[camera_history.model == 'camera+flag+time_to_park']
h8 = v2_results.history('camera_flag_ttp_scaled')
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for label, h, park_col, c in (('6: camera+flag+time_to_park', h6, 'park_loss', 'tab:orange'), ('8: + scaled angles', h8, 'time_to_park_mse', 'tab:purple')):
    ax[0].plot(h.epoch, h.train_wheel_mse, color=c, label=f'{label} train'); ax[0].plot(h.epoch, h.test_wheel_mse, '--', color=c, label=f'{label} test')
    ax[1].plot(h.epoch, h['train_' + park_col], color=c, label=f'{label} train'); ax[1].plot(h.epoch, h['test_' + park_col], '--', color=c, label=f'{label} test')
ax[0].set(title='Wheel commands', ylabel='MSE (rad/s)²'); ax[1].set(title='Time to park', ylabel='MSE of (seconds left / 3 s)')
for a in ax: a.set(xlabel='epoch', yscale='log'); a.legend(fontsize=8)
plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_85_0.png)
    


### Testing: 25 bearings, compared with iteration 6


```python
iter8 = pd.concat([v2_results.tests('camera_flag_ttp', '6: camera+flag+time_to_park'), v2_results.tests('camera_flag_ttp_scaled', '8: + scaled angles')])
v2_results.summary(iter8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parked (of 25)</th>
      <th>dot in view at start (of 11)</th>
      <th>dot out of view at start (of 14)</th>
      <th>test runs (of 4)</th>
      <th>median final error (cm)</th>
      <th>median heading error (°)</th>
    </tr>
    <tr>
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
      <th>6: camera+flag+time_to_park</th>
      <td>16</td>
      <td>7</td>
      <td>9</td>
      <td>2</td>
      <td>14.5</td>
      <td>17.8</td>
    </tr>
    <tr>
      <th>8: + scaled angles</th>
      <td>14</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>19.9</td>
      <td>26.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
iter8.pivot(index=['split', 'bearing', 'seen_at_start'], columns='model', values='parked')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>model</th>
      <th>6: camera+flag+time_to_park</th>
      <th>8: + scaled angles</th>
    </tr>
    <tr>
      <th>split</th>
      <th>bearing</th>
      <th>seen_at_start</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">test</th>
      <th>24</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>108</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>132</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>204</th>
      <th>False</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th rowspan="21" valign="top">train</th>
      <th>12</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>36</th>
      <th>True</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>60</th>
      <th>True</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>72</th>
      <th>True</th>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>84</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>120</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>156</th>
      <th>False</th>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>168</th>
      <th>False</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>180</th>
      <th>False</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>192</th>
      <th>False</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>216</th>
      <th>False</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>228</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>240</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>252</th>
      <th>False</th>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>264</th>
      <th>False</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>288</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>300</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>312</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>336</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>348</th>
      <th>True</th>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>360</th>
      <th>True</th>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



Scaling fixed every bearing where the dot starts in view (11 of 11, up from 7) but made the search for a dot behind the robot worse (3 of 14, down from 9). The total fell from 16 to 14 of 25. One seed is not enough to tell how much of this is the change and how much is chance, which is what iteration 9 measures.

## Iterations 9 to 11: the same model with three new seeds

### What the seed changes

Iteration 8 trained once, with seed 7. The seed only sets the starting weights. Iterations 9, 10 and 11 train the same model on the same data with seeds 1, 2 and 3, and test each on the same 25 bearings. This shows how much of any result is chance.


```python
SEEDS = {'seed 7 (iteration 8)': 'camera_flag_ttp_scaled', 'seed 1 (iteration 9)': 'camera_flag_ttp_scaled_seed1',
         'seed 2 (iteration 10)': 'camera_flag_ttp_scaled_seed2', 'seed 3 (iteration 11)': 'camera_flag_ttp_scaled_seed3'}
seed_tests = pd.concat([v2_results.tests(f, s) for s, f in SEEDS.items()])
seed_table = v2_results.summary(seed_tests)
seed_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parked (of 25)</th>
      <th>dot in view at start (of 11)</th>
      <th>dot out of view at start (of 14)</th>
      <th>test runs (of 4)</th>
      <th>median final error (cm)</th>
      <th>median heading error (°)</th>
    </tr>
    <tr>
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
      <th>seed 7 (iteration 8)</th>
      <td>14</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>19.9</td>
      <td>26.7</td>
    </tr>
    <tr>
      <th>seed 1 (iteration 9)</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12.5</td>
      <td>21.7</td>
    </tr>
    <tr>
      <th>seed 2 (iteration 10)</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>44.3</td>
      <td>15.5</td>
    </tr>
    <tr>
      <th>seed 3 (iteration 11)</th>
      <td>21</td>
      <td>8</td>
      <td>13</td>
      <td>4</td>
      <td>20.6</td>
      <td>3.6</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for (label, f), c in zip(SEEDS.items(), ['tab:purple', 'tab:blue', 'tab:green', 'tab:red']):
    h = v2_results.history(f)
    ax[0].plot(h.epoch, h.test_wheel_mse, color=c, label=label); ax[1].plot(h.epoch, h.test_time_to_park_mse, color=c, label=label)
    mins = [min(l['ttp_s'] for l in json.loads(Path(f'camera_loop/{f}/run_{b}.json').read_text())['log']) for b in ref]
    ax[2].hist(mins, bins=np.arange(-1.6, 2.8, 0.2), histtype='step', lw=2, color=c, label=label)
ax[0].set(title='Wheel loss on the 4 test runs', xlabel='epoch', yscale='log', ylabel='MSE (rad/s)²')
ax[1].set(title='Time to park loss on the 4 test runs', xlabel='epoch', yscale='log', ylabel='MSE of (seconds left / 3 s)')
ax[2].axvline(THRESH, color='k', ls='--', label='PARK threshold 0.25 s')
ax[2].set(title='Lowest predicted time to park in each test drive', xlabel='seconds', ylabel='runs (of 25)')
for a in ax: a.legend(fontsize=8)
plt.tight_layout(); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_92_0.png)
    



```python
pd.DataFrame({'ended within 20 cm of the expert (of 25)': seed_tests.groupby('model', sort=False).final_error_cm.apply(lambda e: int((e < 20).sum())),
              'parked (of 25)': seed_table['parked (of 25)']})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ended within 20 cm of the expert (of 25)</th>
      <th>parked (of 25)</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>seed 7 (iteration 8)</th>
      <td>13</td>
      <td>14</td>
    </tr>
    <tr>
      <th>seed 1 (iteration 9)</th>
      <td>19</td>
      <td>0</td>
    </tr>
    <tr>
      <th>seed 2 (iteration 10)</th>
      <td>8</td>
      <td>4</td>
    </tr>
    <tr>
      <th>seed 3 (iteration 11)</th>
      <td>12</td>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>



### What the seeds show

- Parked counts run from 0 to 21 of 25 with nothing changed but the seed. A difference of a few runs between two single-seed results means nothing.
- The losses on the test runs are almost the same for all four seeds. Training looks equally good offline.
- The driving is better than the parked count suggests. Seed 1 ends within 20 cm of the expert in 19 of 25 runs and parks in none: its predicted time to park never drops below 0.35 s, so PARK never fires.
- So the fixed 0.25 s threshold decides most of the outcome. The model has never seen what being parked looks like, because every expert run stops at the PARK row. Iteration 12 adds that.

## Iteration 12: 100 rows after the expert parks

Every expert run stops at its PARK row, so the network never sees what being parked looks like. Iteration 12 adds 100 rows (3.3 s) after PARK to every training run. In those rows the wheel command is 0 and the time to park is 0.

The rows come from replaying the expert in the fast simulator past its PARK command (`policy_replay` with `"expert_tail_steps": 3400`), with the wheels stopped and the arm held. Before PARK the replay matches the saved expert data to 3e-8. The camera inputs for the new rows come from the same camera, detector and tracker (`expert_tail.py`). Everything else is iteration 8. Because the seeds vary so much, it is trained and tested with all four seeds (7, 1, 2, 3).


```python
tail = np.load('park_tail/tail_24.npz')
print(f"rows per run: {len(tail['x'])}, from {tail['time_since_park_s'][0]:.2f} s to {tail['time_since_park_s'][-1]:.2f} s after PARK")
pd.DataFrame({'largest wheel speed (rad/s)': [float(np.abs(np.load(f)['x'][:, 4:8]).max()) for f in sorted(Path('park_tail').glob('tail_*.npz'))],
              'dot visible (fraction of rows)': [float(np.load(f)['camera'][:, 3].mean()) for f in sorted(Path('park_tail').glob('tail_*.npz'))]},
             index=[f.stem for f in sorted(Path('park_tail').glob('tail_*.npz'))]).describe().loc[['mean', 'min', 'max']].round(2)
```

    rows per run: 100, from 0.01 s to 3.31 s after PARK





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>largest wheel speed (rad/s)</th>
      <th>dot visible (fraction of rows)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>0.20</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.06</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.39</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



### Iteration 12: results over four seeds


```python
pairs = {s: (f'camera_flag_ttp_scaled{"" if s == 7 else f"_seed{s}"}', f'camera_flag_ttp_scaled_tail100{"" if s == 7 else f"_seed{s}"}') for s in (7, 1, 2, 3)}
rows = []
for s, (before, after) in pairs.items():
    for label, f in (('8 to 11: scaled', before), ('12: + parked rows', after)):
        d = v2_results.tests(f)
        mins = [min(l['ttp_s'] for l in json.loads(Path(f'camera_loop/{f}/run_{b}.json').read_text())['log']) for b in ref]
        rows.append({'seed': s, 'model': label, 'parked (of 25)': int(d.parked.sum()),
                     'dot in view at start (of 11)': int(d[d.seen_at_start].parked.sum()), 'dot out of view at start (of 14)': int(d[~d.seen_at_start].parked.sum()),
                     'ended within 20 cm (of 25)': int((d.final_error_cm < 20).sum()), 'median final error (cm)': round(d.final_error_cm.median(), 1),
                     'median lowest time to park (s)': round(float(np.median(mins)), 2)})
iter12 = pd.DataFrame(rows).set_index(['seed', 'model'])
iter12
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>parked (of 25)</th>
      <th>dot in view at start (of 11)</th>
      <th>dot out of view at start (of 14)</th>
      <th>ended within 20 cm (of 25)</th>
      <th>median final error (cm)</th>
      <th>median lowest time to park (s)</th>
    </tr>
    <tr>
      <th>seed</th>
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
      <th rowspan="2" valign="top">7</th>
      <th>8 to 11: scaled</th>
      <td>14</td>
      <td>11</td>
      <td>3</td>
      <td>13</td>
      <td>19.9</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>12: + parked rows</th>
      <td>9</td>
      <td>2</td>
      <td>7</td>
      <td>22</td>
      <td>11.2</td>
      <td>0.47</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>8 to 11: scaled</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>12.5</td>
      <td>0.68</td>
    </tr>
    <tr>
      <th>12: + parked rows</th>
      <td>12</td>
      <td>2</td>
      <td>10</td>
      <td>19</td>
      <td>13.0</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>8 to 11: scaled</th>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>8</td>
      <td>44.3</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>12: + parked rows</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>28.9</td>
      <td>1.43</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>8 to 11: scaled</th>
      <td>21</td>
      <td>8</td>
      <td>13</td>
      <td>12</td>
      <td>20.6</td>
      <td>-0.10</td>
    </tr>
    <tr>
      <th>12: + parked rows</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>39.1</td>
      <td>2.40</td>
    </tr>
  </tbody>
</table>
</div>




```python
iter12.groupby('model').agg(['mean', 'min', 'max'])[['parked (of 25)', 'ended within 20 cm (of 25)']].round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">parked (of 25)</th>
      <th colspan="3" halign="left">ended within 20 cm (of 25)</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
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
      <th>12: + parked rows</th>
      <td>5.75</td>
      <td>0</td>
      <td>12</td>
      <td>13.25</td>
      <td>3</td>
      <td>22</td>
    </tr>
    <tr>
      <th>8 to 11: scaled</th>
      <td>9.75</td>
      <td>0</td>
      <td>21</td>
      <td>13.00</td>
      <td>8</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



Reading the results:

- Averaged over the four seeds, iteration 12 parked 5.75 of 25 against 9.75 before. The seeds still range from 0 to 12.
- Runs ending within 20 cm of the expert hardly changed (13.25 against 13 on average).
- The training losses are not comparable with iteration 8, because a third of the rows are now easy parked rows.
- The parked rows did not make PARK reliable. The bigger problem is that each seed learns a different time-to-park scale, and one fixed threshold cannot suit them all.

## Iteration 13: wheel encoders instead of simulator wheel values

### Iteration 13: what changed

Until now the 12 wheel inputs came straight from the simulator's joint angles and joint speeds. Iteration 13 reads them from simulated wheel encoders (`sensors.py`):

- Each wheel has an incremental encoder with 4096 counts per revolution (an assumed, common 12-bit resolution). The angle moves in steps of 2π/4096 rad.
- Speed is not measured directly. It is the change in counted angle since the previous reading, divided by 1/30 s, so it lags the true speed.
- The 12 inputs are built from these readings exactly as before, and the tracker uses the encoder speeds when the dot is hidden.

The network now sees only sensor readings: wheel encoders and the camera. The outputs are unchanged (four wheel speed commands and time to park). Everything else is iteration 12, trained and tested with the same four seeds.

`sensors.py` collects the sensors in one `Sensors` class. An IMU can be added there with a `reset()` and a `read(frame)`, and its readings added to the inputs.


```python
import sensors
e24 = next(e for e in test if e['bearing'] == 24)
a24 = np.load(DATA / next(r for r in episodes if r['bearing_deg'] == 24)['archive'])
enc = sensors.WheelEncoders.features_for_run(a24['joint_positions'][:, :4])
t24 = a24['time_s']
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
ax[0].plot(t24, e24['x'][:, 4], label='simulator speed FR'); ax[0].plot(t24, enc[:, 4], '--', label='encoder speed FR')
ax[0].set(xlabel='time (s)', ylabel='rad/s', title='24° test run: front right wheel speed input'); ax[0].legend()
ax[1].plot(t24, e24['x'][:, 0] - enc[:, 0])
ax[1].set(xlabel='time (s)', ylabel='rad', title='Angle input: simulator minus encoder (one count is 0.0015 rad)')
plt.tight_layout(); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_102_0.png)
    


### Iteration 13: results over four seeds


```python
rows = []
for s in (7, 1, 2, 3):
    suffix = '' if s == 7 else f'_seed{s}'
    for label, f in (('12: simulator wheel values', 'camera_flag_ttp_scaled_tail100' + suffix), ('13: wheel encoders', 'camera_flag_ttp_scaled_tail100_encoders' + suffix)):
        d = v2_results.tests(f)
        rows.append({'seed': s, 'model': label, 'parked (of 25)': int(d.parked.sum()),
                     'dot in view at start (of 11)': int(d[d.seen_at_start].parked.sum()), 'dot out of view at start (of 14)': int(d[~d.seen_at_start].parked.sum()),
                     'test runs (of 4)': int(d[d.split == 'test'].parked.sum()),
                     'ended within 20 cm (of 25)': int((d.final_error_cm < 20).sum()), 'median final error (cm)': round(d.final_error_cm.median(), 1)})
iter13 = pd.DataFrame(rows).set_index(['seed', 'model'])
iter13
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>parked (of 25)</th>
      <th>dot in view at start (of 11)</th>
      <th>dot out of view at start (of 14)</th>
      <th>test runs (of 4)</th>
      <th>ended within 20 cm (of 25)</th>
      <th>median final error (cm)</th>
    </tr>
    <tr>
      <th>seed</th>
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
      <th rowspan="2" valign="top">7</th>
      <th>12: simulator wheel values</th>
      <td>9</td>
      <td>2</td>
      <td>7</td>
      <td>1</td>
      <td>22</td>
      <td>11.2</td>
    </tr>
    <tr>
      <th>13: wheel encoders</th>
      <td>24</td>
      <td>10</td>
      <td>14</td>
      <td>4</td>
      <td>24</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">1</th>
      <th>12: simulator wheel values</th>
      <td>12</td>
      <td>2</td>
      <td>10</td>
      <td>2</td>
      <td>19</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>13: wheel encoders</th>
      <td>22</td>
      <td>10</td>
      <td>12</td>
      <td>3</td>
      <td>14</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>12: simulator wheel values</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>9</td>
      <td>28.9</td>
    </tr>
    <tr>
      <th>13: wheel encoders</th>
      <td>11</td>
      <td>8</td>
      <td>3</td>
      <td>1</td>
      <td>14</td>
      <td>17.1</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">3</th>
      <th>12: simulator wheel values</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>39.1</td>
    </tr>
    <tr>
      <th>13: wheel encoders</th>
      <td>15</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>14</td>
      <td>17.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
iter13.groupby('model').agg(['mean', 'min', 'max'])[['parked (of 25)', 'ended within 20 cm (of 25)', 'median final error (cm)']].round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">parked (of 25)</th>
      <th colspan="3" halign="left">ended within 20 cm (of 25)</th>
      <th colspan="3" halign="left">median final error (cm)</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>12: simulator wheel values</th>
      <td>5.8</td>
      <td>0</td>
      <td>12</td>
      <td>13.2</td>
      <td>3</td>
      <td>22</td>
      <td>23.0</td>
      <td>11.2</td>
      <td>39.1</td>
    </tr>
    <tr>
      <th>13: wheel encoders</th>
      <td>18.0</td>
      <td>11</td>
      <td>24</td>
      <td>16.5</td>
      <td>14</td>
      <td>24</td>
      <td>15.1</td>
      <td>7.4</td>
      <td>18.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
best = v2_results.tests('camera_flag_ttp_scaled_tail100_encoders', '13: wheel encoders, seed 7')
best[['split', 'bearing', 'seen_at_start', 'parked', 'final_error_cm', 'heading_error_deg', 'park_time_vs_expert_s']].sort_values('final_error_cm', ascending=False).round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>split</th>
      <th>bearing</th>
      <th>seen_at_start</th>
      <th>parked</th>
      <th>final_error_cm</th>
      <th>heading_error_deg</th>
      <th>park_time_vs_expert_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>12</td>
      <td>True</td>
      <td>False</td>
      <td>25.38</td>
      <td>23.98</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train</td>
      <td>60</td>
      <td>True</td>
      <td>True</td>
      <td>15.49</td>
      <td>11.44</td>
      <td>-0.44</td>
    </tr>
    <tr>
      <th>10</th>
      <td>train</td>
      <td>168</td>
      <td>False</td>
      <td>True</td>
      <td>10.16</td>
      <td>7.20</td>
      <td>-0.60</td>
    </tr>
    <tr>
      <th>9</th>
      <td>train</td>
      <td>156</td>
      <td>False</td>
      <td>True</td>
      <td>9.50</td>
      <td>7.06</td>
      <td>-0.36</td>
    </tr>
    <tr>
      <th>12</th>
      <td>train</td>
      <td>192</td>
      <td>False</td>
      <td>True</td>
      <td>9.20</td>
      <td>5.01</td>
      <td>-0.90</td>
    </tr>
    <tr>
      <th>22</th>
      <td>train</td>
      <td>336</td>
      <td>True</td>
      <td>True</td>
      <td>9.00</td>
      <td>9.17</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>14</th>
      <td>train</td>
      <td>216</td>
      <td>False</td>
      <td>True</td>
      <td>8.93</td>
      <td>7.58</td>
      <td>-0.98</td>
    </tr>
    <tr>
      <th>23</th>
      <td>train</td>
      <td>348</td>
      <td>True</td>
      <td>True</td>
      <td>8.73</td>
      <td>17.57</td>
      <td>-0.96</td>
    </tr>
    <tr>
      <th>13</th>
      <td>test</td>
      <td>204</td>
      <td>False</td>
      <td>True</td>
      <td>8.68</td>
      <td>8.25</td>
      <td>-0.98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>test</td>
      <td>132</td>
      <td>False</td>
      <td>True</td>
      <td>8.31</td>
      <td>5.77</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>11</th>
      <td>train</td>
      <td>180</td>
      <td>False</td>
      <td>True</td>
      <td>7.49</td>
      <td>1.86</td>
      <td>-0.68</td>
    </tr>
    <tr>
      <th>7</th>
      <td>train</td>
      <td>120</td>
      <td>False</td>
      <td>True</td>
      <td>7.43</td>
      <td>6.51</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>15</th>
      <td>train</td>
      <td>228</td>
      <td>False</td>
      <td>True</td>
      <td>7.42</td>
      <td>11.68</td>
      <td>-0.92</td>
    </tr>
    <tr>
      <th>16</th>
      <td>train</td>
      <td>240</td>
      <td>False</td>
      <td>True</td>
      <td>7.27</td>
      <td>9.30</td>
      <td>-0.82</td>
    </tr>
    <tr>
      <th>5</th>
      <td>train</td>
      <td>84</td>
      <td>False</td>
      <td>True</td>
      <td>6.58</td>
      <td>36.07</td>
      <td>2.96</td>
    </tr>
    <tr>
      <th>17</th>
      <td>train</td>
      <td>252</td>
      <td>False</td>
      <td>True</td>
      <td>6.32</td>
      <td>11.91</td>
      <td>-0.56</td>
    </tr>
    <tr>
      <th>21</th>
      <td>train</td>
      <td>312</td>
      <td>True</td>
      <td>True</td>
      <td>6.07</td>
      <td>1.57</td>
      <td>-0.31</td>
    </tr>
    <tr>
      <th>6</th>
      <td>test</td>
      <td>108</td>
      <td>False</td>
      <td>True</td>
      <td>5.44</td>
      <td>8.21</td>
      <td>1.22</td>
    </tr>
    <tr>
      <th>24</th>
      <td>train</td>
      <td>360</td>
      <td>True</td>
      <td>True</td>
      <td>5.19</td>
      <td>0.04</td>
      <td>0.76</td>
    </tr>
    <tr>
      <th>18</th>
      <td>train</td>
      <td>264</td>
      <td>False</td>
      <td>True</td>
      <td>5.09</td>
      <td>9.66</td>
      <td>-0.48</td>
    </tr>
    <tr>
      <th>20</th>
      <td>train</td>
      <td>300</td>
      <td>True</td>
      <td>True</td>
      <td>5.03</td>
      <td>1.65</td>
      <td>-0.50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train</td>
      <td>72</td>
      <td>True</td>
      <td>True</td>
      <td>4.55</td>
      <td>6.16</td>
      <td>1.88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train</td>
      <td>36</td>
      <td>True</td>
      <td>True</td>
      <td>4.35</td>
      <td>1.17</td>
      <td>1.85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>test</td>
      <td>24</td>
      <td>True</td>
      <td>True</td>
      <td>3.22</td>
      <td>8.99</td>
      <td>1.85</td>
    </tr>
    <tr>
      <th>19</th>
      <td>train</td>
      <td>288</td>
      <td>True</td>
      <td>True</td>
      <td>2.87</td>
      <td>16.21</td>
      <td>-0.58</td>
    </tr>
  </tbody>
</table>
</div>



Reading the results:

- With encoders the model parked a mean of 18 of 25 over the four seeds (11 to 24), against 5.75 (0 to 12) with simulator wheel values. Every seed improved.
- Seed 7 parks 24 of 25 with a median final error of 7.4 cm, and all 4 test runs park within 9 cm of the expert. The model given the true dot position (iteration 3) parked 23 of 25.
- We do not know yet why encoders help this much. The encoder speed inputs are not smoother row to row than the simulator's (0.156 against 0.166 rad/s mean change). The tracker also switched to encoder speeds, so two things changed together. Testing each separately would show which one matters.
- The seeds still differ (11 to 24 parked), so results should be reported over several seeds.

### Iteration 13: 3D replay of seed 7

Solid robot: the iteration 13 model (seed 7), driving from the camera and wheel encoders. Cyan ghost: the expert. All 25 bearings are in the dropdown, labelled TRAIN or TEST. The panel on the right shows the predicted time to park, the wheel commands and the park and final differences from the expert.

These runs were repeated with the full robot state logged (`FULL_FRAMES=1`, results in `camera_loop_frames/`). The table under the player checks that they end where the runs above ended.

<a href="/files/driving_v2/closed_loop_native/viewer_iter13_seed7.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_v2.build_camera_viewer('camera_flag_ttp_scaled_tail100_encoders', 'iter13_seed7')
lazy_player('/files/driving_v2/closed_loop_native/' + path.name)
```




<div><button onclick="var f=document.getElementById('player95d56d9f');f.src='/files/driving_v2/closed_loop_native/viewer_iter13_seed7.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player95d56d9f" style="display:none;width:100%;height:810px;border:0"></iframe></div>




```python
check = []
for b in v2_results.REF:
    a = json.loads(Path(f'camera_loop/camera_flag_ttp_scaled_tail100_encoders/run_{b}.json').read_text())
    c = json.loads(Path(f'camera_loop_frames/camera_flag_ttp_scaled_tail100_encoders/run_{b}.json').read_text())
    check.append(dict(bearing=int(b), parked_before=a['parked'], parked_now=c['parked'],
                      final_position_difference_mm=1000 * float(np.hypot(*np.subtract(a['final_base'], c['final_base'])))))
pd.DataFrame(check).sort_values('bearing').set_index('bearing').T.round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>bearing</th>
      <th>12</th>
      <th>24</th>
      <th>36</th>
      <th>60</th>
      <th>72</th>
      <th>84</th>
      <th>108</th>
      <th>120</th>
      <th>132</th>
      <th>156</th>
      <th>...</th>
      <th>228</th>
      <th>240</th>
      <th>252</th>
      <th>264</th>
      <th>288</th>
      <th>300</th>
      <th>312</th>
      <th>336</th>
      <th>348</th>
      <th>360</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>parked_before</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>parked_now</th>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>...</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>final_position_difference_mm</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 25 columns</p>
</div>



## Arm: plan and expected results

### What the arm network will do

The expert works in two phases. While it drives, it holds the arm still. Once it has parked, it reaches for the dot with the arm and stops when the gripper is within 1 cm of it. The arm network copies the second phase. It takes over at PARK, from the iteration 13 driving model.

| | Count | Details |
|---|---|---|
| Input: arm joint angles | 7 | arm encoders (`sensors.ArmEncoders`): absolute angle in whole ticks, 3,686,400 per turn on joints 1 to 4 and 2,534,400 on joints 5 to 7 (Kinova datasheet for the Jaco 2 actuators) |
| Input: arm joint speeds | 7 | change in encoder angle over 1/30 s |
| Input: dot x, y and height | 3 | camera and tracker, which now keeps the height |
| Input: dot visible flag | 1 | camera |
| Input: time since PARK | 1 | clock |
| Output: arm joint torques | 7 | clipped at ±30 Nm, like the expert's |
| Output: time to stop | 1 | stop when it is 0.25 s or less, latched |

- The network is a 2×64 ReLU MLP. Inputs and each torque are standardised, and the scaling is folded into the weights, as in iteration 8.
- It runs at 30 Hz, like the driving network. Each torque command is held for 33 ms while the physics runs at 1 kHz.
- The fingers do not move, so they are not inputs.
- Until PARK the arm is held by the same hold controller as the expert. After PARK there is no hold controller: the network's torques move the arm and also keep it up after stop. A run ends 0.5 s after stop, or 1 s after the expert's stop time.
- Training rows run from the expert's PARK to 0.5 s after the expert's stop, with time to stop 0 after the stop. There are 21 training runs, one per training bearing.
- The loss is the one used for driving: mean squared error on the outputs with equal weight per run, plus the weighted time-to-stop term. Results are also reported as the number of bearings where the gripper ends within 1 cm of the dot.
- Every version is trained with four seeds (7, 1, 2 and 3).

### Smoke tests before training

| Test | What it checks | Expected result |
|---|---|---|
| 1. Iteration 13, seed 7, through the changed simulator | the change does not affect driving | the same final positions, to the millimetre |
| 2. The expert's own 1 kHz arm torques sent through the new arm input | the arm input is wired correctly | the same reach as the expert, gripper within 1 cm |
| 3. The expert's torques averaged over each 33 ms and held | 30 Hz is fast enough for the arm | gripper within 1 cm on most bearings; if not, the arm network moves to 100 Hz |
| 5. Dot in the camera view after PARK | where the dot inputs come from during the reach | probably out of view, so the tracker's estimate is used; this is fine because the robot does not move |
| 6. Train on one bearing and test it in the simulator | the inputs carry enough information for one reach | the gripper ends within 1 cm on that bearing |

Test 4, the expert's torques against the real actuator limits, is not run: we take the expert's torques as correct.

### Expected results and what we do if they fail

- The reach should be easier than driving to learn. The dot does not move once the robot has parked, and the arm starts from almost the same pose every time.
- The risk is drift: a small torque error at the shoulder moves the gripper a long way. If the reach drifts, the next steps are the joint torque sensors as 7 more inputs (±0.4 Nm), then extra training runs around the expert's path (DART).
- If the network does not stop, we add more rows after the expert's stop, as iteration 12 did for PARK.
- Later, the arm network will run for the whole run with a parked flag as an input, so the hold controller before PARK is replaced as well.

### Arm: what the expert does

The expert's full runs were recorded for all 30 bearings with the fast simulator's expert collector, every 1 ms, from the start to the expert's stop (`v2_sensors/arm/data/expert_full`, 18 s to collect). Their PARK times match the driving data to the millisecond.

- The expert stops when the gripper is within 1 cm of the dot and more than 8 s have passed since the start (`controller.cpp:91`). So 23 of the 30 runs stop at exactly 8.0 s, however early the gripper got there. The arm network therefore gets the time since the start as an input, as well as the time since PARK.
- On 204° and 228° the expert does not get within 1 cm. It stalls about 11.5 mm from the dot and stops 3 s later. 204° is a test bearing.
- The brief (slide 4) names the outputs `torque_arm` and `wheel_cmds`. The arm network therefore outputs the 7 torques directly. A network that outputs target angles for a fixed PD loop would change the command type.
- The wheels are different. A wheel command is a speed, and the simulator turns it into torque every 1 ms with 30 × (commanded speed − actual speed) (`physics.cpp:106`). This is the simulator's motor model, and it keeps correcting while a 30 Hz command is held. The arm torques go to the joints as they are, with nothing in between.

### Arm: smoke tests

The stepping server (`sensor_parking`) has new options, and without them it behaves as before:

- `arm`: after PARK the client's 7 torques drive the arm and `stop` ends the run 0.5 s later. Until the first arm command after PARK the hold controller keeps the arm.
- `expert_drive`: the expert drives and parks itself, so the arm can be tested on its own.
- `expert_arm_hz`: the expert's own arm controller runs after PARK, but its torque is only taken at this rate and held in between.

| Test | Result |
|---|---|
| 1. Iteration 13, seed 7, through the changed server | the same 24 of 25 parked, final positions identical to the millimetre |
| 2. The expert's own 1 kHz torques sent through the arm input | the expert's reach reproduced exactly on all 30 bearings |
| 3. The expert's torques averaged over each 33 ms and held, without feedback | the gripper ends 0.2 to 1.2 m from the dot |
| 3b. The expert's own controller, its torque taken at a lower rate and held | works only at 1 kHz (table below) |
| 5. Dot in view after PARK | in view on 9 of the 24 parked runs; otherwise the tracker's estimate is within about 1 to 4 cm |

Test 4, the expert's torques against the real actuator limits, is not run, because we take the expert's torques as correct.


```python
ARM = Path('../arm')
t2 = pd.DataFrame(json.loads((ARM / 'smoke/test2_1khz.json').read_text()))
rows = []
for hz in (1000, 500, 250, 100, 30):
    d = pd.DataFrame(json.loads((ARM / f'smoke/test3b_expert_{hz}hz.json').read_text()))
    rows.append({'rate (Hz)': hz, 'matches the expert (of 30)': int((d.at_stop_mm <= np.maximum(10, t2.expert_stop_mm + 1)).sum()),
                 'median distance at the expert stop (mm)': round(d.at_stop_mm.median(), 1),
                 'largest distance (mm)': round(d.at_stop_mm.max(), 1)})
print('Test 2: largest difference from the expert at its stop (mm):', float((t2.at_stop_mm - t2.expert_stop_mm).abs().max()))
pd.DataFrame(rows).set_index('rate (Hz)')
```

    Test 2: largest difference from the expert at its stop (mm): 1.7763568394002505e-15





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>matches the expert (of 30)</th>
      <th>median distance at the expert stop (mm)</th>
      <th>largest distance (mm)</th>
    </tr>
    <tr>
      <th>rate (Hz)</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <td>30</td>
      <td>4.2</td>
      <td>11.7</td>
    </tr>
    <tr>
      <th>500</th>
      <td>0</td>
      <td>65.1</td>
      <td>95.9</td>
    </tr>
    <tr>
      <th>250</th>
      <td>0</td>
      <td>452.1</td>
      <td>629.6</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0</td>
      <td>595.8</td>
      <td>814.4</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>637.8</td>
      <td>55856.2</td>
    </tr>
  </tbody>
</table>
</div>



Test 3b is the main finding. The expert's arm controller has stiff gains (100 on the angle error, 30 on the speed) on light joints. Its torque is right for one millisecond. Held for 2 ms the arm overshoots, and held for 33 ms it swings away from the dot.

### Arm: 30 Hz torque networks (iterations 14 to 16)

#### Iterations 14 to 16: plan and expected results

This entry was added after the runs. The expectation below is the one stated before training.

Three networks output the 7 arm torques once per 30 Hz decision (`v2_sensors/arm/train_arm.py`), to see what happens when torques are held for 33 ms:

| Iteration | Model | Trained on | Torque target |
|---|---|---|---|
| 14 | `arm30_sample_b12` | bearing 12 only | the expert's torque at the decision step |
| 15 | `arm30_sample` | the 21 training bearings | the expert's torque at the decision step |
| 16 | `arm30_mean` | the 21 training bearings | the expert's torque averaged over the 33 ms until the next decision |

- 19 inputs: arm encoder angles (7) and speeds (7), the dot in the robot frame (3, from the simulator for this test, not the camera), time since PARK and time since the start.
- Outputs: the 7 torques and the time to stop. 2×64 ReLU, 5,960 parameters, inputs and torques standardised, Adam with learning rate 0.01, 300 full-batch epochs, seed 7.
- Rows: every 30 Hz decision from the expert's PARK to its stop, about 110 per run.
- Loss: torque mean squared error (in standardised units) plus time-to-stop mean squared error, each training run counted equally.
- Test: in the simulator with `expert_drive`. The expert drives and parks, then only the arm network moves the arm (`v2_sensors/arm/test_arm.py`), on the 25 bearings.
- Iteration 14 checks whether a network that fits one reach almost exactly can repeat it in the simulator.

Expected: all three fail, because test 3b showed the expert's own torques fail when held for 33 ms. A network that copies the expert's torque at each state should fail the same way.


```python
rows = []
for name in ('arm30_sample_b12', 'arm30_sample', 'arm30_mean'):
    h = json.loads((ARM / f'models/{name}_history.json').read_text())[-1]
    for r in json.loads((ARM / f'results/{name}.json').read_text()):
        t0 = r['log'][0]['t']
        rows.append(dict(model=name, split=r['split'], bearing=r['bearing'], train_torque_loss=h['train_torque_mse'],
                         test_torque_loss=np.nan if name.endswith('_b12') else h['test_torque_mse'],
                         at_park_mm=r['log'][0]['ee_mm'], closest_mm=r['min_mm'], at_expert_stop_mm=r['at_expert_stop_mm'],
                         s_until_1m=next((l['t'] - t0 for l in r['log'] if l['ee_mm'] > 1000), np.nan)))
arm30 = pd.DataFrame(rows)
arm30.groupby('model', sort=False).agg(**{
    'runs': ('bearing', 'size'), 'train torque loss': ('train_torque_loss', 'first'), 'test torque loss': ('test_torque_loss', 'first'),
    'within 1 cm at the expert stop': ('at_expert_stop_mm', lambda s: int((s <= 10).sum())),
    'median distance at PARK (mm)': ('at_park_mm', 'median'), 'median closest (mm)': ('closest_mm', 'median'),
    'median time until 1 m away (s)': ('s_until_1m', 'median')}).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>runs</th>
      <th>train torque loss</th>
      <th>test torque loss</th>
      <th>within 1 cm at the expert stop</th>
      <th>median distance at PARK (mm)</th>
      <th>median closest (mm)</th>
      <th>median time until 1 m away (s)</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
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
      <th>arm30_sample_b12</th>
      <td>1</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>0</td>
      <td>224.522</td>
      <td>161.429</td>
      <td>0.533</td>
    </tr>
    <tr>
      <th>arm30_sample</th>
      <td>25</td>
      <td>0.091</td>
      <td>1.456</td>
      <td>0</td>
      <td>227.017</td>
      <td>188.806</td>
      <td>0.534</td>
    </tr>
    <tr>
      <th>arm30_mean</th>
      <td>25</td>
      <td>0.030</td>
      <td>0.062</td>
      <td>0</td>
      <td>227.017</td>
      <td>221.130</td>
      <td>0.467</td>
    </tr>
  </tbody>
</table>
</div>



#### Iterations 14 to 16: loss curves

Torque loss is in standardised units. Iteration 14 is trained on one bearing, so it has no test curve.


```python
def arm_loss_plot(names, title):
    """Per-epoch training (solid) and test (dashed) losses of arm networks, from models/NAME_history.json."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
    for name in names:
        h = pd.DataFrame(json.loads((ARM / f'models/{name}_history.json').read_text()))
        line, = ax[0].plot(h.epoch, h.train_torque_mse, label=f'{name} train')
        stop = 'time_to_stop' if 'train_time_to_stop_mse' in h else 'distance'      # what the stop head learns
        ax[1].plot(h.epoch, h[f'train_{stop}_mse'], color=line.get_color(), label=f'{name} train ({stop})')
        if not name.endswith('_b12'):               # trained on one bearing: no test curve
            ax[0].plot(h.epoch, h.test_torque_mse, '--', color=line.get_color(), label=f'{name} test')
            ax[1].plot(h.epoch, h[f'test_{stop}_mse'], '--', color=line.get_color(), label=f'{name} test ({stop})')
    for a, t in zip(ax, ('torque loss (standardised units)', 'stop loss (scaled 0..1)')):
        a.set(yscale='log', xlabel='epoch', title=t); a.legend(fontsize=8)
    fig.suptitle(title); plt.show()

arm_loss_plot(['arm30_sample_b12', 'arm30_sample', 'arm30_mean'], 'Iterations 14 to 16: 30 Hz torque networks')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_119_0.png)
    



```python
fig, ax = plt.subplots(figsize=(10, 4))
for name in ('arm30_sample_b12', 'arm30_sample', 'arm30_mean'):
    r = next(r for r in json.loads((ARM / f'results/{name}.json').read_text()) if r['bearing'] == 12)
    ax.plot([l['t'] - r['log'][0]['t'] for l in r['log']], [l['ee_mm'] for l in r['log']], label=name)
ax.axhline(10, color='k', ls=':', label='1 cm')
ax.set(yscale='log', xlabel='time since PARK (s)', ylabel='gripper to dot (mm)', title='Bearing 12: 30 Hz torque networks', xlim=(0, 2))
ax.legend(); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_120_0.png)
    


Reading the results:

- No network gets the gripper within 1 cm. The gripper starts about 22.5 cm from the dot at PARK. Each network moves it a few centimetres closer, then the arm swings away and passes 1 m from the dot about 0.5 s after PARK.
- The network trained on bearing 12 alone fits its training rows almost exactly (torque loss 0.0001) and still swings away after 0.53 s. So the failure comes from holding the torques for 33 ms, not from the network learning badly.
- The averaged torques are much easier to learn (test loss 0.06 against 1.46), but they fail in the same way.
- Iteration 15 overfits: its test loss is lowest at epoch 70 (0.87) and rises to 1.46 by epoch 300. Iteration 16 does not: its test loss stays near 0.06.
- This matches test 3b: the expert's own controller fails when its torque is held for 33 ms.

What this means for the arm: to output torques, as the expert does, the arm network has to run at the expert's rate, 1 kHz. The next test is the same network at 1 kHz.

#### Iteration 14: 3D replay (`arm30_sample_b12`)

Solid robot: the arm network's run. Cyan ghost: the expert's own run for the same bearing. The expert drives and parks in both runs, so the two robots overlap until PARK. After PARK the solid arm gets only the network's torques. The side table shows the 7 torques of both. A run is cut once the gripper is more than 1.5 m from the dot after PARK.

The line under the player gives the distance from each gripper to the dot in mm, for the network and the expert. This is the distance the expert itself uses: from its end-effector frame (`robot_j2s7s300_end_effector`) to the dot centre. After PARK the dot is never more than 0.53 m from the shoulder, inside the expert's 0.85 m reach limit, so the expert always aims at the dot centre itself.

<a href="/files/arm/results/viewer_arm30_sample_b12.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
import sys
sys.path.insert(0, str(ARM.resolve()))
import replay_arm
path = replay_arm.build_viewer('arm30_sample_b12')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerb3e88e00');f.src='/files/arm/results/viewer_arm30_sample_b12.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerb3e88e00" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 15: 3D replay (`arm30_sample`)

<a href="/files/arm/results/viewer_arm30_sample.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm30_sample')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('player48598696');f.src='/files/arm/results/viewer_arm30_sample.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player48598696" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 16: 3D replay (`arm30_mean`)

<a href="/files/arm/results/viewer_arm30_mean.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm30_mean')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerdde542c7');f.src='/files/arm/results/viewer_arm30_mean.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerdde542c7" style="display:none;width:100%;height:810px;border:0"></iframe></div>



### Arm: 1 kHz torque network (iteration 17)

#### Iteration 17: plan and expected results

This entry was added after the run. The expectation below is the one stated before training.

The same network as iteration 15 (`arm30_sample`), with one change: it runs every 1 ms, the rate at which the expert outputs its torques (`train_arm.py --hz 1000`).

- Rows: every 1 ms from the expert's PARK to its stop, 75,949 rows over the 21 training runs.
- The encoder speed is the change in angle over 1 ms. The dot position and the times are read every 1 ms too.
- Test: the network's torques are sent every 1 ms through the arm input (`test_arm.py`), on the 25 bearings, with the expert driving.

Expected: the arm should stay stable, because test 3b showed the expert's own torques work at 1 kHz. The risk is drift: the expert's torque depends on a target angle it updates internally every 1 ms (`controller.cpp:81-85`), which the network cannot see, so it has to approximate it from the angles, speeds and times.


```python
rows = []
for name in ('arm30_sample', 'arm1k'):
    h = json.loads((ARM / f'models/{name}_history.json').read_text())[-1]
    for r in json.loads((ARM / f'results/{name}.json').read_text()):
        rows.append(dict(model=name, split=r['split'], bearing=r['bearing'], train=h['train_torque_mse'], test=h['test_torque_mse'],
                         closest_mm=r['min_mm'], at_expert_stop_mm=r['at_expert_stop_mm'], stopped=r['stop_s'] is not None,
                         stop_vs_expert_s=np.nan if r['stop_s'] is None else r['stop_s'] - r['expert_stop_s']))
arm1k = pd.DataFrame(rows)
arm1k.groupby('model', sort=False).agg(**{
    'train torque loss': ('train', 'first'), 'test torque loss': ('test', 'first'),
    'median closest (mm)': ('closest_mm', 'median'), 'closest within 1 cm (of 25)': ('closest_mm', lambda s: int((s <= 10).sum())),
    'median at the expert stop (mm)': ('at_expert_stop_mm', 'median'), 'stopped (of 25)': ('stopped', 'sum'),
    'median stop vs expert (s)': ('stop_vs_expert_s', 'median')}).round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train torque loss</th>
      <th>test torque loss</th>
      <th>median closest (mm)</th>
      <th>closest within 1 cm (of 25)</th>
      <th>median at the expert stop (mm)</th>
      <th>stopped (of 25)</th>
      <th>median stop vs expert (s)</th>
    </tr>
    <tr>
      <th>model</th>
      <th></th>
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
      <th>arm30_sample</th>
      <td>0.091</td>
      <td>1.456</td>
      <td>188.806</td>
      <td>0</td>
      <td>7082.420</td>
      <td>6</td>
      <td>-2.648</td>
    </tr>
    <tr>
      <th>arm1k</th>
      <td>0.078</td>
      <td>0.572</td>
      <td>18.325</td>
      <td>2</td>
      <td>124.103</td>
      <td>22</td>
      <td>-0.969</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 17: loss curves

The 1 kHz network against iteration 15 at 30 Hz, in standardised units.


```python
arm_loss_plot(['arm30_sample', 'arm1k'], 'Iteration 17 (1 kHz) against iteration 15 (30 Hz)')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_131_0.png)
    



```python
fig, ax = plt.subplots(figsize=(10, 4))
for name, style in (('arm30_sample', ':'), ('arm1k', '-')):
    for r in json.loads((ARM / f'results/{name}.json').read_text()):
        if r['bearing'] in (12, 24, 180, 336):
            ax.plot([l['t'] - r['log'][0]['t'] for l in r['log']], [l['ee_mm'] for l in r['log']], style, label=f"{name} {r['bearing']}°")
ax.axhline(10, color='k', lw=0.8, label='1 cm')
ax.set(yscale='log', ylim=(3, 3000), xlabel='time since PARK (s)', ylabel='gripper to dot (mm)', title='30 Hz (dotted) and 1 kHz (solid) torque networks')
ax.legend(ncol=2, fontsize=8); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_132_0.png)
    


#### Iteration 17: 3D replay (`arm1k`)

<a href="/files/arm/results/viewer_arm1k.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerf8908224');f.src='/files/arm/results/viewer_arm1k.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerf8908224" style="display:none;width:100%;height:810px;border:0"></iframe></div>



Reading the results:

- At 1 kHz the arm stays stable. The gripper gets to a median of 18 mm from the dot, against 189 mm at 30 Hz, and within 1 cm on 2 bearings (192° and 336°, both training bearings).
- It does not stay there. After its closest point the gripper drifts away, and at the expert's stop it is a median 124 mm from the dot.
- The network says stop on 22 of 25 bearings, usually early: a median 1.0 s before the expert.
- Test torque loss is 0.57 against 0.08 on training. The test loss is lowest at epoch 120 (0.35) and rises after that, so the network overfits the 21 training reaches. Stopping training earlier is worth testing.
- The likely cause of the drift: the expert's torque depends on a target angle it updates internally every 1 ms (`controller.cpp:81-85`), which the network cannot see. Small torque errors also add up with nothing to correct them.

### Arm: fewer epochs (iteration 18)

#### Iteration 18: plan and expected results

Iteration 17 overfits: its test torque loss is lowest at epoch 120 (0.35) and rises to 0.57 by epoch 300, while the training loss keeps falling. Iteration 18 is the same 1 kHz network trained for fewer epochs.

How the number of epochs is chosen, without using the test bearings:

- The 21 training bearings are already split into 5 folds (`split_plan.json`). For each fold, a network is trained on the other 4 folds for 300 epochs, and its torque loss on the held-out fold is recorded every epoch.
- The number of epochs is the one where the held-out loss, averaged over the 5 folds, is lowest.
- Iteration 18 is then trained on all 21 training bearings for that many epochs. Everything else is as in iteration 17.

Seeds: iteration 18 is trained with seeds 7, 1, 2 and 3. Iteration 17 so far has only seed 7, so it is also trained with seeds 1, 2 and 3 (300 epochs), to compare the two over the same four seeds.

Test: as iteration 17, in the simulator on the 25 bearings with the expert driving.

Expected:

- The chosen number of epochs is between 80 and 200, close to where the iteration 17 test loss is lowest.
- The test torque loss is lower than iteration 17's over the four seeds, about 0.35 against 0.57.
- In the simulator the gripper drifts less after its closest point: the median distance at the expert's stop is below iteration 17's (124 mm with seed 7), and the closest distance is no worse (18 mm).

If the test loss falls but the simulator results do not improve, overfitting is not the main reason for the drift. The more likely reason is then the expert's internal target angle, which the network cannot see.

#### Iteration 18: choosing the number of epochs

For each of the 5 folds of training bearings, a network was trained on the other 4 folds for 300 epochs, and its torque loss on the held-out fold was recorded every epoch. The averaged held-out loss is lowest at epoch 77, so iteration 18 is trained for 77 epochs.


```python
cv = [pd.DataFrame(json.loads((ARM / f'models/arm1k_cv_fold{f}_history.json').read_text())) for f in range(5)]
mean_val = np.mean([h.val_torque_mse for h in cv], 0)
chosen = json.loads((ARM / 'results/cv_epochs_arm1k.json').read_text())['epochs']
fig, ax = plt.subplots(figsize=(10, 4))
for f, h in enumerate(cv):
    ax.plot(h.epoch, h.val_torque_mse, lw=0.8, label=f'held-out fold {f}')
ax.plot(cv[0].epoch, mean_val, 'k', lw=2, label='mean of the 5 folds')
ax.axvline(chosen, color='r', ls=':', label=f'lowest mean: epoch {chosen}')
ax.set(yscale='log', xlabel='epoch', ylabel='held-out torque loss (standardised)', title='Iteration 18: choosing the number of epochs')
ax.legend(fontsize=8); plt.show()
pd.DataFrame([dict(fold=f, best_epoch=int(h.epoch[h.val_torque_mse.idxmin()]),
                   best_loss=h.val_torque_mse.min(), loss_at_300=h.val_torque_mse.iloc[-1]) for f, h in enumerate(cv)]).set_index('fold').round(3)
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_138_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>best_epoch</th>
      <th>best_loss</th>
      <th>loss_at_300</th>
    </tr>
    <tr>
      <th>fold</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>91</td>
      <td>0.040</td>
      <td>0.217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>299</td>
      <td>0.026</td>
      <td>0.026</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77</td>
      <td>0.464</td>
      <td>0.513</td>
    </tr>
    <tr>
      <th>3</th>
      <td>167</td>
      <td>0.028</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>0.571</td>
      <td>13.757</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 18: results over four seeds

Iteration 17 (300 epochs) against iteration 18 (77 epochs), each with seeds 7, 1, 2 and 3, tested in the simulator on the 25 bearings with the expert driving.


```python
NETS = {17: ['arm1k', 'arm1k_seed1', 'arm1k_seed2', 'arm1k_seed3'], 18: ['arm1k_e77', 'arm1k_e77_seed1', 'arm1k_e77_seed2', 'arm1k_e77_seed3']}
rows = []
for it, names in NETS.items():
    for name, seed in zip(names, (7, 1, 2, 3)):
        h = json.loads((ARM / f'models/{name}_history.json').read_text())[-1]
        r = pd.DataFrame(json.loads((ARM / f'results/{name}.json').read_text()))
        rows.append({'iteration': it, 'seed': seed, 'model': name, 'epochs': h['epoch'], 'test torque loss': h['test_torque_mse'],
                     'train time-to-stop loss': h['train_time_to_stop_mse'], 'median closest (mm)': r.min_mm.median(),
                     'closest within 1 cm (of 25)': int((r.min_mm <= 10).sum()), 'median at the expert stop (mm)': r.at_expert_stop_mm.median(),
                     'stopped (of 25)': int(r.stop_s.notna().sum())})
iter18 = pd.DataFrame(rows)
display(iter18.set_index(['iteration', 'seed']).round(3))
iter18.drop(columns=['seed', 'model']).groupby('iteration').mean().round(3)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>model</th>
      <th>epochs</th>
      <th>test torque loss</th>
      <th>train time-to-stop loss</th>
      <th>median closest (mm)</th>
      <th>closest within 1 cm (of 25)</th>
      <th>median at the expert stop (mm)</th>
      <th>stopped (of 25)</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th>seed</th>
      <th></th>
      <th></th>
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
      <th rowspan="4" valign="top">17</th>
      <th>7</th>
      <td>arm1k</td>
      <td>300</td>
      <td>0.572</td>
      <td>0.001</td>
      <td>18.325</td>
      <td>2</td>
      <td>124.103</td>
      <td>22</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arm1k_seed1</td>
      <td>300</td>
      <td>0.482</td>
      <td>0.002</td>
      <td>16.456</td>
      <td>6</td>
      <td>100.728</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arm1k_seed2</td>
      <td>300</td>
      <td>0.811</td>
      <td>0.001</td>
      <td>19.862</td>
      <td>4</td>
      <td>74.951</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arm1k_seed3</td>
      <td>300</td>
      <td>0.694</td>
      <td>0.001</td>
      <td>37.845</td>
      <td>0</td>
      <td>488.239</td>
      <td>22</td>
    </tr>
    <tr>
      <th rowspan="4" valign="top">18</th>
      <th>7</th>
      <td>arm1k_e77</td>
      <td>77</td>
      <td>0.377</td>
      <td>0.004</td>
      <td>32.971</td>
      <td>4</td>
      <td>278.746</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>arm1k_e77_seed1</td>
      <td>77</td>
      <td>0.367</td>
      <td>0.004</td>
      <td>18.923</td>
      <td>3</td>
      <td>209.033</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>arm1k_e77_seed2</td>
      <td>77</td>
      <td>0.377</td>
      <td>0.004</td>
      <td>26.687</td>
      <td>0</td>
      <td>152.882</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>arm1k_e77_seed3</td>
      <td>77</td>
      <td>0.385</td>
      <td>0.003</td>
      <td>17.870</td>
      <td>2</td>
      <td>118.563</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epochs</th>
      <th>test torque loss</th>
      <th>train time-to-stop loss</th>
      <th>median closest (mm)</th>
      <th>closest within 1 cm (of 25)</th>
      <th>median at the expert stop (mm)</th>
      <th>stopped (of 25)</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th></th>
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
      <th>17</th>
      <td>300.0</td>
      <td>0.640</td>
      <td>0.001</td>
      <td>23.122</td>
      <td>3.00</td>
      <td>197.005</td>
      <td>18.25</td>
    </tr>
    <tr>
      <th>18</th>
      <td>77.0</td>
      <td>0.376</td>
      <td>0.004</td>
      <td>24.113</td>
      <td>2.25</td>
      <td>189.806</td>
      <td>7.00</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 18: loss curves

Training (solid) and test (dashed) losses for the four seeds of each iteration, in standardised units.


```python
arm_loss_plot(NETS[17], 'Iteration 17: 300 epochs, four seeds')
arm_loss_plot(NETS[18], 'Iteration 18: 77 epochs, four seeds')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_142_0.png)
    



    
![png](driving_v2_sensors_files/driving_v2_sensors_142_1.png)
    


#### Iteration 18: 3D replays

One player per network: the four seeds of iteration 18, then the three new seeds of iteration 17 (seed 7 is in the iteration 17 section). The line under each player gives the distance from each gripper to the dot in mm.

##### Iteration 18, seed 7 (`arm1k_e77`)

<a href="/files/arm/results/viewer_arm1k_e77.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('player6e811572');f.src='/files/arm/results/viewer_arm1k_e77.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player6e811572" style="display:none;width:100%;height:810px;border:0"></iframe></div>



##### Iteration 18, seed 1 (`arm1k_e77_seed1`)

<a href="/files/arm/results/viewer_arm1k_e77_seed1.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_seed1')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerf65efee6');f.src='/files/arm/results/viewer_arm1k_e77_seed1.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerf65efee6" style="display:none;width:100%;height:810px;border:0"></iframe></div>



##### Iteration 18, seed 2 (`arm1k_e77_seed2`)

<a href="/files/arm/results/viewer_arm1k_e77_seed2.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_seed2')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('player042a5393');f.src='/files/arm/results/viewer_arm1k_e77_seed2.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player042a5393" style="display:none;width:100%;height:810px;border:0"></iframe></div>



##### Iteration 18, seed 3 (`arm1k_e77_seed3`)

<a href="/files/arm/results/viewer_arm1k_e77_seed3.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_seed3')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('player342e2582');f.src='/files/arm/results/viewer_arm1k_e77_seed3.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player342e2582" style="display:none;width:100%;height:810px;border:0"></iframe></div>



##### Iteration 17, seed 1 (`arm1k_seed1`)

<a href="/files/arm/results/viewer_arm1k_seed1.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_seed1')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playera0b600f6');f.src='/files/arm/results/viewer_arm1k_seed1.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playera0b600f6" style="display:none;width:100%;height:810px;border:0"></iframe></div>



##### Iteration 17, seed 2 (`arm1k_seed2`)

<a href="/files/arm/results/viewer_arm1k_seed2.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_seed2')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerf98f4b5f');f.src='/files/arm/results/viewer_arm1k_seed2.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerf98f4b5f" style="display:none;width:100%;height:810px;border:0"></iframe></div>



##### Iteration 17, seed 3 (`arm1k_seed3`)

<a href="/files/arm/results/viewer_arm1k_seed3.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_seed3')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerd2ae263d');f.src='/files/arm/results/viewer_arm1k_seed3.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerd2ae263d" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 18: reading the results

Against the expectation:

- **Number of epochs:** 77, just below the expected 80 to 200. The mean is pulled down mostly by fold 4, whose held-out loss rises from 0.57 at epoch 74 to 13.8 at epoch 300. Folds 1 and 3 keep improving until epochs 299 and 167, so the choice depends heavily on one fold.
- **Test torque loss: as expected.** It falls from a mean of 0.64 to 0.38 over the four seeds, and it is much steadier across seeds (0.37 to 0.39, against 0.48 to 0.81).
- **Simulator: not as expected.** The reach does not improve. Over the four seeds the median closest distance is 24.1 mm against 23.1 mm, the gripper gets within 1 cm on 2.25 bearings against 3, and the median distance at the expert's stop is 190 mm against 197 mm.
- **Stop: worse.** The network says stop on 7 of 25 bearings against 18.25. With fewer epochs the time-to-stop output is fitted less closely (training loss 0.004 against 0.001), so it rarely drops below the 0.25 s threshold.
- The seed still matters more than the number of epochs. With 300 epochs, the median distance at the expert's stop ranges from 75 mm (seed 2) to 488 mm (seed 3).

What it tells us: as the plan said, a lower test loss without a better reach in the simulator means overfitting is not the main reason the arm drifts. The more likely reason is the expert's internal target angle, which the network cannot see. Inputs that let the network infer it, such as its previous torques or a short history of joint angles, are the next thing to test. Fewer epochs is not kept, because it costs the stop.

### Arm: stop on distance (iteration 19)

#### Iteration 19: plan and expected results

In iterations 17 and 18 the arm follows the expert's reach closely, but the network does not know when it has reached the dot. The expert gets within 15 mm of the dot about 2 s after PARK and then holds the arm almost still until its stop rule allows it to stop: within 1 cm and more than 8 s since the start. That hold lasts a median 1.6 s. The network drifts during it, and when it does say stop the gripper is already far away: a median 257 mm for iteration 18, seed 7.

Iteration 19 is iteration 18, seed 7, with one change: the stop head learns the distance from the gripper to the dot instead of the time to stop.

- Stop target: the distance from the expert's end effector to the dot centre at each row, from the expert's recordings, clipped at 0.25 m and scaled to 0..1. The network says stop when its predicted distance is 1 cm or less, latched.
- Everything else is as in iteration 18: the same 19 inputs (arm encoder angles and speeds, the dot position from the simulator, time since PARK and since the start), 2×64 ReLU, 1 kHz, 77 epochs, seed 7, rows from the expert's PARK to its stop.
- The loss is the torque loss plus the distance loss, each training run counted equally.
- Test: the 25 bearings in the simulator with the expert driving, as before.
- New result: the distance from the gripper to the dot when the network says stop. A run ends 0.5 s after stop, so the distance 0.5 s later is also reported.

The network has every input it needs for the distance: the 7 joint angles fix where the gripper is, and the dot position is given. It has to learn that mapping from 21 reaches.

Expected:

- The network says stop on at least 20 of the 25 bearings, against 15 for iteration 18, seed 7.
- When it says stop, the gripper is a median 20 mm or less from the dot, against 257 mm.
- It stops earlier than the expert, about 1.5 s before, because it no longer waits for the expert's 8 s rule.
- The reach itself is unchanged: the median closest distance stays near 33 mm.

The expectation is wrong if the network stops on fewer than 15 bearings or the median distance at stop is above 30 mm. That would mean 21 reaches are not enough to learn the distance from the joint angles.

#### Iteration 19: results

Iteration 18, seed 7 (stop on time) against iteration 19 (stop on distance), on the 25 bearings with the expert driving. A run ends 0.5 s after the network says stop.


```python
rows = []
for it, name, stop in ((18, 'arm1k_e77', 'time'), (19, 'arm1k_e77_dist', 'distance')):
    h = json.loads((ARM / f'models/{name}_history.json').read_text())[-1]
    r = pd.DataFrame(json.loads((ARM / f'results/{name}.json').read_text()))
    if 'at_network_stop_mm' not in r:          # older result files: read it from the log
        r['at_network_stop_mm'] = [next((l['ee_mm'] for l in x['log'] if l['t'] >= x['stop_s']), np.nan) if x['stop_s'] else np.nan
                                   for x in json.loads((ARM / f'results/{name}.json').read_text())]
    rows.append({'iteration': it, 'model': name, 'stop learns': stop, 'test torque loss': h['test_torque_mse'],
                 'stopped (of 25)': int(r.stop_s.notna().sum()), 'median distance at stop (mm)': r.at_network_stop_mm.median(),
                 'median distance at the end of the run (mm)': r.end_mm.median(), 'median closest (mm)': r.min_mm.median(),
                 'median stop vs expert (s)': (r.stop_s - r.expert_stop_s).median()})
pd.DataFrame(rows).set_index('iteration').round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>stop learns</th>
      <th>test torque loss</th>
      <th>stopped (of 25)</th>
      <th>median distance at stop (mm)</th>
      <th>median distance at the end of the run (mm)</th>
      <th>median closest (mm)</th>
      <th>median stop vs expert (s)</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th></th>
      <th></th>
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
      <th>18</th>
      <td>arm1k_e77</td>
      <td>time</td>
      <td>0.377</td>
      <td>15</td>
      <td>257.271</td>
      <td>445.634</td>
      <td>32.971</td>
      <td>-0.191</td>
    </tr>
    <tr>
      <th>19</th>
      <td>arm1k_e77_dist</td>
      <td>distance</td>
      <td>0.368</td>
      <td>25</td>
      <td>35.219</td>
      <td>62.868</td>
      <td>29.332</td>
      <td>-1.881</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 19: predicted against actual distance

The stop head's predicted distance against the real distance from the gripper to the dot, on four bearings. The network says stop when its prediction first reaches 1 cm (dotted line).


```python
r19 = {r['bearing']: r for r in json.loads((ARM / 'results/arm1k_e77_dist.json').read_text())}
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for ax, b in zip(axes, (24, 132, 204, 360)):
    L = r19[b]['log']; t0 = L[0]['t']
    ax.plot([l['t'] - t0 for l in L], [l['ee_mm'] for l in L], label='actual')
    ax.plot([l['t'] - t0 for l in L], [1000 * l['ttp_s'] for l in L], label='predicted')
    ax.axhline(10, color='k', ls=':', lw=0.8)
    if r19[b]['stop_s']:
        ax.axvline(r19[b]['stop_s'] - t0, color='r', lw=0.8, label='network says stop')
    ax.set(title=f"{b}° ({r19[b]['split']})", xlabel='time since PARK (s)', yscale='log', ylim=(1, 400))
axes[0].set_ylabel('gripper to dot (mm)'); axes[0].legend(fontsize=8); plt.show()
near = np.array([1000 * l['ttp_s'] - l['ee_mm'] for r in r19.values() for l in r['log'] if l['ee_mm'] < 100])
print(f'Within 100 mm of the dot: predicted minus actual distance, median {np.median(near):.1f} mm, mean absolute {np.abs(near).mean():.1f} mm')
print('Network stops within 10 mm of the closest point it reaches:', sum(r['at_network_stop_mm'] - r['min_mm'] <= 10 for r in r19.values()), 'of 25')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_163_0.png)
    


    Within 100 mm of the dot: predicted minus actual distance, median -14.2 mm, mean absolute 28.8 mm
    Network stops within 10 mm of the closest point it reaches: 18 of 25


#### Iteration 19: loss curves

Training (solid) and test (dashed) losses. The right panel is the stop loss: time to stop for iteration 18 and distance for iteration 19, each scaled to 0..1, so the two are not on the same scale.


```python
arm_loss_plot(['arm1k_e77', 'arm1k_e77_dist'], 'Iteration 19 (stop on distance) against iteration 18, seed 7 (stop on time)')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_165_0.png)
    


#### Iteration 19: 3D replay (`arm1k_e77_dist`)

<a href="/files/arm/results/viewer_arm1k_e77_dist.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_dist')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerbbb6bd70');f.src='/files/arm/results/viewer_arm1k_e77_dist.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerbbb6bd70" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 19: reading the results

Against the expectation:

- **Stops: as expected.** The network says stop on all 25 bearings, against 15 for iteration 18, seed 7.
- **Distance at stop: better, but not as expected.** The gripper is a median 35 mm from the dot when the network says stop, against 257 mm for iteration 18. The expectation was 20 mm or less, and above 30 mm counts as wrong.
- **When it stops: as expected.** A median 1.9 s before the expert, because it no longer waits for the expert's 8 s rule.
- **Reach: as expected.** The median closest distance is 29 mm, against 33 mm.

Why the distance at stop is 35 mm and not 1 cm:

- On 18 of 25 bearings the network says stop within 10 mm of the closest point the gripper reaches. The stop now comes at the right moment; the gripper simply never gets within 1 cm.
- Near the dot the stop head underestimates the distance: when the gripper is within 100 mm, the prediction is a median 14 mm too low. It seems to have learned "this stage of the reach means close" from the 21 expert reaches, where the gripper always ends within about 1 cm, rather than the real distance.
- After stop the distance grows from a median 35 mm to 63 mm over the last 0.5 s, because nothing holds the arm.
- The distance loss is still falling at epoch 77 (training 0.0003, test 0.0003), so the stop head has not finished learning. On 204°, a test bearing where the expert itself stalls 11.5 mm away, the gripper stays about 60 mm away and the prediction falls below 1 cm anyway.

What it tells us: the stop works on distance far better than on time, so it is kept. The limit is now the reach itself. The gripper gets a median 29 mm from the dot, never within 1 cm, and drifts once it stops moving towards it. Getting closer and holding still are the next problems.

### Arm: dot from the camera (iteration 20)

#### Iteration 20: plan and expected results

In iterations 14 to 19 the arm network got the dot position from the simulator. The robot can only know it from its own sensors, so iteration 20 gives the arm network the dot from the corner camera and the tracker, as the driving network gets it.

- The network is iteration 19's, unchanged: no retraining, so only the source of the dot changes.
- The camera and tracker are the ones the driving network uses, with one change: the tracker now also keeps the dot's height from the camera. While the dot is hidden, the height stays as last seen, because the robot drives on flat ground.
- The camera image is taken every 1/30 s, as in driving. The arm network runs every 1 ms and uses the latest tracker estimate. The tracker moves the dot with the wheel encoder speeds while the robot drives, and after PARK the robot does not move.
- The expert still drives and parks, so the dot source is the only change from iteration 19.
- Test: the 25 bearings in the simulator. New result: the error of the tracker's estimate against the true dot, at PARK and during the reach.

Expected:

- At PARK the dot is in the camera's view on about 9 of 25 bearings. On the others the tracker's estimate is 1 to 4 cm off, from the wheel odometry while the dot was hidden.
- The arm aims at the estimated dot, so the gripper ends up about as far from the true dot as the estimate is off. The median closest distance rises from 29 mm to about 40 to 50 mm.
- The network still stops on all 25 bearings, when it thinks it is within 1 cm of the estimated dot. The real distance at stop rises from a median 35 mm to about 50 mm.

The expectation is wrong if the results match iteration 19 within a few millimetres, which would mean the camera error does not matter, or if the median distance at stop is above 80 mm, which would mean the camera is not good enough for the arm as it is.

All 30 expert targets are at the same height, 0.5 m. The network has only ever seen that height, so this test does not show whether it can reach a dot at another height. That needs new expert runs at other heights and is a later iteration.

#### Iteration 20: results

The same network as iteration 19, with the dot from the simulator (iteration 19) and from the camera and tracker (iteration 20), on the 25 bearings with the expert driving.


```python
rows = []
for it, results, source in ((19, 'arm1k_e77_dist', 'simulator'), (20, 'arm1k_e77_dist_camera', 'camera and tracker')):
    r = pd.DataFrame(json.loads((ARM / f'results/{results}.json').read_text()))
    rows.append({'iteration': it, 'dot from': source, 'stopped (of 25)': int(r.stop_s.notna().sum()),
                 'median distance at stop (mm)': r.at_network_stop_mm.median(), 'median closest (mm)': r.min_mm.median(),
                 'median distance at the end of the run (mm)': r.end_mm.median(), 'median stop vs expert (s)': (r.stop_s - r.expert_stop_s).median()})
display(pd.DataFrame(rows).set_index('iteration').round(1))
cam = pd.DataFrame(json.loads((ARM / 'results/arm1k_e77_dist_camera.json').read_text()))
print('Dot in view at PARK:', int(cam.visible_at_park.sum()), 'of 25. Estimate error at PARK: median', round(cam.est_err_at_park_mm.median(), 1),
      'mm; height error: median', round(cam.height_err_at_park_mm.abs().median(), 1), 'mm.')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dot from</th>
      <th>stopped (of 25)</th>
      <th>median distance at stop (mm)</th>
      <th>median closest (mm)</th>
      <th>median distance at the end of the run (mm)</th>
      <th>median stop vs expert (s)</th>
    </tr>
    <tr>
      <th>iteration</th>
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
      <th>19</th>
      <td>simulator</td>
      <td>25</td>
      <td>35.2</td>
      <td>29.3</td>
      <td>62.9</td>
      <td>-1.9</td>
    </tr>
    <tr>
      <th>20</th>
      <td>camera and tracker</td>
      <td>25</td>
      <td>214.1</td>
      <td>105.3</td>
      <td>501.7</td>
      <td>-2.7</td>
    </tr>
  </tbody>
</table>
</div>


    Dot in view at PARK: 16 of 25. Estimate error at PARK: median 26.8 mm; height error: median 10.9 mm.


#### Iteration 20: the camera's estimate during the reach

Left: how far the tracker's estimate is from the true dot. Right: how far the estimate moves from its value at PARK. The robot is parked, so in training the dot input never changed after PARK.


```python
camruns = json.loads((ARM / 'results/arm1k_e77_dist_camera.json').read_text())
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
for r in camruns:
    L = r['log']; t0 = L[0]['t']; E = np.array([l['est'] for l in L])
    ax[0].plot([l['t'] - t0 for l in L], [l['est_err_mm'] for l in L], lw=0.6, alpha=0.6)
    ax[1].plot([l['t'] - t0 for l in L], 1000 * np.linalg.norm(E - E[0], axis=1), lw=0.6, alpha=0.6)
ax[0].set(xlabel='time since PARK (s)', ylabel='estimate to true dot (mm)', title='Tracker estimate error, 25 bearings')
ax[1].set(xlabel='time since PARK (s)', ylabel='change from the estimate at PARK (mm)', title='How far the estimate moves after PARK')
plt.show()
moves = [1000 * np.abs(np.array([l['est'] for l in r['log']]) - r['log'][0]['est']).max() for r in camruns]
print(f"Estimate during the reach: error median {np.median([l['est_err_mm'] for r in camruns for l in r['log']]):.0f} mm, dot in view "
      f"{np.mean([l['visible'] for r in camruns for l in r['log']]):.0%} of frames; largest change from PARK: median {np.median(moves):.0f} mm, max {max(moves):.0f} mm")
tc = lambda runs: np.median([min(r['log'], key=lambda l: l['ee_mm'])['t'] - r['log'][0]['t'] for r in runs])
print(f"Time from PARK to the closest point: median {tc(camruns):.2f} s with the camera, "
      f"{tc(json.loads((ARM / 'results/arm1k_e77_dist.json').read_text())):.2f} s with the simulator's dot")
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_173_0.png)
    


    Estimate during the reach: error median 11 mm, dot in view 94% of frames; largest change from PARK: median 19 mm, max 72 mm
    Time from PARK to the closest point: median 0.50 s with the camera, 1.63 s with the simulator's dot


#### Iteration 20: loss curves

Iteration 20 uses iteration 19's network without retraining, so these are iteration 19's training curves.


```python
arm_loss_plot(['arm1k_e77_dist'], 'Iteration 20 uses the iteration 19 network (no retraining)')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_175_0.png)
    


#### Iteration 20: 3D replay (`arm1k_e77_dist`, dot from the camera)

<a href="/files/arm/results/viewer_arm1k_e77_dist_camera.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_dist_camera')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('player4798c1d9');f.src='/files/arm/results/viewer_arm1k_e77_dist_camera.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player4798c1d9" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 20: reading the results

Against the expectation:

- **The camera's error: smaller than expected.** The dot is in view at PARK on 16 of 25 bearings, against the expected 9. The estimate is a median 27 mm from the true dot at PARK (11 mm in height), and 11 mm during the reach, when the dot is in view 94% of the time.
- **The reach: much worse than expected.** The gripper gets a median 105 mm from the dot, against 29 mm with the simulator's dot and the expected 40 to 50 mm.
- **The stop: it still stops on all 25, but far from the dot.** A median 214 mm away when it stops, against 35 mm. Above 80 mm means the expectation was wrong: the camera, as it is used now, is not good enough for the arm.

Why it is worse than the estimate error suggests:

- The dot input jumps in the first 0.5 s after PARK. Where the dot was hidden while parking, the estimate starts 30 to 50 mm off and is corrected in one step when the dot comes into view. The network runs every 1 ms and sees the new value as a sudden change.
- In training the robot is parked, so the dot input never changes after PARK. A dot input that jumps is something the network has never seen.
- The arm leaves the expert's path early: the gripper is closest a median 0.5 s after PARK, against 1.6 s with the simulator's dot, and then swings away.
- Once the arm swings, it pushes the robot base: the true dot moves by up to 40 mm relative to the robot. The estimate follows it, staying about 11 mm off, which is why the estimate moves by up to 72 mm in total. This is a result of the swing, not its cause.

What it tells us: the next test is to give the arm network one fixed dot position, taken once the camera has a fresh view of the dot after PARK, and to keep it fixed during the reach. If that works, the camera's error alone, about 1 cm when the dot is in view, sets how close the gripper can get. Training on dot positions with the camera's error added is the other option.

### Arm: no time inputs (iteration 21)

#### Iteration 21: plan and expected results

With the camera (iteration 20), the arm reaches the dot about 0.6 s after PARK instead of 1.6 s, says stop, and then drops: the gripper falls a median 447 mm in the next 0.5 s. With the simulator's dot (iteration 19) it falls a median 5 mm after its closest point, up to about 50 mm on some bearings.

The likely reason is the two time inputs, time since PARK and time since the start. In every training row the arm is at the point of the expert's reach that the clock predicts: about halfway 0.6 s after PARK, at the dot about 1.6 s after. The network can use the clock as a shortcut instead of the joint angles. When the arm arrives early, the clock and the joints disagree, which never happens in training, and the torques it gives there do not hold the arm against gravity.

The time inputs were needed for the old stop, which followed the expert's 8 s rule. Since iteration 19 the stop learns the distance to the dot, so the clock is no longer needed.

Iteration 21 is iteration 19 retrained without the two time inputs:

- 17 inputs: arm encoder angles and speeds (7 each) and the dot position (x, y, height).
- Everything else as in iteration 19: the torques and the distance to the dot as outputs, stop at 1 cm, 2×64 ReLU, 1 kHz, 77 epochs, seed 7, rows from the expert's PARK to its stop.
- Tested twice on the 25 bearings with the expert driving: with the simulator's dot, to compare with iteration 19, and with the camera and tracker, to compare with iteration 20.
- New result: how far the gripper height falls after the gripper's closest point to the dot.

Expected:

- With the simulator's dot: the reach stays as in iteration 19 (median closest near 30 mm) and the fall after the closest point stays small, near 5 mm, and no longer reaches 50 mm.
- With the camera: the fall of a median 447 mm mostly goes away, to below 50 mm. The early, fast reach stays, because it comes from the dot input jumping when a hidden dot comes into view, not from the clock.

The expectation is wrong if the arm still falls by more than 200 mm with the camera. Then the clock is not the cause, and the next suspect is gravity itself: the torque that holds the arm up depends on the joint angles in a way the network has not learned from 21 reaches.

#### Iteration 21: results

Iterations 19 and 21 with the simulator's dot, and iterations 20 and 21 with the camera, on the 25 bearings with the expert driving. The fall is how far the gripper height drops after the gripper's closest point to the dot.


```python
def gripper_fall_mm(run):
    """How far the gripper height drops after its closest point to the dot (from the frames logged every 1/30 s)."""
    F = [f for f in run['frames'] if f['parked']]
    i = int(np.argmin([f['ee_mm'] for f in F]))
    return 1000 * (F[i]['ee'][2] - min(f['ee'][2] for f in F[i:]))

rows = []
for it, results, inputs, source in ((19, 'arm1k_e77_dist', 19, 'simulator'), (21, 'arm1k_e77_dist_notime', 17, 'simulator'),
                                    (20, 'arm1k_e77_dist_camera', 19, 'camera'), (21, 'arm1k_e77_dist_notime_camera', 17, 'camera')):
    runs = json.loads((ARM / f'results/{results}.json').read_text()); r = pd.DataFrame(runs)
    rows.append({'iteration': it, 'inputs': inputs, 'dot from': source, 'stopped (of 25)': int(r.stop_s.notna().sum()),
                 'median distance at stop (mm)': r.at_network_stop_mm.median(), 'median closest (mm)': r.min_mm.median(),
                 'median fall after closest (mm)': np.median([gripper_fall_mm(x) for x in runs]),
                 'largest fall (mm)': max(gripper_fall_mm(x) for x in runs)})
pd.DataFrame(rows).set_index(['iteration', 'dot from']).round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>inputs</th>
      <th>stopped (of 25)</th>
      <th>median distance at stop (mm)</th>
      <th>median closest (mm)</th>
      <th>median fall after closest (mm)</th>
      <th>largest fall (mm)</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th>dot from</th>
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
      <th>19</th>
      <th>simulator</th>
      <td>19</td>
      <td>25</td>
      <td>35.2</td>
      <td>29.3</td>
      <td>5.3</td>
      <td>81.7</td>
    </tr>
    <tr>
      <th>21</th>
      <th>simulator</th>
      <td>17</td>
      <td>24</td>
      <td>79.0</td>
      <td>23.7</td>
      <td>111.4</td>
      <td>345.5</td>
    </tr>
    <tr>
      <th>20</th>
      <th>camera</th>
      <td>19</td>
      <td>25</td>
      <td>214.1</td>
      <td>105.3</td>
      <td>446.9</td>
      <td>578.9</td>
    </tr>
    <tr>
      <th>21</th>
      <th>camera</th>
      <td>17</td>
      <td>3</td>
      <td>493.4</td>
      <td>188.7</td>
      <td>385.8</td>
      <td>682.5</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 21: gripper height after PARK

The gripper height on four bearings, with and without the time inputs. The dot is at 0.5 m.


```python
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
styles = (('arm1k_e77_dist', 'iteration 19, simulator dot', '-'), ('arm1k_e77_dist_notime', 'iteration 21, simulator dot', '--'),
          ('arm1k_e77_dist_camera', 'iteration 20, camera', '-'), ('arm1k_e77_dist_notime_camera', 'iteration 21, camera', '--'))
runs = {name: {r['bearing']: r for r in json.loads((ARM / f'results/{name}.json').read_text())} for name, _, _ in styles}
for ax, b in zip(axes, (24, 60, 180, 300)):
    for name, label, ls in styles:
        F = [f for f in runs[name][b]['frames'] if f['parked']]
        ax.plot([f['time_s'] - F[0]['time_s'] for f in F], [1000 * f['ee'][2] for f in F], ls, label=label)
    ax.axhline(500, color='r', lw=0.8, ls=':'); ax.set(title=f'{b}°', xlabel='time since PARK (s)')
axes[0].set_ylabel('gripper height (mm)'); axes[0].legend(fontsize=7); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_183_0.png)
    


#### Iteration 21: loss curves

Iteration 21 (17 inputs, no time) against iteration 19 (19 inputs).


```python
arm_loss_plot(['arm1k_e77_dist', 'arm1k_e77_dist_notime'], 'Iteration 21 (no time inputs) against iteration 19')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_185_0.png)
    


#### Iteration 21: 3D replay with the simulator's dot (`arm1k_e77_dist_notime`)

<a href="/files/arm/results/viewer_arm1k_e77_dist_notime.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_dist_notime')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('player9ce8125e');f.src='/files/arm/results/viewer_arm1k_e77_dist_notime.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player9ce8125e" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 21: 3D replay with the camera (`arm1k_e77_dist_notime`, dot from the camera)

<a href="/files/arm/results/viewer_arm1k_e77_dist_notime_camera.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_dist_notime_camera')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playereaa2a466');f.src='/files/arm/results/viewer_arm1k_e77_dist_notime_camera.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playereaa2a466" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 21: the dot input at PARK, training against the camera

The spread of the dot position relative to the robot at PARK over the 21 training runs, against the camera's error at PARK.


```python
import train_arm
train_dot = np.array([train_arm.load_run(b, 'sample', 1000)['X'][0, 14:17] for b in sorted(train_arm.SPLIT) if train_arm.SPLIT[b] == 'train'])
cam = json.loads((ARM / 'results/arm1k_e77_dist_camera.json').read_text())
err = np.array([np.subtract(r['log'][0]['est'], r['log'][0]['true']) for r in cam])
spread = 1000 * train_dot.std(0)
pd.DataFrame({'training range (mm)': [f'{1000 * lo:.0f} to {1000 * hi:.0f}' for lo, hi in zip(train_dot.min(0), train_dot.max(0))],
              'training spread, std (mm)': spread.round(1),
              'camera error at PARK, median size (mm)': (1000 * np.median(np.abs(err), 0)).round(1),
              'camera error in training stds': (1000 * np.median(np.abs(err), 0) / spread).round(1)},
             index=['x (forward)', 'y (left)', 'height'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>training range (mm)</th>
      <th>training spread, std (mm)</th>
      <th>camera error at PARK, median size (mm)</th>
      <th>camera error in training stds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>x (forward)</th>
      <td>676 to 730</td>
      <td>13.500000</td>
      <td>17.2</td>
      <td>1.3</td>
    </tr>
    <tr>
      <th>y (left)</th>
      <td>-31 to 26</td>
      <td>16.200001</td>
      <td>18.4</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>height</th>
      <td>502 to 503</td>
      <td>0.200000</td>
      <td>10.9</td>
      <td>61.6</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 21: reading the results

Against the expectation:

- **With the simulator's dot: not as expected.** The gripper gets a little closer (median 24 mm against 29 mm), but then it falls a median 111 mm after its closest point, against 5 mm with the time inputs. When it says stop it is a median 79 mm from the dot, against 35 mm.
- **With the camera: not as expected.** The fall stays large, a median 386 mm against 447 mm, so it is above the 200 mm that the plan set as the test. The reach is worse (median closest 189 mm against 105 mm), and the network says stop on only 3 of 25 bearings.

What it tells us:

- The clock was not the cause of the fall. It was the opposite: with the simulator's dot, the time inputs were what kept the arm up after the reach. In the expert's runs the arm holds still near the dot for the last part of every run, and the network had learned "late in the run, hold" from the clock.
- Without the clock, the network has to work out the holding torque from the joint angles alone, and it cannot from 21 reaches. Holding the arm still means cancelling gravity, and the torque for that changes with every joint angle.
- Iteration 19, with the time inputs, stays the best arm network with the simulator's dot.

Why the camera makes the arm fall (iterations 20 and 21). The height plot shows that with the camera the gripper does not reach and then drop. It falls from the start, straight through the dot's height. The cause is the dot height input:

- Over the 21 training runs the dot height relative to the robot is 502 to 503 mm, a spread of 0.2 mm, because every dot is 0.5 m high and the robot always parks on flat ground. The x and y positions spread by 13 to 16 mm.
- The inputs are standardised, so the network sees a 1 mm change in height as 5 standard deviations.
- At PARK the camera's estimate is a median 11 mm off in height, about 60 standard deviations, while its x and y errors (17 and 18 mm) are about 1 standard deviation. The height input is far outside anything the network has seen, and its torques there make the arm fall.

What to test next: train on dot heights with the camera's error added (or with different dot heights), or leave the height out until the expert data has dots at other heights. Leaving it out is the quickest test that the height input is the cause.

### Arm: no scaling (iteration 22)

#### Iteration 22: plan and expected results

Every arm network so far (iterations 14 to 21) standardised all its inputs and all 7 torques with the training-set mean and standard deviation. For inputs that barely move in training this makes small changes look huge. The dot height spreads by 0.2 mm over the 21 training runs, so a 1 mm change is 5 standard deviations and the camera's 11 mm height error at PARK is about 60. Joint 7's angle spreads by 0.001 rad, so a 0.01 rad drift is 10.

Iteration 22 is iteration 21 retrained with no scaling at all:

- Inputs as in iteration 21 (17), in their own units: arm encoder angles (rad) and speeds (rad/s), 7 each, and the dot position x, y, height (m). No time inputs.
- Outputs: the 7 torques in N m, and the distance from the gripper to the dot (clipped at 0.25 m and divided by 0.25 m, as since iteration 19; this is how the stop target is defined, not feature scaling). Stop at 1 cm.
- Loss: torque MSE in N m² plus the distance MSE, weight 1 each, each training run counted equally.
- Everything else as in iteration 21: 2×64 ReLU, 1 kHz, Adam lr 0.01, 77 full-batch epochs, seed 7, rows from the expert's PARK to its stop, 21 training bearings.
- Tested on the 25 bearings with the expert driving, with the dot from the camera and tracker. From this iteration on the arm is tested with the camera only.

Expected:

- With the camera: the 11 mm height error is now an 0.011 change in an input of about 0.5, so the arm no longer falls straight through the dot. The median fall after the closest point drops from 386 mm to below 200 mm.
- The reach itself stays about as in iteration 21 with the camera (median closest near 189 mm), since without the time inputs the network still has to work out the holding torque from the joint angles.
- Two risks. The torque loss in N m² is tens of times the distance loss, so the stop head may learn less and stop on fewer bearings. And in 77 full-batch steps the outputs have to grow from 0 to the 16 N m that joint 2 needs, so the torques may underfit.

The expectation is wrong if the arm still falls by more than 200 mm with the camera. Then the height scaling was not the cause.

#### Iteration 22: results

Iterations 21 and 22 with the dot from the camera, on the 25 bearings with the expert driving. Both have the same 17 inputs; iteration 22 has no scaling. The fall is how far the gripper height drops after the gripper's closest point to the dot.


```python
rows = []
for it, results, source in ((21, 'arm1k_e77_dist_notime_camera', 'camera'), (22, 'arm1k_e77_dist_notime_raw_camera', 'camera')):
    runs = json.loads((ARM / f'results/{results}.json').read_text()); r = pd.DataFrame(runs)
    rows.append({'iteration': it, 'dot from': source, 'scaling': 'none' if it == 22 else 'mean/std',
                 'stopped (of 25)': int(r.stop_s.notna().sum()),
                 'median distance at stop (mm)': r.at_network_stop_mm.median(), 'median closest (mm)': r.min_mm.median(),
                 'median at expert stop (mm)': r.at_expert_stop_mm.median(),
                 'median fall after closest (mm)': np.median([gripper_fall_mm(x) for x in runs]),
                 'largest fall (mm)': max(gripper_fall_mm(x) for x in runs)})
pd.DataFrame(rows).set_index(['iteration', 'dot from']).round(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>scaling</th>
      <th>stopped (of 25)</th>
      <th>median distance at stop (mm)</th>
      <th>median closest (mm)</th>
      <th>median at expert stop (mm)</th>
      <th>median fall after closest (mm)</th>
      <th>largest fall (mm)</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th>dot from</th>
      <th></th>
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
      <th>21</th>
      <th>camera</th>
      <td>mean/std</td>
      <td>3</td>
      <td>493.4</td>
      <td>188.7</td>
      <td>574.9</td>
      <td>385.8</td>
      <td>682.5</td>
    </tr>
    <tr>
      <th>22</th>
      <th>camera</th>
      <td>none</td>
      <td>25</td>
      <td>238.8</td>
      <td>186.1</td>
      <td>293.5</td>
      <td>194.6</td>
      <td>421.8</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 22: gripper height after PARK

The gripper height on four bearings with the dot from the camera, iteration 21 (mean/std scaling) against iteration 22 (no scaling). The dot is at 0.5 m.


```python
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
styles = (('arm1k_e77_dist_notime_camera', 'iteration 21 (mean/std)', '-'), ('arm1k_e77_dist_notime_raw_camera', 'iteration 22 (no scaling)', '--'))
runs = {name: {r['bearing']: r for r in json.loads((ARM / f'results/{name}.json').read_text())} for name, _, _ in styles}
for ax, b in zip(axes, (24, 60, 180, 300)):
    for name, label, ls in styles:
        F = [f for f in runs[name][b]['frames'] if f['parked']]
        ax.plot([f['time_s'] - F[0]['time_s'] for f in F], [1000 * f['ee'][2] for f in F], ls, label=label)
    ax.axhline(500, color='r', lw=0.8, ls=':'); ax.set(title=f'{b}°', xlabel='time since PARK (s)')
axes[0].set_ylabel('gripper height (mm)'); axes[0].legend(fontsize=7); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_197_0.png)
    


#### Iteration 22: loss curves

Iteration 22 is trained on torques in N m, so its torque loss is in N m², while earlier arm networks' losses are in standardised units and cannot share an axis with it. The table after the plot puts iteration 21 in the same units: its saved network (which takes raw inputs and gives N m) run over the same rows. Predicting each joint's mean torque gives the baseline.


```python
import train_arm
h = pd.DataFrame(json.loads((ARM / 'models/arm1k_e77_dist_notime_raw_history.json').read_text()))
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
ax[0].plot(h.epoch, h.train_torque_mse, label='train'); ax[0].plot(h.epoch, h.test_torque_mse, '--', color='C0', label='test')
ax[1].plot(h.epoch, h.train_distance_mse, label='train'); ax[1].plot(h.epoch, h.test_distance_mse, '--', color='C0', label='test')
for a, t in zip(ax, ('torque loss (N m²)', 'stop loss: distance (scaled 0..1)')):
    a.set(yscale='log', xlabel='epoch', title=t); a.legend(fontsize=8)
fig.suptitle('Iteration 22 (no scaling)'); plt.show()

def torque_mse_Nm2(model, split):
    w = dict(np.load(ARM / f'models/{model}.npz')); out = []
    runs = [train_arm.load_run(b, 'sample', 1000, 'distance') for b in sorted(train_arm.SPLIT) if train_arm.SPLIT[b] == split]
    for r in runs:
        X = r['X'][:, :17]
        for l in ('shared.0', 'shared.2'):
            X = np.maximum(X @ w[l + '.weight'].T + w[l + '.bias'], 0)
        out.append(((X @ w['arm.weight'].T + w['arm.bias'] - r['tau']) ** 2).mean())
    return np.mean(out)                                       # each run counted equally, as in training

def baseline_Nm2(split):
    tr = [train_arm.load_run(b, 'sample', 1000, 'distance') for b in sorted(train_arm.SPLIT) if train_arm.SPLIT[b] == 'train']
    mean = np.concatenate([r['tau'] for r in tr]).mean(0)
    runs = [train_arm.load_run(b, 'sample', 1000, 'distance') for b in sorted(train_arm.SPLIT) if train_arm.SPLIT[b] == split]
    return np.mean([((r['tau'] - mean) ** 2).mean() for r in runs])

pd.DataFrame({'train torque MSE (N m²)': [baseline_Nm2('train'), torque_mse_Nm2('arm1k_e77_dist_notime', 'train'), torque_mse_Nm2('arm1k_e77_dist_notime_raw', 'train')],
              'test torque MSE (N m²)': [baseline_Nm2('test'), torque_mse_Nm2('arm1k_e77_dist_notime', 'test'), torque_mse_Nm2('arm1k_e77_dist_notime_raw', 'test')]},
             index=['baseline: mean torque', 'iteration 21 (mean/std)', 'iteration 22 (no scaling)']).round(2)
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_199_0.png)
    





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>train torque MSE (N m²)</th>
      <th>test torque MSE (N m²)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline: mean torque</th>
      <td>3.68</td>
      <td>4.41</td>
    </tr>
    <tr>
      <th>iteration 21 (mean/std)</th>
      <td>0.37</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>iteration 22 (no scaling)</th>
      <td>2.17</td>
      <td>2.99</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 22: 3D replay (`arm1k_e77_dist_notime_raw`, dot from the camera)

<a href="/files/arm/results/viewer_arm1k_e77_dist_notime_raw_camera.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_dist_notime_raw_camera')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerf3a8a264');f.src='/files/arm/results/viewer_arm1k_e77_dist_notime_raw_camera.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerf3a8a264" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 22: reading the results

Against the expectation:

- **The fall is under 200 mm, but not for the reason expected.** The median fall after the closest point is 195 mm, against 386 mm in iteration 21. The height plot shows why: the network says stop a median 0.64 s after PARK (iteration 21: 2.96 s), and the run ends 0.5 s later. The gripper has less time to fall, but it is already falling when the run ends. It does not hold the arm up.
- **The network now says stop on all 25 bearings, against 3,** but too early: a median 239 mm from the dot. The stop head learned less, as the plan warned. Its loss is in units of 0.25 m and is tens of times smaller than the torque loss in N m².
- **The reach is no better.** The median closest is 186 mm, against 189 mm. The gripper never rises at PARK as the expert's does. It drops straight away.
- **Fewer torques are at the 30 N m limit:** 6.5% of logged torques, against 31% in iteration 21.

What it tells us:

- Without scaling the network fits the expert's torques poorly. After 77 epochs its torque error is 2.17 N m² on the training runs and 2.99 on the test runs. Predicting each joint's mean torque gives 3.68 and 4.41, and iteration 21 got 0.37 and 0.92. In raw units the inputs and torques are on very different scales: joint 2 needs about 16 N m to hold the arm while joint 7 needs 0.01 N m, and the dot height is 0.5 m in every row. With the same learning rate and number of epochs, the network learns mostly the large torques and the average.
- The camera's height error is no longer the main problem, but this network has not learned to hold the arm against gravity either, so this test cannot show whether removing the scaling fixes the camera fall.

### Arm: inputs scaled by physical sizes (iteration 23)

#### Iteration 23: plan and expected results

Iteration 22 removed all scaling. The camera's height error stopped mattering, but the network learned almost nothing beyond each joint's average torque. On the test runs its torque error per joint equals the expert's own spread, while iteration 21 (scaled) gets the error on joints 2, 4, 6 and 7 down by 4 to 17 times. The torque scaling is what made the fit precise. The input scaling is what made the camera fail: dividing by the training spread magnified the dot height by 5,000 and joint 7's angle by 1,000.

Iteration 23 is iteration 21 with only the input scaling changed. Each input is divided by a fixed physical size, with no mean subtracted:

| Input | Iterations 14 to 21 | Iteration 23 |
|---|---|---|
| Joint angles | (angle − mean) ÷ std, e.g. joint 7 ÷ 0.001 rad | ÷ π |
| Joint speeds | (speed − mean) ÷ std, e.g. joint 1 ÷ 0.039 rad/s | ÷ 0.63 rad/s (joints 1 to 4), ÷ 0.84 rad/s (joints 5 to 7), the speed limits in the robot model |
| Dot x, y, height | (value − mean) ÷ std: ÷ 0.014, 0.028, 0.0002 m | ÷ 1 m |
| Torques | (torque − mean) ÷ std per joint | same |
| Stop target | distance ÷ 0.25 m | same |

Everything else as in iteration 21: 17 inputs (arm encoder angles and speeds, dot x, y, height; no time inputs), the 7 torques and the distance to the dot as outputs, stop at 1 cm, 2×64 ReLU, 1 kHz, Adam lr 0.01, 77 full-batch epochs, seed 7, rows from the expert's PARK to its stop, 21 training bearings. Tested on the 25 bearings with the expert driving and the dot from the camera and tracker.

Expected:

- Torque fit close to iteration 21: test torque error about 0.9 N m² (iteration 22: 2.99). It may be a little worse, because the dot's sideways position now spans only ±0.05 of an input and the torques on joints 1 and 5 depend on it.
- With the camera the arm no longer falls straight through the dot. It behaves like iteration 21 did with the simulator's dot: the median closest distance drops from 189 mm (iterations 21 and 22) to below 100 mm.

The expectation is wrong if the median closest stays above 150 mm. Then the input scaling was not the main reason the camera runs failed.

#### Iteration 23: results

Iterations 21, 22 and 23 with the dot from the camera, on the 25 bearings with the expert driving. All three have the same 17 inputs; only the scaling differs. The fall is how far the gripper height drops after the gripper's closest point to the dot.


```python
SC = (('arm1k_e77_dist_notime', 21, 'mean/std'), ('arm1k_e77_dist_notime_raw', 22, 'none'), ('arm1k_e77_dist_notime_phys', 23, 'inputs by physical size, torques mean/std'))
rows = []
for name, it, scaling in SC:
    runs = json.loads((ARM / f'results/{name}_camera.json').read_text()); r = pd.DataFrame(runs)
    rows.append({'iteration': it, 'scaling': scaling, 'stopped (of 25)': int(r.stop_s.notna().sum()),
                 'median stop after PARK (s)': np.median([x['stop_s'] - x['log'][0]['t'] for x in runs if x['stop_s']]) if r.stop_s.notna().any() else None,
                 'median distance at stop (mm)': r.at_network_stop_mm.median(), 'median closest (mm)': r.min_mm.median(),
                 'closest within 50 mm (of 25)': int((r.min_mm <= 50).sum()),
                 'median at expert stop (mm)': r.at_expert_stop_mm.median(),
                 'median fall after closest (mm)': np.median([gripper_fall_mm(x) for x in runs]),
                 'torques at the 30 N m limit (%)': 100 * np.mean(np.abs([l['tau'] for x in runs for l in x['log']]) >= 29.99)})
pd.DataFrame(rows).set_index('iteration').round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>scaling</th>
      <th>stopped (of 25)</th>
      <th>median stop after PARK (s)</th>
      <th>median distance at stop (mm)</th>
      <th>median closest (mm)</th>
      <th>closest within 50 mm (of 25)</th>
      <th>median at expert stop (mm)</th>
      <th>median fall after closest (mm)</th>
      <th>torques at the 30 N m limit (%)</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>21</th>
      <td>mean/std</td>
      <td>3</td>
      <td>2.96</td>
      <td>493.40</td>
      <td>188.69</td>
      <td>1</td>
      <td>574.86</td>
      <td>385.84</td>
      <td>30.63</td>
    </tr>
    <tr>
      <th>22</th>
      <td>none</td>
      <td>25</td>
      <td>0.64</td>
      <td>238.77</td>
      <td>186.09</td>
      <td>0</td>
      <td>293.51</td>
      <td>194.62</td>
      <td>6.55</td>
    </tr>
    <tr>
      <th>23</th>
      <td>inputs by physical size, torques mean/std</td>
      <td>25</td>
      <td>0.81</td>
      <td>104.44</td>
      <td>56.75</td>
      <td>9</td>
      <td>81.03</td>
      <td>0.00</td>
      <td>0.20</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 23: gripper height after PARK

The gripper height on four bearings with the dot from the camera. The dot is at 0.5 m.


```python
fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
runs = {name: {r['bearing']: r for r in json.loads((ARM / f'results/{name}_camera.json').read_text())} for name, _, _ in SC}
for ax, b in zip(axes, (24, 60, 180, 300)):
    for (name, it, scaling), ls in zip(SC, ('-', ':', '--')):
        F = [f for f in runs[name][b]['frames'] if f['parked']]
        ax.plot([f['time_s'] - F[0]['time_s'] for f in F], [1000 * f['ee'][2] for f in F], ls, label=f'iteration {it} ({scaling})')
    ax.axhline(500, color='r', lw=0.8, ls=':'); ax.set(title=f'{b}°', xlabel='time since PARK (s)')
axes[0].set_ylabel('gripper height (mm)'); axes[0].legend(fontsize=7); plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_207_0.png)
    


#### Iteration 23: torque error per joint

The torque error per joint on the 4 test runs (root mean square, N m), against the spread of the expert's own torques. This is the expert's data, not the simulator test.


```python
def torques(model, X):
    w = dict(np.load(ARM / f'models/{model}.npz')); X = X[:, :w['shared.0.weight'].shape[1]]
    for l in ('shared.0', 'shared.2'):
        X = np.maximum(X @ w[l + '.weight'].T + w[l + '.bias'], 0)
    return X @ w['arm.weight'].T + w['arm.bias']
test_runs = [train_arm.load_run(b, 'sample', 1000, 'distance') for b in sorted(train_arm.SPLIT) if train_arm.SPLIT[b] == 'test']
Xt, Tt = np.concatenate([r['X'] for r in test_runs]), np.concatenate([r['tau'] for r in test_runs])
err = {"expert's torque spread (std)": Tt.std(0)}
for name, it, _ in SC:
    err[f'iteration {it}'] = np.sqrt(((torques(name, Xt) - Tt) ** 2).mean(0))
pd.DataFrame(err, index=[f'joint {j}' for j in range(1, 8)]).T.round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>joint 1</th>
      <th>joint 2</th>
      <th>joint 3</th>
      <th>joint 4</th>
      <th>joint 5</th>
      <th>joint 6</th>
      <th>joint 7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>expert's torque spread (std)</th>
      <td>1.50</td>
      <td>1.62</td>
      <td>1.66</td>
      <td>2.55</td>
      <td>2.84</td>
      <td>2.92</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>iteration 21</th>
      <td>0.78</td>
      <td>0.41</td>
      <td>0.80</td>
      <td>0.45</td>
      <td>2.51</td>
      <td>0.58</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>iteration 22</th>
      <td>1.50</td>
      <td>1.34</td>
      <td>1.65</td>
      <td>1.80</td>
      <td>2.84</td>
      <td>1.89</td>
      <td>0.87</td>
    </tr>
    <tr>
      <th>iteration 23</th>
      <td>1.04</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>1.16</td>
      <td>2.61</td>
      <td>1.44</td>
      <td>0.24</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 23: loss curves

Iterations 21 and 23 standardise the torques the same way, so their losses share an axis.


```python
arm_loss_plot(['arm1k_e77_dist_notime', 'arm1k_e77_dist_notime_phys'], 'Iteration 23 (inputs by physical size) against iteration 21 (mean/std)')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_211_0.png)
    


#### Iteration 23: 3D replay (`arm1k_e77_dist_notime_phys`, dot from the camera)

<a href="/files/arm/results/viewer_arm1k_e77_dist_notime_phys_camera.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_dist_notime_phys_camera')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('player59e1bff9');f.src='/files/arm/results/viewer_arm1k_e77_dist_notime_phys_camera.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player59e1bff9" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 23: reading the results

Against the expectation:

- **As expected, the arm no longer falls through the dot with the camera.** The median closest distance is 57 mm, against 189 mm in iteration 21 and 186 mm in iteration 22, below the 100 mm the plan expected. On 9 of 25 bearings the gripper gets within 50 mm (iteration 21: 1). The height plot shows a smooth, steady move down towards the dot, with no rise and no drop.
- **The torques are calm.** 0.2% of logged torques are at the 30 N m limit, against 31% in iteration 21 and 6.5% in iteration 22.
- **The torque fit is between iterations 21 and 22,** not close to iteration 21 as expected. The test loss is 0.59 in standardised units against 0.38. Per joint, iteration 23 is better than iteration 22 everywhere, but on joints 2, 4 and 6 its error is about twice iteration 21's. The loss curves are still falling at epoch 77: with physical scaling the inputs are smaller numbers, and the network learns more slowly. The 77 epochs were chosen for standardised inputs.

What it tells us:

- The input scaling was the main reason the camera runs failed. With the inputs divided by physical sizes, the camera's 11 mm height error no longer throws the network out of its training range.
- The stop signal is now the limit. The network says stop on all 25 bearings, but a median 0.81 s after PARK and 104 mm from the dot. The expert takes about 2.3 s to get within 1 cm. When the run ends 0.5 s later, the gripper is still moving steadily towards the dot. The stop loss is 0.0017 (an average error of about 10 mm) against 0.0003 in iteration 21, so the stop head also learned less in 77 epochs.
- Next: the stop head (its own layer or its own network), and more training, with the number of epochs chosen again on the validation folds.

### Arm: the stop head (iterations 24 to 26)

Iteration 23 reaches the dot smoothly with the camera, but the stop comes too early on every bearing: a median 0.81 s after PARK and 104 mm from the dot, while the expert takes about 2.3 s to get within 1 cm. On the expert's own test runs the stop head predicts 2 to 39 mm at PARK, where the real distance is about 225 mm. At PARK the arm is still, and the only other time it is still in training is the expert's hold at the dot, where the distance is 0. Iterations 24 to 26 work on the stop head one change at a time, each keeping the changes before it.

#### Iteration 24: plan and expected results

Iteration 24 is iteration 23 with a layer of its own for the stop head:

```
17 inputs -> 64 ReLU -> 64 ReLU -+-> 7 torques
                                 +-> 16 ReLU -> distance to the dot
```

- Before, the stop head was one linear unit on the second shared layer, so it could only draw a straight line through the features the torques need. The 16-unit layer lets it bend its own way, for example to tell PARK apart from the hold at the dot. The torque loss does not reach this layer.
- Everything else as in iteration 23: 17 inputs divided by physical sizes, torques standardised, stop at 1 cm, stop loss weight 1, 1 kHz, Adam lr 0.01, 77 full-batch epochs, seed 7, 21 training bearings. Tested on the 25 bearings with the expert driving and the dot from the camera and tracker.

Expected:

- A small gain. The stop loss after 77 epochs drops a little below iteration 23's 0.0017, and the median stop moves later, from 0.81 s to between 1.0 and 1.5 s after PARK, so the gripper stops closer than 104 mm.
- The reach stays as in iteration 23 (median closest near 57 mm), since the torque part is unchanged apart from what the stop layer sends back into the shared layers.

The expectation is wrong if the median stop stays before 1.0 s after PARK. Then a separate layer alone does not help at 77 epochs, and the 5-fold check (iteration 23's settings at 600 epochs moved the first stop from a median 1.39 s to 2.25 s on held-out runs) points to training length instead.

#### Iteration 24: results

The camera test on the 25 bearings with the expert driving, against the iterations before it. A stop in the first 0.5 s means the network said stop almost as soon as the arm started.


```python
def camera_table(SC):
    """One row per network: the camera test on the 25 bearings (expert driving)."""
    rows = []
    for name, it in SC:
        runs = json.loads((ARM / f'results/{name}_camera.json').read_text()); r = pd.DataFrame(runs)
        stops = [x['stop_s'] - x['log'][0]['t'] for x in runs if x['stop_s']]
        rows.append({'iteration': it, 'stopped (of 25)': len(stops), 'median stop after PARK (s)': np.median(stops) if stops else None,
                     'stops in the first 0.5 s': sum(s < 0.5 for s in stops),
                     'median distance at stop (mm)': r.at_network_stop_mm.median(), 'median closest (mm)': r.min_mm.median(),
                     'closest within 50 mm (of 25)': int((r.min_mm <= 50).sum()), 'median at expert stop (mm)': r.at_expert_stop_mm.median(),
                     'median fall after closest (mm)': np.median([gripper_fall_mm(x) for x in runs]),
                     'torques at the 30 N m limit (%)': 100 * np.mean(np.abs([l['tau'] for x in runs for l in x['log']]) >= 29.99)})
    return pd.DataFrame(rows).set_index('iteration').round(2)

def camera_plots(SC, bearings=(24, 60, 180, 300)):
    """Gripper height (top) and predicted against actual gripper-to-dot distance (bottom) after PARK, camera test."""
    runs = {name: {r['bearing']: r for r in json.loads((ARM / f'results/{name}_camera.json').read_text())} for name, _ in SC}
    fig, axes = plt.subplots(2, len(bearings), figsize=(16, 6.6), sharey='row')
    for k, b in enumerate(bearings):
        for (name, it), c in zip(SC, ('C0', 'C1', 'C2', 'C3')):
            F = [f for f in runs[name][b]['frames'] if f['parked']]; L = runs[name][b]['log']
            axes[0, k].plot([f['time_s'] - F[0]['time_s'] for f in F], [1000 * f['ee'][2] for f in F], color=c, label=f'iteration {it}')
            axes[1, k].plot([l['t'] - L[0]['t'] for l in L], [l['ee_mm'] for l in L], color=c, label=f'iteration {it}, actual')
            axes[1, k].plot([l['t'] - L[0]['t'] for l in L], [1000 * l['ttp_s'] for l in L], '--', color=c, label=f'iteration {it}, predicted')
        axes[0, k].axhline(500, color='r', lw=0.8, ls=':'); axes[0, k].set(title=f'{b}°')
        axes[1, k].axhline(10, color='r', lw=0.8, ls=':'); axes[1, k].set(xlabel='time since PARK (s)', ylim=(-50, 300))
    axes[0, 0].set_ylabel('gripper height (mm)'); axes[1, 0].set_ylabel('gripper to dot (mm)')
    axes[0, 0].legend(fontsize=7); axes[1, 0].legend(fontsize=6); plt.show()

camera_table([('arm1k_e77_dist_notime_phys', 23), ('arm1k_e77_phys_stop16', 24)])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>stopped (of 25)</th>
      <th>median stop after PARK (s)</th>
      <th>stops in the first 0.5 s</th>
      <th>median distance at stop (mm)</th>
      <th>median closest (mm)</th>
      <th>closest within 50 mm (of 25)</th>
      <th>median at expert stop (mm)</th>
      <th>median fall after closest (mm)</th>
      <th>torques at the 30 N m limit (%)</th>
    </tr>
    <tr>
      <th>iteration</th>
      <th></th>
      <th></th>
      <th></th>
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
      <th>23</th>
      <td>25</td>
      <td>0.81</td>
      <td>11</td>
      <td>104.44</td>
      <td>56.75</td>
      <td>9</td>
      <td>81.03</td>
      <td>0.0</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>0.78</td>
      <td>12</td>
      <td>137.78</td>
      <td>114.83</td>
      <td>0</td>
      <td>154.68</td>
      <td>0.0</td>
      <td>0.68</td>
    </tr>
  </tbody>
</table>
</div>



#### Iteration 24: gripper height and the stop prediction

Top: the gripper height after PARK (the dot is at 0.5 m). Bottom: the gripper-to-dot distance the stop head predicts (dashed) against the actual distance (solid); the network says stop when the prediction first reaches 10 mm (red line). Four bearings, camera test.


```python
camera_plots([('arm1k_e77_dist_notime_phys', 23), ('arm1k_e77_phys_stop16', 24)])
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_219_0.png)
    


#### Iteration 24: loss curves

The stop loss is shown unweighted for every network, so the curves compare directly.


```python
arm_loss_plot(['arm1k_e77_dist_notime_phys', 'arm1k_e77_phys_stop16'], 'Iteration 24 against iteration 23')
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_221_0.png)
    


#### Iteration 24: 3D replay (`arm1k_e77_phys_stop16`, dot from the camera)

<a href="/files/arm/results/viewer_arm1k_e77_phys_stop16_camera.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
path = replay_arm.build_viewer('arm1k_e77_phys_stop16_camera')
lazy_player('/files/arm/results/' + path.name)
```




<div><button onclick="var f=document.getElementById('playerdb757b63');f.src='/files/arm/results/viewer_arm1k_e77_phys_stop16_camera.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerdb757b63" style="display:none;width:100%;height:810px;border:0"></iframe></div>



#### Iteration 24: reading the results

Against the expectation:

- **Not as expected: the stop does not move later.** The median stop is 0.78 s after PARK, against 0.81 s in iteration 23, below the 1.0 s the plan set as the test. The training stop loss ends at 0.0015 against 0.0017, about the same.
- **The early stops happen at PARK itself.** On 12 of 25 bearings the network says stop within 2 ms of PARK (iteration 23: 11). At PARK the head predicts a median 40 mm (10 to 111 mm), where the real distance is about 225 mm. Where it does not stop at once, the prediction climbs towards the real distance within 0.2 s and then follows it, with dips as on bearings 180 and 300.
- **The reach looks the same; the closest distance is worse.** The gripper height follows iteration 23 closely on all four bearings plotted. The median closest is 115 mm against 57 mm, and no bearing gets within 50 mm (iteration 23: 9). The bearings where iteration 23 got close were the ones where it did not stop at PARK, and in iteration 24 those runs stop sooner after the dips.

What it tells us:

- A layer of its own does not fix the stop at 77 epochs. The problem is the PARK state, which the head still confuses with the hold at the dot, and the stop latches on a single millisecond.
- Iteration 25 keeps the layer and tunes the weight of the stop loss on the validation folds; iteration 26 trains for 300 epochs.

#### Iteration 25: plan and expected results

Iteration 25 is iteration 24 (with the stop head's own 16-unit layer) with the weight of the stop loss tuned. So far the total loss has been the torque loss plus the stop loss, each with weight 1. The torque loss is about 0.3 at the end of training and the stop loss about 0.0015, so the stop head gets a small share of the training signal through the shared layers. A larger weight gives it more.

How the weight is chosen, without the test bearings:

- Candidate weights 1, 3, 10, 30 and 100.
- For each weight, the network is trained 5 times, each time holding out one of the 5 folds of training bearings (`split_plan.json`) as validation, with the iteration 24 settings (77 epochs, seed 7).
- The chosen weight has the lowest validation stop loss, averaged over the 5 folds, among the weights whose average validation torque loss is no more than 10% above weight 1's. The losses compared are unweighted.
- Also reported, but not used to choose: on the held-out runs, how often the stop head would say stop in the first 0.5 s after PARK along the expert's own path.

Then the network is trained on all 21 training bearings with the chosen weight and tested on the 25 bearings with the expert driving and the dot from the camera and tracker.

Expected:

- The validation stop loss falls as the weight rises, and the torque loss rises a little. A weight between 10 and 30 is chosen.
- In the camera test, fewer stops at PARK: from 12 of 25 to fewer than 6, and a median stop later than 1.0 s after PARK.

The expectation is wrong if 10 or more bearings still stop in the first 0.5 s. Then the weight is not the limit at 77 epochs, and iteration 26 (300 epochs) is the next test.

#### Iteration 25: results

The 5-fold validation for each stop loss weight (iteration 24 settings, 77 epochs, seed 7). Losses are unweighted and averaged over the 5 held-out folds. The last column counts the held-out runs (21 in all) where the stop head would say stop in the first 0.5 s after PARK, replayed along the expert's own path.


```python
choice = json.loads((ARM / 'results/stop_weight_choice_25.json').read_text())
t = pd.DataFrame(choice['weights']).T
t.index.name = 'stop loss weight'
t['torque loss vs weight 1'] = t.val_torque / t.val_torque.loc['1']
display(t.rename(columns={'val_torque': 'validation torque loss', 'val_stop': 'validation stop loss',
                          'early_stops': 'held-out runs stopping in the first 0.5 s', 'runs': 'held-out runs'}).round(4))
print('rule:', choice['rule'], '| allowed:', choice['allowed'], '| chosen:', choice['chosen'])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>validation torque loss</th>
      <th>validation stop loss</th>
      <th>held-out runs stopping in the first 0.5 s</th>
      <th>held-out runs</th>
      <th>torque loss vs weight 1</th>
    </tr>
    <tr>
      <th>stop loss weight</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.3706</td>
      <td>0.0023</td>
      <td>10.0</td>
      <td>21.0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.4105</td>
      <td>0.0017</td>
      <td>9.0</td>
      <td>21.0</td>
      <td>1.1078</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.4823</td>
      <td>0.0017</td>
      <td>5.0</td>
      <td>21.0</td>
      <td>1.3015</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.5915</td>
      <td>0.0016</td>
      <td>4.0</td>
      <td>21.0</td>
      <td>1.5962</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.7734</td>
      <td>0.0016</td>
      <td>4.0</td>
      <td>21.0</td>
      <td>2.0872</td>
    </tr>
  </tbody>
</table>
</div>


    rule: lowest mean validation stop loss with mean validation torque loss within 10% of weight 1 | allowed: [1] | chosen: 1


#### Iteration 25: validation loss curves

Validation losses averaged over the 5 folds, for every epoch, for each stop loss weight.


```python
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
for w in (1, 3, 10, 30, 100):
    H = [pd.DataFrame(json.loads((ARM / f'models/arm25_w{w}_fold{f}_history.json').read_text())) for f in range(5)]
    for a, k in zip(ax, ('val_torque_mse', 'val_distance_mse')):
        a.plot(H[0].epoch, np.mean([h[k] for h in H], 0), label=f'weight {w}')
for a, t in zip(ax, ('validation torque loss (standardised units)', 'validation stop loss (scaled 0..1, unweighted)')):
    a.set(yscale='log', xlabel='epoch', title=t); a.legend(fontsize=8)
plt.show()
```


    
![png](driving_v2_sensors_files/driving_v2_sensors_229_0.png)
    


#### Iteration 25: reading the results

Against the expectation:

- **Not as expected: the rule keeps weight 1.** A larger weight lowers the validation stop loss (0.0023 at weight 1, 0.0017 at weight 3, 0.0016 at 30 and 100), but every larger weight raises the validation torque loss by more than 10%: 11% at weight 3, 30% at 10, 60% at 30 and 109% at 100. Weight 3 misses the limit only just.
- So the network for iteration 25 is the same as iteration 24's (same settings, same seed, same weight), and it is not tested again. Its camera result is iteration 24's.

What it tells us:

- With 77 epochs the stop and the torques compete for the same two shared layers. A larger stop weight makes the head say stop at PARK on fewer held-out runs along the expert's path, from 10 of 21 at weight 1 to 4 of 21 at weights 30 and 100, but it costs torque accuracy that the reach needs.
- The validation stop losses all flatten after about 40 epochs while the torque losses are still falling at 77. Both have room to improve with longer training, which is what iteration 26 tests at weight 1.
