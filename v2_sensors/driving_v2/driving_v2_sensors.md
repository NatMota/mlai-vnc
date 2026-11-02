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
from IPython.display import IFrame
def player(name):
    path = replay_v2.build_viewer(name)
    return IFrame('/files/driving_v2/closed_loop_native/' + path.name, width='100%', height=810)
player('wheels+live_dot')
```





<iframe
    width="100%"
    height="810"
    src="/files/driving_v2/closed_loop_native/viewer_wheels_live_dot.html"
    frameborder="0"
    allowfullscreen

></iframe>




### `wheels+start_dot` (dot direction at the start only)

<a href="/files/driving_v2/closed_loop_native/viewer_wheels_start_dot.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
player('wheels+start_dot')
```





<iframe
    width="100%"
    height="810"
    src="/files/driving_v2/closed_loop_native/viewer_wheels_start_dot.html"
    frameborder="0"
    allowfullscreen

></iframe>




### `wheels` (no dot)

<a href="/files/driving_v2/closed_loop_native/viewer_wheels.html" target="_blank" rel="noopener">Open this player in a separate tab</a>


```python
player('wheels')
```





<iframe
    width="100%"
    height="810"
    src="/files/driving_v2/closed_loop_native/viewer_wheels.html"
    frameborder="0"
    allowfullscreen

></iframe>




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





<iframe
    width="100%"
    height="810"
    src="/files/driving_v2/closed_loop_native/viewer_wheels_live_dot_skid.html"
    frameborder="0"
    allowfullscreen

></iframe>




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
    

