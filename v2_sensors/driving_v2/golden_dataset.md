# COMP0245: golden dataset experiments

This notebook starts from the frozen 300-target dataset and its 200 camera traces. It carries forward the three freshly trained baselines, iterations 34–36, as the starting point for the next experiments. The complete earlier notebook, `driving_v2_sensors.ipynb`, is preserved unchanged.

Re-running this notebook reads and plots saved results. It does not retrain models or open reserved test trajectories. The full data package is on university OneDrive; this repository contains its manifests and a small, explicitly labelled training preview. Use `GOLDEN_DATA_ROOT` for the location of the extracted package when running the training commands in `../arm/GOLDEN_DATA.md`.


```python
from pathlib import Path
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
sys.path.insert(0, str(Path('../arm').resolve()))
import report_golden as golden
from report_golden import MODELS, RESULTS
GOLDEN_DATA_ROOT = Path(os.environ['GOLDEN_DATA_ROOT']).expanduser() if os.environ.get('GOLDEN_DATA_ROOT') else None
print('Full dataset:', GOLDEN_DATA_ROOT if GOLDEN_DATA_ROOT else 'not configured; saved manifests, preview and results remain available')
from IPython.display import IFrame, HTML
import uuid

def lazy_player(url, height=810):
    """A 3D player that loads only when its button is pressed. Browsers keep only about 16 3D views (WebGL
    contexts) at once, so loading every player in the notebook together leaves some of them blank."""
    k = uuid.uuid4().hex[:8]
    return HTML(f"""<div><button onclick="var f=document.getElementById('player{k}');f.src='{url}';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player{k}" style="display:none;width:100%;height:{height}px;border:0"></iframe></div>""")

```

    Full dataset: not configured; saved manifests, preview and results remain available


## Dataset and split

There are 100 training, 100 validation and 100 reserved test expert runs. Offline camera observations exist for training and validation. Test cameras will be rendered live along each learned trajectory in ticket #11110, so expert test camera traces are not needed. Target IDs remain fixed across comparisons.


```python
split = golden.read(MODELS/'dataset_split.json')
manifest = golden.read(MODELS/'dataset_manifest.json')
audit = pd.DataFrame(golden.read(MODELS/'data_audit.json')['rows'])
assert all(len(set(v)) == 100 for v in split.values())
assert len(set().union(*map(set, split.values()))) == 300
display(pd.DataFrame([dict(split=k, expert_runs=len(v), offline_camera_runs=0 if k=='test' else len(v), role={'train':'fit weights and learned scaling','val':'choose epochs and run integration checks','test':'final evaluation only'}[k]) for k,v in split.items()]))
display(audit.groupby('split').agg(runs=('dot','count'),driving_rows=('drive_rows','sum'),arm_rows=('arm_rows','sum'),parked_tail_min=('parked_rows','min'),parked_tail_max=('parked_rows','max')))
print('Frozen split SHA256:', golden.read(MODELS/'data_audit.json')['split_sha256'])
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
      <th>expert_runs</th>
      <th>offline_camera_runs</th>
      <th>role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train</td>
      <td>100</td>
      <td>100</td>
      <td>fit weights and learned scaling</td>
    </tr>
    <tr>
      <th>1</th>
      <td>val</td>
      <td>100</td>
      <td>100</td>
      <td>choose epochs and run integration checks</td>
    </tr>
    <tr>
      <th>2</th>
      <td>test</td>
      <td>100</td>
      <td>0</td>
      <td>final evaluation only</td>
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
      <th>runs</th>
      <th>driving_rows</th>
      <th>arm_rows</th>
      <th>parked_tail_min</th>
      <th>parked_tail_max</th>
    </tr>
    <tr>
      <th>split</th>
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
      <td>100</td>
      <td>25928</td>
      <td>295084</td>
      <td>30</td>
      <td>100</td>
    </tr>
    <tr>
      <th>val</th>
      <td>100</td>
      <td>26428</td>
      <td>288818</td>
      <td>13</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>


    Frozen split SHA256: 286f578aec7e3ab0953592466e63014cc1dde7f467106fadf273b30762ac1abf


## Model inputs and training targets

Driving has 16 inputs: four accumulated encoder angles, four encoder speeds, four combinations of those readings, and camera-tracked range/sin bearing/cos bearing/visibility. Reaching has 17 inputs: seven arm encoder angles, seven encoder speeds and camera-tracked target xyz in the robot frame. Values become FP32 tensors. The camera runs at 30 Hz; its latest estimate is held between arm steps at 1 kHz.

Training targets are the expert wheel-speed and arm-torque commands, normalized time to PARK and normalized gripper distance for STOP. Simulator truth is permitted for labels and scoring; it is not fed into the learned policies.

The existing simulator retains its PD-plus-model-gravity arm hold while driving. After PARK, the learned torque network controls the arm, including the 0.5 s following STOP. That fixed pre-PARK holding controller is part of this baseline and is not a learned sensor-only arm controller. STOP does not freeze physics.


```python
preview = dict(np.load(MODELS/'training_preview.npz', allow_pickle=False))
wheel_names = [f'wheel_angle_{j}_rad' for j in range(1,5)] + [f'wheel_speed_{j}_rad_s' for j in range(1,5)] + ['mean_turn_rad','right_minus_left_turn_rad','mean_speed_rad_s','right_minus_left_speed_rad_s']
drive_names = wheel_names + ['tracked_range_m','tracked_bearing_sin','tracked_bearing_cos','visible']
arm_names = [f'arm_angle_{j}_rad' for j in range(1,8)] + [f'arm_speed_{j}_rad_s' for j in range(1,8)] + ['tracked_dot_x_m','tracked_dot_y_m','tracked_dot_z_m']
print('Preview only: training target', int(preview['dot']), '; five source rows per phase, not the whole dataset.')
display(pd.DataFrame(preview['driving_x'], columns=drive_names).assign(step=preview['drive_steps']).set_index('step'))
display(pd.DataFrame(preview['arm_x'], columns=arm_names).assign(step=preview['arm_steps']).set_index('step'))
print('Input types:', preview['driving_x'].dtype, preview['arm_x'].dtype)
```

    Preview only: training target 0 ; five source rows per phase, not the whole dataset.



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
      <th>wheel_angle_1_rad</th>
      <th>wheel_angle_2_rad</th>
      <th>wheel_angle_3_rad</th>
      <th>wheel_angle_4_rad</th>
      <th>wheel_speed_1_rad_s</th>
      <th>wheel_speed_2_rad_s</th>
      <th>wheel_speed_3_rad_s</th>
      <th>wheel_speed_4_rad_s</th>
      <th>mean_turn_rad</th>
      <th>right_minus_left_turn_rad</th>
      <th>mean_speed_rad_s</th>
      <th>right_minus_left_speed_rad_s</th>
      <th>tracked_range_m</th>
      <th>tracked_bearing_sin</th>
      <th>tracked_bearing_cos</th>
      <th>visible</th>
    </tr>
    <tr>
      <th>step</th>
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-0.090505</td>
      <td>0.088971</td>
      <td>0.090505</td>
      <td>-0.092039</td>
      <td>-2.715146</td>
      <td>2.669127</td>
      <td>2.715146</td>
      <td>-2.761165</td>
      <td>-0.000767</td>
      <td>-0.181010</td>
      <td>-0.02301</td>
      <td>-5.430292</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>67</th>
      <td>-0.181010</td>
      <td>0.179476</td>
      <td>0.185612</td>
      <td>-0.187146</td>
      <td>-2.715146</td>
      <td>2.715146</td>
      <td>2.853204</td>
      <td>-2.853204</td>
      <td>-0.000767</td>
      <td>-0.366621</td>
      <td>0.00000</td>
      <td>-5.568350</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>-0.268447</td>
      <td>0.266913</td>
      <td>0.277651</td>
      <td>-0.279184</td>
      <td>-2.623107</td>
      <td>2.623107</td>
      <td>2.761165</td>
      <td>-2.761165</td>
      <td>-0.000767</td>
      <td>-0.546097</td>
      <td>0.00000</td>
      <td>-5.384273</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>133</th>
      <td>-0.354350</td>
      <td>0.352816</td>
      <td>0.369689</td>
      <td>-0.371223</td>
      <td>-2.577088</td>
      <td>2.577088</td>
      <td>2.761165</td>
      <td>-2.761165</td>
      <td>-0.000767</td>
      <td>-0.724039</td>
      <td>0.00000</td>
      <td>-5.338253</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <th>arm_angle_1_rad</th>
      <th>arm_angle_2_rad</th>
      <th>arm_angle_3_rad</th>
      <th>arm_angle_4_rad</th>
      <th>arm_angle_5_rad</th>
      <th>arm_angle_6_rad</th>
      <th>arm_angle_7_rad</th>
      <th>arm_speed_1_rad_s</th>
      <th>arm_speed_2_rad_s</th>
      <th>arm_speed_3_rad_s</th>
      <th>arm_speed_4_rad_s</th>
      <th>arm_speed_5_rad_s</th>
      <th>arm_speed_6_rad_s</th>
      <th>arm_speed_7_rad_s</th>
      <th>tracked_dot_x_m</th>
      <th>tracked_dot_y_m</th>
      <th>tracked_dot_z_m</th>
    </tr>
    <tr>
      <th>step</th>
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
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6230</th>
      <td>-0.014631</td>
      <td>1.515667</td>
      <td>0.000216</td>
      <td>1.014285</td>
      <td>-0.006637</td>
      <td>1.000131</td>
      <td>-0.001473</td>
      <td>-0.107379</td>
      <td>0.006818</td>
      <td>0.049428</td>
      <td>0.001704</td>
      <td>-0.081812</td>
      <td>0.000000</td>
      <td>0.029750</td>
      <td>0.42766</td>
      <td>-0.034332</td>
      <td>1.238958</td>
    </tr>
    <tr>
      <th>6231</th>
      <td>-0.014738</td>
      <td>1.515657</td>
      <td>0.000269</td>
      <td>1.014328</td>
      <td>-0.006721</td>
      <td>1.000200</td>
      <td>-0.001448</td>
      <td>-0.107379</td>
      <td>-0.010227</td>
      <td>0.052837</td>
      <td>0.042611</td>
      <td>-0.084291</td>
      <td>0.069417</td>
      <td>0.024792</td>
      <td>0.42766</td>
      <td>-0.034332</td>
      <td>1.238958</td>
    </tr>
    <tr>
      <th>6232</th>
      <td>-0.014849</td>
      <td>1.515638</td>
      <td>0.000320</td>
      <td>1.014396</td>
      <td>-0.006803</td>
      <td>1.000322</td>
      <td>-0.001423</td>
      <td>-0.110788</td>
      <td>-0.018749</td>
      <td>0.051133</td>
      <td>0.068177</td>
      <td>-0.081812</td>
      <td>0.121479</td>
      <td>0.024792</td>
      <td>0.42766</td>
      <td>-0.034332</td>
      <td>1.238958</td>
    </tr>
    <tr>
      <th>6233</th>
      <td>-0.014963</td>
      <td>1.515610</td>
      <td>0.000373</td>
      <td>1.014485</td>
      <td>-0.006885</td>
      <td>1.000483</td>
      <td>-0.001401</td>
      <td>-0.114196</td>
      <td>-0.027271</td>
      <td>0.052837</td>
      <td>0.088630</td>
      <td>-0.081812</td>
      <td>0.161145</td>
      <td>0.022312</td>
      <td>0.42619</td>
      <td>-0.038042</td>
      <td>1.239363</td>
    </tr>
    <tr>
      <th>6234</th>
      <td>-0.015081</td>
      <td>1.515576</td>
      <td>0.000424</td>
      <td>1.014587</td>
      <td>-0.006966</td>
      <td>1.000674</td>
      <td>-0.001376</td>
      <td>-0.117605</td>
      <td>-0.034088</td>
      <td>0.051133</td>
      <td>0.102265</td>
      <td>-0.081812</td>
      <td>0.190895</td>
      <td>0.024792</td>
      <td>0.42619</td>
      <td>-0.038042</td>
      <td>1.239363</td>
    </tr>
  </tbody>
</table>
</div>


    Input types: float32 float32



```python
display(pd.DataFrame(preview['driving_commands'], columns=[f'expert_wheel_{j}_rad_s' for j in range(1,5)]).assign(time_to_park_scaled=preview['driving_ttp'],step=preview['drive_steps']).set_index('step'))
display(pd.DataFrame(preview['arm_commands'], columns=[f'expert_torque_{j}_Nm' for j in range(1,8)]).assign(stop_distance_scaled=preview['arm_stop_target'],step=preview['arm_steps']).set_index('step'))
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
      <th>expert_wheel_1_rad_s</th>
      <th>expert_wheel_2_rad_s</th>
      <th>expert_wheel_3_rad_s</th>
      <th>expert_wheel_4_rad_s</th>
      <th>time_to_park_scaled</th>
    </tr>
    <tr>
      <th>step</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>33</th>
      <td>-3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>67</th>
      <td>-3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>100</th>
      <td>-3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>133</th>
      <td>-3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>1.0</td>
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
      <th>expert_torque_1_Nm</th>
      <th>expert_torque_2_Nm</th>
      <th>expert_torque_3_Nm</th>
      <th>expert_torque_4_Nm</th>
      <th>expert_torque_5_Nm</th>
      <th>expert_torque_6_Nm</th>
      <th>expert_torque_7_Nm</th>
      <th>stop_distance_scaled</th>
    </tr>
    <tr>
      <th>step</th>
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
      <th>6230</th>
      <td>5.316199</td>
      <td>28.082699</td>
      <td>-1.365561</td>
      <td>-6.598114</td>
      <td>3.302955</td>
      <td>22.673250</td>
      <td>-0.699793</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6231</th>
      <td>5.320834</td>
      <td>28.591558</td>
      <td>-1.470603</td>
      <td>-7.839973</td>
      <td>3.385210</td>
      <td>20.551292</td>
      <td>-0.551579</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6232</th>
      <td>5.459431</td>
      <td>28.937777</td>
      <td>-1.435724</td>
      <td>-8.689305</td>
      <td>3.328572</td>
      <td>19.007622</td>
      <td>-0.613395</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6233</th>
      <td>5.542691</td>
      <td>29.200615</td>
      <td>-1.472973</td>
      <td>-9.291978</td>
      <td>3.338620</td>
      <td>17.821453</td>
      <td>-0.567547</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6234</th>
      <td>5.643270</td>
      <td>29.394722</td>
      <td>-1.480658</td>
      <td>-9.713266</td>
      <td>3.327830</td>
      <td>16.905899</td>
      <td>-0.562102</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


## Frozen baselines

The following results are copied from the completed golden-data training and validation work. No new training was started to create this notebook. The source notebook remains unchanged. Each seed has the same analysis and its own lazy player, so the browser does not load every 3D view at once.

### Golden data: driving and reaching, three seeds (iterations 34 to 36)

#### Iterations 34 to 36: plan and expected results

Written before training. Ticket #11109 follows the 4 November roadmap: retrain the iteration 13 driving architecture and iteration 27 reaching architecture on 100 training runs, choose epochs on 100 validation runs, and reserve the 100 test runs for ticket #11110. Seeds 7, 1 and 2 are iterations 34, 35 and 36 respectively. Earlier notebook results stay unchanged.

Driving uses 16 encoder/camera inputs at 30 Hz, two 16-unit ReLU layers, four wheel-speed outputs and a separate time-to-PARK head. The target is seconds to the recorded PARK, clipped at 3 s and divided by 3; the PARK threshold remains 0.25 s. The loss is wheel-speed MSE plus normalized time-to-PARK MSE, with each run weighted equally. Include at most 100 available, real parked frames after PARK. Fit the six angle-feature scales on training only.

Reaching uses 17 inputs at 1 kHz: seven encoder angles, seven encoder speeds and camera/tracker xyz, held between camera frames. Its torque path has two 64-unit ReLU layers and seven torque outputs; the independent stop network is 17 -> 32 -> 16 -> 1. Preserve the inherited auxiliary distance loss when training the torque path; the independent stop network supplies runtime STOP. Both distance targets use the aligned expert distance, clipped at 0.25 m and divided by 0.25; STOP remains 10 mm. Each run has equal weight. Encoder resolutions, fixed input scales and runtime clipping remain unchanged.

Every network starts from fresh weights. Use FP32, Adam at 0.01, up to 300 epochs for driving/torques and 600 for independent stop. Select the lowest declared validation objective and restore that checkpoint. Any chunking accumulates the same full-batch gradient before one optimizer update. Validate exported NumPy predictions against PyTorch before simulation.

Expected result: validation losses should fall below their epoch-zero values, and the broader range of target heights should reduce the single-height behaviour seen in iteration 30. This is a hypothesis, not a claimed result. It is contradicted if validation does not improve, or if recorded validation reaches continue to converge to one height regardless of the camera estimate. A lower offline loss alone does not establish successful parking or reaching.

Use a fixed validation subset, selected before training, for full learned drive/PARK/reach/STOP demonstrations. The expert is only a recorded reference/ghost. Report every seed, camera visibility, PARK, STOP, distance at STOP and timeouts; never tune against the reserved test runs. There is no PPO, weight perturbation, new CNN, DART or DAgger in this experiment.

Status: all three seeds trained on the GPU in 87.7 seconds including loading and export checks. The results below are validation simulations only. The 100 test trajectories were not opened.

#### Golden data: checks and training setup

The source and camera hashes matched for all 100 training and 100 validation runs. Camera observations were replayed at their recorded 30 Hz capture times and held between frames; arm encoders use a 1 ms backward difference. The wheel speeds retain the existing nominal 1/30 s denominator at rounded frame steps, matching the recorded features and runtime. No new scaling convention was introduced.

Some runs end before 100 parked camera frames are available. We use only the real recorded tail: 30–100 frames in training and 13–100 in validation. No synthetic tail or repeated row was added. Each run contributes equally to the objective.

This clean notebook contains the golden-data experiment only. Its analysis rereads the frozen outputs; training is a separate explicit command.


```python
from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
sys.path.insert(0, str(Path('../arm').resolve()))
import report_golden as golden
from report_golden import MODELS, RESULTS
training = golden.read(MODELS/'summary.json')
audit = pd.DataFrame(golden.read(MODELS/'data_audit.json')['rows'])
display(audit.groupby('split').agg(runs=('dot','count'), driving_rows=('drive_rows','sum'), arm_rows=('arm_rows','sum'), min_parked_frames=('parked_rows','min'), max_parked_frames=('parked_rows','max')))
print('Training wall time:', round(training['elapsed_s'],1), 'seconds; representative seed:', training['representative_seed'])
print('Test trajectories opened:', golden.read(MODELS/'data_audit.json')['test_trajectories_opened'])
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
      <th>driving_rows</th>
      <th>arm_rows</th>
      <th>min_parked_frames</th>
      <th>max_parked_frames</th>
    </tr>
    <tr>
      <th>split</th>
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
      <td>100</td>
      <td>25928</td>
      <td>295084</td>
      <td>30</td>
      <td>100</td>
    </tr>
    <tr>
      <th>val</th>
      <td>100</td>
      <td>26428</td>
      <td>288818</td>
      <td>13</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>


    Training wall time: 87.7 seconds; representative seed: 7
    Test trajectories opened: 0


#### Iteration 34: golden data, seed 7, results

The same 10 validation targets were fixed before training. Both learned networks control the robot; the recorded expert is only the comparison. Success requires PARK, STOP and actual gripper distance at STOP of at most 10 mm. The subsequent hold and drift do not change that score.


```python
display(golden.table(7).round(3))
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
      <th>parked</th>
      <th>park_s</th>
      <th>park_error_cm</th>
      <th>stop_s</th>
      <th>at_stop_mm</th>
      <th>closest_mm</th>
      <th>end_mm</th>
      <th>success</th>
      <th>camera_visible_fraction</th>
      <th>elapsed_s</th>
    </tr>
    <tr>
      <th>dot</th>
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
      <th>6</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4208.246</td>
      <td>False</td>
      <td>0.000</td>
      <td>19.529</td>
    </tr>
    <tr>
      <th>131</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4790.008</td>
      <td>False</td>
      <td>0.000</td>
      <td>20.079</td>
    </tr>
    <tr>
      <th>204</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4529.255</td>
      <td>False</td>
      <td>0.000</td>
      <td>19.265</td>
    </tr>
    <tr>
      <th>268</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4295.174</td>
      <td>False</td>
      <td>0.000</td>
      <td>18.708</td>
    </tr>
    <tr>
      <th>374</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4063.391</td>
      <td>False</td>
      <td>0.000</td>
      <td>18.454</td>
    </tr>
    <tr>
      <th>506</th>
      <td>True</td>
      <td>8.133</td>
      <td>42.273</td>
      <td>9.427</td>
      <td>662.505</td>
      <td>577.694</td>
      <td>702.714</td>
      <td>False</td>
      <td>0.923</td>
      <td>14.422</td>
    </tr>
    <tr>
      <th>587</th>
      <td>True</td>
      <td>4.167</td>
      <td>12.980</td>
      <td>4.790</td>
      <td>124.919</td>
      <td>102.076</td>
      <td>222.339</td>
      <td>False</td>
      <td>0.862</td>
      <td>8.880</td>
    </tr>
    <tr>
      <th>676</th>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3413.383</td>
      <td>False</td>
      <td>0.000</td>
      <td>17.313</td>
    </tr>
    <tr>
      <th>823</th>
      <td>True</td>
      <td>5.233</td>
      <td>12.807</td>
      <td>6.512</td>
      <td>514.380</td>
      <td>446.754</td>
      <td>676.080</td>
      <td>False</td>
      <td>1.000</td>
      <td>10.663</td>
    </tr>
    <tr>
      <th>998</th>
      <td>True</td>
      <td>7.100</td>
      <td>9.359</td>
      <td>8.458</td>
      <td>507.962</td>
      <td>97.909</td>
      <td>487.271</td>
      <td>False</td>
      <td>0.918</td>
      <td>12.630</td>
    </tr>
  </tbody>
</table>
</div>


#### Iteration 34: loss curves

Solid lines are training and dashed lines are validation. Vertical markers show the independently selected checkpoints. The torque path retains its auxiliary distance loss, while the separate distance network supplies runtime STOP. The final test set is absent from these curves.


```python
golden.losses(7); plt.show()
```



![png](golden_dataset_files/golden_dataset_14_0.png)



#### Iteration 34: parking, reaching and camera visibility

Blank parking bars mean the model did not PARK. Distance statistics apply only where reaching started; the full-task denominator remains 10. The visibility fraction is measured from rendered images, not from known target coordinates.


```python
golden.outcomes(7); plt.show()
```



![png](golden_dataset_files/golden_dataset_16_0.png)



#### Iteration 34: predicted and actual stopping distance

Actual distance is used only for scoring. The predicted distance comes from the stop network. The vertical line is STOP; the dotted horizontal line is 10 mm. The historical log key `ttp_s` now carries distance in metres for this distance-stop model, so it is converted to mm here.


```python
golden.distances(7); plt.show()
```



![png](golden_dataset_files/golden_dataset_18_0.png)



#### Iteration 34: 3D replay with ghost expert and camera

The player contains every recorded validation trajectory without cutting off failures. It loads only when requested. The top-right image is the camera frame captured along the model's actual motion, reduced from 256 px to 128 px for display; detection used the full image.


```python
player_path = golden.build_viewer(7)
player_url = '/files/arm/results/golden_11109/' + player_path.name
display(HTML(f'<a href="{player_url}" target="_blank">Open the seed 7 player separately</a>'))
display(lazy_player(player_url))
```


<a href="/files/arm/results/golden_11109/viewer_golden_seed7.html" target="_blank">Open the seed 7 player separately</a>



<div><button onclick="var f=document.getElementById('playerf31506b5');f.src='/files/arm/results/golden_11109/viewer_golden_seed7.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="playerf31506b5" style="display:none;width:100%;height:810px;border:0"></iframe></div>


#### Iteration 34: reading the results


```python
display(Markdown(golden.reading(7)))
```


Validation losses fell: driving 3.824 to 0.05626 (epoch 300); torque 1.038 to 0.0382 (epoch 300); stop 0.2512 to 0.001658 (epoch 600). In the 10 predetermined simulation checks, 4/10 parked, 4/10 issued STOP, and 0/10 stopped within 10 mm. There was 511 mm median distance at STOP across the 4 runs that stopped. Missing PARK or STOP is a failed full task, not a missing observation to discard. These are integration checks on validation targets, not the final 100-target test score. A better offline fit has not established reliable end-to-end reaching. The untouched test evaluation is ticket #11110.


#### Iteration 35: golden data, seed 1, results

The same 10 validation targets were fixed before training. Both learned networks control the robot; the recorded expert is only the comparison. Success requires PARK, STOP and actual gripper distance at STOP of at most 10 mm. The subsequent hold and drift do not change that score.


```python
display(golden.table(1).round(3))
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
      <th>parked</th>
      <th>park_s</th>
      <th>park_error_cm</th>
      <th>stop_s</th>
      <th>at_stop_mm</th>
      <th>closest_mm</th>
      <th>end_mm</th>
      <th>success</th>
      <th>camera_visible_fraction</th>
      <th>elapsed_s</th>
    </tr>
    <tr>
      <th>dot</th>
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
      <th>6</th>
      <td>True</td>
      <td>8.833</td>
      <td>31.177</td>
      <td>10.470</td>
      <td>134.386</td>
      <td>116.114</td>
      <td>127.963</td>
      <td>False</td>
      <td>0.685</td>
      <td>15.756</td>
    </tr>
    <tr>
      <th>131</th>
      <td>True</td>
      <td>7.333</td>
      <td>52.446</td>
      <td>9.318</td>
      <td>1477.819</td>
      <td>617.023</td>
      <td>1065.524</td>
      <td>False</td>
      <td>0.607</td>
      <td>14.632</td>
    </tr>
    <tr>
      <th>204</th>
      <td>True</td>
      <td>8.400</td>
      <td>38.241</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>445.983</td>
      <td>728.904</td>
      <td>False</td>
      <td>0.808</td>
      <td>19.280</td>
    </tr>
    <tr>
      <th>268</th>
      <td>True</td>
      <td>10.033</td>
      <td>32.301</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>391.244</td>
      <td>512.963</td>
      <td>False</td>
      <td>0.578</td>
      <td>19.101</td>
    </tr>
    <tr>
      <th>374</th>
      <td>True</td>
      <td>7.033</td>
      <td>18.590</td>
      <td>10.657</td>
      <td>566.591</td>
      <td>296.321</td>
      <td>571.398</td>
      <td>False</td>
      <td>0.833</td>
      <td>15.909</td>
    </tr>
    <tr>
      <th>506</th>
      <td>True</td>
      <td>5.033</td>
      <td>39.056</td>
      <td>8.935</td>
      <td>1010.400</td>
      <td>565.693</td>
      <td>1094.893</td>
      <td>False</td>
      <td>0.930</td>
      <td>14.263</td>
    </tr>
    <tr>
      <th>587</th>
      <td>True</td>
      <td>4.600</td>
      <td>2.000</td>
      <td>6.540</td>
      <td>981.281</td>
      <td>71.855</td>
      <td>1031.870</td>
      <td>False</td>
      <td>0.958</td>
      <td>10.845</td>
    </tr>
    <tr>
      <th>676</th>
      <td>True</td>
      <td>5.533</td>
      <td>16.005</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.541</td>
      <td>256.931</td>
      <td>False</td>
      <td>0.808</td>
      <td>16.898</td>
    </tr>
    <tr>
      <th>823</th>
      <td>True</td>
      <td>4.200</td>
      <td>5.481</td>
      <td>9.385</td>
      <td>283.887</td>
      <td>161.203</td>
      <td>161.125</td>
      <td>False</td>
      <td>0.980</td>
      <td>14.471</td>
    </tr>
    <tr>
      <th>998</th>
      <td>True</td>
      <td>4.800</td>
      <td>12.653</td>
      <td>6.231</td>
      <td>227.288</td>
      <td>60.592</td>
      <td>60.726</td>
      <td>False</td>
      <td>0.995</td>
      <td>10.491</td>
    </tr>
  </tbody>
</table>
</div>


#### Iteration 35: loss curves

Solid lines are training and dashed lines are validation. Vertical markers show the independently selected checkpoints. The torque path retains its auxiliary distance loss, while the separate distance network supplies runtime STOP. The final test set is absent from these curves.


```python
golden.losses(1); plt.show()
```



![png](golden_dataset_files/golden_dataset_26_0.png)



#### Iteration 35: parking, reaching and camera visibility

Blank parking bars mean the model did not PARK. Distance statistics apply only where reaching started; the full-task denominator remains 10. The visibility fraction is measured from rendered images, not from known target coordinates.


```python
golden.outcomes(1); plt.show()
```



![png](golden_dataset_files/golden_dataset_28_0.png)



#### Iteration 35: predicted and actual stopping distance

Actual distance is used only for scoring. The predicted distance comes from the stop network. The vertical line is STOP; the dotted horizontal line is 10 mm. The historical log key `ttp_s` now carries distance in metres for this distance-stop model, so it is converted to mm here.


```python
golden.distances(1); plt.show()
```



![png](golden_dataset_files/golden_dataset_30_0.png)



#### Iteration 35: 3D replay with ghost expert and camera

The player contains every recorded validation trajectory without cutting off failures. It loads only when requested. The top-right image is the camera frame captured along the model's actual motion, reduced from 256 px to 128 px for display; detection used the full image.


```python
player_path = golden.build_viewer(1)
player_url = '/files/arm/results/golden_11109/' + player_path.name
display(HTML(f'<a href="{player_url}" target="_blank">Open the seed 1 player separately</a>'))
display(lazy_player(player_url))
```


<a href="/files/arm/results/golden_11109/viewer_golden_seed1.html" target="_blank">Open the seed 1 player separately</a>



<div><button onclick="var f=document.getElementById('player7693669c');f.src='/files/arm/results/golden_11109/viewer_golden_seed1.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player7693669c" style="display:none;width:100%;height:810px;border:0"></iframe></div>


#### Iteration 35: reading the results


```python
display(Markdown(golden.reading(1)))
```


Validation losses fell: driving 3.734 to 0.05583 (epoch 300); torque 0.9775 to 0.03844 (epoch 288); stop 0.1533 to 0.001015 (epoch 598). In the 10 predetermined simulation checks, 10/10 parked, 7/10 issued STOP, and 0/10 stopped within 10 mm. There was 567 mm median distance at STOP across the 7 runs that stopped. Missing PARK or STOP is a failed full task, not a missing observation to discard. These are integration checks on validation targets, not the final 100-target test score. A better offline fit has not established reliable end-to-end reaching. The untouched test evaluation is ticket #11110.


#### Iteration 36: golden data, seed 2, results

The same 10 validation targets were fixed before training. Both learned networks control the robot; the recorded expert is only the comparison. Success requires PARK, STOP and actual gripper distance at STOP of at most 10 mm. The subsequent hold and drift do not change that score.


```python
display(golden.table(2).round(3))
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
      <th>parked</th>
      <th>park_s</th>
      <th>park_error_cm</th>
      <th>stop_s</th>
      <th>at_stop_mm</th>
      <th>closest_mm</th>
      <th>end_mm</th>
      <th>success</th>
      <th>camera_visible_fraction</th>
      <th>elapsed_s</th>
    </tr>
    <tr>
      <th>dot</th>
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
      <th>6</th>
      <td>True</td>
      <td>5.233</td>
      <td>48.670</td>
      <td>13.296</td>
      <td>1768.190</td>
      <td>525.444</td>
      <td>1166.853</td>
      <td>False</td>
      <td>0.800</td>
      <td>20.509</td>
    </tr>
    <tr>
      <th>131</th>
      <td>True</td>
      <td>8.000</td>
      <td>54.903</td>
      <td>10.214</td>
      <td>479.916</td>
      <td>479.268</td>
      <td>505.060</td>
      <td>False</td>
      <td>0.652</td>
      <td>15.707</td>
    </tr>
    <tr>
      <th>204</th>
      <td>True</td>
      <td>10.267</td>
      <td>3.843</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>143.221</td>
      <td>572.360</td>
      <td>False</td>
      <td>0.843</td>
      <td>18.966</td>
    </tr>
    <tr>
      <th>268</th>
      <td>True</td>
      <td>11.400</td>
      <td>8.799</td>
      <td>12.958</td>
      <td>269.493</td>
      <td>121.851</td>
      <td>560.140</td>
      <td>False</td>
      <td>0.520</td>
      <td>18.387</td>
    </tr>
    <tr>
      <th>374</th>
      <td>True</td>
      <td>10.133</td>
      <td>31.693</td>
      <td>13.324</td>
      <td>782.514</td>
      <td>291.282</td>
      <td>883.650</td>
      <td>False</td>
      <td>0.904</td>
      <td>18.872</td>
    </tr>
    <tr>
      <th>506</th>
      <td>True</td>
      <td>7.300</td>
      <td>11.482</td>
      <td>12.455</td>
      <td>1080.763</td>
      <td>24.126</td>
      <td>1030.321</td>
      <td>False</td>
      <td>0.992</td>
      <td>18.791</td>
    </tr>
    <tr>
      <th>587</th>
      <td>True</td>
      <td>6.500</td>
      <td>20.838</td>
      <td>7.015</td>
      <td>258.240</td>
      <td>226.452</td>
      <td>246.977</td>
      <td>False</td>
      <td>0.681</td>
      <td>11.941</td>
    </tr>
    <tr>
      <th>676</th>
      <td>True</td>
      <td>9.733</td>
      <td>36.906</td>
      <td>9.786</td>
      <td>437.288</td>
      <td>358.159</td>
      <td>357.977</td>
      <td>False</td>
      <td>0.722</td>
      <td>14.588</td>
    </tr>
    <tr>
      <th>823</th>
      <td>True</td>
      <td>5.900</td>
      <td>9.060</td>
      <td>8.577</td>
      <td>108.622</td>
      <td>106.571</td>
      <td>119.832</td>
      <td>False</td>
      <td>0.905</td>
      <td>13.009</td>
    </tr>
    <tr>
      <th>998</th>
      <td>True</td>
      <td>6.333</td>
      <td>6.869</td>
      <td>8.621</td>
      <td>317.891</td>
      <td>21.385</td>
      <td>328.472</td>
      <td>False</td>
      <td>0.993</td>
      <td>12.633</td>
    </tr>
  </tbody>
</table>
</div>


#### Iteration 36: loss curves

Solid lines are training and dashed lines are validation. Vertical markers show the independently selected checkpoints. The torque path retains its auxiliary distance loss, while the separate distance network supplies runtime STOP. The final test set is absent from these curves.


```python
golden.losses(2); plt.show()
```



![png](golden_dataset_files/golden_dataset_38_0.png)



#### Iteration 36: parking, reaching and camera visibility

Blank parking bars mean the model did not PARK. Distance statistics apply only where reaching started; the full-task denominator remains 10. The visibility fraction is measured from rendered images, not from known target coordinates.


```python
golden.outcomes(2); plt.show()
```



![png](golden_dataset_files/golden_dataset_40_0.png)



#### Iteration 36: predicted and actual stopping distance

Actual distance is used only for scoring. The predicted distance comes from the stop network. The vertical line is STOP; the dotted horizontal line is 10 mm. The historical log key `ttp_s` now carries distance in metres for this distance-stop model, so it is converted to mm here.


```python
golden.distances(2); plt.show()
```



![png](golden_dataset_files/golden_dataset_42_0.png)



#### Iteration 36: 3D replay with ghost expert and camera

The player contains every recorded validation trajectory without cutting off failures. It loads only when requested. The top-right image is the camera frame captured along the model's actual motion, reduced from 256 px to 128 px for display; detection used the full image.


```python
player_path = golden.build_viewer(2)
player_url = '/files/arm/results/golden_11109/' + player_path.name
display(HTML(f'<a href="{player_url}" target="_blank">Open the seed 2 player separately</a>'))
display(lazy_player(player_url))
```


<a href="/files/arm/results/golden_11109/viewer_golden_seed2.html" target="_blank">Open the seed 2 player separately</a>



<div><button onclick="var f=document.getElementById('player0bb1996c');f.src='/files/arm/results/golden_11109/viewer_golden_seed2.html';f.style.display='block';this.style.display='none'"
 style="padding:8px 14px;font-size:14px">Load the 3D player</button>
<iframe id="player0bb1996c" style="display:none;width:100%;height:810px;border:0"></iframe></div>


#### Iteration 36: reading the results


```python
display(Markdown(golden.reading(2)))
```


Validation losses fell: driving 3.887 to 0.05693 (epoch 300); torque 0.9929 to 0.03982 (epoch 300); stop 0.2839 to 0.00179 (epoch 598). In the 10 predetermined simulation checks, 10/10 parked, 9/10 issued STOP, and 0/10 stopped within 10 mm. There was 437 mm median distance at STOP across the 9 runs that stopped. Missing PARK or STOP is a failed full task, not a missing observation to discard. These are integration checks on validation targets, not the final 100-target test score. A better offline fit has not established reliable end-to-end reaching. The untouched test evaluation is ticket #11110.


#### Golden data: three-seed comparison and next step

Seed 7 is the median of the sum of the three selected validation objectives, chosen before simulation. It is not the best parking result. Keep that distinction when comparing later fixed-model interventions. These three initialisations are not three independent datasets.

The offline learning check passed, but none of these 30 validation attempts achieved the 10 mm full-task goal. This experiment does not establish reliable end-to-end reaching. Final test evaluation remains a separate frozen run in ticket #11110; no new optimisation was performed in response to these validation outcomes.


```python
summary=[]
for seed in (7,1,2):
    d=golden.table(seed)
    summary.append(dict(seed=seed,attempts=len(d),parked=int(d.parked.sum()),stopped=int(d.stop_s.notna().sum()),success_10mm=int(d.success.sum()),median_park_error_cm=d.park_error_cm.median(),median_stop_mm=d.at_stop_mm.median()))
display(pd.DataFrame(summary).set_index('seed').round(2))
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
      <th>attempts</th>
      <th>parked</th>
      <th>stopped</th>
      <th>success_10mm</th>
      <th>median_park_error_cm</th>
      <th>median_stop_mm</th>
    </tr>
    <tr>
      <th>seed</th>
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
      <th>7</th>
      <td>10</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>12.89</td>
      <td>511.17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10</td>
      <td>10</td>
      <td>7</td>
      <td>0</td>
      <td>24.88</td>
      <td>566.59</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>10</td>
      <td>9</td>
      <td>0</td>
      <td>16.16</td>
      <td>437.29</td>
    </tr>
  </tbody>
</table>
</div>


## Next experiments

The baseline models are fixed. Ticket #11110 evaluates all 100 reserved test targets for each seed. Later tickets add the declared noise and visibility conditions, with validation used for any training choice. Add each new hypothesis, result and replay below this point; keep the baselines above intact.

The lighthouse experiment will alternate target visibility during driving and keep it visible after PARK. This is a visual-only intervention. The training losses remain MSE against the expert targets; the 10 mm-at-STOP criterion is an evaluation measure.
