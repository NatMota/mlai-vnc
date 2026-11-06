# Reproduce the golden-arm experiment

## 1. Use the checked source branch and course environment

Use the source attached to release `comp0245-reproduction-2026-11-06-r2`.
For a fixed checkout:

```powershell
git clone --branch comp0245-reproduction-2026-11-06-r2 https://github.com/NatMota/mlai-vnc.git golden-reproduction
cd golden-reproduction
git rev-parse HEAD
```

The development branch is `experiments/v2-golden-arm`. The release tag fixes the reviewed version.
Prepare the environment using [ENVIRONMENT.md](ENVIRONMENT.md). Keep archived results outside
the new checkout's `outputs/golden/` directory.

Supply the course `models/` folder, including `models/summit_xl_arm_description/summit_xl_complete.urdf` and its meshes. Activate the course Python environment with PyTorch, NumPy, PyBullet, Pinocchio and `simulation_and_control`. Pinocchio is imported as `pinocchio`; its recorded distribution name is `pin`. The repository's `pixi.toml` alone does not declare the full simulator environment.

The archived run recorded Python 3.11.18, NumPy 2.4.6, PyTorch 2.14.1, PyBullet 3.2.5, pin 4.1.0, and simulation-and-control 0.1. These are recorded versions, not a new environment verified by this package.

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONIOENCODING = 'utf-8'
$env:ROBOENV_HEADLESS = '1'
$env:CAMERA_RENDERER = 'tiny'
python -B -c "import numpy, torch, pybullet, pinocchio, simulation_and_control; print('Imports OK')"
```

## 2. Obtain the source checkpoint

In the team's MLflow UI, find the run whose ID starts with `540bdcf9`. Download its `model/policy.pt` artifact and save it as `outputs/ablation/all_world/policy.pt` in the new checkout. Use the full run ID shown by MLflow if using a download API; the eight-character prefix is not a complete run ID. The source run ID is `540bdcf942a5448baefdc2802271bbf3`; downloading is read-only.

The expected source checkpoint SHA256 is `0fbba5bc46e0b64bf9fbfe7235e0b7b54bffcbdc35cc9c0b585472e86f57f525`. Package 2 also contains a local copy with this hash.

```powershell
Get-FileHash -Algorithm SHA256 outputs/ablation/all_world/policy.pt
```

## 3. Export the arm without world-coordinate inputs

```powershell
python -B -c "import v2_eval; v2_eval.ensure_arm()"
```

This reads `outputs/ablation/all_world/policy.pt` and writes `outputs/v2_arm/arm_no_world_inputs.pt`. It reuses an existing export only when its source hash and export version match. The archived export hash is `332278bb96ef563106b6b5724ea831e33fa540c77d907f37dae557c63739f58e`; serialization hashes can depend on the environment, so also check the stored source hash and export version.

## 4. Run the golden contract tests

```powershell
python -B -m unittest tests.test_golden
```

This checks Nat's copied source files and the golden sensor/scoring rules. The source table below contains ten hashes verified against the inspected branch blobs while preparing this package. The release verification includes these contract tests; see VERIFICATION.md.

## 5. Run S, freeze the protocol, then P and G

Start in the new checkout with no prior experiment ledger in `outputs/golden/`. Do not copy this package's archived summaries or protocol there. S uses the smallest 12 validation IDs. P uses the smallest 12 test IDs with driving only. G uses all 100 test IDs with the arm.

```powershell
python -B golden_eval.py S --workers 12
```

There is no `--freeze` option in the inspected code. Freeze a protocol after S finishes, using the following PowerShell block. It keeps the required 5 mm / 100 ms rule, records the new S summary hash, and refuses to replace an existing protocol. It does not select a threshold from test results.

```powershell
@'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

out = Path('outputs/golden')
raw = (out / 'S.summary.json').read_bytes()
s = json.loads(raw)
assert s['complete'] and s['episodes'] == 12 and s['process_errors'] == 0
protocol = {
    'version': 1,
    'stop_distance_m': 0.005,
    'stop_hold_steps': 100,
    'selected_from': 'S',
    'reason': 'Reproduction: keep the original fixed 5 mm / 100 ms rule.',
    'validation_summary_sha256': hashlib.sha256(raw).hexdigest(),
    'frozen_utc': datetime.now(timezone.utc).isoformat(),
    'validation_case_ids': s['case_ids'],
    'alternative_used': False,
}
with (out / 'protocol.json').open('x', encoding='utf-8') as f:
    json.dump(protocol, f, indent=2)
    f.write('\n')
'@ | python -B -
python -B golden_eval.py P --workers 12
python -B golden_compare.py --run P --out-dir outputs/golden
python -B golden_eval.py G --workers 12
python -B golden_compare.py --run G --out-dir outputs/golden
```

Check each command's exit status before continuing. P and G must use `outputs/golden/`; the evaluator rejects other output directories for them. Keep the experiment source and model files fixed between P and G. The evaluator records attempts and receipts and will not silently repeat attempted cases. Do not remove ledgers or rerun G to tune parameters. The comparison writes `P_compare.csv`/`P_compare_summary.json` for P and `compare.csv`/`compare_summary.json` for G.

## 6. Fixed inputs and scoring

Nat's source snapshot is commit `d89051b4f971c54d59c81bf71f9f92cb4f8e2833` on `v2-sensor`. The following table is copied from the inspected branch's `nat_golden/SOURCE.md`:

| File here | Source path | sha256 |
| --- | --- | --- |
| `camera_loop.py` | `v2_sensors/driving_v2/camera_loop.py` (reference only, not imported) | `1fa00e61e61c3b05bd2bc2bd3fb23dd0e6462ee14e728ea5b93b2221ad1a26ec` |
| `sensors.py` | `v2_sensors/driving_v2/sensors.py` | `e8d59c74670ab0643edee7e4d5598bd47d93b1846cc2f16af9dd67235415959e` |
| `wrist_camera.py` | `v2_sensors/driving_v2/wrist_camera.py` | `53b325fd5a8360b8c26436a9454b113704ead71b5f7e51837a3dc5e45357314e` |
| `full_run.py` | `v2_sensors/arm/full_run.py` (reference only, not imported) | `a099cabfe4479cf25bd59665e16717b8634a7feeb1b647f1dc39df9bf674514d` |
| `models/driving_seed7.npz` | `v2_sensors/arm/results/overnight_20261105/basics_standard/models/driving_seed7.npz` (iteration 43) | `458fc825c3d3c1e5d39d3630655eb872bb46f3ddc34e0a850ee4eb5da52a18c3` |
| `expert_test.json` | `.../overnight_20261105/basics_standard/test/expert_test.json` | `12eb02ece040351c7b7fec4f551d99017b77fd4ef81a5565882564764ae87075` |
| `expert_val.json` | `.../overnight_20261105/basics_standard/val/expert_test.json` | `11d0b4da2fee075fd77b7799e80e2fceb7a3dccb00f75d998220b97c4a4b163e` |
| `seed7_metrics.csv` | `.../overnight_20261105/basics_standard/test/seed7_metrics.csv` | `ba2c8b8f162400ebd9460447b52d9f23d8d9f0bf54888fedca2f8d8dfe548e13` |
| `test_config.json` | `.../overnight_20261105/basics_standard/test/config.json` | `b2bf96e7a358ee9a5f4b205c2330d52d22f8efef3514ca159f4ed2e447f24ec1` |
| `publication_scores.csv` | `v2_sensors/arm/results/overnight_20261105/publication_scores.csv` | `97d81b07942156fef0c5ee7de0160ff095721dd7312e1416c640dbe625adf376` |

`golden_sensor.py` sets `STOP_DISTANCE_M = 0.005` and `STOP_HOLD_STEPS = 100`. STOP occurs when the sensor-estimated hand-to-goal distance stays at or below 5 mm for 100 consecutive 1 ms steps. This is 100 ms. Success additionally requires PARK, an issued STOP, and true end-effector distance at STOP at or below 10 mm. The simulation continues 0.5 s after STOP. Without STOP, the limit is the expert stop time plus 4 s.

The archived `outputs/golden/protocol.json` records this rule and the original validation selection. It was copied unchanged. Its `validation_summary_sha256` identifies the original S summary before text cleaning. The packaged S/P/G summaries have cleaned branch and executable-path fields, so their file hashes are different. Historical provenance and input-hash values are intentionally retained. Use `SHA256SUMS.txt` to verify the packaged files, and the cleaning report to relate them to original hashes. These archived summaries are reference evidence, not a resumable run or a protocol for a new S result.

The source branch uses neutral module names such as `sim.py` and `memory_policy.py`. Historical summaries still record the names used in the original execution. New provenance hashes and timestamps are expected to differ; do not rewrite old evidence to make the hashes match.

## 7. Expected historical results

Values below were read from the copied summaries, not obtained by a fresh simulation.

| Run | Episodes | PARK | STOP | Success within 10 mm at STOP |
| --- | ---: | ---: | ---: | ---: |
| S, validation | 12 | 12 | 10 | 1 |
| P, driving only | 12 | 12 | 0 | 0 |
| G, full test | 100 | 100 | 68 | 1 |

`compare_summary.json` records 1/100 successes for the two-rate arm and median distance at STOP of **16.95189884006594 mm** (about 17 mm). That median is over the 68 stopped cases, not all 100. Nat iteration 43 has 0/100 successes and a median of 168.80902936971373 mm over its 91 stopped cases. Iteration 57 is a reference with a different driving network.

## Included archive

- `outputs/golden/protocol.json`
- `outputs/golden/S.summary.json`
- `outputs/golden/P.summary.json`
- `outputs/golden/G.summary.json`
- `outputs/golden/compare_summary.json`
- `outputs/golden/compare.csv`

The pointwise golden JSONL files, attempt ledgers and receipts are not included in the requested archive. Run the steps above in the separate checkout to generate new ones. The package does not claim a fresh reproduction of the historical numbers.
