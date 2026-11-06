# Robot environment

The root `pixi.toml` is for the tracking server. Use `environments/robotnik/`
for robot simulation. Its Windows lock file is copied from the environment used
for release tests. The new environment was not independently installed from
scratch during this revision; platform and package-server availability still matter.
Linux/macOS need a separately resolved environment and are not verified here.

From the repository root, with Git and Pixi installed:

```powershell
git clone https://github.com/VModugno/simulation_and_control.git environments/robotnik/.robot-deps/simulation_and_control
git -C environments/robotnik/.robot-deps/simulation_and_control checkout df5850526affcc33b25b15d3096c4d21b614bacd
pixi install --manifest-path environments/robotnik/pixi.toml --locked
pixi shell --manifest-path environments/robotnik/pixi.toml
```

Return to the repository root before running project commands. The library must
be installed editable: its wheel omits required namespace subdirectories. The
checkout is a course dependency, not a generated project file. Do not substitute
the similarly named `pinocchio` PyPI package for distribution `pin`.

The verified runtime reports Python 3.11.18, NumPy 2.4.6, torch 2.14.1,
PyBullet 3.2.5, pin 4.1.0, simulation-and-control 0.1 and mlflow-skinny 3.17.2.
The lock records transitive dependencies and package hashes. Course `models/`
assets are supplied separately, including the URDF and meshes; they are not in Git.
Keep `configs/robotnik_arm.json` in its published location.

```powershell
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONIOENCODING = 'utf-8'
$env:ROBOENV_HEADLESS = '1'
$env:CAMERA_RENDERER = 'tiny'
python -B -c "import numpy, torch, pybullet, pinocchio, simulation_and_control; from simulation_and_control import pb, MotorCommands, PinWrapper; print('Robot imports OK')"
python -B -m unittest discover -s tests
python -B -m unittest discover -s tools -p 'test_*.py'
```

Model-dependent tests skip when the source arm checkpoint is absent. For the full
suite, supply package 2's `outputs/ablation/all_world/policy.pt` before testing.
Tests export that arm into a temporary directory. A small robot integration test
checks controller operation; it is not a new 100-point experiment.

For ablation evaluation, also supply the original 800 `data_dart/dart_*.npy`
trajectories from the course project. Package 1 supplies the split and NPZ training
data, not those original trajectories. `prepare_team.py` downloads the pinned
teammate model from run `f978e3431988470188317b3c6efd9e29` and computes limits
from the training split. Tailscale access to the team tracker is required.
The memory-v5 donor is also required; its expected SHA256 is
`c272f03b626ac086fd09d07780934e8a4999f17cff658aff7b26e9e509a6b147`.
See package 2's README for complete dependency and result tables.

Unselected optional integration metadata is omitted from the published lock files.
Selected packages, versions, dependency requirements and artifact hashes are unchanged.
