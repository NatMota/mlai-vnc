# Golden sensor driving + two-rate arm

This branch evaluates an exported two-rate arm behind Nat's iteration 43
basics_standard sensor-only driving network on the 100 standard test points.

Compared with Nat iteration 43, the median distance at STOP decreased from
**169 mm to 17 mm** (168.8090 to 16.9519 mm). These medians use each controller's
own stopped cases: 91 for Nat and 68 for the two-rate arm.

| Controller | Success within 10 mm at STOP | PARK | STOP |
| --- | ---: | ---: | ---: |
| Two-rate arm + iteration 43 driver | 1/100 | 100/100 | 68/100 |
| Nat iteration 43 | 0/100 | 100/100 | 91/100 |
| Nat iteration 57, different driver | 4/100 | 87/100 | 66/100 |

The STOP rule was frozen using 12 validation points: estimated distance at most
5 mm for 100 consecutive milliseconds. Each full-arm test episode was run once.
See [the full report](docs/golden_results.md) for all 100 pointwise comparisons,
training and simulator limitations, and the exact success definition.

## Reviewer starting point

1. Download the [reproduction release](https://github.com/NatMota/mlai-vnc/releases/tag/comp0245-reproduction-2026-11-06-r2) and read its `DOWNLOAD_GUIDE.md`.
2. Prepare the [locked robot environment](docs/ENVIRONMENT.md).
3. Follow the [code map and iteration guide](docs/ITERATIONS.md), then the [golden reproduction steps](docs/REPRODUCE_GOLDEN.md) or [fused evaluation guide](docs/REPRODUCE.md).
4. Read [revision verification](docs/VERIFICATION.md). Archived results remain separate from new output directories.

## Environment and verification

Use the course robot simulation Python environment with NumPy, PyTorch, PyBullet,
Pinocchio and simulation_and_control. Supply the course models/ assets and the
exported arm at outputs/v2_arm/arm_no_world_inputs.pt. The arm is referenced by
source run 540bdcf9. Weights and course assets are not stored in this Git branch.

Set KMP_DUPLICATE_LIB_OK=TRUE, PYTHONIOENCODING=utf-8, ROBOENV_HEADLESS=1 and
CAMERA_RENDERER=tiny. The evaluator sets these values for its workers.

```powershell
python -B -m unittest discover -s tests
python -B golden_eval.py --validation-case 6 --workers 1 --out-dir work/validation6
```

The release verification repeats a validation point only. Do not repeat G for
parameter selection. Saved publication results are archives: source_* fields
identify original execution evidence; publication hashes identify the exported
files after neutral labels and portable paths have been applied.

## Results and source provenance

MLflow experiment: v2-sensors (ID 3). Run name: golden-tworate-arm-iter43.
The tracker is available through the team's Tailscale network. Publication uses
only a newly created run and does not modify any existing run.

Nat source snapshot: d89051b4f971c54d59c81bf71f9f92cb4f8e2833.
Frozen snapshot files and checksums are documented in nat_golden/SOURCE.md.
The release starts from commit 80e409277744ac238352bbe3d7babc5650c2d97a.
Legacy implementation guides remain in docs/ with neutral module names.
