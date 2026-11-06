# Reproduction revision verification

Date: 2026-11-06. Baseline: `e407006882467c942ef8da85553a1fd18a6fe36d`.
Release: `comp0245-reproduction-2026-11-06-r2`.

## Changes checked

- Published ablation instructions now use `ablation.py` and `parallel_eval.py`.
- Evaluation refuses existing outputs without explicit, matching resume. A
  manifest binds settings, sources, model and selected split targets. Only missing
  indices run again. Completed JSONL and summary files are preserved.
- Ablation refuses silent reuse. Matching completed groups can be reused with
  `--resume`; unfinished groups restart training. Current configuration and donor
  configuration are recorded separately, with data and source hashes.
- New reviewer, environment and iteration guides explain dependencies, commands,
  safe interruption, fixed golden ledgers and historical results.
- Publication text uses neutral descriptions and relative paths. Unselected
  optional integration metadata was removed from dependency lock files without
  changing installed package versions or hashes.

## Evidence

Before implementation, the 12 new regression tests produced 3 failures and 9
errors: existing-output protection and current training settings were absent,
and neither entry accepted `--resume`. The test checkpoint is `ccda3f2`.
After the fix, all 16 iteration tests passed, including four additional cases
for simultaneous writers, modified results, failed workers and worker identity.
The fix checkpoint is `7290eff`.

```powershell
python -B -m unittest discover -s tests
python -B -m unittest discover -s tools -p 'test_*.py'
```

- Project suite: **96 passed**, zero failures, zero skips.
- Publication tool suite: **17 passed**, zero failures, zero skips.
- Coverage of `ablation.py`, `parallel_eval.py`, `iteration_io.py`: **85% overall**
  (95%, 76%, 100% respectively), measured with coverage 7.17.2.
- Ruff 0.17.4 syntax/undefined-name checks (`E9,F63,F7,F82`) passed for changed
  Python modules and tests. This is targeted lint, not a whole-repository style rewrite.
- Python AST parsing and `git diff --check` passed. No project type-checking
  configuration exists; a full static type check was not performed.
- `pixi lock --check --dry-run` passed for both the root tracking environment and
  `environments/robotnik/pixi.toml`; neither lock required changes.
- The tests used the existing Windows course runtime described in ENVIRONMENT.md.
  A clean installation on another machine was not performed.

Tests use temporary synthetic training/evaluation files. The existing robot
integration test exercises a small controller sequence and exports to temporary
storage. No full S/P/G run, model retraining, or new success-rate measurement was
performed. The saved 1/100 golden result is historical evidence, not a new result.

Original data, model weights and experimental results are outside the code diff.
The release cleaning report records byte-hash checks against the protected
checkout and original material sources. Revised packages retain every historical
payload byte; only explanatory documents, inventories and checksums are revised.

GitHub publication and MLflow artifact readback are recorded in the final upload
receipt. MLflow publication creates a new run and compares existing run metadata
before and after uploading; it does not update previous published runs.
