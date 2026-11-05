"""Local-only MLflow tracking for the memory-policy experiments.

Everything is written to ./mlruns in this folder. The tracking URI is set explicitly here, so the
user-level MLFLOW_TRACKING_URI (the team server) is never used: nothing leaves this computer.

    One MLflow run per trained checkpoint (params = training settings, metrics = validation MSE).
    Evaluations add their metrics (e.g. val40_success_rate) to the run of the checkpoint they used;
    the expert, or several checkpoints averaged as an ensemble, get a run of their own.

View with:  pixi run mlflow ui --backend-store-uri ./mlruns   (needs MLFLOW_ALLOW_FILE_STORE=true)
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACKING_URI = (ROOT / "mlruns").as_uri()
EXPERIMENT = "memory-policy"


def _mlflow():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"   # file store is fine for a single local user
    os.environ.pop("MLFLOW_TRACKING_URI", None)      # never fall back to the team server
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def _key(path):
    """Checkpoint identity used to find its run: path relative to this folder, forward slashes."""
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()


def _flat(settings):
    return {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in settings.items()}


def _find(mlflow, key):
    found = mlflow.search_runs(filter_string=f"tags.checkpoint = '{key}'", output_format="list", max_results=1)
    return found[0].info.run_id if found else None


def log_training(checkpoint, settings, nets_summary, files=()):
    """Log a trained checkpoint (into its existing run, if an evaluation created one first)."""
    try:
        mlflow = _mlflow()
        key = _key(checkpoint)
        run_id = _find(mlflow, key)
        with mlflow.start_run(run_id=run_id, run_name=None if run_id else Path(key).parent.name) as run:
            mlflow.set_tags({"checkpoint": key, "kind": "training"})
            mlflow.log_params(_flat(settings))
            for name, info in nets_summary.items():
                mlflow.log_metrics({f"{name}_val_mse": info["best_val"], f"{name}_rows": info["rows"],
                                    f"{name}_parameters": info["parameters"]})
            for f in files:
                if Path(f).exists():
                    mlflow.log_artifact(str(f))
            return run.info.run_id
    except Exception as exc:   # tracking must never lose a finished training run
        print(f"WARNING: local MLflow logging failed: {exc!r}", flush=True)
        return None


def log_evaluation(policy, summary, files=()):
    """Add evaluation metrics to the checkpoint's run (or a new run for expert/ensembles)."""
    try:
        mlflow = _mlflow()
        prefix = f"{summary['group']}{summary['n']}"
        metrics = {f"{prefix}_success_rate": summary["success_rate"],
                   f"{prefix}_successes": summary["successes"],
                   f"{prefix}_process_errors": summary["process_errors"]}
        for k in ("mean_final_error_cm", "median_final_error_cm", "mean_success_time_s"):
            if summary.get(k) is not None:
                metrics[f"{prefix}_{k}"] = summary[k]
        kind, _, paths = policy.partition(":")
        # Other control rates get their own run, so they never overwrite the checkpoint's own metrics.
        goal_error = summary.get("goal_error_m") or 0.0
        single = kind == "memory" and "+" not in paths and not goal_error
        run_id = _find(mlflow, _key(paths)) if single else None
        name = kind if not paths else ("ensemble" if "+" in paths else Path(_key(paths)).parent.name)
        name = f"{name}-{kind}" if paths and kind != "memory" else name
        name = f"{name}-error{goal_error * 1000:g}mm" if goal_error else name
        with mlflow.start_run(run_id=run_id, run_name=None if run_id else name):
            if run_id is None:
                mlflow.set_tags({"policy": policy, "kind": "evaluation", "goal_error_m": goal_error})
                if single:
                    mlflow.set_tag("checkpoint", _key(paths))
            mlflow.log_metrics(metrics)
            for f in files:
                if Path(f).exists():
                    mlflow.log_artifact(str(f), artifact_path=f"eval_{prefix}")
    except Exception as exc:
        print(f"WARNING: local MLflow logging failed: {exc!r}", flush=True)
