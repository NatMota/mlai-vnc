"""Shared helpers for the parallel simulation scripts (one subprocess per episode).

A fresh process per episode is deliberate: creating a second PyBullet simulation inside one
process ran ~30x slower in testing. 12 workers suit a 16 GB machine (each process ~250-350 MB).
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SUB_ENV = dict(os.environ, ROBOENV_HEADLESS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
EPISODE_TIMEOUT_S = 900
# Random target box of the expert data scripts (robotnik_arm_task_dart.py).
TARGET_LOW = (-1.5, -1.5, 0.1)
TARGET_HIGH = (1.5, 1.5, 1.25)
# Row layout written by chester_sim.run_episode, before any labeler columns.
STATE, EE, REACH, LABEL_START = slice(0, 21), slice(21, 24), 24, 36


# Expert base standoff table (robotnik_arm_task*.py): parking distance from the dot vs dot height.
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]


def parking_goal(target):
    """The expert's parking spot (x, y) and final bearing for a dot, exactly as the expert computes them."""
    target = np.asarray(target, dtype=float)
    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    xy = target[:2]
    bearing = float(np.arctan2(xy[1], xy[0])) if np.linalg.norm(xy) > 1e-4 else 0.0
    return xy - standoff * np.array([np.cos(bearing), np.sin(bearing)]), bearing


def sample_target(seed):
    rng = np.random.default_rng(seed)
    return np.array([rng.uniform(lo, hi) for lo, hi in zip(TARGET_LOW, TARGET_HIGH)])


def run_pool(jobs, workers, prefix, on_result):
    """Run (key, argv) jobs as subprocesses; parse the first stdout line starting with prefix.

    on_result(key, text_after_prefix_or_None, error_text_or_None) is called in job order.
    Returns the number of failed jobs (crash, timeout or missing result line).
    """
    def job(item):
        key, argv = item
        try:
            out = subprocess.run([sys.executable, "-B", *argv], env=SUB_ENV, capture_output=True,
                                 encoding="utf-8", errors="replace", timeout=EPISODE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return key, None, f"timeout after {EPISODE_TIMEOUT_S} s"
        for line in out.stdout.splitlines():
            if line.startswith(prefix):
                return key, line[len(prefix):], None
        return key, None, f"exit {out.returncode}: {out.stderr[-1500:]}"

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        for key, text, error in pool.map(job, jobs):
            failed += error is not None
            on_result(key, text, error)
    return failed


def save_npz_atomic(path, **arrays):
    """Write path via a temporary file, so a killed process never leaves a half-written run."""
    path = Path(path)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    for attempt in range(10):   # Windows: an indexer or antivirus can briefly lock the new file
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    tmp.replace(path)


def write_manifest(folder, **info):
    """Record how a data folder was produced (appended, one JSON object per invocation)."""
    with (Path(folder) / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(info) + "\n")
