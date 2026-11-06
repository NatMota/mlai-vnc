# =============================================================================
# collect.py
# This is for: collecting training data. The expert drives to a random target many times.
# How to run:  python3 collect.py balanced 1500 good 4  (ONE balanced dataset: keep going until
#                                                  data_v2/balanced has 1500 GOOD runs, 4 robots at once)
#              python3 collect.py finish 300 good 4  (like balanced, but the expert keeps going to 5 mm;
#                                                  saved in data_v2/finish_5mm)
#              python3 collect.py randomstart 500 good 4  (like balanced, but the robot starts at a random
#                                                  place near the parking spot; saved in data_v2/random_start)
#              python3 collect.py big 1000 good 4  (keep going until data_v2/big_push has 1000 GOOD runs,
#                                                  running 4 robots at the same time)
#              python3 collect.py medium 1000 good 4
#              python3 collect.py big 700 good     (the same, one robot at a time)
#              python3 collect.py big 800          (exactly 800 more runs, good or not)
#              python3 collect.py normal 300       (300 runs with no pushes, saved in data_normal)
#              If stopped, run the same "good" command again: it counts the good runs
#              already in the folder and carries on.
# Result:      one .npy file per run.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   KIND           "balanced", "randomstart", "big", "medium", "mixed" or "normal".
#   N              how many runs (or how many GOOD runs, with "good").
#   good run       the expert's end-effector ended within 2 cm of the target
#                  (the same rule as ONLY_GOOD_RUNS in train.py).
#   bad run        the expert got lost or missed. It is NOT saved (to save disk space).
#   robot_file     which robot program to run.
#   env            settings passed to the robot program (here: no window).
#   subprocess     runs another Python file from this one.
#   headless       no window, so it runs faster.
#   DART           adding random pushes, so the data shows the expert recovering.
#   worker         one robot simulation running on its own. Several workers run at the same
#                  time, each on its own CPU core (idea from Nat's fast collector).
# -----------------------------------------------------------------------------

import glob
import os
import subprocess
import sys
import tempfile
import time
import numpy as np


# =============================================================================
# SETTINGS
# Type these after the command, for example: python3 collect.py big 700 good
# =============================================================================

# SETTING: kind of data (first word)
#   "normal" = the expert on its own. Saved in data_normal.
#   "big"    = the expert with big pushes, like data_dart. Saved in data_v2/big_push.
#   "medium" = the expert with medium pushes, like data_dart2. Saved in data_v2/medium_push.
#   "mixed"  = small wheel pushes while driving, big ones after parking, big arm pushes.
#              The good parts of both in one folder. Saved in data_v2/mixed_push.
#              The push sizes are in robotnik_arm_task_dart.py.
#   "balanced" = ONE balanced dataset: "mixed" pushes, and the targets spread evenly by distance,
#              so close targets are as common as far ones. Saved in data_v2/balanced.
#              (check_rare.py showed the model failed on close targets because they were rare.)
#   "randomstart" = like "balanced", but the robot starts at a random place near the parking spot,
#              facing a random direction, so the data shows the expert recovering from awkward
#              places (like facing the target while too close). Saved in data_v2/random_start.
KIND = sys.argv[1] if len(sys.argv) > 1 else "normal"

# SETTING: number (second word)
#   If nothing is typed, 10.
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# SETTING: good runs only (third word, optional)
#   "good" = keep going until the folder has N good runs. Only for "big" and "medium".
#   nothing = collect exactly N more runs.
UNTIL_GOOD = len(sys.argv) > 3 and sys.argv[3] == "good"

# SETTING: workers (fourth word, optional, only with "good")
#   How many robots run at the same time. Use at most the number "nproc" shows.
#   If nothing is typed, 1.
WORKERS = int(sys.argv[4]) if len(sys.argv) > 4 else 1

GOOD_RUN_CM = 2.0   # the same rule as ONLY_GOOD_RUNS in train.py


# =============================================================================
# STEP 1: Choose the robot file and the folder
# =============================================================================

if KIND == "balanced":
    robot_file = "robotnik_arm_task_dart.py"
    folder = os.path.join("data_v2", "balanced")
elif KIND == "randomstart":
    robot_file = "robotnik_arm_task_dart.py"
    folder = os.path.join("data_v2", "random_start")
elif KIND == "finish":
    robot_file = "robotnik_arm_task_dart.py"
    folder = os.path.join("data_v2", "finish_5mm")
elif KIND in ("big", "medium", "mixed"):
    robot_file = "robotnik_arm_task_dart.py"
    folder = os.path.join("data_v2", {"big": "big_push", "mixed": "mixed_push"}.get(KIND, "medium_push"))
else:
    robot_file = "robotnik_arm_task.py"
    folder = "data_normal"
    UNTIL_GOOD = False   # normal runs do not save the end-effector, so they cannot be checked here


def is_good(f):
    """True if the expert's end-effector ended within 2 cm of the target (columns 36-38 = real end-effector)."""
    d = np.load(f, mmap_mode="r")
    return np.linalg.norm(d[-1, 36:39] - d[-1, 18:21]) * 100 < GOOD_RUN_CM


def count_good():
    return sum(is_good(f) for f in glob.glob(f"{folder}/*.npy"))


# =============================================================================
# STEP 2: Run the robot again and again, with no window (faster)
# =============================================================================

env = os.environ.copy()
env["ROBOENV_HEADLESS"] = "1"
env["ROBOENV_PUSH"] = "mixed" if KIND in ("balanced", "randomstart", "finish") else KIND   # which push size to use
env["ROBOENV_TARGETS"] = "even" if KIND in ("balanced", "randomstart", "finish") else "square"   # how the targets are spread
if KIND == "finish":
    env["ROBOENV_REACH_TOLERANCE"] = "0.005"   # like balanced, but the expert keeps going to 5 mm
env["ROBOENV_START"] = "random" if KIND == "randomstart" else "centre"   # where the robot starts

if UNTIL_GOOD:
    good = count_good()
    start_good = good
    total = 0
    print(f"{folder} already has {good} good runs. Target: {N} good runs.")
    print(f"Running {WORKERS} robot(s) at the same time.")
    running = []   # the robots running now: (process, its output file)
    while good < N or running:
        # Start new robots until WORKERS are running (but not more than still needed).
        while len(running) < WORKERS and good + len(running) < N:
            out = tempfile.TemporaryFile(mode="w+")
            p = subprocess.Popen([sys.executable, robot_file], env=env, stdout=out, stderr=subprocess.DEVNULL, text=True)
            running.append((p, out))
        # Check which robots have finished.
        time.sleep(0.2)
        for p, out in list(running):
            if p.poll() is None:
                continue   # still running
            running.remove((p, out))
            out.seek(0)
            ok = "SAVED good run" in out.read()   # the robot file prints this when it saves a good run
            out.close()
            total += 1
            good += ok
            bad = total - (good - start_good)
            # Print results: one line per finished run, with how many failed so far.
            print(f"Run {total}: {'good' if ok else 'bad (not saved)'}   good runs: {good} of {N}   "
                  f"bad so far: {bad} of {total} ({bad / total * 100:.0f}%)")
    print(f"Done. {folder} has {good} good runs.")
else:
    for i in range(N):
        # Print results: which run is going.
        print(f"Run {i + 1} of {N} ({KIND})")
        subprocess.run([sys.executable, robot_file], env=env)