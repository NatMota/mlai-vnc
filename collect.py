# =============================================================================
# collect.py
# This is for: collecting training data. The expert drives to a random dot many times.
# How to run:  python3 collect.py normal 300    (300 normal runs, saved in data_normal)
#              python3 collect.py dart 300      (300 runs with pushes, saved in data_dart)
# Result:      one .npy file per run.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   KIND           "normal" or "dart".
#   N_RUNS         how many runs to collect.
#   robot_file     which robot program to run.
#   env            settings passed to the robot program (here: no window).
#   subprocess     runs another Python file from this one.
#   headless       no window, so it runs faster.
#   DART           adding random pushes, so the data shows the expert recovering.
# -----------------------------------------------------------------------------

import os
import subprocess
import sys


# =============================================================================
# SETTINGS (change these, or type them after the command)
# =============================================================================

# Change data type: "normal" = plain expert, "dart" = expert with random pushes.
# The pushes help the model learn how to recover from mistakes.
KIND = sys.argv[1] if len(sys.argv) > 1 else "normal"

# Change number of runs.
N_RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 10


# =============================================================================
# STEP 1: Choose the robot file
# =============================================================================

if KIND == "dart":
    robot_file = "robotnik_arm_task_dart.py"
else:
    robot_file = "robotnik_arm_task.py"


# =============================================================================
# STEP 2: Run the robot again and again, with no window (faster)
# =============================================================================

env = os.environ.copy()
env["ROBOENV_HEADLESS"] = "1"

for i in range(N_RUNS):
    # Print results: which run is going.
    print(f"Run {i + 1} of {N_RUNS} ({KIND})")
    subprocess.run([sys.executable, robot_file], env=env)
