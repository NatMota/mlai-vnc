# =============================================================================
# test.py
# This is for: checking how good the model is, on many test dots it has never seen.
# How to run:  python3 test.py        (20 test dots)
#              python3 test.py 50     (50 test dots)
# Result:      one RESULT line per dot, then a summary with averages.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   env            settings passed to drive.py (here: no window).
#   subprocess     runs another Python file (drive.py) from this one.
#   out            everything drive.py printed.
#   RESULT line    the last line drive.py prints, with the scores for one dot.
#   reached        True if the end-effector ended within 1 cm of the dot.
#   best_error_cm  closest the end-effector got to the dot, in cm.
#   final_error_cm how far it was when the run stopped, in cm.
#   get()          reads one number out of a RESULT line.
# -----------------------------------------------------------------------------

import os
import subprocess
import sys


# =============================================================================
# SETTINGS (change these)
# =============================================================================

# Change data folder: must match train.py.
DATA_FOLDER = "data_dart"

# Change number of test dots: type it after the command, or change the 20 here.
N_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


# =============================================================================
# STEP 1: Drive to each test dot, one after another
# This runs drive.py once per dot, with no window, so it is faster.
# =============================================================================

env = os.environ.copy()
env["ROBOENV_HEADLESS"] = "1"

results = []
for i in range(N_TARGETS):
    print(f"--- Target {i + 1} of {N_TARGETS} ---")
    out = subprocess.run([sys.executable, "drive.py", DATA_FOLDER, str(i)],
                         env=env, capture_output=True, text=True)
    # Keep only the RESULT line from each run.
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            print(line)
            results.append(line)
    # Print results: show any error, so it can be fixed.
    if out.returncode != 0:
        print(out.stderr[-1000:])


# =============================================================================
# STEP 2: Print results: summary and averages
# =============================================================================

def get(line, key):
    return float(line.split(key + "=")[1].split()[0])

print("\n===== SUMMARY =====")
for line in results:
    print(line)

if results:
    reached = sum("reached=True" in line for line in results)
    closest = [get(line, "best_error_cm") for line in results]
    final = [get(line, "final_error_cm") for line in results]
    print(f"Reached within 1 cm: {reached} of {len(results)}")
    print(f"Average closest point: {sum(closest) / len(closest):.1f} cm")
    print(f"Average final error:   {sum(final) / len(final):.1f} cm")
    print(f"Within 2 cm at its closest: {sum(c < 2 for c in closest)} of {len(closest)}")
    print(f"Within 5 cm at its closest: {sum(c < 5 for c in closest)} of {len(closest)}")
