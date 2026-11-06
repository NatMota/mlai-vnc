# =============================================================================
# test.py
# This is for: checking how good the model is, on many test targets it has never seen.
# How to run:  python3 test.py            (20 test targets)
#              python3 test.py 50         (50 test targets)
#              python3 test.py 20 val     (20 validation targets: use these to choose settings,
#                                          so the test targets stay a fair final exam)
# Result:      one RESULT line per target, then a summary with averages.
#              Saved in a new folder every run:
#              results/<model>_seed<seed>/test_<date>_<time>/
#              (test_20targets.txt, test_20targets.png, and one drive graph per target).
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   env            settings passed to drive.py (here: no window).
#   subprocess     runs another Python file (drive.py) from this one.
#   out            everything drive.py printed.
#   RESULT line    the last line drive.py prints, with the scores for one target.
#   reached        True if the end-effector ended within 1 cm of the target.
#   best_error_cm  closest the end-effector got to the target, in cm.
#   final_error_cm how far it was when the run stopped, in cm.
#   get()          reads one number out of a RESULT line.
# -----------------------------------------------------------------------------

import os
import datetime
import subprocess
import sys


# =============================================================================
# SETTINGS
# =============================================================================

# SETTING: model folder
#   Where the model is. Must be the same as DATA_FOLDER in train.py.
DATA_FOLDER = "data_dart"
#   The best model so far is in "data_dart". For the data_v2 tests: "data_v2/big_push".

# SETTING: targets folder
#   Where the test and validation targets come from.
#   Keep it at "data_dart", so every model is tested on the same targets and scores can be compared.
TARGETS_FOLDER = "data_dart"

# SETTING: number of test targets
#   Type it after the command, for example: python3 test.py 50
#   If nothing is typed, 20 targets are tested. There are 120 test targets in total.
N_TARGETS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

# SETTING: which targets
#   Type it after the number, for example: python3 test.py 20 val
#   "test" = test targets (the final exam). Used if nothing is typed.
#   "val"  = validation targets (the practice exam). Use these to choose settings like EARLY_ROWS.
GROUP = sys.argv[2] if len(sys.argv) > 2 else "test"


# =============================================================================
# STEP 1: Drive to each test target, one after another
# This runs drive.py once per target, with no window, so it is faster.
# =============================================================================

env = os.environ.copy()
env["ROBOENV_HEADLESS"] = "1"
# The date and time of this test run. drive.py uses it too, so all graphs go in one folder.
STAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")   # for example 2026-10-27_23-15
env["TEST_STAMP"] = STAMP

results = []
for i in range(N_TARGETS):
    print(f"--- Target {i + 1} of {N_TARGETS} ---")
    out = subprocess.run([sys.executable, "drive.py", DATA_FOLDER, str(i), GROUP, TARGETS_FOLDER],
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

summary = ["===== SUMMARY ====="] + results

if results:
    # Which model and seed were tested (written on the RESULT line by drive.py).
    model = results[0].split("model=")[1].split()[0] if "model=" in results[0] else "model"
    seed = results[0].split("seed=")[1].split()[0] if "seed=" in results[0] else "unknown"
    reached = sum("reached=True" in line for line in results)
    closest = [get(line, "best_error_cm") for line in results]
    final = [get(line, "final_error_cm") for line in results]
    summary += [
        f"Model: {model}   seed: {seed}   targets: {GROUP}",
        f"Reached within 1 cm: {reached} of {len(results)}",
        f"Average closest point: {sum(closest) / len(closest):.1f} cm",
        f"Average final error:   {sum(final) / len(final):.1f} cm",
        f"Within 2 cm at its closest: {sum(c < 2 for c in closest)} of {len(closest)}",
        f"Within 5 cm at its closest: {sum(c < 5 for c in closest)} of {len(closest)}",
    ]

print()
print("\n".join(summary))

# Save the summary as a text file in this test run's folder.
if results:
    results_folder = f"results/{model}_seed{seed}/{GROUP}_{STAMP}"
    os.makedirs(results_folder, exist_ok=True)
    with open(f"{results_folder}/test_{len(results)}targets.txt", "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"Results saved in {results_folder}/")


# =============================================================================
# STEP 3: Graph: closest distance for each test target
# This is for: seeing at a glance which targets were reached, and how far the others missed.
# Saved in the results folder. Open it in VS Code to see it.
# Each run of drive.py also saves its own graph there.
# =============================================================================

# Graph colours and style (shared look for all graphs)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

def graph_style(ax):
    ax.set_facecolor("#fcfcfb")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)

if results:
    try:
        import matplotlib
        matplotlib.use("Agg")   # draw to a file, no window
        import matplotlib.pyplot as plt
        import matplotlib.ticker

        targets = [int(get(line, "target")) for line in results]
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#fcfcfb")
        graph_style(ax)
        ax.grid(False, axis="x")   # no up-and-down lines behind the bars
        ax.bar([str(d) for d in targets], closest, color=BLUE, width=0.6)
        ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
        ax.text(len(targets) - 0.5, 1.0, "1 cm", color=INK_2, ha="right", va="bottom", fontsize=9)
        ax.set_yscale("log")   # log scale, so small and big misses both show
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))   # plain numbers, not 10^1
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Validation target" if GROUP == "val" else "Test target", color=INK)
        ax.set_ylabel("Closest distance (cm)", color=INK)
        ax.set_title(f"{model}, seed {seed}, {GROUP} targets:  reached within 1 cm {reached} of {len(results)}   "
                     f"(average closest {sum(closest) / len(closest):.1f} cm)", color=INK, loc="left")
        fig.tight_layout()
        graph_file = f"{results_folder}/test_{len(results)}targets.png"
        fig.savefig(graph_file, dpi=150)
        print(f"Graph saved as {graph_file}")
    except ImportError:
        print("No graph: matplotlib is missing. Run with the pixi line to get the graph.")
