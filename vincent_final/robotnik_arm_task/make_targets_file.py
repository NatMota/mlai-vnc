# =============================================================================
# make_targets_file.py
# This is for: saving the position of every validation, test and golden target in one small file,
#              so test_drive_net.py can run without the big data folders.
# How to run:  python3 make_targets_file.py      (only needed again if the data folders change)
# Result:      saves report_targets.json. It needs data_dart and the golden dataset, and does not change them.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   target      the x, y, z position the arm tip must reach (m).
#   group       a named set of targets: val, test, golden_val or golden_test.
#   split.json  a file that lists which saved runs belong to each group.
# -----------------------------------------------------------------------------

import json
import os
import numpy as np

# SETTING: where the team's golden data folder is.
GOLDEN_DATA = os.path.expanduser("~/COMP0245_Golden_300")

# SETTING: the file this script makes.
OUTPUT_FILE = "report_targets.json"

targets = {}

# The validation and test targets are in the first row of each saved run.
with open("data_dart/split.json") as f:
    split = json.load(f)
for group in ("val", "test"):
    targets[group] = [np.load(run_file, mmap_mode="r")[0, 18:21].astype(float).tolist() for run_file in split[group]]

# The golden targets are in the manifest of each saved run.
with open(os.path.join(GOLDEN_DATA, "split.json")) as f:
    split = json.load(f)
for group in ("val", "test"):
    targets[f"golden_{group}"] = []
    for run_number in split[group]:
        with np.load(os.path.join(GOLDEN_DATA, "expert", group, f"run_{run_number}.npz"), allow_pickle=False) as run:
            targets[f"golden_{group}"].append(json.loads(str(run["manifest"]))["target_position_world_m"])

with open(OUTPUT_FILE, "w") as f:
    json.dump(targets, f, indent=0)

# Print results: how many targets each group has.
for group, positions in targets.items():
    print(f"{group}: {len(positions)} targets")
print(f"Saved as {OUTPUT_FILE}")
