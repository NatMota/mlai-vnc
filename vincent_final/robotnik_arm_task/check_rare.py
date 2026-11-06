# =============================================================================
# check_rare.py
# This is for: checking if the training data is balanced (evenly spread).
#              A copying model is weakest where it has seen few examples.
#              For each run, it looks at the target at the start: how far away it was,
#              how much the robot had to turn to face it, and how high it was.
#              It also counts how many runs the expert got right in each group, because
#              only the good runs are used for training (ONLY_GOOD_RUNS). If the expert fails
#              more in one group, that group becomes even rarer in training.
#              Nothing is trained or changed.
# How to run:  python3 check_rare.py                  (also prints the 20 validation targets)
#              python3 check_rare.py 9 13 14           (prints only these validation targets)
#              python3 check_rare.py data_v2/balanced  (checks that folder instead; needs add_ee.py first)
# Result:      three tables (distance, turn, height), a distance x turn table, the "too close"
#              count, a voxel map (3D boxes around the robot, one layer per height),
#              then the validation targets with their groups and boxes.
# =============================================================================

# -----------------------------------------------------------------------------
# WORDS USED IN THIS FILE
#   start distance  floor distance from the robot to the target at the start (m).
#   turn            how much the robot had to turn at the start to face the target (degrees).
#                   0 = straight ahead, 180 = straight behind.
#   height          the target's height above the floor (m).
#   all runs        every training run in the folders (good and bad).
#   good runs       the expert ended within 2 cm. Only these are used for training.
#   kept %          good runs / all runs. Low = the expert often failed in this group.
#   share %         this group's good runs / all good runs. Low = rare in training.
#   bar             one # for each 1% of the good runs.
#   too close       the target starts closer than the expert's parking distance for its height
#                   (35 to 80 cm), so the expert must back away before reaching.
#   voxel           a 3D box in the space around the robot (0.5 m x 0.5 m, 0.4 to 0.5 m high).
#                   The number in each box = good training runs whose target started in it.
#   thin box        a box with fewer than 5 good runs: the model has few examples there.
# -----------------------------------------------------------------------------

import glob
import json
import os
import sys
import numpy as np

# SETTING: which data to check. Type a folder name to check only that folder (all its runs),
#   for example: python3 check_rare.py data_v2/balanced
#   If no folder is typed: the training runs of data_dart + data_dart2 (the same data as the best model).
_folder_args = [a for a in sys.argv[1:] if not a.isdigit()]
TRAIN_FOLDERS = _folder_args if _folder_args else ["data_dart", "data_dart2"]
TARGETS_FOLDER = "data_dart"                  # where the validation targets come from
_numbers = [int(a) for a in sys.argv[1:] if a.isdigit()]
SHOW = _numbers if _numbers else list(range(20))
GOOD_RUN_CM = 2.0

DIST_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 99.0]   # metres
TURN_EDGES = [0, 30, 60, 90, 120, 150, 181]                            # degrees
HEIGHT_EDGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 99.0]               # metres


def describe(f):
    """Start distance (m), turn (degrees, 0 to 180) and height (m) for one run."""
    d = np.load(f, mmap_mode="r")
    base_xy, bearing, target = np.asarray(d[0, 0:2]), float(d[0, 3]), np.asarray(d[0, 18:21])
    gap = target[:2] - base_xy
    turn = np.degrees(np.arctan2(gap[1], gap[0]) - bearing)
    turn = abs((turn + 180) % 360 - 180)
    return float(np.hypot(*gap)), float(turn), float(target[2])


def is_good(f):
    ee = np.load(os.path.join(os.path.dirname(f) + "_ee", os.path.basename(f)), mmap_mode="r")
    target = np.load(f, mmap_mode="r")[-1, 18:21]
    return np.linalg.norm(ee[-1] - target) * 100 < GOOD_RUN_CM


def bin_of(value, edges):
    i = int(np.searchsorted(edges, value, side="right")) - 1
    return min(max(i, 0), len(edges) - 2)


def label(edges, i, unit):
    last = i + 1 == len(edges) - 1
    top = "more" if (last and edges[i + 1] == 99.0) else f"{min(edges[i + 1], 180):g}"
    return f"{edges[i]:g} to {top} {unit}"


# =============================================================================
# STEP 1: Describe every training run (the train part of data_dart, and all of data_dart2)
# =============================================================================

with open(f"{TARGETS_FOLDER}/split.json") as f:
    split = json.load(f)
if _folder_args:
    files = []
    for folder in TRAIN_FOLDERS:            # every run in the typed folder(s)
        files += sorted(glob.glob(f"{folder}/*.npy"))
else:
    files = list(split["train"])            # the train part of data_dart ...
    for folder in TRAIN_FOLDERS[1:]:        # ... and all of data_dart2
        files += sorted(glob.glob(f"{folder}/*.npy"))
print(f"Data checked: {', '.join(TRAIN_FOLDERS)}")

runs = np.array([describe(f) + (is_good(f),) for f in files])   # distance, turn, height, good
dist, turn, height, good = runs[:, 0], runs[:, 1], runs[:, 2], runs[:, 3].astype(bool)
n_good = good.sum()
print(f"Training runs: {len(files)} in total, {n_good} good (used for training)")


# =============================================================================
# STEP 2: One table for each way the targets can differ
# =============================================================================

def table(title, values, edges, unit):
    print(f"\n{title:22s}  all runs  good runs  kept %  share %")
    for i in range(len(edges) - 1):
        in_group = np.array([bin_of(v, edges) == i for v in values])
        n_all, n_ok = int(in_group.sum()), int((in_group & good).sum())
        kept = n_ok / n_all * 100 if n_all else 0.0
        share = n_ok / n_good * 100
        # Print results: one line per group, with a bar for the share.
        print(f"  {label(edges, i, unit):20s} {n_all:8d} {n_ok:10d} {kept:7.0f} {share:8.1f}  " + "#" * min(int(round(share)), 60))


table("START DISTANCE", dist, DIST_EDGES, "m")
table("TURN", turn, TURN_EDGES, "deg")
table("HEIGHT", height, HEIGHT_EDGES, "m")

# Distance x turn, good runs only (where the close and behind cases are).
counts = np.zeros((len(DIST_EDGES) - 1, len(TURN_EDGES) - 1), dtype=int)
for dv, tv in zip(dist[good], turn[good]):
    counts[bin_of(dv, DIST_EDGES), bin_of(tv, TURN_EDGES)] += 1
print("\nGOOD RUNS: start distance x turn")
print("  distance           turn:" + "".join(f"{label(TURN_EDGES, j, ''):>11s}" for j in range(len(TURN_EDGES) - 1)))
for i in range(len(DIST_EDGES) - 1):
    print(f"  {label(DIST_EDGES, i, 'm'):20s}    " + "".join(f"{counts[i, j]:11d}" for j in range(len(TURN_EDGES) - 1)))


# =============================================================================
# STEP 2b: "Too close" targets: the target starts closer than the expert's parking distance,
# so the expert has to back away before it reaches. The parking distance depends on the
# target's height (the same numbers as in the expert's code).
# =============================================================================

STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]     # target height (m)
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]    # parking distance from the target (m)
park_distance = np.interp(height, STANDOFF_Z_POINTS, STANDOFF_POINTS)
too_close = dist < park_distance
print(f"\nTOO CLOSE (target starts closer than its parking distance, so the expert backs away first)")
print(f"  all runs: {too_close.sum()}   good runs: {(too_close & good).sum()}   "
      f"kept %: {(too_close & good).sum() / max(too_close.sum(), 1) * 100:.0f}   "
      f"share %: {(too_close & good).sum() / n_good * 100:.1f}")


# =============================================================================
# STEP 2c: Voxels: cut the space around the robot into boxes and count the good runs in each.
# The boxes use the robot's view at the start: ahead, right (minus = left), and height.
# Each layer is printed like the top view in plot_targets.py: ahead is up, right is right,
# and the robot is at the centre (between the 4 middle boxes), facing up.
# =============================================================================

AXIS_EDGES = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]   # metres, for "ahead" and "right" (0.5 m boxes)
LAYER_EDGES = [0.0, 0.4, 0.8, 1.3]                    # metres, for the height (0.4 to 0.5 m layers)
THIN = 5                                              # a box with fewer good runs than this is "thin"


def robot_view(f):
    """Target at the start, seen from the robot: ahead (m), right (m), height (m)."""
    d = np.load(f, mmap_mode="r")
    base_xy, bearing, target = np.asarray(d[0, 0:2]), float(d[0, 3]), np.asarray(d[0, 18:21])
    gap = target[:2] - base_xy
    c, s = np.cos(bearing), np.sin(bearing)
    return c * gap[0] + s * gap[1], -(-s * gap[0] + c * gap[1]), float(target[2])


def box_of(ahead, right, h):
    """Box numbers (ahead, right, layer). Values outside the edges go in the outer boxes."""
    return (bin_of(ahead, AXIS_EDGES), bin_of(right, AXIS_EDGES), bin_of(h, LAYER_EDGES))


views = np.array([robot_view(f) for f in files])
voxels = np.zeros((len(AXIS_EDGES) - 1, len(AXIS_EDGES) - 1, len(LAYER_EDGES) - 1), dtype=int)
for a, r, h in views[good]:
    voxels[box_of(a, r, h)] += 1

n_boxes = voxels.size
print(f"\nVOXELS: good runs in each 0.5 m x 0.5 m box ({n_boxes} boxes in total). "
      f"Empty: {(voxels == 0).sum()}.  Thin (fewer than {THIN}): {((voxels > 0) & (voxels < THIN)).sum()}.")
for k in range(len(LAYER_EDGES) - 1):
    print(f"\n  Height {LAYER_EDGES[k]:g} to {LAYER_EDGES[k + 1]:g} m"
          "      right ->  (columns: -1.5 to 1.5 m in 0.5 m steps)")
    for i in reversed(range(len(AXIS_EDGES) - 1)):   # ahead: top row = furthest ahead
        row = "".join(f"{voxels[i, j, k]:5d}" for j in range(len(AXIS_EDGES) - 1))
        print(f"    ahead {AXIS_EDGES[i]:+.1f} to {AXIS_EDGES[i + 1]:+.1f} m {row}")
        if AXIS_EDGES[i] == 0.0:
            print("    " + " " * 22 + "-" * 13 + " robot " + "-" * 13)


# =============================================================================
# STEP 3: The validation targets, with their groups
# =============================================================================

print("\nValidation target   distance (m)   turn (deg)   height (m)   too close?   good runs in same     good runs in same")
print("                                                                           distance x turn group  voxel (box)")
for k in SHOW:
    dv, tv, hv = describe(split["val"][k])
    close_flag = "yes" if dv < np.interp(hv, STANDOFF_Z_POINTS, STANDOFF_POINTS) else "no"
    print(f"  {k:3d}               {dv:6.2f}        {tv:5.0f}        {hv:5.2f}        {close_flag:3s}          "
          f"{counts[bin_of(dv, DIST_EDGES), bin_of(tv, TURN_EDGES)]:8d}             "
          f"{voxels[box_of(*robot_view(split['val'][k]))]:8d}")