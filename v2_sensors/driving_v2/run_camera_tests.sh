#!/bin/bash
# Test one model on every bearing in expert_reference.json, with the camera in the loop.
# Usage: ./run_camera_tests.sh models/camera_flag_ttp.npz camera_loop/camera_flag_ttp [parallel runs, default 5]
# Needs: a Python with PyBullet (PYTHON, default python3) and collection/native built (sensor_parking target).
set -e
cd "$(dirname "$0")"
MODEL=$1; OUT=$2; JOBS=${3:-5}; PY=${PYTHON:-python3}
mkdir -p "$OUT"
export CAMERA_RENDERER=tiny OMP_NUM_THREADS=1
$PY -c "import json; print(' '.join(json.load(open('expert_reference.json'))['bearings']))" | tr ' ' '\n' |
  xargs -P "$JOBS" -I{} sh -c "$PY camera_loop.py $MODEL {} $OUT/run_{}.json > $OUT/log_{}.txt 2>&1"
echo "done: $(ls $OUT/run_*.json | wc -l) runs in $OUT"
