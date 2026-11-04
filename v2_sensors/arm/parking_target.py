"""Supervised distance to the expert's PARK position; never a deployment input."""
import numpy as np

DISTANCE_SCALE_M = 3.0
PARK_THRESHOLD_M = 0.05


def parking_distance(base_position, park_step, sample_steps):
    xy = np.asarray(base_position)[:, :2]
    distance = np.linalg.norm(xy[sample_steps] - xy[park_step], axis=1)
    return (np.clip(distance, 0, DISTANCE_SCALE_M) / DISTANCE_SCALE_M).astype(np.float32)
