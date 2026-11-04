"""Supervised distance to the expert's PARK position; never a deployment input."""
import numpy as np

DISTANCE_SCALE_M = 3.0
PARK_THRESHOLD_M = 0.05


def parking_distance(base_position, park_step, sample_steps, scale_m=DISTANCE_SCALE_M):
    if not np.isfinite(scale_m) or scale_m <= 0:
        raise ValueError("Distance scale must be finite and positive")
    xy = np.asarray(base_position)[:, :2]
    distance = np.linalg.norm(xy[sample_steps] - xy[park_step], axis=1)
    return (np.clip(distance, 0, scale_m) / scale_m).astype(np.float32)
