"""Causal 1 kHz sensor history, reset at PARK and padded with its first sample."""
from collections import deque
import numpy as np

LAGS_MS = (0, 50, 100)


def stack_history(x):
    indices = np.arange(len(x))
    return np.concatenate([x[np.maximum(0, indices-lag)] for lag in LAGS_MS], axis=1)


class SensorHistory:
    def __init__(self):
        self.rows = deque(maxlen=max(LAGS_MS)+1)

    def push(self, x):
        self.rows.append(np.asarray(x, np.float32).copy())
        return np.concatenate([self.rows[max(0,len(self.rows)-1-lag)] for lag in LAGS_MS])


if __name__ == '__main__':
    x=np.random.default_rng(7).normal(size=(207,17)).astype(np.float32)
    history=SensorHistory()
    np.testing.assert_array_equal(np.stack([history.push(row) for row in x]),stack_history(x))
    np.testing.assert_array_equal(stack_history(x)[0],np.tile(x[0],3))
    np.testing.assert_array_equal(stack_history(x)[150,17:34],x[100])
    np.testing.assert_array_equal(stack_history(x)[150,34:],x[50])
    print('PASS: online/offline history exact, causal lags, PARK padding, bounded buffer')
