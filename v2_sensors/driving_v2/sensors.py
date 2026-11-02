"""Robot sensors for v2. Each sensor turns the simulator's true state into what a real sensor would report.

WheelEncoders: one incremental encoder per wheel.
  - The angle is counted in whole ticks: COUNTS_PER_REV ticks per wheel revolution, so the reading moves in steps
    of 2*pi / COUNTS_PER_REV and anything smaller is lost.
  - The speed is not measured directly. It is the change in counted angle since the last reading, divided by the
    time between readings (1/30 s). This lags the true speed and is quantised.
  - The 12 network inputs are built from these readings exactly as before: angle since the start, speed, and
    the average and right minus left of both.

To add a sensor (for example an IMU): give it a reset() and a read(frame) that returns a dict of readings,
then add it to Sensors.
"""
import numpy as np

DT = 1 / 30                     # one network decision
COUNTS_PER_REV = 4096           # assumed encoder resolution per wheel revolution (a common 12-bit encoder)
TICK = 2 * np.pi / COUNTS_PER_REV


def count(q):
    """True wheel angle (rad) -> angle the encoder reports (whole ticks)."""
    return np.floor(np.asarray(q) / TICK) * TICK


def wheel_features(angle, speed):
    """12 inputs from wheel angles since the start (4) and wheel speeds (4). Wheel order FR, FL, BL, BR."""
    angle, speed = np.atleast_2d(angle), np.atleast_2d(speed)
    r, l = angle[:, [0, 3]].mean(1), angle[:, [1, 2]].mean(1)
    vr, vl = speed[:, [0, 3]].mean(1), speed[:, [1, 2]].mean(1)
    return np.c_[angle, speed, (r + l) / 2, r - l, (vr + vl) / 2, vr - vl].astype(np.float32)


class WheelEncoders:
    def reset(self, q0):
        self.start = count(q0); self.last = self.start.copy()

    def read(self, q):
        now = count(q)
        speed = (now - self.last) / DT
        self.last = now
        return dict(wheel_angle=now - self.start, wheel_speed=speed)

    @staticmethod
    def features_for_run(q):
        """Encoder inputs for a recorded run: q is the true wheel angle at each 30 Hz decision, shape (T, 4)."""
        enc = WheelEncoders(); enc.reset(q[0]); out = []
        for row in q:
            r = enc.read(row); out.append(wheel_features(r['wheel_angle'], r['wheel_speed'])[0])
        return np.array(out, np.float32)


class Sensors:
    """All sensors together. The network only ever sees what these return."""
    def __init__(self):
        self.encoders = WheelEncoders()

    def reset(self, frame):
        self.encoders.reset(np.array(frame['joints'][:4]))

    def read(self, frame):
        return self.encoders.read(np.array(frame['joints'][:4]))
