"""Robot sensors for v2. Each sensor turns the simulator's true state into what a real sensor would report.

WheelEncoders: one incremental encoder per wheel.
  - The angle is counted in whole ticks: COUNTS_PER_REV ticks per wheel revolution, so the reading moves in steps
    of 2*pi / COUNTS_PER_REV and anything smaller is lost.
  - The speed is not measured directly. It is the change in counted angle since the last reading, divided by the
    time between readings (1/30 s). This lags the true speed and is quantised.
  - The 12 network inputs are built from these readings exactly as before: angle since the start, speed, and
    the average and right minus left of both.

ArmEncoders: the position sensor in each of the 7 arm actuators (Kinova Jaco 2, KA-75+ on joints 1-4,
KA-58 on joints 5-7; resolutions from Kinova's actuator datasheet, AS-ACT-KA58-KA75-SP-INT-EN 201804-1.2).
  - The sensor is absolute, so the angle is the joint angle itself, counted in whole ticks. We assume the joints
    have already indexed, so the +-1.5 deg start-up error does not apply.
  - The speed is the change in counted angle since the last reading: over 1/30 s, as for the wheels, or over 1 ms
    when the arm network runs at 1 kHz.

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


ARM_COUNTS_PER_REV = np.array([3_686_400] * 4 + [2_534_400] * 3)    # KA-75+ x4, KA-58 x3
ARM_TICK = 2 * np.pi / ARM_COUNTS_PER_REV
ARM = slice(4, 11)              # arm joints in the simulator's joint list (after the 4 wheels)


class ArmEncoders:
    def __init__(self, dt=DT):
        self.dt = dt                # time between readings: 1/30 s, or 1 ms for an arm network running at 1 kHz

    def reset(self, q0):
        self.last = np.floor(np.asarray(q0) / ARM_TICK) * ARM_TICK

    def read(self, q):
        now = np.floor(np.asarray(q) / ARM_TICK) * ARM_TICK
        speed = (now - self.last) / self.dt
        self.last = now
        return dict(arm_angle=now, arm_speed=speed)

    @staticmethod
    def readings_for_run(q, dt=DT):
        """Encoder angles and speeds for a recorded run: q is the true arm angle at each reading, shape (T, 7)."""
        enc = ArmEncoders(dt); enc.reset(q[0]); out = [enc.read(row) for row in q]
        return np.array([r['arm_angle'] for r in out]), np.array([r['arm_speed'] for r in out])


class Sensors:
    """All sensors together. The network only ever sees what these return."""
    def __init__(self):
        self.encoders = WheelEncoders()
        self.arm = ArmEncoders()

    def reset(self, frame):
        q = np.array(frame['joints'])
        self.encoders.reset(q[:4]); self.arm.reset(q[ARM])

    def read(self, frame):
        q = np.array(frame['joints'])
        return {**self.encoders.read(q[:4]), **self.arm.read(q[ARM])}
