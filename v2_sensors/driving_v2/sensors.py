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
from pathlib import Path
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


# End-effector IMU, modelled on the Bosch BMI088 (accelerometer and gyroscope, 3 axes each), read every 1 ms.
# Typical noise densities from the datasheet: 175 ug/sqrt(Hz) and 0.014 deg/s/sqrt(Hz); at 1 kHz (bandwidth 500 Hz)
# that is about 0.038 m/s^2 and 0.0055 rad/s of white noise per reading. Ranges clipped to +-24 g and +-2000 deg/s.
G = 9.81
IMU_ACC_NOISE = 175e-6 * G * np.sqrt(500)
IMU_GYRO_NOISE = np.radians(0.014) * np.sqrt(500)
IMU_ACC_RANGE, IMU_GYRO_RANGE = 24 * G, np.radians(2000)


class HandPose:
    """Position and rotation of the end effector from the joint angles and the base pose, with the robot model
    (URDF) in PyBullet. The IMU is computed from these, the same way in training and in testing."""
    def __init__(self):
        import json, pybullet as p
        from wrist_camera import URDF
        self.p = p; self.cid = p.connect(p.DIRECT)
        self.robot = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=self.cid)
        info = [p.getJointInfo(self.robot, j, physicsClientId=self.cid) for j in range(p.getNumJoints(self.robot, physicsClientId=self.cid))]
        joint = {i[1].decode(): i[0] for i in info}
        self.hand = next(i[0] for i in info if i[12].decode() == 'robot_j2s7s300_end_effector')
        names = json.loads((Path(__file__).resolve().parent / 'closed_loop_native/result_wheels_live_dot.json').read_text())['joint_names']
        self.ids = [joint[n] for n in names]          # the simulator's 17 joints, in its order

    def __call__(self, base_xyz, base_quat, q):
        p = self.p
        p.resetBasePositionAndOrientation(self.robot, list(base_xyz), list(base_quat), physicsClientId=self.cid)
        for j, v in zip(self.ids, q):
            p.resetJointState(self.robot, j, float(v), physicsClientId=self.cid)
        s = p.getLinkState(self.robot, self.hand, computeForwardKinematics=True, physicsClientId=self.cid)
        return np.array(s[4]), np.array(p.getMatrixFromQuaternion(s[5])).reshape(3, 3)


class EndEffectorIMU:
    """Accelerometer (hand acceleration minus gravity, in the hand's frame, m/s^2) and gyroscope (hand rotation rate,
    in the hand's frame, rad/s), from the hand's position and rotation every dt."""
    def __init__(self, dt=1e-3, seed=0, noise=True):
        self.dt, self.noise, self.rng = dt, noise, np.random.default_rng(seed)

    def reset(self, pos, R):
        self.p1 = self.p2 = np.asarray(pos); self.R1 = np.asarray(R)

    @staticmethod
    def rotvec(D):
        """Rotation vector (axis times angle) of a rotation matrix; NumPy only (SciPy clashes with PyTorch's OpenMP)."""
        angle = np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))
        v = 0.5 * np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
        return v if angle < 1e-8 else v * angle / np.sin(angle)

    def read(self, pos, R):
        pos, R = np.asarray(pos), np.asarray(R)
        acc_world = (pos - 2 * self.p1 + self.p2) / self.dt ** 2
        acc = R.T @ (acc_world + np.array([0, 0, G]))
        gyro = self.rotvec(self.R1.T @ R) / self.dt
        self.p2, self.p1, self.R1 = self.p1, pos, R
        if self.noise:
            acc = acc + self.rng.normal(0, IMU_ACC_NOISE, 3); gyro = gyro + self.rng.normal(0, IMU_GYRO_NOISE, 3)
        return dict(imu_acc=np.clip(acc, -IMU_ACC_RANGE, IMU_ACC_RANGE), imu_gyro=np.clip(gyro, -IMU_GYRO_RANGE, IMU_GYRO_RANGE))
