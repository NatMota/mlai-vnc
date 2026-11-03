"""Wrist 180-degree fisheye RGB-D camera: render saved expert poses, find the dot by colour, estimate where it is.

Run with any Python that has PyBullet:  python wrist_camera.py OUT.npz
Robot and dot are exactly as in robotnik_arm_task.py (no repainting; dot radius 3 cm, colour [1, 0, 0, 0.85]).
No physics is stepped: each frame resets the robot to a recorded expert pose and renders.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('EXPERT_DATA', HERE.parent / 'driving_30/data/expert_30hz'))
URDF = Path(os.environ.get('CAMERA_URDF', HERE.parents[1] / 'collection/native/assets/summit_xl_arm_description/summit_xl_complete.urdf'))
# CAMERA_RENDERER=egl uses the GPU (Linux + NVIDIA); default is the CPU TinyRenderer.
RENDERER = os.environ.get('CAMERA_RENDERER', 'tiny')

# ---- camera definition (all in the hand frame 'robot_j2s7s300_end_effector'; +Z points along the fingers) ----
MOUNT = np.array([0.0, 0.045, -0.07])   # on top of the hand, near the finger roots
SIZE = 256                              # fisheye image is SIZE x SIZE; 180 degrees across the circle
FACE = 192                              # resolution of each 90-degree cube face used to build the fisheye
NEAR, FAR = 0.01, 6.0
DOT_RADIUS, DOT_RGBA = 0.03, [1.0, 0.0, 0.0, 0.85]
STRIDE = 6                              # analyse every 6th saved row (30 Hz -> 5 Hz)

# Camera frame: z forward (hand +Z), y down (= -hand +Y), x right (= y cross z).
def camera_axes(R_hand):
    z = R_hand[:, 2]; y = -R_hand[:, 1]; x = np.cross(y, z)
    return np.c_[x, y, z]               # columns: camera x, y, z in world

# Fisheye lookup: for every output pixel, its unit ray in the camera frame (equidistant: angle = radius * 90 deg).
c = (SIZE - 1) / 2
jj, ii = np.meshgrid(np.arange(SIZE), np.arange(SIZE))
u, v = (jj - c) / c, (ii - c) / c
rr = np.hypot(u, v); INSIDE = rr <= 1
theta, phi = rr * np.pi / 2, np.arctan2(v, u)
RAYS = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)
# Five 90-degree faces (camera frame): forward, right, left, down, up. Each: (forward axis, up axis).
FACES = [((0, 0, 1), (0, -1, 0)), ((1, 0, 0), (0, -1, 0)), ((-1, 0, 0), (0, -1, 0)),
         ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, -1))]
FACES = [(np.array(f, float), np.array(up, float)) for f, up in FACES]
dots = np.stack([RAYS @ f for f, _ in FACES], -1)
FACE_OF = np.argmax(dots, -1)


def pixel_to_ray(row, col):
    uu, vv = (col - c) / c, (row - c) / c
    r = np.hypot(uu, vv); t, ph = r * np.pi / 2, np.arctan2(vv, uu)
    return np.array([np.sin(t) * np.cos(ph), np.sin(t) * np.sin(ph), np.cos(t)])


class Scene:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        if RENDERER == 'egl':
            import importlib.util
            p.loadPlugin(importlib.util.find_spec('eglRenderer').origin, '_eglRendererPlugin', physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.loadURDF('plane.urdf', physicsClientId=self.cid)
        self.robot = p.loadURDF(str(URDF), useFixedBase=True, physicsClientId=self.cid)
        n = p.getNumJoints(self.robot, physicsClientId=self.cid)
        info = [p.getJointInfo(self.robot, j, physicsClientId=self.cid) for j in range(n)]
        self.joint = {i[1].decode(): i[0] for i in info}
        self.hand = next(i[0] for i in info if i[12].decode() == 'robot_j2s7s300_end_effector')
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=DOT_RADIUS, rgbaColor=DOT_RGBA, physicsClientId=self.cid)
        self.dot = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape, physicsClientId=self.cid)

    def pose(self, base_pose, joint_names, q, target):
        p.resetBasePositionAndOrientation(self.robot, base_pose[:3], base_pose[3:], physicsClientId=self.cid)
        for name, value in zip(joint_names, q):
            p.resetJointState(self.robot, self.joint[name], value, physicsClientId=self.cid)
        p.resetBasePositionAndOrientation(self.dot, target, [0, 0, 0, 1], physicsClientId=self.cid)
        s = p.getLinkState(self.robot, self.hand, computeForwardKinematics=True, physicsClientId=self.cid)
        R_hand = np.array(p.getMatrixFromQuaternion(s[5])).reshape(3, 3)
        eye = np.array(s[4]) + R_hand @ MOUNT
        return eye, camera_axes(R_hand)

    def fisheye(self, eye, C):
        rgb = np.zeros((SIZE, SIZE, 3), np.uint8); depth = np.full((SIZE, SIZE), np.inf); seg = np.full((SIZE, SIZE), -1)
        proj = p.computeProjectionMatrixFOV(90, 1, NEAR, FAR)
        for k, (f, up) in enumerate(FACES):
            sel = INSIDE & (FACE_OF == k)
            if not sel.any():
                continue
            fw, upw = C @ f, C @ up
            view = p.computeViewMatrix(eye.tolist(), (eye + fw).tolist(), upw.tolist())
            _, _, img, zbuf, mask = p.getCameraImage(FACE, FACE, view, proj, renderer=p.ER_BULLET_HARDWARE_OPENGL if RENDERER == 'egl' else p.ER_TINY_RENDERER,
                                                     physicsClientId=self.cid)
            img = np.asarray(img, np.uint8).reshape(FACE, FACE, 4)[..., :3]
            zbuf = np.asarray(zbuf).reshape(FACE, FACE); mask = np.asarray(mask).reshape(FACE, FACE)
            right, down = np.cross(f, up), -up
            d = RAYS[sel]; along = d @ f
            a, b = (d @ right) / along, (d @ down) / along             # in [-1, 1] on this face
            col = np.clip(np.rint((a + 1) / 2 * (FACE - 1)), 0, FACE - 1).astype(int)
            row = np.clip(np.rint((b + 1) / 2 * (FACE - 1)), 0, FACE - 1).astype(int)
            rgb[sel] = img[row, col]
            z = FAR * NEAR / (FAR - (FAR - NEAR) * zbuf[row, col])     # metres along the face axis
            depth[sel] = z / along                                      # metres along the pixel's ray
            seg[sel] = mask[row, col] & ((1 << 24) - 1)
        return rgb, depth, seg


def detect(rgb, depth):
    """Colour threshold -> centroid -> ray; depth at the dot pixels -> distance. Returns (found, ray, dist, n_pixels)."""
    r, g, b = (rgb[..., k].astype(int) for k in range(3))
    red = INSIDE & (r > 100) & (r > 1.7 * g) & (r > 1.7 * b)
    n = int(red.sum())
    if n == 0:
        return False, None, None, 0
    rows, cols = red.nonzero()
    ray = pixel_to_ray(rows.mean(), cols.mean())
    dist = float(np.median(depth[red])) + DOT_RADIUS      # surface -> centre
    return True, ray, dist, n


def planar(point_world, base_pose):
    """Dot relative to the robot base: horizontal distance and direction (degrees) from where the base faces."""
    x, y, z, w = base_pose[3:]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dx, dy = point_world[0] - base_pose[0], point_world[1] - base_pose[1]
    ang = np.arctan2(dy, dx) - yaw
    return float(np.hypot(dx, dy)), float(np.rad2deg(np.arctan2(np.sin(ang), np.cos(ang))))


def episode(r):
    scene = Scene()
    names = r['manifest']['joint_names']
    target = np.array(r['manifest']['target_position_world_m'])
    with np.load(DATA / r['archive']) as a:
        poses, joints, times = a['base_pose'], a['joint_positions'], a['time_s']
    n = len(poses); idx = sorted(set(range(0, n, STRIDE)) | {n // 2, n - 1})
    key = {0: 'start', n // 2: 'middle', n - 1: 'park'}
    rows, images = [], {}
    for i in idx:
        eye, C = scene.pose(poses[i], names, joints[i], target)
        rgb, depth, seg = scene.fisheye(eye, C)
        truly_visible = int((seg == scene.dot).sum())
        found, ray, dist, npx = detect(rgb, depth)
        false_px = int(((rgb[..., 0] > 100) & (rgb[..., 0] > 1.7 * rgb[..., 1]) & (rgb[..., 0] > 1.7 * rgb[..., 2])
                        & INSIDE & (seg != scene.dot)).sum())
        true_d, true_a = planar(target, poses[i])
        row = dict(bearing=r['bearing_deg'], index=i, time_s=float(times[i]), true_visible_px=truly_visible,
                   found=found, dot_px=npx, false_red_px=false_px, true_dist_m=true_d, true_dir_deg=true_a,
                   est_dist_m=np.nan, est_dir_deg=np.nan)
        if found:
            est = eye + C @ (ray * dist)                 # camera ray -> world point (camera pose from arm joints)
            row['est_dist_m'], row['est_dir_deg'] = planar(est, poses[i])
        rows.append(row)
        if i in key:
            images[key[i]] = rgb
    return rows, images


if __name__ == '__main__':
    out = Path(sys.argv[1])
    records = [json.loads(line) for line in open(DATA / 'episodes.jsonl')]
    unique = {}
    for r in records:
        unique.setdefault(r['trajectory_hash'], r)
    eps = sorted(unique.values(), key=lambda r: r['bearing_deg'])
    with Pool(4) as pool:
        results = pool.map(episode, eps)
    rows = [row for rs, _ in results for row in rs]
    bearings = np.array([e['bearing_deg'] for e in eps])
    imgs = {k: np.stack([im[k] for _, im in results]) for k in ('start', 'middle', 'park')}
    # One outside view showing where the camera sits (hold pose, bearing 12).
    scene = Scene(); e0 = eps[0]
    with np.load(DATA / e0['archive']) as a:
        eye, C = scene.pose(a['base_pose'][0], e0['manifest']['joint_names'], a['joint_positions'][0],
                            e0['manifest']['target_position_world_m'])
    marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[0, 0.8, 0, 1], physicsClientId=scene.cid)
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker, basePosition=eye.tolist(), physicsClientId=scene.cid)
    view = p.computeViewMatrix((eye + np.array([-0.25, 0.45, 0.25])).tolist(), eye.tolist(), [0, 0, 1])
    _, _, ext, _, _ = p.getCameraImage(512, 384, view, p.computeProjectionMatrixFOV(45, 512 / 384, 0.01, 10),
                                       renderer=p.ER_TINY_RENDERER, physicsClientId=scene.cid)
    np.savez_compressed(out, rows=json.dumps(rows), bearings=bearings, outside=np.asarray(ext, np.uint8).reshape(384, 512, 4)[..., :3],
                        camera_mount=MOUNT, size=SIZE, **{f'img_{k}': v for k, v in imgs.items()})
    print('frames', len(rows), '->', out)
