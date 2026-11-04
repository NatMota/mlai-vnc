"""Sample identical recorded poses in SEPARATE Tiny/EGL processes to audit portability.

CAMERA_RENDERER=tiny python audit_camera_renderer.py --source ... --ids 0 13 --out tiny.json
CAMERA_RENDERER=egl  python audit_camera_renderer.py --source ... --ids 0 13 --out egl.json
Do not switch renderer in a client with the EGL plugin already loaded: it can still route through EGL.
Ground truth segmentation and target position here are diagnostic only, never detector inputs.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'driving_v2'))
import pybullet as p
from camera_loop import Scene, MOUNT, see_dot, camera_to_robot, INSIDE
from wrist_camera import RENDERER, pixel_to_ray
from render_golden import png


def audit(source, ids, out):
    rows = []; scene = Scene()
    try:
        for dot in ids:
            with np.load(source / f'run_{dot}.npz', allow_pickle=False) as d:
                m = json.loads(str(d['manifest'])); pos, quat, q = d['base_position'], d['base_quaternion'], d['q']
            target = np.array(m['target_position_world_m'])
            for k in sorted(set([0, len(q)//2, max(0, m['parking_step']), len(q)-1])):
                B = np.array(p.getMatrixFromQuaternion(quat[k])).reshape(3, 3)
                scene.pose(np.r_[pos[k], quat[k]], m['joint_names'], q[k], target)
                rgb, depth, seg = scene.fisheye(pos[k]+B@MOUNT, np.c_[-B[:,1],-B[:,2],B[:,0]])
                hit = see_dot(rgb, depth)
                r, g, b = (rgb[..., i].astype(int) for i in range(3))
                red = INSIDE & (r>100) & (r>1.7*g) & (r>1.7*b)
                row = dict(dot=int(dot),step=int(k),nred=int(red.sum()),dot_pixels=int((seg==scene.dot).sum()),
                           false_red=int((red & (seg!=scene.dot)).sum()),detection=None)
                if hit is not None:
                    y,x=red.nonzero(); point=camera_to_robot(hit)
                    row.update(centroid_rc=[float(y.mean()),float(x.mean())],ray=pixel_to_ray(y.mean(),x.mean()).tolist(),
                               detection=point.tolist(),range_m=float(np.linalg.norm(hit)),
                               true_error_mm=float(np.linalg.norm(point-B.T@(target-pos[k]))*1000))
                rows.append(row)
                if dot==ids[0] and k==m['parking_step']:
                    png(out.with_suffix('.png'),rgb)
    finally:
        p.disconnect(scene.cid)
    out.write_text(json.dumps(dict(renderer=RENDERER,rows=rows),indent=2))
    print(f'{RENDERER}: {len(rows)} samples -> {out}')


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--ids',nargs='+',type=int,required=True); ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args(); a.out.parent.mkdir(exist_ok=True,parents=True); audit(a.source,a.ids,a.out)
