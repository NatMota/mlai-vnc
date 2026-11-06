"""Offline rendered sensor-calibration diagnostic. Truth used for scoring/rendering only."""
import os,sys,json
from pathlib import Path
import numpy as np
R=Path(__file__).parent
os.environ.update(CAMERA_URDF=str(R/'native/assets/summit_xl_arm_description/summit_xl_complete.urdf'),CAMERA_RENDERER='egl',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(R/'runtime/v2_sensors/driving_v2'))
from camera_loop import Scene,see_dot,camera_to_robot,MOUNT
from wrist_camera import RAYS,INSIDE,DOT_RADIUS,FACES,FACE_LOOKUP,FACE,NEAR,FAR,SIZE
import pybullet as p
# Reconstruct each actual sampled cube-face pixel ray from the calibrated projection.
# This uses camera intrinsics only, never target or robot truth.
Q=np.zeros((SIZE,SIZE,3))
proj=list(p.computeProjectionMatrixFOV(90,1,NEAR,FAR));proj[8]-=1/FACE;proj[9]-=1/FACE
for (f,up),(sel,row,col,along) in zip(FACES,FACE_LOOKUP):
 view=p.computeViewMatrix([0,0,0],f.tolist(),up.tolist())
 P=np.array(proj).reshape(4,4,order='F');V=np.array(view).reshape(4,4,order='F')
 ndc=np.c_[2*(col+.5)/FACE-1,2*(FACE-row-.5)/FACE-1,np.zeros(len(row)),np.ones(len(row))]
 xyz=(np.linalg.inv(P@V)@ndc.T).T;xyz=xyz[:,:3]/xyz[:,3,None]
 Q[sel]=xyz/(xyz@f)[:,None]*along[:,None]
def fit(rgb,depth):
 init=see_dot(rgb,depth)
 if init is None:return None,0
 r,g,b=(rgb[...,k].astype(int) for k in range(3));mask=INSIDE&(r>100)&(r>1.7*g)&(r>1.7*b)&np.isfinite(depth)
 P=Q[mask]*depth[mask,None]
 if len(P)<6:return init,len(P)
 c=init.copy()
 for _ in range(12):
  v=c-P;d=np.linalg.norm(v,axis=1);A=v/np.maximum(d[:,None],1e-8);res=d-DOT_RADIUS
  # Radius is a known physical property of the marker, never its position.
  delta=np.linalg.solve(A.T@A+np.eye(3)*.02,-A.T@res)
  norm=np.linalg.norm(delta)
  if norm>.02:delta*=.02/norm
  c+=delta
  if np.linalg.norm(delta)<1e-6:break
 if np.linalg.norm(c-init)>.06 or not np.isfinite(c).all():return init,len(P)
 return c,len(P)
def main():
 scene=Scene();names=[p.getJointInfo(scene.robot,j)[1].decode() for j in range(p.getNumJoints(scene.robot)) if p.getJointInfo(scene.robot,j)[2]!=p.JOINT_FIXED]
 rows=[]
 for dot in [6,16,23,45,55,80]:
  run=json.loads((R/f'egl_batch01/baseline/run_{dot}.json').read_text());a=np.load(R/f'data/expert/val/run_{dot}.npz');target=np.array(json.loads(str(a['manifest']))['target_position_world_m'])
  for fr in run['frames'][::6]:
   B=np.array(p.getMatrixFromQuaternion(fr['quaternion'])).reshape(3,3);base=np.array(fr['base'])
   scene.pose(np.r_[base,fr['quaternion']],names,fr['joints'],target)
   rgb,depth,_=scene.fisheye(base+B@MOUNT,np.c_[-B[:,1],-B[:,2],B[:,0]])
   old=see_dot(rgb,depth);new,n=fit(rgb,depth)
   if old is None:continue
   truth=B.T@(target-base)
   rows.append(dict(dot=dot,t=fr['time_s'],n_pixels=n,centroid_error_mm=float(1000*np.linalg.norm(camera_to_robot(old)-truth)),sphere_error_mm=float(1000*np.linalg.norm(camera_to_robot(new)-truth))))
 stats=[]
 for dot in [6,16,23,45,55,80]:
  a=[x for x in rows if x['dot']==dot]
  stats.append(dict(dot=dot,frames=len(a),centroid_median_mm=float(np.median([x['centroid_error_mm'] for x in a])),sphere_median_mm=float(np.median([x['sphere_error_mm'] for x in a])),centroid_p90_mm=float(np.quantile([x['centroid_error_mm'] for x in a],.9)),sphere_p90_mm=float(np.quantile([x['sphere_error_mm'] for x in a],.9))))
 (R/'camera_fit_v2_audit.json').write_text(json.dumps(dict(kind='offline same-frame sensor estimator comparison; no policy improvement claimed',stats=stats,rows=rows),indent=2));print(json.dumps(stats,indent=2));p.disconnect()
if __name__=='__main__':main()
