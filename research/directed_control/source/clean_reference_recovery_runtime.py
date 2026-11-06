"""Single architectural prior: active-set DLS within identified q6lower plus observed-training q2lower posture regulariser."""
from pathlib import Path
import numpy as np
from geometry_runtime import feature,position_jacobian
class GeometryPolicy:
 def __init__(self,pw,tw):
  self.pw=pw;self.tw=tw;self.reference=None;self.filtered=None;self.last=None
  w=np.load('/media/nm/Fast Drive/comp0245-directed-edit-20261105/learned_geometry_branch/hybrid_gain15_stop5_v1/models/hybrid_bounds.npz');self.lower=w['lower'];self.upper=w['upper']
 def __call__(self,w,x):
  q=np.asarray(x[:7],float);qd=np.asarray(x[7:14],float);target=np.asarray(x[14:17],float)
  if self.reference is None:self.reference=q.copy();self.filtered=qd.copy()
  else:self.filtered+=(qd-self.filtered)/6
  pos,J=position_jacobian(self.pw,q);error=target-pos;v=np.zeros(7);blocked=np.zeros(7,bool)
  if x[17]>.5:
   for iteration in range(8):
    active=~blocked;A=J[:,active];v[:]=0
    if np.any(active):v[active]=A.T@np.linalg.solve(A@A.T+.02**2*np.eye(3),6.0*error)
    new=((q<=self.lower+.005)&(v<0))|((q>=self.upper-.005)&(v>0))
    if not np.any(new&~blocked):break
    blocked|=new
   v*=min(1.,1.0/max(float(np.max(abs(v))),1e-12))
  # No instantaneous envelope projection, including at first PARK.
  # Blocked joints receive zero integration; sensed-q relative bound is unchanged.
  # Rate-limited recovery toward the existing learned lower-reference priors.
  v=np.maximum(v,np.minimum(.1,(self.lower-self.reference)/.001))
  self.reference=np.clip(self.reference+.001*v,q-.15,q+.15)
  g=feature(q,self.tw['axes'])@self.tw['coef'];tau=g+self.tw['Kp']*(self.reference-q)+self.tw['Kd']*(v-self.filtered)
  tau+=40.0*J.T@(J@(v-self.filtered))
  self.last=dict(reference=self.reference.copy(),velocity=v,position=pos,condition=float(np.linalg.cond(J@J.T+.02**2*np.eye(3))),feedforward=g,filtered_velocity=self.filtered.copy(),blocked=blocked)
  return tau,float(np.linalg.norm(error))
