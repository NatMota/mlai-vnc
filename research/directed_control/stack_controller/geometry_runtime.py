"""Engineered controller using only learned geometry/torque and sensor state."""
import numpy as np

def feature(q,axes,derivative=None):
 f=np.ones(1)
 for j in axes:
  v=np.array([0,np.cos(q[j]),-np.sin(q[j])]) if j==derivative else np.array([1,np.sin(q[j]),np.cos(q[j])])
  f=(f[:,None]*v).reshape(-1)
 return f

def position_jacobian(w,q):
 pos=feature(q,w['axes'])@w['coef']
 J=np.column_stack([feature(q,w['axes'],j)@w['coef'] if j in w['axes'] else np.zeros(3) for j in range(7)])
 return pos,J

class GeometryPolicy:
 def __init__(self,pw,tw):self.pw=pw;self.tw=tw;self.reference=None;self.last=None
 def __call__(self,w,x):
  q=np.asarray(x[:7],dtype=float);qd=np.asarray(x[7:14],dtype=float);target=np.asarray(x[14:17],dtype=float)
  if self.reference is None:self.reference=q.copy()
  pos,J=position_jacobian(self.pw,q);error=target-pos;v=np.zeros(7)
  if x[17]>.5:
   v=J.T@np.linalg.solve(J@J.T+(.02**2)*np.eye(3),.8*error)
   v*=min(1.,.5/max(float(np.max(np.abs(v))),1e-12))
  self.reference=np.clip(self.reference+.001*v,q-.15,q+.15)
  g=feature(q,self.tw['axes'])@self.tw['coef']
  tau=g+self.tw['Kp']*(self.reference-q)+self.tw['Kd']*(v-qd)
  self.last=dict(reference=self.reference.copy(),velocity=v,position=pos,condition=float(np.linalg.cond(J@J.T+.02**2*np.eye(3))),feedforward=g)
  return tau,float(np.linalg.norm(error))
