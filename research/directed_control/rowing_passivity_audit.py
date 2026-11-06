"""Check free-wheel physics before attributing motion to the rowing gait."""
import pybullet as p, numpy as np,json
from pathlib import Path
R=Path(__file__).parent
out=[]
for stiffness in [True,False]:
 for aniso in [True,False]:
  p.connect(p.DIRECT);p.setGravity(0,0,-9.81);p.setTimeStep(.001);p.setPhysicsEngineParameter(numSolverIterations=30,enableConeFriction=0)
  floor=p.loadURDF(str(R/'native/assets/plane.urdf'));r=p.loadURDF(str(R/'native/assets/summit_xl_arm_description/summit_xl_complete.urdf'),flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_IMPLICIT_CYLINDER)
  active=[j for j in range(p.getNumJoints(r)) if p.getJointInfo(r,j)[2]!=p.JOINT_FIXED]
  for b in [floor,r]:p.changeDynamics(b,-1,linearDamping=0,angularDamping=0,jointDamping=0)
  p.changeDynamics(floor,-1,lateralFriction=1)
  for j in active[:4]:
   if stiffness:p.changeDynamics(r,j,contactStiffness=30000,contactDamping=1000)
   if aniso:p.changeDynamics(r,j,anisotropicFriction=[1,.01,1])
  p.setJointMotorControlArray(r,active,p.VELOCITY_CONTROL,forces=[0]*17)
  for j,x in zip(active,[0,0,0,0,0,1.57,0,1,0,1,0,.1,.1,.1,.1,.1,.1]):p.resetJointState(r,j,x,0)
  p.resetBasePositionAndOrientation(r,[0,0,0],[0,0,0,1])
  start=p.getBasePositionAndOrientation(r)[0]
  for t in range(5000):
   p.setJointMotorControlArray(r,active,p.TORQUE_CONTROL,forces=[0]*17);p.stepSimulation()
  end=p.getBasePositionAndOrientation(r)[0];vel=p.getBaseVelocity(r)[0];q=p.getJointStates(r,active)
  out.append(dict(stiffness=stiffness,anisotropic=aniso,initial=start,final=end,velocity=vel,wheel_speed=[x[1] for x in q[:4]],motor_torque=[x[3] for x in q],all_commands_zero=True));print(json.dumps(out[-1]),flush=True);p.disconnect()
(R/'rowing_passivity_audit.json').write_text(json.dumps(out,indent=2))
