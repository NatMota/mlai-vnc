"""Separate non-imitation locomotion probe. Wheel actuator torque is identically zero."""
import argparse, json, time
from pathlib import Path
import numpy as np
import pybullet as p
from geometry_fast import position_jacobian
R=Path(__file__).parent

def feature(q, axes):
    f=np.ones(1)
    for j in axes: f=(f[:,None]*[1,np.sin(q[j]),np.cos(q[j])]).reshape(-1)
    return f

def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['hold','stroke'],required=True);a.add_argument('--seconds',type=float,default=25);a.add_argument('--out',required=True);a.add_argument('--ground-z',type=float,default=.10);a.add_argument('--damping-scale',type=float,default=1);a.add_argument('--hard-contact-diagnostic',action='store_true');args=a.parse_args()
    out=R/args.out;out.mkdir(exist_ok=False)
    pw=dict(np.load(R/'trig_position_training/selected.npz'));tw=dict(np.load(R/'identified_torque_training/identified_torque.npz'))
    p.connect(p.DIRECT);p.setTimeStep(.001);p.setGravity(0,0,-9.81);p.setPhysicsEngineParameter(numSolverIterations=30,enableConeFriction=0)
    assets=R/'native/assets';floor=p.loadURDF(str(assets/'plane.urdf'))
    robot=p.loadURDF(str(assets/'summit_xl_arm_description/summit_xl_complete.urdf'),flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_IMPLICIT_CYLINDER)
    for body in [floor,robot]:p.changeDynamics(body,-1,linearDamping=0,angularDamping=0,jointDamping=0)
    p.changeDynamics(floor,-1,lateralFriction=1)
    infos=[p.getJointInfo(robot,j) for j in range(p.getNumJoints(robot))];active=[x[0] for x in infos if x[2]!=p.JOINT_FIXED];assert len(active)==17
    wheels,arm,fingers=active[:4],active[4:11],active[11:]
    ee=next(x[0] for x in infos if x[12].decode()=='robot_j2s7s300_end_effector')
    for j in wheels:
        p.changeDynamics(robot,j,anisotropicFriction=[1,.01,1])
        if not args.hard_contact_diagnostic:p.changeDynamics(robot,j,contactStiffness=30000,contactDamping=1000)
    qhome=np.array([0,1.57,0,1,0,1,0.]);q0=np.r_[np.zeros(4),qhome,np.full(6,.1)]
    p.setJointMotorControlArray(robot,active,p.VELOCITY_CONTROL,forces=[0]*17)
    for j,x in zip(active,q0):p.resetJointState(robot,j,float(x),0)
    p.setJointMotorControlArray(robot,fingers,p.POSITION_CONTROL,targetPositions=[.1]*6,positionGains=[.1]*6,velocityGains=[1]*6,forces=[5]*6)
    # References are generated with the frozen learned map, not simulator IK.
    low=np.array([-6,.8203047,-6,.5235988,-6,1.134464,-6]);high=np.array([6,5.46288,6,5.75959,6,5.14872,6])
    def ik(target, start):
        q=start.copy();ix=np.array([1,3,5])
        for _ in range(150):
            pos,J=position_jacobian(pw,q);A=J[:,ix];e=target-pos
            d=A.T@np.linalg.solve(A@A.T+1e-6*np.eye(3),e);q[ix]+=np.clip(d,-.1,.1);q=np.clip(q,low,high)
            if np.linalg.norm(e)<.0001:break
        return q
    seed=np.array([0,1.60,0,3.73,0,1.385,0])
    targets=[np.array([.75,-.0098,.30]),np.array([.75,-.0098,args.ground_z]),np.array([.55,-.0098,args.ground_z]),np.array([.55,-.0098,.30])]
    poses=[]
    for xyz in targets:seed=ik(xyz,seed);poses.append(seed.copy())
    # Initial-state diagnostic only. Reset before dynamic simulation starts.
    calibration=[]
    for xyz,q in zip(targets,poses):
        for j,x in zip(arm,q):p.resetJointState(robot,j,float(x),0)
        actual=np.array(p.getLinkState(robot,ee,computeForwardKinematics=True)[4]);pred,_=position_jacobian(pw,q)
        calibration.append(dict(target=xyz.tolist(),q=q.tolist(),learned=pred.tolist(),actual=actual.tolist(),error_mm=float(np.linalg.norm(pred-actual)*1000)))
    for j,x in zip(active,q0):p.resetJointState(robot,j,float(x),0)
    p.resetBasePositionAndOrientation(robot,[0,0,0],[0,0,0,1]);p.resetBaseVelocity(robot,[0,0,0],[0,0,0])
    cfg=dict(mode=args.mode,seconds=args.seconds,ground_z=args.ground_z,damping_scale=args.damping_scale,hard_contact_diagnostic=args.hard_contact_diagnostic,physics_hz=1000,wheel_torque=0,torque_limit_Nm=30,self_collision=False,geometry_calibration=calibration,description='Engineered arm-reference gait, frozen learned feedforward torque, no powered wheels or external force. Self-collision setting matches baseline; not a navigation or imitation result.')
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    rows=[];frames=[];wheel_work=0.;arm_work=0.;contact_ticks=0;reference=qhome.copy();filtered=np.zeros(7);start=time.monotonic()
    for tick in range(int(args.seconds*1000)):
        t=tick*.001;st=p.getJointStates(robot,active);q=np.array([x[0] for x in st]);qd=np.array([x[1] for x in st]);qa=q[4:11];va=qd[4:11]
        if args.mode=='hold': desired=qhome
        elif t<7:
            s=np.clip(t/7,0,1);s=s*s*(3-2*s);desired=qhome+(poses[0]-qhome)*s
        else:
            u=(t-7)%6;seg=int(u/1.5);s=(u%1.5)/1.5;s=s*s*(3-2*s);desired=poses[seg]+s*(poses[(seg+1)%4]-poses[seg])
        v=np.clip((desired-reference)/.001,-.6,.6);reference=np.clip(reference+.001*v,qa-.15,qa+.15)
        filtered+=(va-filtered)/6
        tau=np.clip(feature(qa,tw['axes'])@tw['coef']+tw['Kp']*(reference-qa)+tw['Kd']*np.array([args.damping_scale,1,args.damping_scale,.5,args.damping_scale,.5,args.damping_scale])*(v-filtered),-30,30)
        sent=np.r_[np.zeros(4),tau];assert np.all(sent[:4]==0)
        p.setJointMotorControlArray(robot,active[:11],p.TORQUE_CONTROL,forces=sent.tolist());p.stepSimulation()
        contacts=[c for c in p.getContactPoints(robot,floor) if c[3] not in wheels and c[3]>=arm[0]]
        force=sum(c[9] for c in contacts);contact_ticks+=force>1e-6;arm_work+=float(tau@va)*.001;wheel_work+=float(sent[:4]@qd[:4])*.001
        base,quat=p.getBasePositionAndOrientation(robot);ep=p.getLinkState(robot,ee,computeForwardKinematics=True)[4]
        rows.append([t,*base,*quat,*q,*qd,*sent,*ep,force,arm_work,wheel_work])
        if tick%50==0:
            view=p.computeViewMatrix([base[0]+1.8,base[1]-2,1.3],[base[0]+.2,base[1],.35],[0,0,1]);proj=p.computeProjectionMatrixFOV(55,1.5,.05,10)
            im=p.getCameraImage(480,320,view,proj,renderer=p.ER_TINY_RENDERER)[2]
            frames.append(np.asarray(im,dtype=np.uint8)[:,:,:3])
        if tick%5000==0:print(json.dumps(dict(t=t,base=list(base),contact_force=force)),flush=True)
    arr=np.array(rows);np.savez_compressed(out/'trajectory.npz',data=arr,frames=np.array(frames))
    summary=dict(mode=args.mode,seconds=args.seconds,wall_s=time.monotonic()-start,base_final=arr[-1,1:4].tolist(),forward_displacement_m=float(arr[-1,1]-arr[0,1]),lateral_displacement_m=float(arr[-1,2]-arr[0,2]),arm_floor_contact_seconds=contact_ticks*.001,arm_net_work_J=arm_work,wheel_actuator_work_J=wheel_work,wheel_sent_torque_max_Nm=float(abs(arr[:,42:46]).max()),max_contact_force_N=float(arr[:,-3].max()),columns='time, base_xyz, base_quat, q17, qd17, sent_wheel4_arm7, ee_xyz, arm_floor_normal_force, arm_net_work, wheel_work')
    assert summary['wheel_sent_torque_max_Nm']==0 and wheel_work==0
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);p.disconnect()
if __name__=='__main__':main()
