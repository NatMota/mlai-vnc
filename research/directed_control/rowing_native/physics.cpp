#include "expert.hpp"
#include "bullet_api.hpp"
#include <stdexcept>
#include <filesystem>
struct Physics::Impl {
    b3PhysicsClientHandle c=nullptr;
    int robot=-1, saved=-1;
    std::vector<b3JointInfo> joints;
    V17 prev=V17::Zero();
    bool fingers_configured=false;
    bool owned=true;
    b3SharedMemoryStatusHandle send(b3SharedMemoryCommandHandle cmd) {
        auto s=b3SubmitClientCommandAndWaitStatus(c,cmd);
        if(!s) throw std::runtime_error("Bullet command returned no status");
        return s;
    }
    int load(const std::string& path,int flags) {
        auto cmd=b3LoadUrdfCommandInit(c,path.c_str());
        b3LoadUrdfCommandSetFlags(cmd,flags);
        int id=b3GetStatusBodyIndex(send(cmd));
        if(id<0) throw std::runtime_error("Cannot load "+path);
        return id;
    }
};
Physics::Physics(const std::string& assets, Physics* shared, bool self_collision):p(new Impl) {
    p->owned=!shared;p->c=shared?shared->p->c:b3ConnectPhysicsDirect();
    if(!p->c) throw std::runtime_error("Cannot connect Bullet DIRECT");
    b3SharedMemoryCommandHandle cmd;int floor=0;
    if(!shared){cmd=b3InitPhysicsParamCommand(p->c);
    b3PhysicsParamSetTimeStep(cmd,dt);
    b3PhysicsParamSetGravity(cmd,0,0,-9.81);
    b3PhysicsParamSetNumSolverIterations(cmd,30);
    b3PhysicsParamSetEnableConeFriction(cmd,0);
    p->send(cmd);
    floor=p->load(assets+"/plane.urdf",0);}
    int visual_flag=std::getenv("EXPERT_SKIP_VISUALS")?URDF_IGNORE_VISUAL_SHAPES:0;
    p->robot=p->load(assets+"/summit_xl_arm_description/summit_xl_complete.urdf",URDF_USE_INERTIA_FROM_FILE|URDF_USE_IMPLICIT_CYLINDER|visual_flag|(self_collision?URDF_USE_SELF_COLLISION:0));
    // Preserve the original wrapper's effective damping settings (body IDs 0,1).
    for(int body : {floor,p->robot}) {
        cmd=b3InitChangeDynamicsInfo(p->c);
        b3ChangeDynamicsInfoSetLinearDamping(cmd,body,0);
        b3ChangeDynamicsInfoSetAngularDamping(cmd,body,0);
        b3ChangeDynamicsInfoSetJointDamping(cmd,body,-1,0);
        p->send(cmd);
    }
    cmd=b3InitChangeDynamicsInfo(p->c);
    b3ChangeDynamicsInfoSetLateralFriction(cmd,floor,-1,1);
    p->send(cmd);
    for(int j=0;j<b3GetNumJoints(p->c,p->robot);j++) {
        b3JointInfo info{};
        b3GetJointInfo(p->c,p->robot,j,&info);
        if(info.m_jointType==eFixedType) continue;
        p->joints.push_back(info); names.emplace_back(info.m_jointName);
        if(names.back().find("wheel")!=std::string::npos) {
            cmd=b3InitChangeDynamicsInfo(p->c);
            double friction[3]={1,.01,1};
            b3ChangeDynamicsInfoSetContactStiffnessAndDamping(cmd,p->robot,j,30000,1000);
            b3ChangeDynamicsInfoSetAnisotropicFriction(cmd,p->robot,j,friction);
            p->send(cmd);
        }
    }
    if(p->joints.size()!=17) throw std::runtime_error("Expected 17 active joints");
    for(int i=0;i<7;i++) {lower[i]=p->joints[i+4].m_jointLowerLimit; upper[i]=p->joints[i+4].m_jointUpperLimit;}
    double q0[17]={0,0,0,0,0,1.57,0,1,0,1,0,.1,.1,.1,.1,.1,.1};
    for(int i=0;i<17;i++) {
        auto j=p->joints[i];
        cmd=b3JointControlCommandInit2(p->c,p->robot,CONTROL_MODE_VELOCITY);
        b3JointControlSetDesiredVelocity(cmd,j.m_uIndex,0);
        b3JointControlSetKd(cmd,j.m_uIndex,1);
        b3JointControlSetMaximumForce(cmd,j.m_uIndex,0);p->send(cmd);
        cmd=b3CreatePoseCommandInit(p->c,p->robot);
        b3CreatePoseCommandSetJointPosition(p->c,cmd,j.m_jointIndex,q0[i]);
        b3CreatePoseCommandSetJointVelocity(p->c,cmd,j.m_jointIndex,0);p->send(cmd);
    }
    cmd=b3CreatePoseCommandInit(p->c,p->robot);
    double zero[3]={0,0,0};
    b3CreatePoseCommandSetBasePosition(cmd,0,0,0);
    b3CreatePoseCommandSetBaseOrientation(cmd,0,0,0,1);
    b3CreatePoseCommandSetBaseLinearVelocity(cmd,zero);
    b3CreatePoseCommandSetBaseAngularVelocity(cmd,zero);p->send(cmd);
    p->saved=b3GetStatusGetStateId(p->send(b3SaveStateCommandInit(p->c)));
}
Physics::~Physics(){if(p->c&&p->owned)b3DisconnectSharedMemory(p->c);}
void Physics::isolate(){
    // Co-located independent robots see the ground (group 1), never one another (group 2).
    // This avoids translating origins, but shared-world solver ordering must still be validated.
    auto floor=b3CollisionFilterCommandInit(p->c);b3SetCollisionFilterGroupMask(floor,0,-1,1,2);p->send(floor);
    for(int j=-1;j<b3GetNumJoints(p->c,p->robot);j++){
        auto cmd=b3CollisionFilterCommandInit(p->c);b3SetCollisionFilterGroupMask(cmd,p->robot,j,2,1);p->send(cmd);
    }
}
void Physics::reset(){auto cmd=b3LoadStateCommandInit(p->c);b3LoadStateSetStateId(cmd,p->saved);p->send(cmd);p->prev.setZero();p->fingers_configured=false;}
State Physics::state(){
    State s;
    auto status=p->send(b3RequestActualStateCommandInit(p->c,p->robot));
    int id,nq,nv; const double *root,*q,*v,*reaction;
    if(!b3GetStatusActualState(status,&id,&nq,&nv,&root,&q,&v,&reaction)) throw std::runtime_error("State retrieval failed");
    s.pos=Eigen::Map<const V3>(q);s.quat=Eigen::Map<const Eigen::Vector4d>(q+3);
    s.lin=Eigen::Map<const V3>(v);s.ang=Eigen::Map<const V3>(v+3);
    for(int i=0;i<17;i++){s.q[i]=q[p->joints[i].m_qIndex];s.qd[i]=v[p->joints[i].m_uIndex];}
    s.qdd=(s.qd-p->prev)/dt;p->prev=s.qd;
    return s;
}
V17 Physics::step(const State& s,const Action& a,bool advance_world,const V17* extra){
    V17 torque=V17::Zero();
    torque.head<4>().setZero(); // Explicit unpowered-wheel demonstration only.
    torque.segment<7>(4)=a.tau;
    if(extra)torque.head<11>()+=extra->head<11>();
    auto cmd=b3JointControlCommandInit2(p->c,p->robot,CONTROL_MODE_TORQUE);
    for(int i=0;i<11;i++) b3JointControlSetDesiredForceTorque(cmd,p->joints[i].m_uIndex,torque[i]);
    p->send(cmd);
    // Bullet retains servo targets and gains; these six constant commands need setting once.
    if(!p->fingers_configured){
      cmd=b3JointControlCommandInit2(p->c,p->robot,CONTROL_MODE_POSITION_VELOCITY_PD);
      for(int i=11;i<17;i++) {
        auto j=p->joints[i];
        b3JointControlSetDesiredPosition(cmd,j.m_qIndex,.1);
        b3JointControlSetDesiredVelocity(cmd,j.m_uIndex,0);
        b3JointControlSetKp(cmd,j.m_uIndex,.1);b3JointControlSetKd(cmd,j.m_uIndex,1);
        b3JointControlSetMaximumForce(cmd,j.m_uIndex,5);
      }
      p->send(cmd);p->fingers_configured=true;
    }
    if(advance_world)advance();
    torque.tail<6>().setConstant(missing);return torque;
}
void Physics::advance(){p->send(b3InitStepSimulationCommand(p->c));}
Eigen::MatrixXd Physics::mass(const State& s){
    Eigen::Matrix<double,24,1> full;full<<s.pos,s.quat,s.q;
    auto cmd=b3CalculateMassMatrixCommandInit(p->c,p->robot,full.data(),24);
    b3CalculateMassMatrixSetFlags(cmd,1);
    auto status=p->send(cmd);int n=0;
    b3GetStatusMassMatrix(p->c,status,&n,nullptr);
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> m(n,n);b3GetStatusMassMatrix(p->c,status,&n,m.data());
    // Bullet flags=1 includes the redundant quaternion coordinate at index 6.
    if(n==24){Eigen::MatrixXd compact(23,23);for(int r=0;r<23;r++)for(int c=0;c<23;c++)compact(r,c)=m(r+(r>=6),c+(c>=6));return compact;}
    return m;
}
