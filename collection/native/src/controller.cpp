#include "expert.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/math/rpy.hpp>
#include <Eigen/SVD>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include "numerics.hpp"
static double clip(double x,double lo,double hi){return std::clamp(x,lo,hi);}
static double wrap(double a){double x=std::fmod(a+M_PI,2*M_PI);return (x<0?x+2*M_PI:x)-M_PI;}
struct Expert::Impl {
    pinocchio::Model model;
    std::unique_ptr<pinocchio::Data> data;
    std::array<int,17> idx;
    pinocchio::FrameIndex ee,shoulder;
    V7 hold,desired,lower,upper;
    int phase=0;bool done=false;
    double best=std::numeric_limits<double>::infinity(),best_time=0;
};
Expert::Expert(const std::string& assets,const Physics& physics,const State& s,V3 tgt):p(new Impl),target(tgt){
    pinocchio::urdf::buildModel(assets+"/summit_xl_arm_description/summit_xl_complete_pin.urdf",pinocchio::JointModelFreeFlyer(),p->model);
    p->data=std::make_unique<pinocchio::Data>(p->model);
    for(int i=0;i<17;i++) p->idx[i]=p->model.joints[p->model.getJointId(physics.names[i])].idx_q();
    p->ee=p->model.getFrameId("robot_j2s7s300_end_effector");p->shoulder=p->model.getFrameId("robot_j2s7s300_link_2");
    p->hold=s.q.segment<7>(4);p->desired=p->hold;p->lower=physics.lower;p->upper=physics.upper;
    double z=target.z();standoff=z<=.1?.8:z<.5?((.65-.8)/(.5-.1))*(z-.1)+.8:z<=1.1?.65:z<1.25?((.35-.65)/(1.25-1.1))*(z-1.1)+.65:.35;
    bearing=numpy_norm(target.head<2>())>1e-4?std::atan2(target.y(),target.x()):0;
    goal=target.head<2>()-standoff*Eigen::Vector2d(std::cos(bearing),std::sin(bearing));
}
Expert::~Expert()=default;
Action Expert::act(const State& s,double t){
    Action a;auto& m=p->model;auto& d=*p->data;
    a.qpin=Eigen::VectorXd::Zero(m.nq);a.qpin.head<3>()=s.pos;a.qpin.segment<4>(3)=s.quat;
    for(int i=0;i<17;i++) a.qpin[p->idx[i]]=s.q[i];
    pinocchio::forwardKinematics(m,d,a.qpin);pinocchio::updateFramePlacements(m,d);
    a.ee=d.oMf[p->ee].translation();a.shoulder=d.oMf[p->shoulder].translation();a.rotation=d.oMf[p->ee].rotation();
    a.error=target-a.ee;a.distance=numpy_norm(a.error);
    Eigen::Quaterniond quat(s.quat.w(),s.quat.x(),s.quat.y(),s.quat.z());quat.normalize();
    a.yaw=pinocchio::rpy::matrixToRpy(quat.toRotationMatrix())[2];
    if(!p->done){
        auto delta=(goal-s.pos.head<2>()).eval();a.goal_dist=numpy_norm(delta);a.heading=std::atan2(delta.y(),delta.x());a.v=0;a.w=0;
        if(p->phase==0){a.err=wrap(a.heading-a.yaw);if(a.goal_dist<.08||std::abs(a.err)<.05)p->phase=a.goal_dist>=.08?1:2;else a.w=clip(3*a.err,-1,1);}
        if(p->phase==1){a.err=wrap(a.heading-a.yaw);bool passed=a.goal_dist<.15&&std::cos(a.err)<0;
            if(a.goal_dist<.08||passed)p->phase=2;
            else{a.v=clip(a.goal_dist,.05,.25)*std::max(0.,std::cos(a.err));a.w=clip(1.5*a.err,-.3,.3);}}
        if(p->phase==2){a.err=wrap(bearing-a.yaw);if(std::abs(a.err)<.05)p->done=true;else a.w=clip(3*a.err,-1,1);}
        a.lw=(a.v-a.w*.66/2)/.11;a.rw=(a.v+a.w*.66/2)/.11;a.wheel<<a.rw,a.lw,a.lw,a.rw;
        if(t>25)p->done=true;
    }
    a.phase=p->phase;a.done=p->done;
    V7 q=s.q.segment<7>(4),qd=s.qd.segment<7>(4);
    if(!p->done)p->desired=p->hold;
    else{
        Eigen::Matrix<double,6,Eigen::Dynamic> J(6,m.nv);J.setZero();
        // computeFrameJacobian already computes the joint Jacobians it needs.
        pinocchio::computeFrameJacobian(m,d,a.qpin,p->ee,pinocchio::LOCAL_WORLD_ALIGNED,J);
        a.jac=J.block<3,7>(0,6);
        a.sigma=numpy_sigma_min(a.jac);
        a.damping=a.sigma<=0?.10:a.sigma>=.15?.01:((.01-.10)/.15)*a.sigma+.10;
        V3 offset=target-a.shoulder;double offset_norm=numpy_norm(offset);if(offset_norm>.85)offset*=.85/offset_norm;
        a.ik=a.shoulder+offset-a.ee;a.vc=1.5*a.ik;double cartesian_speed=numpy_norm(a.vc);if(cartesian_speed>.25)a.vc*=.25/cartesian_speed;
        std::array<bool,7> free;free.fill(true);
        for(int it=0;it<7;it++){
            a.qddes=numpy_ik(a.jac,free,a.damping,a.vc);
            bool blocked=false;
            for(int j=0;j<7;j++)if(free[j]&&p->lower[j]<p->upper[j]&&((q[j]<=p->lower[j]+.05&&a.qddes[j]<0)||(q[j]>=p->upper[j]-.05&&a.qddes[j]>0))){free[j]=false;blocked=true;}
            if(!blocked)break;a.qddes.setZero();
        }
        double speed=a.qddes.cwiseAbs().maxCoeff();if(speed>1)a.qddes*=1./speed;
        p->desired+=a.qddes*dt;
        for(int j=0;j<7;j++){p->desired[j]=clip(p->desired[j],q[j]-.15,q[j]+.15);if(p->lower[j]<p->upper[j])p->desired[j]=clip(p->desired[j],p->lower[j]+.05,p->upper[j]-.05);}
    }
    a.qdes=p->desired;a.gravity=pinocchio::computeGeneralizedGravity(m,d,a.qpin).segment<7>(6);
    a.pd=100*(a.qdes-q)+30*(a.qddes-qd);a.tau=(a.pd+a.gravity).cwiseMax(-30).cwiseMin(30);
    return a;
}
std::string Expert::stop(const Action& a,double t,int cap){
    if(cap&&t/dt>=cap)return "step_cap";
    if(a.done){if(a.distance<p->best-.002){p->best=a.distance;p->best_time=t;}else if(t-p->best_time>3&&t>8)return "stalled";}
    if(a.done&&a.distance<.01&&t>8)return "reached";
    return "";
}
