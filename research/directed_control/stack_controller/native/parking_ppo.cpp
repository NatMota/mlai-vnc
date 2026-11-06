// Action-space rollouts for PPO. No parameter perturbations or command labels.
#include "expert.hpp"
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <iostream>
#include <random>
#include <map>
#include <cmath>
using json=nlohmann::json;
template<class T> json arr(const T& v){json a=json::array();for(int i=0;i<v.size();i++)a.push_back(v[i]);return a;}
double yaw(const State& s){return std::atan2(2*(s.quat.w()*s.quat.z()+s.quat.x()*s.quat.y()),1-2*(s.quat.y()*s.quat.y()+s.quat.z()*s.quat.z()));}
double wrapped(double a){return std::atan2(std::sin(a),std::cos(a));}
struct Net {
    std::vector<Eigen::MatrixXf> w;std::vector<Eigen::VectorXf> b;
    explicit Net(const json& weights){for(auto name:{"shared.0","shared.2","wheels","park"}){
        const auto& ww=weights.at(std::string(name)+".weight");const auto& bb=weights.at(std::string(name)+".bias");
        Eigen::MatrixXf m(ww.size(),ww[0].size());Eigen::VectorXf v(bb.size());
        for(int i=0;i<m.rows();i++){v[i]=bb[i];for(int j=0;j<m.cols();j++)m(i,j)=ww[i][j];}w.push_back(m);b.push_back(v);
    }}
    Eigen::VectorXf features(const State& s,const Eigen::Vector4d& initial,double angle){
        Eigen::VectorXf f(14);f.head<4>()=(s.q.head<4>()-initial).cast<float>();f.segment<4>(4)=s.qd.head<4>().cast<float>();
        float r=(f[0]+f[3])/2,l=(f[1]+f[2])/2,vr=(f[4]+f[7])/2,vl=(f[5]+f[6])/2;
        f[8]=(r+l)/2;f[9]=r-l;f[10]=(vr+vl)/2;f[11]=vr-vl;f[12]=std::sin(angle);f[13]=std::cos(angle);return f;
    }
    std::pair<Eigen::Vector4d,double> predict(const Eigen::VectorXf& f){
        Eigen::VectorXf h=(w[0]*f+b[0]).cwiseMax(0);h=(w[1]*h+b[1]).cwiseMax(0);
        return {(w[2]*h+b[2]).cast<double>(),(w[3]*h+b[3])[0]};
    }
};
struct Metric {double distance,heading,range_error,dl,hl;};
Metric metric(const State& s,const V3& target,double reference){
    // URDF robot_base_docking_joint: existing front contact reference.
    Eigen::Quaterniond q(s.quat.w(),s.quat.x(),s.quat.y(),s.quat.z());q.normalize();
    V3 prow=s.pos+q*V3(.284,.012,.178); // base_footprint -> base_link adds 0.11 m
    double d=(target-prow).head<2>().norm();
    double bearing=std::atan2(target.y()-s.pos.y(),target.x()-s.pos.x());
    double heading=std::abs(wrapped(bearing-yaw(s)))*180/M_PI;
    double error=std::abs(d-reference);
    double dl=std::pow(std::max(0.,error-.05)/.10,2);
    double hl=10*std::pow(std::max(0.,heading-20.)/160.,2);
    return {d,heading,error,dl,hl};
}
json frame(const State& s,const Action& a,int step,bool decision,bool parked,double prob){
    return {{"step",step},{"time_s",step*dt},{"decision",decision},{"base",arr(s.pos)},{"quaternion",arr(s.quat)},
      {"joints",arr(s.q)},{"wheel_velocity",arr(s.qd.head<4>())},{"command",arr(a.wheel)},
      {"park_probability",prob},{"parked",parked},{"yaw",yaw(s)},{"speed_m_s",s.lin.head<2>().norm()},
      {"yaw_rate_rad_s",s.ang.z()},{"ee",arr(a.ee)}};
}
struct Reference {json run;double distance;int park_step;};
Reference reference(Physics& physics,const std::string& assets,const V3& target){
    physics.reset();State s=physics.state();Expert expert(assets,physics,s,target);V7 hold=s.q.segment<7>(4);
    json frames=json::array(),park=nullptr;int parked=-1,fi=0;
    for(int step=0;;step++){
        if(step)s=physics.state();Action a=expert.act(s,step*dt);
        if(a.done&&parked<0)parked=step;
        if(parked>=0){a.wheel.setZero();a.tau=(100*(hold-s.q.segment<7>(4))-30*s.qd.segment<7>(4)+a.gravity).cwiseMax(-30).cwiseMin(30);}
        auto f=frame(s,a,step,true,parked>=0,parked>=0?1:0);
        if(step==parked)park=f;
        bool end=parked>=0&&step>=parked+500;
        if(step==std::lround(fi*1000./30)){frames.push_back(f);fi++;}else if(step==parked||end)frames.push_back(f);
        if(end){return {{{"frames",frames},{"park",park},{"final",f},{"termination","parked"},{"clipped_decisions",0}},metric(s,target,0).distance,parked};}
        physics.step(s,a);
    }
}
json episode(Physics& physics,const std::string& assets,const json& c,Net& net,const Reference& ref,const json& req){
    V3 target(c["target"][0],c["target"][1],c["target"][2]);double angle=c["bearing_deg"].get<double>()*M_PI/180;
    bool train=req.value("train",false),record=req.value("record",false);
    std::mt19937 rng(c.value("seed",7));std::normal_distribution<double> normal(0,1);std::uniform_real_distribution<double> uniform(0,1);
    Eigen::Vector4d sd=Eigen::Vector4d::Ones();if(train)for(int i=0;i<4;i++)sd[i]=std::exp(req.at("log_std")[i].get<double>());
    physics.reset();State s=physics.state();Expert expert(assets,physics,s,target);
    Eigen::Vector4d initial=s.q.head<4>(),wheel=Eigen::Vector4d::Zero();V7 hold=s.q.segment<7>(4);
    int park_step=-1,decision=0,fi=0,clipped=0,stable_steps=0;double probability=0;
    json rows=json::array(),frames=json::array(),park=nullptr,final;
    double limit=req.value("max_duration_s",-1.);
    if(limit!=-1.&&(!std::isfinite(limit)||limit<=0||limit>600))throw std::runtime_error("Evaluation duration must be in (0,600] seconds");
    const int deadline=limit>0?std::lround(limit*1000):ref.park_step+3000;
    for(int step=0;;step++){
        if(step)s=physics.state();Action a=expert.act(s,step*dt);
        bool update=step==std::lround(decision*1000./30);
        if(update&&park_step<0&&step<deadline){
            Eigen::VectorXf obs=net.features(s,initial,angle);auto [mean,logit]=net.predict(obs);
            probability=1/(1+std::exp(-std::clamp(logit,-80.,80.)));
            bool stop=train?uniform(rng)<probability:probability>=.5;
            Eigen::Vector4d raw=mean;if(train)for(int i=0;i<4;i++)raw[i]+=sd[i]*normal(rng);
            if(!raw.allFinite()||!std::isfinite(probability))throw std::runtime_error("Nonfinite policy");
            wheel=raw.cwiseMax(-3).cwiseMin(3);if(raw.cwiseAbs().maxCoeff()>3)clipped++;
            if(stop)park_step=step;
            if(train){auto m=metric(s,target,ref.distance);rows.push_back({{"obs",arr(obs)},{"action",arr(raw)},{"stop",stop},
                {"mean",arr(mean)},{"logit",logit},{"potential",m.dl+m.hl},{"time_s",step*dt}});}
        }
        if(update)decision++;
        if(park_step>=0)wheel.setZero();a.wheel=wheel;
        a.tau=(100*(hold-s.q.segment<7>(4))-30*s.qd.segment<7>(4)+a.gravity).cwiseMax(-30).cwiseMin(30);
        bool stable=s.lin.head<2>().norm()<=.02&&std::abs(s.ang.z())<=2*M_PI/180;
        stable_steps=stable?stable_steps+1:0;
        bool end=park_step>=0?step>=park_step+500:step>=deadline;
        if(record||end||step==park_step){final=frame(s,a,step,update,park_step>=0,probability);
            if(step==park_step)park=final;
            if(record&&step==std::lround(fi*1000./30)){frames.push_back(final);fi++;}
            else if(record&&(step==park_step||end))frames.push_back(final);
        }
        if(end)break;
        physics.step(s,a);
    }
    auto m=metric(s,target,ref.distance);bool stopped=park_step>=0&&stable_steps>=200;
    double stop_loss=stopped?0:10;
    json result={{"bearing_deg",c["bearing_deg"]},{"split",c["split"]},{"target",c["target"]},
      {"total",m.dl+m.hl+stop_loss},{"distance_loss",m.dl},{"heading_loss",m.hl},{"stop_loss",stop_loss},
      {"prow_distance_m",m.distance},{"expert_prow_distance_m",ref.distance},{"distance_error_m",m.range_error},
      {"heading_error_deg",m.heading},{"parked",park_step>=0},{"physically_stopped",stopped},
      {"suitable",stopped&&m.range_error<=.05&&m.heading<=20},
      {"model_final",final},{"expert_final",ref.run["final"]},{"model_park",park},{"expert_park",ref.run["park"]},
      {"clipped_decisions",clipped},{"decisions",decision}};
    if(train)result["trajectory"]=std::move(rows);
    if(record){result["expert"]=ref.run;result["model"]={{"frames",frames},{"park",park},{"final",final},
        {"termination",park_step<0?(limit>0?"no_park_within_evaluation_limit":"no_park_within_grace_period"):"parked"},{"clipped_decisions",clipped}};}
    return result;
}
int main(int argc,char**argv){try{
    if(argc!=2)throw std::runtime_error("Usage: parking_ppo ASSETS");std::string assets=argv[1];
    Physics physics(assets);std::map<std::string,Reference> refs;
    std::cout<<json({{"type","ready"},{"engine_sha256",ENGINE_SHA}}).dump()<<std::endl;
    std::string line;while(std::getline(std::cin,line)){try{
        json req=json::parse(line);Net net(req.at("weights"));json results=json::array();
        for(const auto& c:req.at("cases")){
            std::string key=c["target"].dump();V3 target(c["target"][0],c["target"][1],c["target"][2]);
            if(!refs.count(key))refs.emplace(key,reference(physics,assets,target));
            results.push_back(episode(physics,assets,c,net,refs.at(key),req));
        }
        std::cout<<json({{"type","result"},{"cases",results},{"joint_names",physics.names},{"engine_sha256",ENGINE_SHA}}).dump()<<std::endl;
    }catch(const std::exception& e){std::cout<<json({{"type","error"},{"error",e.what()}}).dump()<<std::endl;}}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
