#include "expert.hpp"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
using json=nlohmann::json;
namespace fs=std::filesystem;
template<class T> json values(const T& v){json a=json::array();for(int i=0;i<v.size();i++)a.push_back(v[i]);return a;}
struct Policy {
    json weights;
    // Input set: "start_dot" (original 14 inputs), "wheels" (12) or "live_dot" (15:
    // distance and sin/cos of the dot direction relative to the robot, from the simulator pose).
    std::string inputs="start_dot";
    Eigen::VectorXf layer(const Eigen::VectorXf& x,const std::string& name){
        auto& w=weights.at(name+".weight");auto& b=weights.at(name+".bias");
        if(w.at(0).size()!=size_t(x.size()))throw std::runtime_error("Weights expect "+std::to_string(w.at(0).size())+" inputs, got "+std::to_string(x.size()));
        Eigen::MatrixXf m(w.size(),x.size());Eigen::VectorXf bias(w.size());
        for(int i=0;i<m.rows();i++){bias[i]=b.at(i).get<float>();for(int j=0;j<m.cols();j++)m(i,j)=w.at(i).at(j).get<float>();}
        return m*x+bias;
    }
    std::pair<Eigen::Vector4d,double> predict(const State& s,const Eigen::Vector4d& initial,double angle,const V3& target,double yaw){
        int n=inputs=="wheels"?12:inputs=="live_dot"?15:inputs=="start_dot"?14:-1;
        if(n<0)throw std::runtime_error("Unknown inputs: "+inputs);
        Eigen::VectorXd f(n);
        f.head<4>()=s.q.head<4>()-initial;f.segment<4>(4)=s.qd.head<4>();
        double r=(f[0]+f[3])/2,l=(f[1]+f[2])/2,vr=(f[4]+f[7])/2,vl=(f[5]+f[6])/2;
        f[8]=(r+l)/2;f[9]=r-l;f[10]=(vr+vl)/2;f[11]=vr-vl;
        if(inputs=="start_dot"){f[12]=std::sin(angle);f[13]=std::cos(angle);}
        if(inputs=="live_dot"){
            double dx=target.x()-s.pos.x(),dy=target.y()-s.pos.y(),a=std::atan2(dy,dx)-yaw;
            f[12]=std::hypot(dx,dy);f[13]=std::sin(a);f[14]=std::cos(a);
        }
        auto h=layer(layer(f.cast<float>(),"shared.0").cwiseMax(0),"shared.2").cwiseMax(0).eval();
        auto wheel=layer(h,"wheels");float logit=layer(h,"park")[0];
        return {wheel.cast<double>(),1./(1.+std::exp(-double(logit)))};
    }
};
// Steps the expert keeps running after its PARK command (500 = 0.5 s). Set per request with "expert_tail_steps".
int expert_tail_steps=500;
json rollout(Physics& physics,const std::string& assets,V3 target,double angle,Policy* policy,int comparison_end_step=-1,bool record=true){
    State s=physics.state();Expert expert(assets,physics,s,target);
    Eigen::Vector4d initial=s.q.head<4>(),wheel=Eigen::Vector4d::Zero();V7 hold=s.q.segment<7>(4);
    int park_step=-1,decision=0,frame=0,clipped=0;double probability=0,t=0;json rows=json::array(),park=nullptr;
    // The expert terminates on its own PARK event, never on a fixed duration.
    // The model gets three additional seconds after the expert PARK event.
    for(int step=0;;step++,t+=dt){
        if(step)s=physics.state();
        Action a=expert.act(s,t);
        bool update=step==std::lround(decision*1000./30);
        if(policy){
            if(update&&park_step<0){
                auto prediction=policy->predict(s,initial,angle,target,a.yaw);wheel=prediction.first;probability=prediction.second;
                if(!wheel.allFinite()||!std::isfinite(probability))throw std::runtime_error("Non-finite model output");
                if(wheel.cwiseAbs().maxCoeff()>3)clipped++;
                wheel=wheel.cwiseMax(-3).cwiseMin(3);
                if(probability>=.5)park_step=step;
            }
            if(park_step>=0)wheel.setZero();a.wheel=wheel;
            // The wheel/park heads are learned. The unchanged arm holding PD +
            // gravity controller remains supplied, with no arm-reaching phase.
            a.tau=(100*(hold-s.q.segment<7>(4))-30*s.qd.segment<7>(4)+a.gravity).cwiseMax(-30).cwiseMin(30);
        }else{
            probability=a.done?1:0;
            if(a.done&&park_step<0)park_step=step;
            if(park_step>=0){
                a.wheel.setZero();
                a.tau=(100*(hold-s.q.segment<7>(4))-30*s.qd.segment<7>(4)+a.gravity).cwiseMax(-30).cwiseMin(30);
            }
        }
        if(update)decision++;
        auto state_json=[&](){return json{{"step",step},{"time_s",t},{"decision",policy?update:true},{"base",values(s.pos)},{"quaternion",values(s.quat)},
            {"joints",values(s.q)},{"wheel_velocity",values(s.qd.head<4>())},{"command",values(a.wheel)},
            {"park_probability",probability},{"parked",park_step>=0},{"yaw",a.yaw},
            {"speed_m_s",s.lin.head<2>().norm()},{"ee",values(a.ee)}};};
        if(step==park_step)park=state_json();
        bool end=park_step>=0 ? step>=park_step+(policy?500:expert_tail_steps) : policy&&step>=comparison_end_step;
        if(record&&step==std::lround(frame*1000./30)){rows.push_back(state_json());frame++;}
        else if((record&&step==park_step)||end)rows.push_back(state_json());
        if(end)break;
        physics.step(s,a);
    }
    return {{"frames",rows},{"park",park},{"final",rows.back()},{"termination",park_step<0?"no_park_within_grace_period":"parked"},
        {"clipped_decisions",clipped},{"decision_hz",policy?30:1000},{"physics_hz",1000},{"settling_tail_s",.5}};
}
// Persistent, process-isolated evaluator for terminal-pose policy search.
// It accepts no expert wheel labels. The expert supplies only the parking pose
// and its PARK time; policy inputs remain wheel observations and initial bearing.
void serve(const std::string& assets){
    Physics physics(assets);std::map<std::string,json> references;
    std::cout<<json({{"type","ready"},{"engine_sha256",ENGINE_SHA}}).dump()<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)){
        try{
            json request=json::parse(line);Policy policy{request.at("weights"),request.value("inputs",std::string("start_dot"))};
            double ps=request.at("position_scale_m"),hs=request.at("heading_scale_deg"),miss=request.at("missing_park_cost");
            if(!(ps>0&&hs>0&&miss>=0))throw std::runtime_error("Invalid pose-loss scales");
            bool record=request.value("record",false);json results=json::array();double total=0;
            for(const auto& c:request.at("cases")){
                V3 target(c["target"][0],c["target"][1],c["target"][2]);
                double angle=c["bearing_deg"].get<double>()*std::acos(-1.)/180;
                std::string key=c["target"].dump();
                if(!references.count(key)){
                    physics.reset();references[key]=rollout(physics,assets,target,angle,nullptr);
                }
                const json& expert=references.at(key);
                physics.reset();json model=rollout(physics,assets,target,angle,&policy,expert["park"]["step"].get<int>()+3000,record);
                const json& m=model["final"];const json& e=expert["final"];
                double dx=(m["base"][0].get<double>()-target.x())-(e["base"][0].get<double>()-target.x());
                double dy=(m["base"][1].get<double>()-target.y())-(e["base"][1].get<double>()-target.y());
                double delta=m["yaw"].get<double>()-e["yaw"].get<double>();
                double heading=std::atan2(std::sin(delta),std::cos(delta)),position2=dx*dx+dy*dy;
                double pl=position2/(ps*ps),hl=std::pow(heading/(hs*std::acos(-1.)/180),2);
                double ml=model["park"].is_null()?miss:0,loss=pl+hl+ml;
                if(!std::isfinite(loss))throw std::runtime_error("Non-finite terminal-pose loss");
                json result={{"bearing_deg",c["bearing_deg"]},{"split",c["split"]},{"target",c["target"]},
                    {"total",loss},{"position_loss",pl},{"heading_loss",hl},{"missing_park_loss",ml},
                    {"position_error_m",std::sqrt(position2)},{"heading_error_deg",std::abs(heading)*180/std::acos(-1.)},
                    {"parked",!model["park"].is_null()},{"model_final",model["final"]},{"expert_final",expert["final"]},
                    {"expert_park",expert["park"]},{"model_park",model["park"]}};
                if(record){result["model"]=std::move(model);result["expert"]=expert;}
                results.push_back(std::move(result));total+=loss;
            }
            if(results.empty())throw std::runtime_error("No evaluation cases");
            std::cout<<json({{"type","pose_result"},{"mean_loss",total/results.size()},
                {"cases",results},{"joint_names",physics.names},{"engine_sha256",ENGINE_SHA}}).dump()<<std::endl;
        }catch(const std::exception& e){std::cout<<json({{"type","pose_error"},{"error",e.what()}}).dump()<<std::endl;}
    }
}
int main(int argc,char**argv){try{
    if(argc==3&&std::string(argv[2])=="--serve"){serve(argv[1]);return 0;}
    if(argc!=4)throw std::runtime_error("Usage: policy_replay ASSETS REQUEST.json NEW_OUTPUT.json");
    if(fs::exists(argv[3]))throw std::runtime_error("Refusing to overwrite replay output");
    json request;std::ifstream(argv[2])>>request;Policy policy{request.at("weights"),request.value("inputs",std::string("start_dot"))};expert_tail_steps=request.value("expert_tail_steps",500);
    Physics physics(argv[1]);json runs=json::array();bool first=true;
    for(auto& c:request.at("cases")){
        V3 target(c["target"][0],c["target"][1],c["target"][2]);double angle=c["bearing_deg"].get<double>()*std::acos(-1.)/180;
        if(!first)physics.reset();first=false;
        auto reference=rollout(physics,argv[1],target,angle,nullptr);
        physics.reset();auto model=rollout(physics,argv[1],target,angle,&policy,reference["park"]["step"].get<int>()+3000);
        runs.push_back({{"bearing_deg",c["bearing_deg"]},{"split",c["split"]},{"target",c["target"]},{"expert",reference},{"model",model}});
        std::cout<<c["split"]<<" "<<c["bearing_deg"]<<" model "<<model["termination"]<<std::endl;
    }
    json result={{"runs",runs},{"joint_names",physics.names},{"engine_sha256",ENGINE_SHA},{"noise_enabled",false},{"termination_rule","expert_park_plus_3s"},
      {"recording_description","The expert runs until its PARK command, plus 0.5 s to show drift. The model gets 3 additional seconds after the expert PARK command to issue its own PARK. If it parks, 0.5 s of drift is recorded. Otherwise it is labelled no PARK within the grace period; the recording ending is not a stop command. Each robot freezes at its last recorded state."},
      {"policy_inputs","Wheel positions/velocities and requested initial bearing only"},{"arm","Original hold controller throughout; no learned reaching"}};
    std::ofstream out(argv[3]);out.exceptions(std::ios::failbit|std::ios::badbit);out<<result.dump();
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
