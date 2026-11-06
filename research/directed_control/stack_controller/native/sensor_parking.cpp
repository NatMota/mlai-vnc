// Interactive sensor-policy stepping; existing collectors/trainers are unchanged.
#define main original_parking_main
#include "parking_ppo.cpp"
#undef main
int main(int argc,char**argv){try{
    if(argc!=2)throw std::runtime_error("Usage: sensor_parking ASSETS");
    std::string assets=argv[1];Physics physics(assets);std::map<std::string,Reference> refs;
    std::unique_ptr<Expert> expert;State s;V3 target;V7 hold;Reference* ref=nullptr;
    Eigen::Vector4d applied=Eigen::Vector4d::Zero();
    int step=0,decisions=0,park_step=-1,stable_steps=0,stop_step=-1;double limit=60;bool done=false,arm_mode=false,expert_drive=false;int expert_arm_hz=0,arm_updates=0;bool arm_live=false;
    V7 arm_tau=V7::Zero();
    std::cout<<json({{"type","ready"},{"engine_sha256",ENGINE_SHA},{"joint_names",physics.names}}).dump()<<std::endl;
    auto packet=[&](){
        auto a=expert->act(s,step*dt,true);auto m=metric(s,target,ref->distance);
        a.wheel=applied;bool stopped=park_step>=0&&stable_steps>=200;
        return json{{"type","state"},{"frame",frame(s,a,step,true,park_step>=0,0)},
          {"potential",m.dl+m.hl},{"total",m.dl+m.hl+(stopped?0:10)},
          {"distance_error_m",m.range_error},{"heading_error_deg",m.heading},
          {"parked",park_step>=0},{"physically_stopped",stopped},
          {"suitable",stopped&&m.range_error<=.05&&m.heading<=20},{"done",done},
          {"expert_prow_distance_m",ref->distance},{"expert_park_time_s",ref->park_step*dt},
          {"ee_distance_m",(target-a.ee).norm()},{"arm_stopped",stop_step>=0},{"park_step",park_step}};
    };
    std::string line;while(std::getline(std::cin,line)){try{
        auto r=json::parse(line);std::string op=r.at("op");
        if(op=="reset"){
            auto t=r.at("target");target=V3(t[0],t[1],t[2]);std::string key=t.dump();
            if(!refs.count(key))refs.emplace(key,reference(physics,assets,target));ref=&refs.at(key);
            limit=r.value("max_duration_s",60.);if(limit<=0||limit>600)throw std::runtime_error("Invalid duration");
            physics.reset();s=physics.state();expert=std::make_unique<Expert>(assets,physics,s,target);
            hold=s.q.segment<7>(4);step=decisions=stable_steps=0;park_step=stop_step=-1;done=false;applied.setZero();
            // Arm mode: after PARK the client's arm torques drive the arm (no hold controller) and "stop" ends the run
            // 0.5 s later. expert_drive: the expert drives and parks itself, so the arm can be tested on its own.
            arm_mode=r.value("arm",false);expert_drive=r.value("expert_drive",false);arm_tau.setZero();
            // expert_arm_hz (smoke test): after PARK the expert's own arm controller runs closed loop, but its torque is
            // only sampled at this rate and held in between, like a network deciding at that rate.
            expert_arm_hz=r.value("expert_arm_hz",0);arm_updates=0;arm_live=false;
        }else if(op=="step"){
            if(!expert||done)throw std::runtime_error("Reset before stepping a finished episode");
            Eigen::Vector4d wheels;for(int i=0;i<4;i++)wheels[i]=std::clamp(r.at("wheel")[i].get<double>(),-3.,3.);
            if(r.value("park",false)&&park_step<0&&!expert_drive)park_step=step;
            if(arm_mode&&r.contains("arm"))for(int j=0;j<7;j++)arm_tau[j]=std::clamp(r.at("arm")[j].get<double>(),-30.,30.);
            // Client torques take over from the first request sent after PARK (or at once in 1 ms replay); until then
            // the hold controller keeps the arm, so the gap between PARK and the next decision is not left unpowered.
            if(arm_mode&&r.contains("arm")&&(park_step>=0||r.contains("ms")))arm_live=true;
            if(arm_mode&&park_step>=0&&r.value("stop",false)&&stop_step<0)stop_step=step;
            // Default: run to the next 30 Hz decision. "ms" runs that many 1 ms physics steps instead (smoke tests).
            int next=r.contains("ms")?step+r.at("ms").get<int>():std::lround((++decisions)*1000./30);
            while(step<next&&!done){
                auto a=expert->act(s,step*dt,true);
                if(expert_drive&&park_step<0){auto e=expert->act(s,step*dt,false);if(e.done){park_step=step;if(expert_arm_hz>0){arm_tau=e.tau;arm_updates=1;}}else a.wheel=e.wheel;}
                else if(expert_arm_hz>0&&park_step>=0){
                    auto e=expert->act(s,step*dt,false);    // every 1 ms, so the expert's own integrators stay exact
                    if(step-park_step>=std::lround(arm_updates*1000./expert_arm_hz)){arm_tau=e.tau;arm_updates++;}
                }
                else a.wheel=park_step>=0?Eigen::Vector4d::Zero().eval():wheels;
                if(park_step>=0)a.wheel.setZero();
                applied=a.wheel;
                if(arm_mode&&park_step>=0&&(arm_live||expert_arm_hz>0))a.tau=arm_tau;
                else a.tau=(100*(hold-s.q.segment<7>(4))-30*s.qd.segment<7>(4)+a.gravity).cwiseMax(-30).cwiseMin(30);
                physics.step(s,a);step++;s=physics.state();
                stable_steps=(s.lin.head<2>().norm()<=.02&&std::abs(s.ang.z())<=2*M_PI/180)?stable_steps+1:0;
                if(arm_mode)done=stop_step>=0?step>=stop_step+500:step>=std::lround(limit*1000);
                else done=park_step>=0?step>=park_step+500:step>=std::lround(limit*1000);
            }
        }else throw std::runtime_error("Unknown operation");
        std::cout<<packet().dump()<<std::endl;
    }catch(const std::exception&e){std::cout<<json({{"type","error"},{"error",e.what()}}).dump()<<std::endl;}}
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}return 0;}
