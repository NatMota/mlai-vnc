// Interactive sensor-policy stepping; existing collectors/trainers are unchanged.
#define main original_parking_main
#include "parking_ppo.cpp"
#undef main
#include <deque>
int main(int argc,char**argv){try{
    if(argc!=2)throw std::runtime_error("Usage: sensor_parking ASSETS");
    std::string assets=argv[1];Physics physics(assets);std::map<std::string,Reference> refs;
    std::unique_ptr<Expert> expert;State s;V3 target;V7 hold;Reference* ref=nullptr;
    Eigen::Vector4d applied=Eigen::Vector4d::Zero();
    int step=0,decisions=0,park_step=-1,stable_steps=0,stop_step=-1;double limit=60;bool done=false,arm_mode=false,expert_drive=false;int expert_arm_hz=0,arm_updates=0;bool arm_live=false,arm_from_reset=false;
    V7 arm_tau=V7::Zero(),last_arm=V7::Zero();
    double damping=0,elasticity=0,inertia=0,motor_sd=0;int motor_delay=0;V17 extra=V17::Zero(),sent=V17::Zero();
    std::mt19937_64 motor_rng;std::normal_distribution<double> normal(0,1);std::deque<State> motor_history;
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
          {"motor_extra_torque",arr(extra.head<11>())},{"sent_arm_torque",arr(sent.segment<7>(4))},{"applied_arm_torque",arr(last_arm)},{"arm_from_reset",arm_from_reset},{"ee_distance_m",(target-a.ee).norm()},{"arm_stopped",stop_step>=0},{"park_step",park_step}};
    };
    std::string line;while(std::getline(std::cin,line)){try{
        auto r=json::parse(line);std::string op=r.at("op");
        if(op=="reset"){
            auto t=r.at("target");target=V3(t[0],t[1],t[2]);std::string key=t.dump();
            if(!refs.count(key))refs.emplace(key,reference(physics,assets,target));ref=&refs.at(key);
            damping=r.value("damping",0.);elasticity=r.value("elasticity",0.);inertia=r.value("inertia",0.);
            if(!std::isfinite(damping+elasticity+inertia)||damping<0||elasticity<0||inertia<0)throw std::runtime_error("Invalid motor effects");extra.setZero();
            motor_sd=r.value("motor_measurement_sd",0.);motor_delay=r.value("delay_ms",0);
            if(!std::isfinite(motor_sd)||motor_sd<0||motor_delay<0)throw std::runtime_error("Invalid motor sensing");
            motor_rng.seed(r.value("sensor_seed",0ULL));normal.reset();motor_history.clear();sent.setZero();
            limit=r.value("max_duration_s",60.);if(limit<=0||limit>600)throw std::runtime_error("Invalid duration");
            if(r.contains("start_pose")){auto pose=r.at("start_pose");physics.reset_pose(pose[0],pose[1],pose[2]);}else physics.reset();
            s=physics.state();expert=std::make_unique<Expert>(assets,physics,s,target);
            hold=s.q.segment<7>(4);step=decisions=stable_steps=0;park_step=stop_step=-1;done=false;applied.setZero();
            // Arm mode: after PARK the client's arm torques drive the arm (no hold controller) and "stop" ends the run
            // 0.5 s later. expert_drive: the expert drives and parks itself, so the arm can be tested on its own.
            arm_mode=r.value("arm",false);expert_drive=r.value("expert_drive",false);arm_tau.setZero();last_arm.setZero();arm_from_reset=r.value("arm_from_reset",false);if(arm_from_reset&&(!arm_mode||expert_drive))throw std::runtime_error("Full episode requires client arm mode");
            // expert_arm_hz (smoke test): after PARK the expert's own arm controller runs closed loop, but its torque is
            // only sampled at this rate and held in between, like a network deciding at that rate.
            expert_arm_hz=r.value("expert_arm_hz",0);arm_updates=0;arm_live=false;
        }else if(op=="step"){
            if(!expert||done)throw std::runtime_error("Reset before stepping a finished episode");
            if(arm_from_reset&&(!r.contains("arm")||r.value("ms",1)!=1))throw std::runtime_error("Full episode needs arm torque each 1 ms tick");
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
                if(arm_mode&&(arm_from_reset||park_step>=0)&&(arm_live||expert_arm_hz>0))a.tau=arm_tau;
                else a.tau=(100*(hold-s.q.segment<7>(4))-30*s.qd.segment<7>(4)+a.gravity).cwiseMax(-30).cwiseMin(30);
                extra.setZero();
                if(damping!=0||elasticity!=0){
                    // Match the course motor wrapper: measured q/qd, actual mass matrix.
                    // This is separate from the policy's quantised encoder interface.
                    motor_history.push_back(s);while(int(motor_history.size())>motor_delay+1)motor_history.pop_front();
                    State measured=motor_history.front();
                    if(motor_sd)for(int j=0;j<17;j++){measured.q[j]+=motor_sd*normal(motor_rng);measured.qd[j]+=motor_sd*normal(motor_rng);}
                    extra=physics.mass(s).bottomRightCorner(17,17)*(-damping*measured.qd-elasticity*measured.q);
                }
                if(inertia!=0)extra-=inertia*s.qdd;
                if(!extra.allFinite())throw std::runtime_error("Nonfinite motor correction");
                last_arm=a.tau;sent=physics.step(s,a,true,&extra);step++;s=physics.state();
                stable_steps=(s.lin.head<2>().norm()<=.02&&std::abs(s.ang.z())<=2*M_PI/180)?stable_steps+1:0;
                if(arm_mode)done=stop_step>=0?step>=stop_step+500:step>=std::lround(limit*1000);
                else done=park_step>=0?step>=park_step+500:step>=std::lround(limit*1000);
            }
        }else throw std::runtime_error("Unknown operation");
        std::cout<<packet().dump()<<std::endl;
    }catch(const std::exception&e){std::cout<<json({{"type","error"},{"error",e.what()}}).dump()<<std::endl;}}
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}return 0;}
