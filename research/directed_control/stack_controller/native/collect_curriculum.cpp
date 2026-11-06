// New collection only. Reuse the verified exporter and motor-effect equations.
#define main original_collector_main
#include "main.cpp"
#undef main

int main(int argc,char**argv){try{
    if(argc!=4)throw std::runtime_error("Usage: collect_curriculum ASSETS PLAN.json OUTPUT");
    std::string assets=fs::absolute(argv[1]).string();fs::path output=argv[3];
    if(fs::exists(output))throw std::runtime_error("Output already exists");
    json plan;std::ifstream input(argv[2]);input>>plan;binary_output=true;
    if(plan.contains("effects_file"))load_effects(plan.at("effects_file"));
    Physics physics(assets);auto started=Clock::now();
    for(const auto& run:plan.at("runs")){
        int id=run.at("id"),cap=run.value("max_steps",40000);auto tp=run.at("target");V3 target(tp[0],tp[1],tp[2]);
        auto pose=run.value("start_pose",json::array({0.,0.,0.}));
        physics.reset_pose(pose[0],pose[1],pose[2]);State s=physics.state();Expert expert(assets,physics,s,target);
        unsigned long long seed=run.value("seed",0ULL);std::mt19937_64 rng(seed);std::normal_distribution<double> normal(0,1);
        double wheel_sd=run.value("wheel_noise_sd",0.),arm_sd=run.value("arm_noise_sd",0.);
        int noise_hold=run.value("noise_hold_steps",33);
        if(wheel_sd<0||arm_sd<0||noise_hold<1||cap<1)throw std::runtime_error("Invalid collection setting");
        Eigen::Vector4d nw=Eigen::Vector4d::Zero();V7 na=V7::Zero();
        Measure measure(seed);std::vector<Row> rows;rows.reserve(12000);std::vector<double> applied_commands;int park=-1,step=0;Action label;std::string reason;
        auto begin=Clock::now();double t=0;
        while(true){
            State measured=measure(s);
            label=expert.act(s,t);if(label.done&&park<0)park=step;
            if(step%noise_hold==0){for(int j=0;j<4;j++)nw[j]=wheel_sd*normal(rng);for(int j=0;j<7;j++)na[j]=arm_sd*normal(rng);}
            Action applied=label;
            if(wheel_sd>0)applied.wheel=(label.wheel+nw).cwiseMax(-3).cwiseMin(3);
            if(label.done)applied.wheel.setZero();
            applied.tau=(label.tau+na).cwiseMax(-30).cwiseMin(30);
            for(int j=0;j<4;j++)applied_commands.push_back(applied.wheel[j]);
            for(int j=0;j<7;j++)applied_commands.push_back(applied.tau[j]);
            V17 extra=motor_torques(physics,s,measured);auto before=Clock::now();V17 sent=physics.step(s,applied,true,&extra);
            rows.push_back({s,label,sent,t,elapsed(before),elapsed(begin),{},measured});
            reason=label.done&&(label.phase!=2||std::abs(label.err)>=.05)?"navigation_timeout":expert.stop(label,t,cap);step++;
            if(!reason.empty())break;
            t+=dt;s=physics.state();if(!s.q.allFinite()||!s.qd.allFinite())throw std::runtime_error("Nonfinite collection state");
        }
        json manifest={{"episode",id},{"steps",step},{"dt_s",dt},{"parking_step",park},{"termination",reason},
            {"target_position_world_m",tp},{"start_pose",pose},{"joint_names",physics.names},{"final_pre_step_ee_error_m",label.distance},
            {"effects",effects.raw},{"seed",seed},{"action_noise",{{"wheel_sd_rad_s",wheel_sd},{"arm_sd_Nm",arm_sd},{"hold_steps",noise_hold}}},
            {"label_source","clean stateful expert at the actual perturbed pre-step state"},
            {"sent_torque_source","actual noisy actuator command plus motor effects"},{"sample_alignment","pre-step state and corresponding clean label"},
            {"collector_source_sha256",COLLECTOR_SHA},{"engine_sha256",ENGINE_SHA},{"compiler",COMPILER_VERSION}};
        write_run(output/("run_"+std::to_string(id)),rows,expert,manifest);
        npy(output/("run_"+std::to_string(id))/"applied_command.npy",applied_commands,{rows.size(),11});
        std::cout<<"RESULT "<<manifest.dump()<<std::endl;
    }
    std::cout<<"COMPLETE "<<elapsed(started)<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
