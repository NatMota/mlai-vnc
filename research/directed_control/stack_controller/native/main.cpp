#include "expert.hpp"
#include <Eigen/Geometry>
#include "bullet_api.hpp"
#include <charconv>
#include <nlohmann/json.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <sys/wait.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <deque>
#include <random>
using json=nlohmann::json;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
struct Row { State s; Action a; V17 torque; double t,step_wall,wall; Eigen::MatrixXd mass; State m; };
// The course simulator's optional effects (RoboEnv robot_pybullet config), all off by default.
// Measurement noise and delay change what the expert measures (its q, qd, base position and orientation), as the
// course's GetMotorAngles/GetMotorVelocities/GetBasePosition/GetBaseOrientation do. The motor model adds
// M*(-c*qd_measured - k*q_measured) - ci*qdd_previous to the wheel and arm torques (ServoMotorModel.compute_torque;
// the inertia term is commented out in the course code and is switched on here only if its coefficient is set).
struct Effects {
    bool any=false,self_collision=false,noise=false;int delay=0;double damping=0,elastic=0,inertia=0;
    double joint_sd=1e-4,joint_vel_sd=1e-4,base_pos_sd=.1,base_ori_sd=.1;unsigned long long seed=0;json raw=json::object();
};
static Effects effects;
static std::string effects_file;
static void load_effects(const std::string& path){
    std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open effects file");json j;f>>j;effects.raw=j;
    effects.self_collision=j.value("collision_enable",0)!=0;effects.noise=j.value("noise_flag",0)!=0;
    effects.delay=j.value("delay_measure_flag",0)?j.value("delay_measure_steps",3):0;
    effects.damping=j.value("motor_damping",0)?j.value("motor_damping_coeff",0.0):0;
    effects.elastic=j.value("motor_elastic_torque",0)?j.value("motor_elastic_coeff",0.0):0;
    effects.inertia=j.value("motor_inertia",0)?j.value("motor_inertia_coeff",0.0):0;
    if(j.contains("robot_noise")){auto& n=j["robot_noise"];effects.joint_sd=n.value("joint_cov",1e-4);effects.joint_vel_sd=n.value("joint_vel_cov",1e-4);effects.base_pos_sd=n.value("base_pos_cov",.1);effects.base_ori_sd=n.value("base_ori_cov",.1);}
    effects.seed=j.value("seed",0ULL);
    effects.any=effects.self_collision||effects.noise||effects.delay>0||effects.damping!=0||effects.elastic!=0||effects.inertia!=0;
    if(effects.delay<0)throw std::runtime_error("Negative delay");
}
struct Measure {
    std::deque<State> buffer;std::mt19937_64 rng;std::normal_distribution<double> normal{0,1};
    explicit Measure(unsigned long long seed):rng(seed){}
    State operator()(const State& s){
        if(!effects.noise&&effects.delay==0)return s;
        buffer.push_back(s);while(int(buffer.size())>effects.delay+1)buffer.pop_front();
        State m=buffer.front();
        if(effects.noise){
            for(int i=0;i<3;i++)m.pos[i]+=effects.base_pos_sd*normal(rng);
            double r=effects.base_ori_sd*normal(rng),pi=effects.base_ori_sd*normal(rng),y=effects.base_ori_sd*normal(rng);
            Eigen::Quaterniond noise=Eigen::AngleAxisd(y,V3::UnitZ())*Eigen::AngleAxisd(pi,V3::UnitY())*Eigen::AngleAxisd(r,V3::UnitX());
            Eigen::Quaterniond q(m.quat[3],m.quat[0],m.quat[1],m.quat[2]);q=(noise*q).normalized();m.quat<<q.x(),q.y(),q.z(),q.w();
            for(int i=0;i<17;i++)m.q[i]+=effects.joint_sd*normal(rng);
            for(int i=0;i<17;i++)m.qd[i]+=effects.joint_vel_sd*normal(rng);
        }
        return m;
    }
};
static V17 motor_torques(Physics& physics,const State& truth,const State& m){
    V17 extra=V17::Zero();
    if(effects.damping!=0||effects.elastic!=0){Eigen::MatrixXd M=physics.mass(truth);extra=M.bottomRightCorner(17,17)*(-effects.damping*m.qd-effects.elastic*m.q);}
    if(effects.inertia!=0)extra-=effects.inertia*truth.qdd;
    return extra;
}
static bool binary_output=false;
static void npy(const fs::path&,const std::vector<double>&,const std::vector<size_t>&);
// Numerical simulation stays double precision, matching the reference. Cast to float32 at training ETL.
struct Csv {
    std::ofstream f;fs::path path;bool header=true;size_t rows=0;std::vector<std::string> keys,values,last_labels;std::vector<double> numbers,last_codes;json categories=json::object();size_t column=0;
    explicit Csv(const fs::path& p):path(p){if(!binary_output){f.exceptions(std::ios::failbit|std::ios::badbit);f.open(p);f<<std::setprecision(17);}else numbers.reserve(8001*60);}
    void finish(){if(binary_output){auto p=path;p.replace_extension(".npy");npy(p,numbers,{rows,keys.size()});p.replace_extension(".schema.json");std::ofstream schema;schema.exceptions(std::ios::failbit|std::ios::badbit);schema.open(p);schema<<json({{"columns",keys},{"categories",categories}}).dump(2)<<'\n';schema.close();}else f.close();}
    template<class T> void add(std::string k,const T& v){
        if(header){keys.push_back(k);last_labels.emplace_back();last_codes.push_back(0);}
        if(binary_output){
            if constexpr(std::is_arithmetic_v<T>)numbers.push_back(double(v));
            else {std::string_view label(v);if(header||label!=last_labels[column]){auto& key=keys[column];if(!categories.contains(key))categories[key]=json::array();auto& labels=categories[key];auto it=std::find(labels.begin(),labels.end(),std::string(label));last_codes[column]=it==labels.end()?double(labels.size()):double(std::distance(labels.begin(),it));if(it==labels.end())labels.push_back(std::string(label));last_labels[column]=label;}numbers.push_back(last_codes[column]);}
            column++;return;
        }
        if constexpr(std::is_arithmetic_v<T>){char buf[64];auto result=std::to_chars(buf,buf+64,double(v),std::chars_format::general,17);if(result.ec!=std::errc())throw std::runtime_error("Number formatting failed");values.emplace_back(buf,result.ptr);}
        else values.emplace_back(v);
    }
    template<class Derived> void vec(std::string_view prefix,const Eigen::MatrixBase<Derived>& v,const std::vector<std::string>& labels){for(int i=0;i<v.size();i++)add(header?std::string(prefix)+"_"+labels[i]:"",v[i]);}
    void end(){rows++;column=0;if(binary_output){header=false;return;}auto line=[&](auto& x){for(size_t i=0;i<x.size();i++)f<<(i?",":"")<<x[i];f<<'\n';};if(header){line(keys);header=false;}line(values);values.clear();}
};
static std::vector<std::string> numbered(std::string prefix,int n,int pad=0){std::vector<std::string> s;for(int i=1;i<=n;i++)s.push_back(prefix+(pad&&i<10?"0":"")+std::to_string(i));return s;}
static void npy(const fs::path& path,const std::vector<double>& values,const std::vector<size_t>& shape){
    std::ostringstream shape_text;for(auto n:shape)shape_text<<n<<", ";
    std::string h="{'descr': '<f8', 'fortran_order': False, 'shape': ("+shape_text.str()+"), }";
    h+=std::string((64-(10+h.size()+1)%64)%64,' ');h+='\n';
    std::ofstream f;f.exceptions(std::ios::failbit|std::ios::badbit);f.open(path,std::ios::binary);f.write("\x93NUMPY\x01\x00",8);uint16_t len=h.size();f.write(reinterpret_cast<char*>(&len),2);f<<h;f.write(reinterpret_cast<const char*>(values.data()),values.size()*8);f.close();
}
static void write_run(const fs::path& out,const std::vector<Row>& rows,const Expert& expert,json manifest){
    if(fs::exists(out))throw std::runtime_error("Refusing to overwrite "+out.string());fs::create_directories(out);
    auto joint=numbered("j",17,1),arm=numbered("arm",7);
    const std::vector<std::string> xyz{"x","y","z"},xy{"x","y"},quat{"x","y","z","w"},wheel_labels{"front_right","front_left","back_left","back_right"};
    std::vector<std::string> rot;for(int r=1;r<=3;r++)for(int c=1;c<=3;c++)rot.push_back("r"+std::to_string(r)+std::to_string(c));
    std::unique_ptr<Csv> measured;if(effects.noise||effects.delay>0)measured=std::make_unique<Csv>(out/"measured_state.csv");
    Csv target(out/"target.csv"),state(out/"joint_state.csv"),base(out/"base_state.csv"),kin(out/"kinematics.csv"),bc(out/"base_controller.csv"),ac(out/"arm_controller.csv"),cmd(out/"commands.csv"),timing(out/"timing.csv");
    std::vector<double> qp,jac,mass;
    for(size_t i=0;i<rows.size();i++){
        auto& r=rows[i];auto& s=r.s;auto& a=r.a;
        for(auto f:{&target,&state,&base,&kin,&bc,&ac,&cmd,&timing}){f->add("step",i);f->add("time_s",r.t);}
        if(measured){measured->add("step",i);measured->add("time_s",r.t);measured->vec("q_rad",r.m.q,joint);measured->vec("qd_rad_s",r.m.qd,joint);measured->vec("position_m",r.m.pos,xyz);measured->vec("quaternion",r.m.quat,quat);measured->end();}
        state.vec("q_rad",s.q,joint);state.vec("qd_rad_s",s.qd,joint);state.vec("qdd_backward_rad_s2",s.qdd,joint);
        base.vec("position_m",s.pos,xyz);base.vec("quaternion",s.quat,quat);base.vec("linear_velocity_world_m_s",s.lin,xyz);base.vec("angular_velocity_world_rad_s",s.ang,xyz);
        double zero[3]={0,0,0},identity[4]={0,0,0,1},invpos[3],invq[4],tmpq[4];V3 linbody,angbody;
        b3InvertTransform(zero,s.quat.data(),invpos,invq);
        b3MultiplyTransforms(zero,invq,s.lin.data(),identity,linbody.data(),tmpq);
        b3MultiplyTransforms(zero,invq,s.ang.data(),identity,angbody.data(),tmpq);
        base.vec("linear_velocity_body_m_s",linbody,xyz);base.vec("angular_velocity_body_rad_s",angbody,xyz);base.add("yaw_rad",a.yaw);
        target.add("target_yaw_not_specified","True");target.vec("ee_target_world_m",expert.target,xyz);target.vec("base_standoff_goal_world_m",expert.goal,xy);target.add("base_goal_yaw_rad",expert.bearing);target.add("standoff_m",expert.standoff);
        kin.add("ee_distance_error_m",a.distance);kin.vec("ee_world_m",a.ee,xyz);kin.vec("ee_error_world_m",a.error,xyz);kin.vec("shoulder_world_m",a.shoulder,xyz);
        Eigen::Matrix<double,9,1> rotation;for(int x=0;x<3;x++)for(int y=0;y<3;y++)rotation[x*3+y]=a.rotation(x,y);kin.vec("ee_rotation",rotation,rot);
        bc.add("phase_after_call",std::array<const char*,3>{"align","drive","orient"}[a.phase]);bc.add("navigation_done",a.done?"True":"False");
        bc.add("dist_to_goal",a.goal_dist);bc.add("heading_to_goal",a.heading);bc.add("v_cmd",a.v);bc.add("w_cmd",a.w);bc.add("err",a.err);bc.add("lw",a.lw);bc.add("rw",a.rw);bc.vec("wheel_velocity_command_rad_s",a.wheel,wheel_labels);
        ac.add("reach_active",a.done?"True":"False");ac.add("sigma_min",a.sigma);ac.add("ik_damping",a.damping);
        ac.vec("q_input_rad",s.q.segment<7>(4),arm);ac.vec("qd_input_rad_s",s.qd.segment<7>(4),arm);ac.vec("q_des_rad",a.qdes,arm);ac.vec("qd_des_rad_s",a.qddes,arm);ac.vec("tau_command_Nm",a.tau,arm);ac.vec("gravity_Nm",a.gravity,arm);ac.vec("pd_term_Nm",a.pd,arm);ac.vec("ik_error",a.ik,xyz);ac.vec("v_cartesian",a.vc,xyz);
        V17 mixed=V17::Zero();mixed.head<4>()=a.wheel;mixed.segment<7>(4)=a.tau;cmd.vec("mixed_command",mixed,joint);cmd.vec("sent_torque_Nm",r.torque,joint);for(int j=0;j<17;j++)cmd.add("mode_"+joint[j],j<4?"velocity_to_torque":j<11?"torque":"position_hold");
        // The original wrapper's post-step clock is step_count * dt, while
        // its controller time_s is accumulated. Preserve both conventions.
        timing.add("post_step_time_s",(i+1)*dt);timing.add("physics_step_wall_s",r.step_wall);timing.add("wall_elapsed_s",r.wall);
        for(auto f:{&target,&state,&base,&kin,&bc,&ac,&cmd,&timing})f->end();
        qp.insert(qp.end(),a.qpin.data(),a.qpin.data()+a.qpin.size());
        for(int x=0;x<3;x++)for(int y=0;y<7;y++)jac.push_back(a.jac(x,y));
        for(int x=0;x<r.mass.rows();x++)for(int y=0;y<r.mass.cols();y++)mass.push_back(r.mass(x,y));
    }
    npy(out/"q_pin.npy",qp,{rows.size(),24});npy(out/"arm_jacobian.npy",jac,{rows.size(),3,7});
    if(!mass.empty())npy(out/"mass_matrix.npy",mass,{rows.size(),23,23});
    for(auto f:{&target,&state,&base,&kin,&bc,&ac,&cmd,&timing})f->finish();
    if(measured)measured->finish();
    manifest["tensor_format"]="Separate standard NPY arrays; same tensor names and dimensions as reference NPZ";
    manifest["mass_matrix_recorded"]=!mass.empty();
    manifest["table_format"]=binary_output?"NPY float64 + JSON column/category schemas":"CSV";
    manifest["export_complete"]=true;
    std::ofstream complete;complete.exceptions(std::ios::failbit|std::ios::badbit);complete.open(out/"manifest.json");complete<<manifest.dump(2)<<'\n';complete.close();
}
static void shared_world(const std::string& assets,const std::string& out,V3 target,int count,int cap){
    auto start=Clock::now();std::vector<std::unique_ptr<Physics>> robots;
    std::vector<std::unique_ptr<Expert>> experts;std::vector<std::vector<Row>> rows(count);std::vector<int> parking(count,-1);std::vector<bool> done(count,false);std::vector<std::string> reasons(count);
    for(int i=0;i<count;i++){robots.push_back(std::make_unique<Physics>(assets,i?robots[0].get():nullptr));robots.back()->isolate();experts.push_back(std::make_unique<Expert>(assets,*robots.back(),robots.back()->state(),target));}
    double setup=elapsed(start),t=0;int tick=0,finished=0;auto loop=Clock::now();
    while(finished<count){
        std::vector<State> state(count);std::vector<Action> action(count);std::vector<V17> torque(count);
        for(int i=0;i<count;i++)if(!done[i]){state[i]=robots[i]->state();action[i]=experts[i]->act(state[i],t);torque[i]=robots[i]->step(state[i],action[i],false);}
        auto stepping=Clock::now();robots[0]->advance();double step_time=elapsed(stepping);
        for(int i=0;i<count;i++)if(!done[i]){
            if(action[i].done&&parking[i]<0)parking[i]=tick;
            if(!out.empty())rows[i].push_back({state[i],action[i],torque[i],t,step_time,elapsed(start),{}});
            reasons[i]=experts[i]->stop(action[i],t,cap);
            if(!reasons[i].empty()){done[i]=true;finished++;std::cout<<"WORLD_RESULT "<<json({{"expert",i},{"steps",tick+1},{"parking_step",parking[i]},{"termination",reasons[i]},{"final_error",action[i].distance}}).dump()<<std::endl;}
        }
        tick++;t+=dt;
    }
    double loop_time=elapsed(loop);auto exporting=Clock::now();
    for(int i=0;i<count;i++)if(!out.empty())write_run(fs::path(out)/("expert_"+std::to_string(i)),rows[i],*experts[i],{{"shared_world",true},{"experimental",true},{"dt_s",dt},{"steps",rows[i].size()},{"parking_step",parking[i]},{"termination",reasons[i]},{"joint_names",robots[i]->names}});
    std::cout<<"BATCH "<<json({{"world_experts",count},{"episodes",count},{"setup_seconds",setup},{"loop_seconds",loop_time},{"export_seconds",elapsed(exporting)},{"wall_seconds",elapsed(start)},{"episodes_per_second",count/elapsed(start)},{"ok",true}}).dump()<<std::endl;
}
int main(int argc,char** argv){try{
    std::string assets="assets",out,states_file,targets_file,queue_file;V3 target(1.1,0,.5);int cap=40000,repeats=1,workers=1,world_experts=1,episode_offset=0,total_jobs=0;bool diagnostics=false,dynamic=false,fork_workers=false,driving_only=false;
    for(int i=1;i<argc;i++){
        std::string arg=argv[i];auto next=[&](){if(++i>=argc)throw std::runtime_error("Missing value for "+arg);return std::string(argv[i]);};
        if(arg=="--help"){std::cout<<"CPU C++ expert, original 0.001 s physics/control step\n"
          "  --target X Y Z       target in metres (default 1.1 0 0.5)\n"
          "  --targets FILE      JSON array of target xyz positions\n"
          "  --out NEW_FOLDER    save all 1000 Hz observations and labels\n"
          "  --binary            faster NPY tables + column schemas (otherwise CSV)\n"
          "  --driving-only      stop after first navigation_done tick (1000 Hz prefix)\n"
          "  --diagnostics       also save 23x23 mass matrices at every tick\n"
          "  --workers N         independent CPU processes (default 1)\n"
          "  --fork-workers      preload one world, fork isolated copies (POSIX)\n"
          "  --repeat N          complete episodes per worker (default 1)\n"
          "  --dynamic           distribute workers*repeat jobs through a queue\n"
          "  --max-steps N       original stop cap (default 40000)\n"
          "  --assets FOLDER     model directory (default assets)\n"
          "  --world-experts N   EXPERIMENTAL shared world; does not match baseline\n"
          "  --states FILE       controller-only replay audit, not a new rollout\n"
          "  --effects FILE      course simulator effects (collision, noise, delay, motor damping/elasticity/inertia)\n";return 0;}
        if(arg=="--effects"){effects_file=fs::absolute(next()).string();load_effects(effects_file);}
        else if(arg=="--assets")assets=next();else if(arg=="--out")out=next();else if(arg=="--states")states_file=next();else if(arg=="--targets")targets_file=next();else if(arg=="--episode-offset")episode_offset=std::stoi(next());else if(arg=="--target"){for(int j=0;j<3;j++)target[j]=std::stod(next());}
        else if(arg=="--max-steps")cap=std::stoi(next());else if(arg=="--repeat")repeats=std::stoi(next());else if(arg=="--workers")workers=std::stoi(next());else if(arg=="--world-experts")world_experts=std::stoi(next());else if(arg=="--driving-only")driving_only=true;else if(arg=="--diagnostics")diagnostics=true;else if(arg=="--binary")binary_output=true;else if(arg=="--dynamic")dynamic=true;else if(arg=="--fork-workers")fork_workers=true;else if(arg=="--queue")queue_file=next();else if(arg=="--jobs")total_jobs=std::stoi(next());
        else throw std::runtime_error("Unknown argument "+arg);
    }
    if(!target.allFinite()||cap<1||repeats<1||workers<1||workers>64||world_experts<1||episode_offset<0)throw std::runtime_error("Invalid arguments");
    if(!out.empty()&&fs::exists(out)&&queue_file.empty())throw std::runtime_error("Output already exists");
    assets=fs::absolute(assets).string();
    std::vector<V3> targets;
    if(!targets_file.empty()){
        std::ifstream f(targets_file);if(!f)throw std::runtime_error("Cannot open targets file");json j;f>>j;if(!j.is_array()||j.empty())throw std::runtime_error("Targets must be a nonempty array");
        for(auto& x:j){if(!x.is_array()||x.size()!=3)throw std::runtime_error("Target needs xyz");V3 v(x[0].get<double>(),x[1].get<double>(),x[2].get<double>());if(!v.allFinite())throw std::runtime_error("Nonfinite target");targets.push_back(v);}
        targets_file=fs::absolute(targets_file).string();
    }
    if(world_experts>1){if(driving_only)throw std::runtime_error("Driving-only requires independent worlds");if(workers!=1||repeats!=1||diagnostics||!states_file.empty()||world_experts>16||!targets.empty())throw std::runtime_error("Shared-world comparison requires one target, one worker, one repeat, no diagnostics/replay, at most 16 robots");shared_world(assets,out,target,world_experts,cap);return 0;}
    std::vector<State> replay;
    if(!states_file.empty()){
        if(driving_only)throw std::runtime_error("Driving-only is a physics rollout, not state replay");
        if(workers!=1||repeats!=1)throw std::runtime_error("State replay is a single-run controller audit");
        std::ifstream f(states_file);if(!f)throw std::runtime_error("Cannot open state replay");std::string line;std::getline(f,line);
        while(std::getline(f,line)){
            std::istringstream stream(line);std::vector<double> v;std::string part;while(std::getline(stream,part,','))v.push_back(std::stod(part));
            if(v.size()!=65)throw std::runtime_error("State replay needs 65 columns");
            State s;s.pos=Eigen::Map<V3>(v.data()+1);s.quat=Eigen::Map<Eigen::Vector4d>(v.data()+4);s.lin=Eigen::Map<V3>(v.data()+8);s.ang=Eigen::Map<V3>(v.data()+11);s.q=Eigen::Map<V17>(v.data()+14);s.qd=Eigen::Map<V17>(v.data()+31);s.qdd=Eigen::Map<V17>(v.data()+48);replay.push_back(s);
        }
        if(replay.empty())throw std::runtime_error("State replay is empty");
    }
    // Process workers isolate Bullet global caches and collision worlds. Each has the same origin.
    auto total=Clock::now();
    std::unique_ptr<Physics> loaded_physics;
    if(workers>1){
        if(fork_workers){loaded_physics=std::make_unique<Physics>(assets,nullptr,effects.self_collision);std::cout.flush();std::cerr.flush();std::fflush(nullptr);}
        if(dynamic){char path[]="/tmp/native-expert-queue-XXXXXX";int fd=mkstemp(path);if(fd<0)throw std::runtime_error("Cannot create work queue");int zero=0;if(write(fd,&zero,sizeof(zero))!=sizeof(zero))throw std::runtime_error("Cannot initialise queue");close(fd);queue_file=path;total_jobs=workers*repeats;}
        std::vector<pid_t> children;for(int w=0;w<workers;w++){
            pid_t pid=fork();if(pid<0)throw std::runtime_error("fork failed");if(pid==0){
                if(fork_workers){if(!out.empty()&&!dynamic)out+="/worker_"+std::to_string(w);episode_offset+=w*repeats;workers=1;children.clear();break;}
                auto exact=[](double x){std::ostringstream s;s<<std::setprecision(17)<<x;return s.str();};
                std::vector<std::string> a{argv[0],"--assets",assets,"--target",exact(target.x()),exact(target.y()),exact(target.z()),"--max-steps",std::to_string(cap),"--repeat",std::to_string(repeats)};
                if(!effects_file.empty())a.insert(a.end(),{"--effects",effects_file});
                if(driving_only)a.push_back("--driving-only");if(diagnostics)a.push_back("--diagnostics");if(binary_output)a.push_back("--binary");if(!out.empty()){a.push_back("--out");a.push_back(dynamic?out:out+"/worker_"+std::to_string(w));}
                if(dynamic)a.insert(a.end(),{"--queue",queue_file,"--jobs",std::to_string(total_jobs)});
                if(!targets_file.empty()){a.insert(a.end(),{"--targets",targets_file,"--episode-offset",std::to_string(episode_offset+w*repeats)});}
                std::vector<char*> raw;for(auto& x:a)raw.push_back(x.data());raw.push_back(nullptr);execvp(argv[0],raw.data());_exit(127);
            }children.push_back(pid);
        }
        if(workers>1){bool failed=false;for(auto pid:children){int status;waitpid(pid,&status,0);failed|=!WIFEXITED(status)||WEXITSTATUS(status)!=0;}
        if(dynamic)fs::remove(queue_file);
        double seconds=elapsed(total);std::cout<<"BATCH "<<json({{"workers",workers},{"dynamic",dynamic},{"fork_workers",fork_workers},{"episodes",workers*repeats},{"wall_seconds",seconds},{"episodes_per_second",workers*repeats/seconds},{"ok",!failed}}).dump()<<std::endl;return failed?1:0;}
    }
    if(!loaded_physics)loaded_physics=std::make_unique<Physics>(assets,nullptr,effects.self_collision);Physics& physics=*loaded_physics;double setup=elapsed(total);
    int completed=0;int queue_fd=-1;if(!queue_file.empty()){queue_fd=open(queue_file.c_str(),O_RDWR);if(queue_fd<0)throw std::runtime_error("Cannot open job queue");if(total_jobs<1)throw std::runtime_error("Invalid job count");}
    for(int local_episode=0;queue_fd>=0||local_episode<repeats;local_episode++){
        int episode=local_episode;
        if(queue_fd>=0){if(flock(queue_fd,LOCK_EX))throw std::runtime_error("Cannot lock queue");if(pread(queue_fd,&episode,sizeof(episode),0)!=sizeof(episode))throw std::runtime_error("Cannot read queue");int next=episode+1;if(pwrite(queue_fd,&next,sizeof(next),0)!=sizeof(next))throw std::runtime_error("Cannot update queue");flock(queue_fd,LOCK_UN);if(episode>=total_jobs)break;}
        if(!targets.empty())target=targets[(queue_fd>=0?episode:episode_offset+episode)%targets.size()];
        if(completed)physics.reset();auto begin=Clock::now();State s=replay.empty()?physics.state():replay[0];Expert expert(assets,physics,s,target);double model_setup=elapsed(begin);
        std::vector<Row> rows;if(!out.empty())rows.reserve(12000);
        double t=0;int park=-1;std::string reason;Action action;auto loop=Clock::now();int steps=0;
        Measure measure(effects.seed*1000003ULL+episode);State m;
        while(true){
            m=measure(s);
            action=expert.act(m,t);if(action.done&&park<0)park=steps;
            Eigen::MatrixXd mass;if(diagnostics&&!out.empty())mass=physics.mass(s);
            auto before=Clock::now();V17 torque;
            V17 extra=effects.any?motor_torques(physics,s,m):V17::Zero();
            if(replay.empty())torque=physics.step(m,action,true,effects.any?&extra:nullptr);else{torque.head<4>()=30*(action.wheel-s.qd.head<4>());torque.segment<7>(4)=action.tau;torque.tail<6>().setConstant(missing);}
            double wall=elapsed(before);
            if(!out.empty())rows.push_back({s,action,torque,t,wall,elapsed(begin),std::move(mass),m});
            steps++;
            // Save and execute the first done-labelled tick before terminating.
            // The original expert also sets done on timeout; do not call that parking.
            if(driving_only&&action.done){reason=(action.phase==2&&std::abs(action.err)<.05)?"parked_command":"navigation_timeout";break;}
            reason=expert.stop(action,t,cap);if(!reason.empty())break;if(!replay.empty()&&steps>=replay.size()){reason="replay_end";break;}t+=dt;s=replay.empty()?physics.state():replay[steps];
        }
        json result={{"episode",episode},{"steps",steps},{"dt_s",dt},{"simulated_duration_s",steps*dt},{"loop_seconds",elapsed(loop)},{"physics_setup_seconds",setup},{"model_setup_seconds",model_setup},{"termination",reason},{"parking_step",park},{"final_pre_step_time_s",t},{"final_pre_step_ee_error_m",action.distance},{"base_final",{s.pos.x(),s.pos.y(),s.pos.z()}},{"target_position_world_m",{target.x(),target.y(),target.z()}},{"joint_names",physics.names},{"runtime","C++ CPU Bullet DIRECT + Pinocchio"}};
        result["driving_only"]=driving_only;
        result["effects"]=effects.raw;result["effects_seed"]=effects.seed*1000003ULL+episode;
        result["controller_audit_only"]=!replay.empty();
        result["fork_workers"]=fork_workers;
        result["collector_source_sha256"]=COLLECTOR_SHA;result["compiler"]=COMPILER_VERSION;
        result["visual_meshes_skipped"]=bool(std::getenv("EXPERT_SKIP_VISUALS"));
        result["sample_alignment"]="pre-step state with command applied for that tick";
#ifdef REFERENCE_ENGINE
        result["engine"]="Pinned PyBullet 3.25 native C API, no Python execution";
        result["default_engine_binary_sha256"]=ENGINE_SHA;
        result["engine_library_override"]=bool(std::getenv("EXPERT_PYBULLET"));
#else
        result["engine"]="Source-built Bullet 3.25: experimental numerical variant";
#endif
        if(!out.empty()){auto writing=Clock::now();write_run(repeats==1&&queue_fd<0?fs::path(out):fs::path(out)/("run_"+std::to_string(episode)),rows,expert,result);result["export_seconds"]=elapsed(writing);}
        std::cout<<"RESULT "<<result.dump()<<std::endl;
        completed++;
    }
    if(queue_fd>=0)close(queue_fd);
    std::cout<<"BATCH "<<json({{"workers",1},{"episodes",completed},{"wall_seconds",elapsed(total)},{"episodes_per_second",completed/elapsed(total)},{"ok",true}}).dump()<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<std::endl;return 1;}}
