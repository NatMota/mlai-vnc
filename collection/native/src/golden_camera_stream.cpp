// Baseline expert, exactly main.cpp's pre-step loop, streamed at 30 Hz for camera parity.
// No policy, effects, new controller, physics changes or target information fed into the camera.
#include "expert.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <cmath>
using json = nlohmann::json;
template<class V> json values(const V& v) { json a=json::array(); for(int i=0;i<v.size();i++)a.push_back(v[i]); return a; }
int main(int argc, char** argv) { try {
    if(argc!=6)throw std::runtime_error("Usage: golden_camera_stream ASSETS X Y Z MAX_STEPS");
    Physics physics(argv[1]); State s=physics.state();
    V3 target(std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]));
    Expert expert(argv[1],physics,s,target); int cap=std::stoi(argv[5]);
    std::cout<<"CAMERA "<<json({{"type","ready"},{"joint_names",physics.names},{"engine_sha256",ENGINE_SHA}}).dump()<<std::endl;
    double t=0; int frame=0,park=-1;
    for(int step=0;;step++,t+=dt){
        auto a=expert.act(s,t); if(a.done&&park<0)park=step;
        if(step==std::lround(frame*1000./30)){
            std::cout<<"CAMERA "<<json({{"type","frame"},{"step",step},{"base",values(s.pos)},
                {"quaternion",values(s.quat)},{"joints",values(s.q)}}).dump()<<std::endl;
            std::string ack; if(!std::getline(std::cin,ack))return 0; frame++;
        }
        physics.step(s,a);
        auto reason=expert.stop(a,t,cap);
        if(!reason.empty()){
            std::cout<<"CAMERA "<<json({{"type","done"},{"steps",step+1},{"parking_step",park},{"termination",reason}}).dump()<<std::endl;
            break;
        }
        s=physics.state();
    }
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;} }
