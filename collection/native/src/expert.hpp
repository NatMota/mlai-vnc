#pragma once
#include <Eigen/Core>
#include <array>
#include <string>
#include <vector>
#include <memory>
#include <limits>
constexpr double dt = .001;
constexpr double missing = std::numeric_limits<double>::quiet_NaN();
using V3 = Eigen::Vector3d;
using V7 = Eigen::Matrix<double,7,1>;
using V17 = Eigen::Matrix<double,17,1>;
struct State {
    V3 pos=V3::Zero(), lin=V3::Zero(), ang=V3::Zero();
    Eigen::Vector4d quat{0,0,0,1};
    V17 q=V17::Zero(), qd=V17::Zero(), qdd=V17::Zero();
};
struct Action {
    Eigen::Vector4d wheel=Eigen::Vector4d::Zero();
    V7 qdes=V7::Zero(), qddes=V7::Zero(), tau=V7::Zero(), gravity=V7::Zero(), pd=V7::Zero();
    V3 ee, shoulder, error, ik=V3::Constant(missing), vc=V3::Constant(missing);
    Eigen::Matrix3d rotation;
    Eigen::Matrix<double,3,7> jac=Eigen::Matrix<double,3,7>::Constant(missing);
    Eigen::VectorXd qpin;
    double yaw=0, distance=0, sigma=missing, damping=missing;
    double goal_dist=missing, heading=missing, v=missing, w=missing, err=missing, lw=missing, rw=missing;
    int phase=0;
    bool done=false;
};
class Physics {
    struct Impl;
    std::unique_ptr<Impl> p;
public:
    std::vector<std::string> names;
    V7 lower, upper;
    explicit Physics(const std::string& assets, Physics* shared=nullptr);
    ~Physics();
    State state();
    V17 step(const State&, const Action&, bool advance=true);
    void advance();
    void isolate();
    void reset();
    Eigen::MatrixXd mass(const State&);
};
class Expert {
    struct Impl;
    std::unique_ptr<Impl> p;
public:
    V3 target;
    Eigen::Vector2d goal;
    double bearing, standoff;
    Expert(const std::string& assets, const Physics&, const State&, V3 target);
    ~Expert();
    Action act(const State&, double t);
    std::string stop(const Action&, double t, int cap);
};
