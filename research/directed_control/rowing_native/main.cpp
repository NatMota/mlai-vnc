#include "expert.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
using json=nlohmann::json;
int main(int argc,char**argv){Physics physics(argv[1]);State s=physics.state();std::string line;std::cout<<"{\"ready\":true}"<<std::endl;
while(std::getline(std::cin,line)){auto j=json::parse(line);Action a;for(int k=0;k<7;k++)a.tau[k]=j.at("tau")[k];int ticks=j.value("ticks",1);V17 sent;for(int n=0;n<ticks;n++){sent=physics.step(s,a);s=physics.state();}std::cout<<json{{"pos",std::vector<double>(s.pos.data(),s.pos.data()+3)},{"q",std::vector<double>(s.q.data(),s.q.data()+17)},{"qd",std::vector<double>(s.qd.data(),s.qd.data()+17)},{"sent",std::vector<double>(sent.data(),sent.data()+11)}}.dump()<<std::endl;}}
