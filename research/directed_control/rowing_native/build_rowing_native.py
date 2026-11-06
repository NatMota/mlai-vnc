from pathlib import Path
import json,subprocess,shutil,hashlib
R=Path('/media/nm/Fast Drive/comp0245-directed-edit-20261105');src=R/'recovered_clean_native';out=R/'rowing_native';out.mkdir(exist_ok=True)
for n in ['physics.cpp','expert.hpp','bullet_api.hpp']:shutil.copy2(src/n,out/n)
p=out/'physics.cpp';s=p.read_text();old='torque.head<4>()=30*(a.wheel-s.qd.head<4>());';assert old in s;p.write_text(s.replace(old,'torque.head<4>().setZero(); // Explicit unpowered-wheel demonstration only.'))
(out/'main.cpp').write_text('''#include "expert.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
using json=nlohmann::json;
int main(int argc,char**argv){Physics physics(argv[1]);State s=physics.state();std::string line;std::cout<<"{\\"ready\\":true}"<<std::endl;
while(std::getline(std::cin,line)){auto j=json::parse(line);Action a;for(int k=0;k<7;k++)a.tau[k]=j.at("tau")[k];int ticks=j.value("ticks",1);V17 sent;for(int n=0;n<ticks;n++){sent=physics.step(s,a);s=physics.state();}std::cout<<json{{"pos",std::vector<double>(s.pos.data(),s.pos.data()+3)},{"q",std::vector<double>(s.q.data(),s.q.data()+17)},{"qd",std::vector<double>(s.qd.data(),s.qd.data()+17)},{"sent",std::vector<double>(sent.data(),sent.data()+11)}}.dump()<<std::endl;}}
''')
r=json.loads((src/'build_receipt.json').read_text());commands=r.get('commands',r.get('build_commands'));assert commands,r.keys()
base=next(c for c in commands if str(src/'physics.cpp') in c)
for name in ['physics.cpp','main.cpp']:
 cmd=[x.replace(str(src/'physics.cpp'),str(out/name)) for x in base];subprocess.run(cmd,check=True,capture_output=True)
link=commands[-1];link=[x.replace(str(src/'physics.cpp.o'),str(out/'physics.cpp.o')).replace(str(src/'sensor_curriculum.cpp.o'),str(out/'main.cpp.o')).replace(str(src/'rebuilt_sensor_curriculum'),str(out/'rowing_native')) for x in link];subprocess.run(link,check=True,capture_output=True)
(out/'build_receipt.json').write_text(json.dumps(dict(commands=[base,link],binary_sha256=hashlib.sha256((out/'rowing_native').read_bytes()).hexdigest(),change='Wheel torque servo bypass only; frozen native source otherwise unchanged'),indent=2));print(out/'rowing_native')
