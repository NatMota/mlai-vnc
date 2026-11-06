import json,subprocess
from pathlib import Path
R=Path('/media/nm/Fast Drive/comp0245-directed-edit-20261105');exe=R/'rowing_native/rowing_native';assets=R/'native/assets'
p=subprocess.Popen([str(exe),str(assets)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
while True:
 s=p.stdout.readline()
 try:
  r=json.loads(s[s.index("{"):])
  if r.get('ready'):break
 except ValueError:pass
rows=[]
for k in range(10):
 p.stdin.write(json.dumps(dict(tau=[0]*7,ticks=500))+'\n');p.stdin.flush();row=json.loads(p.stdout.readline());assert row['sent'][:4]==[0]*4;row['time_s']=(k+1)*.5;rows.append(row)
p.stdin.close();p.wait();(R/'rowing_native/passive_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-1]))
