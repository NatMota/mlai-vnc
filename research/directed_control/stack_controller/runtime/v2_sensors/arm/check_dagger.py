"""Verify the stateful oracle against a complete recorded expert episode."""
import argparse
import json
from pathlib import Path
import numpy as np
from check_curriculum import Simulator


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    with np.load(a.run,allow_pickle=False) as z:
        m=json.loads(str(z['manifest']));q=z['q'];tau=z['arm_torque_command'];wheel=z['wheel_command'];qdes=z['arm_q_des']
    sim=Simulator(a.binary);errors=np.zeros(4)
    try:
        reset=dict(op='reset',target=m['target_position_world_m'],arm=True,arm_from_reset=True,collect_oracle=True,max_duration_s=m['steps']/1000+1)
        state=sim.call(reset)
        assert sim.call(dict(op='step',wheel=[0]*4,arm=[0]*7,ms=1))['type']=='error'
        for k in range(m['steps']):
            parked=k>=m['parking_step'];teacher=sim.call(dict(op='teacher',park=parked))
            assert teacher['type']=='teacher' and teacher['step']==k
            if k==0:assert sim.call(dict(op='teacher',park=parked))['type']=='error'
            for i,(got,want) in enumerate(((state['frame']['joints'],q[k]),(teacher['arm'],tau[k]),(teacher['wheel'],wheel[k]),(teacher['q_des'],qdes[k]))):
                errors[i]=max(errors[i],float(np.max(np.abs(np.asarray(got)-want))))
                np.testing.assert_allclose(got,want,atol=2e-4,rtol=1e-5,err_msg=f'tick {k}, field {i}')
            state=sim.call(dict(op='step',wheel=teacher['wheel'],arm=teacher['arm'],park=parked,ms=1))
        result=dict(steps=m['steps'],maximum_errors=dict(zip(('q','arm_torque','wheels','q_des'),errors.tolist())),
                    duplicate_query_rejected=True,missing_query_rejected=True,oracle_phase='matched recorded PARK',
                    limitation='Parity on the expert path; corrective policy quality is assessed separately')
        a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
    finally:sim.close()


if __name__=='__main__':main()
