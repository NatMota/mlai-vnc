"""Check no-hold command handoff, zero-effect parity and motor-effect accounting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from curriculum_sensors import check


class Simulator:
    def __init__(self,binary):
        native=Path(os.environ['NATIVE_ROOT'])
        self.p=subprocess.Popen([str(Path.home()/'.local/bin/pixi'),'run',str(binary),str(native/'assets')],cwd=native,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,
            env=dict(os.environ,EXPERT_SKIP_VISUALS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        self.call()

    def call(self,msg=None):
        if msg is not None:self.p.stdin.write(json.dumps(msg)+'\n');self.p.stdin.flush()
        while True:
            s=self.p.stdout.readline()
            if not s:raise RuntimeError('Simulator exited')
            if '{"' in s:return json.loads(s[s.index('{"'):])

    def close(self):self.p.kill();self.p.wait()


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    check();old,new=Simulator(a.old),Simulator(a.new)
    reset=dict(op='reset',target=[1.1,0,.5],arm=True,arm_from_reset=True,max_duration_s=2)
    command=[0,18,0,0,0,2.5,0]
    try:
        states=[s.call(reset) for s in (old,new)]
        for t in range(100):
            req=dict(op='step',wheel=[1.]*4,arm=command,ms=1)
            states=[s.call(req) for s in (old,new)]
            for k in ('base','quaternion','joints','ee'):
                np.testing.assert_array_equal(states[0]['frame'][k],states[1]['frame'][k])
            np.testing.assert_array_equal(states[1]['applied_arm_torque'],command)
            np.testing.assert_array_equal(states[1]['sent_arm_torque'],command)
        assert new.call(dict(op='step',wheel=[0]*4,ms=1))['type']=='error'
        assert new.call(dict(op='step',wheel=[0]*4,arm=command,ms=2))['type']=='error'
        for effects in (dict(damping=1.),dict(damping=1.,elasticity=1.),dict(damping=1.,elasticity=1.,inertia=.001)):
            new.call(dict(reset,**effects))
            for t in range(100):
                r=new.call(dict(op='step',wheel=[1.]*4,arm=command,ms=1))
                assert r['type']=='state'
                np.testing.assert_allclose(r['sent_arm_torque'],np.array(command)+np.array(r['motor_extra_torque'])[4:11],atol=1e-10)
                assert np.isfinite(r['frame']['joints']).all()
        result=dict(zero_effect_steps_bit_identical=100,command_handoff_exact=True,missing_arm_rejected=True,batching_rejected=True,
                    motor_effects_accounted=True,delivery_causality=True,
                    limitation='Motor-equation accounting check, not full trajectory parity against the Python course wrapper')
        a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
    finally:old.close();new.close()


if __name__=='__main__':main()
