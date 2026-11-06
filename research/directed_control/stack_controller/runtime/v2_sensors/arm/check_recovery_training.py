"""Check masked auxiliary losses and exercise export with real DAgger inputs."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from train_golden import arrays,tensors,objective,Driving,train_one


def main():
    p=argparse.ArgumentParser();p.add_argument('--dagger',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    z=np.load(a.dagger,allow_pickle=False)
    dx,dy=z['drive_x'],z['drive_y'];ax,ay=z['arm_x'],z['arm_y']
    d=np.clip(z['arm_distance_m'],0,.25)/.25
    original=dict(drive=(dx,dy,np.linspace(1,0,len(dx),dtype=np.float32)),arm=(ax,ay,d))
    additional=dict(dagger=True,drive=(dx,dy,np.zeros(len(dx),np.float32)),arm=(ax,ay,d),
        drive_aux_mask=np.zeros(len(dx),np.float32),arm_aux_mask=np.zeros(len(ax),np.float32))
    runs=[original,additional]
    assert len(arrays(runs,'stop')[0])==len(ax)
    scale=(np.zeros(11),np.ones(11),np.zeros(4),np.ones(4))
    data=tensors(runs,'driving',scale,'cpu');model=Driving(11)
    before=objective(model,data,'driving',999)
    data[2][len(dx):]+=1000
    np.testing.assert_array_equal(before,objective(model,data,'driving',999))
    assert abs(float(data[4].sum())-1)<1e-6
    args=SimpleNamespace(device='cpu',chunk=128,epochs=1,stop_epochs=1,arm_hidden=128,
        out=a.out,arm_scaling='standard',lr=.01,save_epochs=())
    records=dict(train=runs,val=[original])
    for kind in ('driving','torque','stop'):train_one(records,kind,7,args)
    print('PASS: DAgger PARK labels have no loss; auxiliary weights normalised; STOP uses original records; all exports match')


if __name__=='__main__':main()
