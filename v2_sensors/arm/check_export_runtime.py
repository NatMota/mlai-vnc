"""Check exported inference, including independent STOP scaling and old archives."""
import argparse
from pathlib import Path
import sys
import numpy as np
from train_arm import policy as arm_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--dagger',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--write-reference',action='store_true',help='Run in training environment, then check in simulation environment')
    a=p.parse_args();z=np.load(a.dagger,allow_pickle=False)
    torque=dict(np.load(a.models/'torque_seed7.npz'))
    stop=dict(np.load(a.models/'stop_seed7.npz'))
    drive=dict(np.load(a.models/'driving_seed7.npz'))
    ax=z['arm_x'][::max(1,len(z['arm_x'])//256)]
    dx=z['drive_x'][::max(1,len(z['drive_x'])//256)]
    if a.write_reference:
        from train_golden import infer_export
        np.savez(a.reference,arm_x=ax,drive_x=dx,
                 torque=infer_export(torque,ax,'torque')[0],
                 stop=infer_export(stop,ax,'stop')*float(torque['ttp_clip']),
                 wheel=np.clip(infer_export(drive,dx,'driving')[0],-3,3),
                 park=infer_export(drive,dx,'driving')[1]*float(drive['ttp_clip']))
        return
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'driving_v2'))
    from camera_loop import policy as drive_policy
    reference=np.load(a.reference,allow_pickle=False)
    np.testing.assert_array_equal(ax,reference['arm_x'])
    np.testing.assert_array_equal(dx,reference['drive_x'])
    combined=dict(torque)
    combined.update({'stopnet.'+k:v for k,v in stop.items()
                     if k.endswith(('.weight','.bias')) or k.startswith('export_')})
    for i,x in enumerate(ax):
        expected=reference['torque'][i]
        predicted,dist=arm_policy(combined,x)
        np.testing.assert_allclose(predicted,expected,atol=1e-4,rtol=1e-5)
        stop_expected=reference['stop'][i]
        np.testing.assert_allclose(dist,stop_expected,atol=1e-4,rtol=1e-5)
    for i,x in enumerate(dx):
        got,park=drive_policy(drive,x)
        np.testing.assert_allclose(got,reference['wheel'][i],atol=1e-4,rtol=1e-5)
        np.testing.assert_allclose(drive_policy.ttp,reference['park'][i],atol=1e-4,rtol=1e-5)
    print('PASS: batched export and live single-row inference agree, including separate STOP scaler')


if __name__=='__main__':main()
