"""Check legacy input parity and causal camera-height assembly; no simulation or fitting."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'driving_v2'))
from camera_loop import driving_inputs, Tracker


def main():
    old={'shared.0.weight':np.zeros((16,16),np.float32)}
    new={'shared.0.weight':np.zeros((16,17),np.float32),'info_camera_height':1}
    rng=np.random.default_rng(7)
    for _ in range(100):
        wheel=rng.normal(size=12).astype(np.float32);dot=rng.normal(size=2);visible=bool(rng.integers(2));height=float(rng.uniform(.1,1.25))
        a=np.arctan2(dot[1],dot[0]);legacy=np.r_[wheel,[float(np.hypot(*dot)),np.sin(a),np.cos(a)],float(visible)].astype(np.float32)
        np.testing.assert_array_equal(driving_inputs(old,wheel,dot,visible),legacy)
        np.testing.assert_array_equal(driving_inputs(new,wheel,dot,visible,height),np.r_[legacy,np.float32(height)])
    tracker=Tracker(keep_height=True);wheel=np.zeros(12,np.float32)
    assert tracker.update(None,np.zeros(4)) is None
    np.testing.assert_array_equal(driving_inputs(new,wheel,None,False),np.zeros(17,np.float32))
    tracker.update(np.array([1.,.2,.8]),np.zeros(4))
    held=tracker.update(None,np.ones(4))
    x=driving_inputs(new,wheel,held[:2],False,held[2])
    assert x[-2]==0 and x[-1]==np.float32(.8)
    print('PASS: legacy 16-input parity; appended height order/metres; zero before sighting; held height while hidden')


if __name__=='__main__': main()
