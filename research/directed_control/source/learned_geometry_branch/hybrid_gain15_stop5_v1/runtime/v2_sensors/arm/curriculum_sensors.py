"""Noisy encoder captures and causal delivery. Truth never enters delivered features."""
from collections import deque
import numpy as np


class Delivery:
    def __init__(self, delay_ms=0):
        if not isinstance(delay_ms,int) or delay_ms<0:raise ValueError('Delay must be nonnegative whole ms')
        self.delay=delay_ms;self.pending=deque();self.latest=None;self.capture_step=None

    def capture(self,step,value):
        self.pending.append((step+self.delay,step,value))

    def read(self,step):
        while self.pending and self.pending[0][0]<=step:
            _,self.capture_step,self.latest=self.pending.popleft()
        return self.latest


class EncoderNoise:
    def __init__(self,sd,seed):
        if not np.isfinite(sd) or sd<0:raise ValueError('Invalid angle standard deviation')
        self.sd=sd;self.rng=np.random.default_rng(seed)

    def __call__(self,q):
        q=np.asarray(q,dtype=float)
        return q if not self.sd else q+self.rng.normal(0,self.sd,q.shape)


def check():
    q=Delivery(3)
    for t in range(10):
        q.capture(t,t);got=q.read(t)
        assert got==(None if t<3 else t-3)
        if got is not None:assert q.capture_step<=t-3
    q=Delivery(0);q.capture(0,'same');assert q.read(0)=='same'
    x=np.arange(7,dtype=float);np.testing.assert_array_equal(EncoderNoise(0,7)(x),x)
    a,b=EncoderNoise(.0001,7),EncoderNoise(.0001,7)
    np.testing.assert_array_equal(a(x),b(x))
    print('PASS: delivery never uses future captures; zero noise identity; reproducible noise')


if __name__=='__main__':check()
