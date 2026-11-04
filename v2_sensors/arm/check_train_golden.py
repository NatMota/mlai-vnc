"""Check run weighting and chunked full-batch gradients before a long run."""
import copy
import numpy as np
import torch
from train_golden import Driving, arrays, objective, sensors, arm_readings


def main():
    torch.manual_seed(7)
    runs = []
    for n, target in ((1, 1.), (9, 3.)):
        runs.append(dict(drive=(np.zeros((n,16),np.float32), np.full((n,4),target,np.float32), np.zeros(n,np.float32))))
    data = [torch.from_numpy(x) for x in arrays(runs, 'driving')]
    zero = Driving()
    with torch.no_grad():
        for p in zero.parameters(): p.zero_()
        measured = objective(zero, data, 'driving', 3)
    np.testing.assert_allclose(measured, [5.,0.], atol=1e-6)

    data[0] = torch.randn(10,16)
    full, chunked = Driving(), Driving()
    chunked.load_state_dict(copy.deepcopy(full.state_dict()))
    objective(full, data, 'driving', 10, backward=True)
    objective(chunked, data, 'driving', 3, backward=True)
    for a,b in zip(full.parameters(),chunked.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=1e-6,rtol=1e-5)
    for model in (full,chunked): torch.optim.Adam(model.parameters(),lr=.01).step()
    for a,b in zip(full.parameters(),chunked.parameters()):
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)

    q=np.random.default_rng(7).normal(0,.01,(50,7))
    got=arm_readings(q)
    want=sensors.ArmEncoders.readings_for_run(q,.001)
    for a,b in zip(got,want): np.testing.assert_array_equal(a,b)
    print('PASS: equal run weighting, one full-batch update across chunks, runtime encoder parity')


if __name__ == '__main__': main()
