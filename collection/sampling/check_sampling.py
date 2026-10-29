"""Check a between-samples PARK event and exact-repeat handling without physics."""
import json
import tempfile
from pathlib import Path
import numpy as np
from dataset import arrays_from_run, prepare, split_plan


def check():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        n = 1943  # PARK at 1.942 s, between the regular 30 Hz sample times.
        step = np.arange(n)
        time_s = step * .001

        def table(name, columns, values, categories=None):
            np.save(root / f'{name}.npy', np.column_stack(values))
            (root / f'{name}.schema.json').write_text(json.dumps(
                dict(columns=columns, categories=categories or {})))

        wheel_rates = np.array([1., 2., 4., 8.])
        q = [10 + j + time_s * (wheel_rates[j] if j < 4 else j + 1)
             for j in range(17)]
        qd = [np.full(n, rate) for rate in wheel_rates]
        table('joint_state', ['step', 'time_s'] +
              [f'q_rad_j{j:02}' for j in range(1, 18)] +
              [f'qd_rad_s_j{j:02}' for j in range(1, 5)],
              [step, time_s] + q + qd)
        wheel_names = ['front_right', 'front_left', 'back_left', 'back_right']
        commands = [np.r_[np.full(n - 1, j + 1.), 0.] for j in range(4)]
        park = np.r_[np.zeros(n - 1), 1.]
        table('base_controller', ['step', 'time_s'] +
              [f'wheel_velocity_command_rad_s_{name}' for name in wheel_names] +
              ['navigation_done'], [step, time_s] + commands + [park],
              {'navigation_done': ['False', 'True']})
        pose_columns = ([f'position_m_{a}' for a in 'xyz'] +
                        [f'quaternion_{a}' for a in 'xyzw'])
        table('base_state', pose_columns,
              [np.zeros(n) for _ in range(6)] + [np.ones(n)])

        sampled, full, fingerprint, _ = arrays_from_run(root, 180)
        assert sampled['x'].shape == (60, 14)
        assert sampled['x'].dtype == sampled['wheels'].dtype == np.float32
        assert sampled['step'][-1] == 1942 and sampled['step'][-2] == 1933
        assert sampled['time_s'][-1] == time_s[-1]
        assert sampled['park'].sum() == 1 and sampled['park'][-1, 0] == 1
        assert np.all(sampled['wheels'][-1] == 0)
        assert np.all(sampled['x'][0, :4] == 0)
        assert np.allclose(sampled['x'][-1, :4], 1.942 * wheel_rates)
        assert np.allclose(sampled['x'][:, 8], time_s[sampled['step']] * 3.75)
        assert np.allclose(sampled['x'][:, 9], time_s[sampled['step']] * 1.5)
        assert np.allclose(sampled['x'][:, 10], 3.75)
        assert np.allclose(sampled['x'][:, 11], 1.5)
        assert np.allclose(sampled['x'][:, 12:], [0, -1], atol=1e-6)
        assert full['wheel_state'].dtype == np.float64 and len(full['step']) == n

        np.savez_compressed(root / 'sample.npz', **sampled)
        record = dict(bearing_deg=180, trajectory_hash=fingerprint,
                      manifest={'termination': 'parked_command'}, source_host='check',
                      archive='sample.npz', full_rate_archive=None)
        records = [dict(record, episode_id=f'check_{i}') for i in range(2)]
        (root / 'episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        audit, rows = prepare(root)
        assert len(audit) == 2 and len(rows) == 60
        assert rows.episode_id.nunique() == 1 and rows.split.nunique() == 1
        plan = split_plan()
        assert len({r['bearing_deg'] for r in plan}) == 30
        assert [sum(r['split'] == s for r in plan)
                for s in ['train', 'test', 'holdout']] == [21, 4, 5]
    print('PASS: 30 Hz FP32 features, timestamps, terminal PARK, full-rate fields, duplicates and bearing splits')


if __name__ == '__main__':
    check()
