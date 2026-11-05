"""Instrument one frozen validation replay; reject differences from its saved evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recover(root, data, native, out, dot=6):
    frozen = root / 'ladder_runtime'
    reference_dir = root / 'encoder_noise/preflight'
    config = json.loads((reference_dir / 'config.json').read_text())
    assert config['split'] == 'val' and dot in config['cases']
    assert sha(data / 'split.json') == config['split_sha256']
    hashes = json.loads((frozen / 'hashes.json').read_text())
    for relative, digest in hashes.items():
        if relative.endswith('.py') and not Path(relative).name.startswith('._'):
            assert sha(frozen / relative) == digest, relative
    for name, digest in config['source_hashes'].items():
        assert sha(frozen / 'v2_sensors/arm' / name) == digest, name
    for path, digest in config['native_hashes'].items():
        assert sha(frozen / 'native' / Path(path).name) == digest, path
    models = Path(config['models'])
    for name, digest in config['model_hashes'].items():
        assert sha(models / name) == digest, name
    os.environ.update(NATIVE_ROOT=str(native), CURRICULUM_BINARY=str(frozen / 'native/sensor_curriculum'),
                      CAMERA_RENDERER=config['renderer'],
                      CAMERA_URDF=str(native / 'assets/summit_xl_arm_description/summit_xl_complete.urdf'))
    sys.path.insert(0, str(frozen / 'v2_sensors/arm'))
    import full_run
    import curriculum_sensors
    from validate_golden import case_for
    sensors = full_run.sensors
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((reference_dir / f'seed7_dot{dot}.json').read_text())
    arrays = {'arm': [], 'wheel': []}
    captured = {}
    created = []
    old_init = curriculum_sensors.EncoderNoise.__init__
    old_noise = curriculum_sensors.EncoderNoise.__call__
    old_arm = sensors.ArmEncoders.read
    old_wheel = sensors.WheelEncoders.read

    def noise_init(self, sd, seed):
        old_init(self, sd, seed)
        self.trace_kind = ('arm', 'wheel')[len(created)]
        created.append(self.trace_kind)

    def noise(self, q):
        result = old_noise(self, q)
        captured[self.trace_kind] = (np.asarray(q).copy(), result.copy())
        return result

    def arm_read(self, q):
        result = old_arm(self, q)
        if self.dt == .001:
            truth, noisy = captured['arm']
            np.testing.assert_array_equal(q, noisy)
            arrays['arm'].append((truth, noisy, np.floor(noisy / sensors.ARM_TICK).astype(np.int64),
                                  result['arm_angle'].copy(), result['arm_speed'].copy()))
        return result

    def wheel_read(self, q):
        result = old_wheel(self, q)
        truth, noisy = captured['wheel']
        np.testing.assert_array_equal(q, noisy[:4])
        arrays['wheel'].append((truth, noisy, np.floor(q / sensors.TICK).astype(np.int64),
                                result['wheel_angle'].copy(), result['wheel_speed'].copy()))
        return result

    curriculum_sensors.EncoderNoise.__init__ = noise_init
    curriculum_sensors.EncoderNoise.__call__ = noise
    sensors.ArmEncoders.read = arm_read
    sensors.WheelEncoders.read = wheel_read
    try:
        case, _ = case_for(data, dot, 'val')
        wd = dict(np.load(models / 'driving_seed7.npz', allow_pickle=False))
        wa = dict(np.load(models / 'reaching_seed7.npz', allow_pickle=False))
        replay = full_run.one(wd, wa, dot, True, case=case,
                              camera_dir=out / f'seed7_dot{dot}_camera', condition=config['condition'])
    finally:
        curriculum_sensors.EncoderNoise.__init__ = old_init
        curriculum_sensors.EncoderNoise.__call__ = old_noise
        sensors.ArmEncoders.read = old_arm
        sensors.WheelEncoders.read = old_wheel
    # Metadata added by the evaluator is excluded; every value returned by one() is compared.
    differences = [key for key in replay if replay[key] != original.get(key)]
    (out / 'comparison.json').write_text(json.dumps({'differing_keys': differences}, indent=2))
    assert not differences, differences
    cameras = list((out / f'seed7_dot{dot}_camera').glob('*.png'))
    for path in cameras:
        assert sha(path) == sha(reference_dir / path.parent.name / path.name), path.name
    seed = np.random.SeedSequence([config['condition']['sensor_seed'], dot])
    seeds = dict(zip(('arm', 'wheel'), seed.spawn(2)))
    saved = {}
    statistics = {}
    for kind, rows in arrays.items():
        truth, noisy, counts, angle, speed = [np.stack(x) for x in zip(*rows)]
        rng = np.random.default_rng(seeds[kind])
        regenerated = np.stack([q + rng.normal(0, config['condition']['encoder_noise_sd'], q.shape) for q in truth])
        np.testing.assert_array_equal(regenerated, noisy)
        ticks = sensors.ARM_TICK if kind == 'arm' else np.full(4, sensors.TICK)
        if kind == 'wheel':
            # The original wheel stream consumes 17 draws per capture, not four.
            saved['wheel_rng_draws_per_capture'] = np.array(truth.shape[1])
            truth, noisy = truth[:, :4], noisy[:, :4]
        dt = .001 if kind == 'arm' else sensors.DT
        clean = np.floor(truth / ticks) * ticks
        initial = clean[0]
        clean_speed = np.diff(np.vstack([initial, clean]), axis=0) / dt
        counted = counts * ticks
        expected_speed = np.diff(np.vstack([initial, counted]), axis=0) / dt
        np.testing.assert_array_equal(speed, expected_speed)
        expected_angle = counted if kind == 'arm' else counted - initial
        np.testing.assert_array_equal(angle, expected_angle)
        steps = np.arange(len(rows)) if kind == 'arm' else np.rint(np.arange(len(rows)) * 1000 / 30).astype(int)
        saved.update({f'{kind}_{k}': v for k, v in dict(step=steps, truth_rad=truth, noisy_rad=noisy,
                      counts=counts, angle_rad=angle, speed_rad_s=speed,
                      clean_speed_rad_s=clean_speed, tick_rad=ticks, dt_s=np.array(dt)).items()})
        statistics[kind] = dict(captures=len(rows), angle_noise_sd_rad=np.std(noisy-truth, axis=0).tolist(),
                               velocity_error_sd_rad_s=np.std(speed-clean_speed, axis=0).tolist())
    # Confirm reconstructed wheel outputs at every originally retained camera frame.
    np.testing.assert_array_equal(saved['wheel_speed_rad_s'], [f['wheel_speed_input'] for f in original['frames']])
    np.savez_compressed(out / 'encoder_trace.npz', **saved)
    summary = dict(dot=dot, split='val', seed=7, condition=config['condition'],
                   original_record_sha256=sha(reference_dir / f'seed7_dot{dot}.json'),
                   original_config=config, frozen_runtime_hashes_sha256=sha(frozen / 'hashes.json'),
                   numpy_version=np.__version__, replay_output_exact=True,
                   retained_frames_exact=len(replay['frames']), retained_log_rows_exact=len(replay['log']),
                   camera_pngs_byte_identical=len(cameras), random_streams_exact=True,
                   count_and_velocity_equations_exact=True, statistics=statistics,
                   provenance='New instrumented deterministic replay; every retained original value and camera PNG matches exactly. '
                              'Original logs did not save 1 kHz joint states, so those intermediate states cannot be independently compared.',
                   noise_source='robotnik_arm.json robot_noise.joint_cov=0.0001; pybullet_robot_interface.py angle-reading method '
                                'passes this as np.random.normal scale (standard deviation, radians). '
                                'The course separately adds velocity noise; this encoder experiment derives velocity from noisy counts.',
                   wheel_timing='Original wheel encoder divides by fixed 1/30 s although captures alternate 33/34 ms; preserved unchanged.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def plot(directory):
    """Two sensor chains, using a fixed first 200 ms window and all samples for distributions."""
    import matplotlib.pyplot as plt
    directory = Path(directory)
    a = np.load(directory / 'encoder_trace.npz', allow_pickle=False)
    figs = []
    for kind, label in [('arm', 'Arm joint 1, 1 kHz'), ('wheel', 'Front-right wheel, 30 Hz')]:
        t = a[f'{kind}_step'] / 1000
        q, noisy = a[f'{kind}_truth_rad'][:, 0], a[f'{kind}_noisy_rad'][:, 0]
        counts = a[f'{kind}_counts'][:, 0]
        tick = a[f'{kind}_tick_rad'][0]
        mask = t <= .2
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
        axes[0, 0].plot(t[mask], q[mask], label='True angle (audit only)')
        axes[0, 0].plot(t[mask], noisy[mask], '.', label='After noise', alpha=.7)
        axes[0, 0].step(t[mask], counts[mask]*tick, where='post', label='After counting')
        axes[0, 0].set(ylabel='Angle (rad)', xlabel='Time (s)'); axes[0, 0].legend(fontsize=8)
        axes[0, 1].step(t[mask], counts[mask]-counts[0], where='post')
        axes[0, 1].set(ylabel='Encoder counts relative to first sample', xlabel='Time (s)')
        axes[1, 0].plot(t[mask], a[f'{kind}_clean_speed_rad_s'][mask, 0], label='Same states, no added noise')
        axes[1, 0].plot(t[mask], a[f'{kind}_speed_rad_s'][mask, 0], label='Noisy-count velocity', alpha=.8)
        axes[1, 0].set(ylabel='Velocity (rad/s)', xlabel='Time (s)'); axes[1, 0].legend(fontsize=8)
        errors = a[f'{kind}_speed_rad_s']-a[f'{kind}_clean_speed_rad_s']
        axes[1, 1].hist(errors.ravel(), bins=60)
        axes[1, 1].set(xlabel='Velocity error from added noise (rad/s)', ylabel='Samples across all joints and times')
        fig.suptitle(f'Iteration 42 sensor audit: {label} | validation target 6, seed 7')
        fig.savefig(directory / f'{kind}_encoder_trace.png', dpi=140)
        figs.append(fig)
    return figs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'data', 'native', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    recover(args.root, args.data, args.native, args.out)
    plot(args.out)
