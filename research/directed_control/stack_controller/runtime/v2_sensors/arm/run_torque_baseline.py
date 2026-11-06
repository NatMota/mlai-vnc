"""One torque-only baseline and six validation episodes. No notebook or MLflow writes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from run_curriculum_ladder import freeze, save
from render_golden import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('out', 'data', 'reference', 'native', 'cuda'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    ids = [6, 16, 23, 45, 55, 80]
    save(a.out / 'plan.json', dict(
        seed=7, change='Remove auxiliary arm distance head and loss; retrain torque from scratch',
        reference=str(a.reference), architecture='18->128->128->7 ReLU',
        arm_scaling='standard', epochs=1200, learning_rate=.01,
        frozen_components=['driving', 'independent_STOP'],
        reference_hashes={n: sha(a.reference / n) for n in ('driving_seed7.npz', 'reaching_seed7.npz', 'stop_seed7.npz')},
        validation_ids=ids, complete_episodes=True, learned_arm_from_reset=True,
        hypothesis='Removing the redundant objective may improve torque control; six episodes are preliminary, not a success claim',
        checkpoint_selection='Minimum validation torque MSE on the 100 validation recordings',
        test_evaluation=False, recovery_collection=False, notebook_writes=False, mlflow_writes=False))
    runtime = a.out / 'runtime'
    freeze(Path(__file__).resolve().parent.parent, runtime, a.native)
    arm = runtime / 'v2_sensors/arm'
    models = a.out / 'models'
    env = dict(os.environ, NATIVE_ROOT=str(a.native), CAMERA_RENDERER='egl',
               CURRICULUM_BINARY=str(runtime / 'native/sensor_curriculum'),
               CURRICULUM_SOURCE=str(runtime / 'native/sensor_curriculum.cpp'),
               CAMERA_URDF=str(a.native / 'assets/summit_xl_arm_description/summit_xl_complete.urdf'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONUNBUFFERED='1')

    def status(stage, **extra):
        save(a.out / 'status.json', dict(stage=stage, elapsed_s=time.monotonic()-start, **extra))

    def run(stage, cmd):
        status(stage)
        with (a.out / (stage + '.log')).open('x') as log:
            subprocess.run([str(v) for v in cmd], cwd=arm, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        status('waiting_for_gpu_lock')
        with open('/tmp/comp0245-validation-gpu.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run('training', [a.cuda, arm / 'train_torque_only.py', '--data', a.data,
                '--out', models, '--driving', a.reference / 'driving_seed7.npz',
                '--frozen-stop', a.reference, '--arm-scaling', 'standard', '--epochs', 1200,
                '--arm-hidden', 128, '--seed', 7, '--device', 'cuda'])
            for component in ('driving', 'stop'):
                filename = f'{component}_seed7.npz'
                assert sha(models / filename) == sha(a.reference / filename)
            weights = dict(np.load(models / 'reaching_seed7.npz', allow_pickle=False))
            assert not any(k.startswith('stop.') for k in weights)
            assert 'stopnet.0.weight' in weights and int(weights['info_full_episode']) == 1
            history = json.loads((models / 'torque_seed7_history.json').read_text())
            assert len(history) == 1201
            assert all(r['train_aux'] == 0 and r['val_aux'] == 0 for r in history)
            # Deleting the unused old head alone must preserve runtime predictions.
            from train_arm import policy
            original = dict(np.load(a.reference / 'reaching_seed7.npz', allow_pickle=False))
            stripped = {k: v for k, v in original.items() if not k.startswith('stop.')}
            rng = np.random.default_rng(7)
            for x in rng.normal(size=(512, 18)).astype(np.float32):
                old, new = policy(original, x), policy(stripped, x)
                np.testing.assert_array_equal(old[0], new[0])
                assert old[1] == new[1]
            save(a.out / 'parity.json', dict(deletion_only_samples=512,
                torque_and_independent_stop_bit_equal=True, driving_and_stop_files_unchanged=True,
                auxiliary_training_loss_zero=True,
                trained_export_checks=json.loads((models / 'torque_seed7_summary.json').read_text())['export_checks']))
            run('validation', [a.native / '.pixi/envs/default/bin/python', arm / 'evaluate_curriculum.py',
                '--data', a.data, '--models', models, '--out', a.out / 'val',
                '--split', 'val', '--ids', *ids, '--jobs', 2])
            rows = json.loads((a.out / 'val/summary.json').read_text())
            assert len(rows) == 6
            assert not any(r['status'] == 'execution_failure' for r in rows)
            status('complete', attempts=6, parked=sum(r['parked'] for r in rows),
                   stopped=sum(r['stop_s'] is not None for r in rows),
                   next='Results ready for review; no further jobs queued')
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
