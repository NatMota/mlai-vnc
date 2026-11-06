"""Four matched arm trainings, without simulation, notebook or MLflow writes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
import train_golden as tg
from pilot_torque_only import TorqueOnly, objective, infer_export
from render_golden import sha
from run_curriculum_ladder import freeze, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data', 'out', 'reference', 'native'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    if not a.worker:
        import subprocess
        import sys
        a.out.mkdir(parents=True, exist_ok=False)
        freeze(Path(__file__).resolve().parent.parent, a.out/'runtime', a.native)
        script = a.out/'runtime/v2_sensors/arm/train_paired_arms.py'
        args = [sys.executable, str(script), '--worker', '--data', str(a.data),
                '--out', str(a.out), '--reference', str(a.reference), '--native', str(a.native)]
        with (a.out/'training.log').open('x') as log:
            subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
        return
    torch.set_num_threads(4)
    started = time.monotonic()
    original = tg.ArmPolicy, tg.objective, tg.infer_export
    completed = []

    def status(stage, **extra):
        save(a.out/'status.json', dict(stage=stage, completed=completed,
                                      elapsed_s=time.monotonic()-started, **extra))

    try:
        status('waiting_for_gpu_lock')
        with open('/tmp/comp0245-validation-gpu.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            plan = dict(seeds=[1,2], architectures=['A_auxiliary', 'B_torque_only'],
                reference='Iteration 43, standardised inputs', epochs=1200, stop_epochs=600,
                lr=.01, chunk=32768, device='cuda', checkpoint_metric='val_command',
                shared_and_torque_initialisation='bit-identical within each seed pair',
                stop='trained once per seed and shared across the two architectures',
                driving='frozen iteration-43 seed-7 checkpoint for every model',
                data='golden expert training only; original 100 validation recordings select epochs',
                simulation=False, test_evaluation=False, recovery_data=False,
                notebook_writes=False, mlflow_writes=False,
                hypothesis='Compare whether auxiliary distance supervision improves arm control across matched seeds',
                expected_result='No assumed winner; later paired simulation is needed for a conclusion')
            save(a.out/'plan.json', plan)
            status('loading_data')
            records, audit = tg.load_data(a.data, driving_height=True, full_arm=True,
                                          allowed_conditions=['baseline'])
            for split in ('train','val'):
                assert len(records[split]) == 100
                assert all(r['arm'][0].shape[1] == 18 for r in records[split])
            save(a.out/'data_audit.json', audit)
            base_history = json.loads((a.reference/'torque_seed7_history.json').read_text())
            assert min(base_history,key=lambda r:r['val_command'])['epoch'] == min(base_history,key=lambda r:r['val_total'])['epoch']
            for seed in (1,2):
                torch.manual_seed(seed)
                with_aux = original[0](18, hidden=128)
                torch.manual_seed(seed)
                without_aux = TorqueOnly(18, hidden=128)
                for key,value in without_aux.state_dict().items():
                    assert torch.equal(value,with_aux.state_dict()[key])
                save(a.out/f'initialisation_seed{seed}.json', dict(seed=seed, bit_identical=True,
                     compared_parameters=list(without_aux.state_dict())))
                stop_weights = stop_meta = None
                for architecture in ('A_auxiliary','B_torque_only'):
                    out = a.out/architecture/f'seed{seed}'
                    out.mkdir(parents=True)
                    args = argparse.Namespace(out=out, arm_hidden=128, arm_scaling='standard',
                         epochs=1200, stop_epochs=600, device='cuda', seed=seed, lr=.01,
                         chunk=32768, save_epochs=(300,600,1200), checkpoint_metric='val_command')
                    save(out/'config.json', dict(**plan, seed=seed, architecture=architecture,
                         split_sha256=audit['split_sha256'], trainer_sha256=sha(__file__)))
                    status('training', architecture=architecture, seed=seed)
                    tg.ArmPolicy,tg.objective,tg.infer_export = ((TorqueOnly,objective,infer_export)
                         if architecture == 'B_torque_only' else original)
                    torque, meta = tg.train_one(records,'torque',seed,args)
                    tg.ArmPolicy,tg.objective,tg.infer_export = original
                    if architecture == 'B_torque_only':
                        assert not any(k.startswith('stop.') for k in torque)
                        torque['info_auxiliary_loss_weight'] = np.float32(0)
                        meta.update(loss_terms=['standardized_torque_mse'], auxiliary_loss_weight=0.)
                        np.savez(out/f'torque_seed{seed}.npz',**torque)
                        tg.write_json(out/f'torque_seed{seed}_summary.json',meta)
                        checkpoint = torch.load(out/f'torque_seed{seed}.pt',weights_only=False)
                        checkpoint['config'] = meta
                        torch.save(checkpoint,out/f'torque_seed{seed}.pt')
                    if architecture == 'A_auxiliary':
                        stop_weights,stop_meta = tg.train_one(records,'stop',seed,args)
                    else:
                        for suffix in ('.npz','.pt','_selected.pt','_history.json','_summary.json'):
                            src=a.out/'A_auxiliary'/f'seed{seed}'/f'stop_seed{seed}{suffix}'
                            shutil.copy2(src,out/src.name)
                    shutil.copy2(a.reference/'driving_seed7.npz',out/f'driving_seed{seed}.npz')
                    combined=dict(torque)
                    combined.update({'stopnet.'+k:v for k,v in stop_weights.items()
                                     if k.endswith(('.weight','.bias')) or k.startswith('export_')})
                    combined.update(info_stopnet='18->32->16->1 relu',
                                    info_stopnet_epochs=np.int64(stop_meta['selected_epoch']))
                    np.savez(out/f'reaching_seed{seed}.npz',**combined)
                    save(out/'summary.json',dict(seed=seed,architecture=architecture,
                         torque=meta,stop=stop_meta,driving_sha256=sha(out/f'driving_seed{seed}.npz'),
                         reaching_sha256=sha(out/f'reaching_seed{seed}.npz'),
                         status='trained; simulation not queued'))
                    completed.append(dict(seed=seed,architecture=architecture,models=str(out)))
                    status('training_completed_one')
                for filename in (f'stop_seed{seed}.npz',f'driving_seed{seed}.npz'):
                    assert sha(a.out/'A_auxiliary'/f'seed{seed}'/filename) == sha(a.out/'B_torque_only'/f'seed{seed}'/filename)
            status('complete',models_trained=4,simulation_queued=False)
    except Exception as exc:
        status('failed',error=repr(exc))
        raise


if __name__ == '__main__':
    main()
