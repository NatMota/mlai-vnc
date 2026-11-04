"""Log official frozen-model test iterations, including original training histories. Smoke checks do not create runs."""
import argparse
import json
import re
from pathlib import Path
import subprocess
import time

import mlflow
from mlflow.entities import Metric

HERE=Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--notebook-html',type=Path,required=True)
    ap.add_argument('--run-record',type=Path,required=True)
    ap.add_argument('--source-commit',help='Pushed revision for a hash-verified remote copy without .git')
    a=ap.parse_args()
    models=HERE/'models/golden_11109'; results=HERE/'results/golden_11110'
    config=read(models/'config.json'); summary=read(models/'summary.json')
    commit=a.source_commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    assert re.fullmatch('[0-9a-f]{40}',commit), 'A complete source commit SHA is required'
    done=read(results/'completion.json')
    assert done['all_accounted'] and done['attempts']==done['expected']==300
    from report_test_golden import score, table
    client=mlflow.MlflowClient(); ids=read(a.run_record) if a.run_record.exists() else []
    for iteration,seed in ((34,7),(35,1),(36,2)):
        rows=read(results/f'seed{seed}.json')
        url=f'https://github.com/NatMota/mlai-vnc/blob/{commit}/v2_sensors/driving_v2/golden_dataset.md#iteration-{iteration}-golden-data-seed-{seed}-results'
        scores=score(seed)
        description=(f'Frozen golden-data driving and reaching, seed {seed}. Models trained on 100 training targets; '
            f'100 validation targets selected checkpoints, before this 100-target test. '
            f'Driving: 16→16→16, wheel-speed and time-to-PARK MSE. '
            f'Reaching: 17→64→64 torque path with auxiliary distance MSE, independent 17→32→16→1 distance stop. '
            f'FP32 Adam 0.01, equal weight per training run. '
            f'Full test: {scores["parked"]}/100 PARK, {scores["stopped"]}/100 STOP, '
            f'{scores["success_10mm"]}/100 within 10 mm at exact STOP; all failures retained. '
            f'[Notebook and plots]({url}). Original training of all three seeds took {summary["elapsed_s"]:.1f} seconds. '
            f'Representative seed {summary["representative_seed"]} was fixed by validation before simulation outcomes. '
            'Camera/driving 30 Hz; arm torque/encoders 1 kHz. Fixed native PD plus model gravity holds the arm before PARK. '
            'No added noise/delay/motor corrections; existing quantisation and camera error remain. '
            'Old validation-only entries are archived smoke checks. Replay ZIP includes all recorded test trajectories and camera previews.')
        record=next((r for r in ids if r['seed']==seed),None)
        if record is not None and record.get('verified'):
            print('Already verified',record,flush=True);continue
        if record is None:
            run=client.create_run('3',tags={'mlflow.runName':f'v2-iter{iteration}-golden-full-test-seed{seed}'})
            record=dict(iteration=iteration,seed=seed,run_id=run.info.run_id,verified=False)
            ids.append(record);a.run_record.write_text(json.dumps(ids,indent=2)+'\n')
        run_id=record['run_id']
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags({'iteration':str(iteration),'git.commit':commit,'mlflow.source.git.commit':commit,
                            'mlflow.note.content':description,'evaluation_split':'test','ticket':'11110','run_role':'official_test_iteration',
                            'test_status':'100 frozen attempts completed','source':'golden300-camera200'})
            mlflow.log_params(dict(seed=seed,train_runs=100,val_runs=100,test_runs_evaluated=100,validation_smoke_cases=10,
                lr=.01,dtype='float32',device=config['device'],torch_version=config['torch'],epochs=300,stop_epochs=600,
                driving_hz=30,camera_hz=30,arm_hz=1000,drive_inputs=16,arm_inputs=17,run_weighting='equal mean per run',
                representative_seed=summary['representative_seed'],renderer='egl',park_threshold_s=.25,stop_threshold_m=.01,
                trainer_sha256=config['trainer_sha256'],sensor_sha256=config['sensor_sha256'],
                split_sha256=read(models/'data_audit.json')['split_sha256'],prepark_arm_control='native PD hold plus model gravity'))
            metrics=[]; stamp=int(time.time()*1000); existing={}
            for kind in ('driving','torque','stop'):
                history=read(models/f'{kind}_seed{seed}_history.json')
                for h in history:
                    for key,value in h.items():
                        if key not in ('epoch','seconds'):
                            name=f'{kind}/{key}'
                            if name not in existing: existing[name]={m.step for m in client.get_metric_history(run_id,name)}
                            if int(h['epoch']) not in existing[name]: metrics.append(Metric(name,value,stamp,int(h['epoch'])))
                chosen=read(models/f'{kind}_seed{seed}_summary.json')
                mlflow.log_params({f'{kind}_selected_epoch':chosen['selected_epoch']})
                mlflow.log_metrics({f'{kind}/selected_val_loss':chosen['selected_val_loss']})
                for suffix in ('.npz','.pt','_history.json','_summary.json'):
                    mlflow.log_artifact(str(models/f'{kind}_seed{seed}{suffix}'),'models')
            for i in range(0,len(metrics),900): client.log_batch(run_id,metrics=metrics[i:i+900])
            import math
            mlflow.log_metrics({'test/'+k:float(v) for k,v in scores.items() if k!='seed' and math.isfinite(float(v))})
            mlflow.log_metrics(dict(training_all_seeds_seconds=summary['elapsed_s'],test_all_seeds_wall_seconds=done['elapsed_s']))
            mlflow.log_artifact(str(models/f'reaching_seed{seed}.npz'),'models')
            for name in ('config.json','data_audit.json','environment.json','summary.json','dataset_manifest.json','dataset_split.json','training_preview.npz'):
                mlflow.log_artifact(str(models/name),'provenance')
            for name in ('train_golden.py','check_train_golden.py','validate_golden.py','evaluate_golden.py','full_run.py','report_golden.py','report_test_golden.py','check_full_run_speed.py','log_golden.py'):
                mlflow.log_artifact(str(HERE/name),'source')
            mlflow.log_artifact(str(results/f'seed{seed}.json'),'test')
            mlflow.log_artifact(str(results/'expert_test.json'),'test')
            for filename in ('config.json','resume_config.json','completion.json','source_integrity.json','speed_parity.json'):
                mlflow.log_artifact(str(results/filename),'test')
            mlflow.log_artifacts(str(results/'speed_checks'),'test/speed_checks')
            csv=results/f'seed{seed}_metrics.csv';table(seed).to_csv(csv);mlflow.log_artifact(str(csv),'test')
            mlflow.log_artifact(str(results/f'replay_test_seed{seed}.zip'),'replay')
            mlflow.log_artifact(str(a.notebook_html),'notebook')
            mlflow.log_artifact(str(HERE.parent/'driving_v2/golden_dataset.ipynb'),'notebook')
        checks={}
        for kind,last in (('driving',300),('torque',300),('stop',600)):
            history=client.get_metric_history(run_id,f'{kind}/val_total')
            assert {m.step for m in history}==set(range(last+1)), (run_id,kind,'missing epochs')
            checks[kind]=len(history)
        assert client.get_run(run_id).info.status=='FINISHED'
        assert any(x.path.endswith(f'reaching_seed{seed}.npz') for x in client.list_artifacts(run_id,'models'))
        assert len(client.list_artifacts(run_id,'replay'))==1
        record.update(verified=True,verified_epoch_counts=checks)
        a.run_record.write_text(json.dumps(ids,indent=2)+'\n')
        print(record,flush=True)


if __name__=='__main__': main()
