"""Log frozen golden-data models, epoch losses and validation checks to MLflow."""
import argparse
import json
from pathlib import Path
import statistics
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
    a=ap.parse_args()
    if a.run_record.exists(): raise FileExistsError('Do not duplicate logged runs: '+str(a.run_record))
    models=HERE/'models/golden_11109'; results=HERE/'results/golden_11109'
    config=read(models/'config.json'); summary=read(models/'summary.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    client=mlflow.MlflowClient(); ids=[]
    for iteration,seed in ((34,7),(35,1),(36,2)):
        rows=read(results/f'seed{seed}.json')
        stopped=[r['at_network_stop_mm'] for r in rows if r['at_network_stop_mm'] is not None]
        parked=[100*r['park_error_vs_expert_m'] for r in rows if r['parked']]
        success=sum(r['parked'] and r['at_network_stop_mm'] is not None and r['at_network_stop_mm']<=10 for r in rows)
        url=f'https://github.com/NatMota/mlai-vnc/blob/{commit}/v2_sensors/driving_v2/golden_dataset.md#iteration-{iteration}-golden-data-seed-{seed}-results'
        description=(f'Fresh golden-data driving and reaching, seed {seed}. 100 training and 100 validation runs; '
            f'no test trajectories opened. Driving: 16→16→16, wheel-speed and time-to-PARK MSE. '
            f'Reaching: 17→64→64 torque path with auxiliary distance MSE, independent 17→32→16→1 distance stop. '
            f'FP32 Adam 0.01, equal weight per run, checkpoints chosen by validation only. '
            f'Full learned-policy validation: {sum(r["parked"] for r in rows)}/10 PARK, {len(stopped)}/10 STOP, '
            f'{success}/10 within 10 mm at STOP. This is validation, not the final 100-target test. '
            f'[Notebook and plots]({url}). All three seeds trained in {summary["elapsed_s"]:.1f} seconds on GTX 1060 CUDA. '
            f'The representative seed is {summary["representative_seed"]}, chosen by the declared validation-loss rule.')
        with mlflow.start_run(experiment_id='3',run_name=f'v2-iter{iteration}-golden-drive-reach-seed{seed}') as run:
            run_id=run.info.run_id
            mlflow.set_tags({'iteration':str(iteration),'git.commit':commit,'mlflow.source.git.commit':commit,
                            'mlflow.note.content':description,'evaluation_split':'validation','ticket':'11109',
                            'test_status':'reserved for 11110','source':'golden300-camera200'})
            mlflow.log_params(dict(seed=seed,train_runs=100,val_runs=100,test_runs_evaluated=0,validation_simulations=10,
                lr=.01,dtype='float32',device=config['device'],torch_version=config['torch'],epochs=300,stop_epochs=600,
                driving_hz=30,camera_hz=30,arm_hz=1000,drive_inputs=16,arm_inputs=17,run_weighting='equal mean per run',
                representative_seed=summary['representative_seed'],renderer='egl',park_threshold_s=.25,stop_threshold_m=.01,
                trainer_sha256=config['trainer_sha256'],sensor_sha256=config['sensor_sha256'],
                split_sha256=read(models/'data_audit.json')['split_sha256'],prepark_arm_control='native PD hold plus model gravity'))
            metrics=[]; stamp=int(time.time()*1000)
            for kind in ('driving','torque','stop'):
                history=read(models/f'{kind}_seed{seed}_history.json')
                for h in history:
                    for key,value in h.items():
                        if key not in ('epoch','seconds'): metrics.append(Metric(f'{kind}/{key}',value,stamp,int(h['epoch'])))
                chosen=read(models/f'{kind}_seed{seed}_summary.json')
                mlflow.log_params({f'{kind}_selected_epoch':chosen['selected_epoch']})
                mlflow.log_metrics({f'{kind}/selected_val_loss':chosen['selected_val_loss']})
                for suffix in ('.npz','.pt','_history.json','_summary.json'):
                    mlflow.log_artifact(str(models/f'{kind}_seed{seed}{suffix}'),'models')
            for i in range(0,len(metrics),900): client.log_batch(run_id,metrics=metrics[i:i+900])
            scores=dict(validation_attempts=len(rows),validation_parked=sum(r['parked'] for r in rows),validation_stopped=len(stopped),
                        validation_success_10mm=success,training_all_seeds_seconds=summary['elapsed_s'])
            if stopped: scores['validation_median_stop_mm']=statistics.median(stopped)
            if parked: scores['validation_median_park_error_cm']=statistics.median(parked)
            mlflow.log_metrics(scores)
            mlflow.log_artifact(str(models/f'reaching_seed{seed}.npz'),'models')
            for name in ('config.json','data_audit.json','environment.json','summary.json','dataset_manifest.json','dataset_split.json','training_preview.npz'):
                mlflow.log_artifact(str(models/name),'provenance')
            for name in ('train_golden.py','check_train_golden.py','validate_golden.py','full_run.py','report_golden.py','log_golden.py'):
                mlflow.log_artifact(str(HERE/name),'source')
            mlflow.log_artifact(str(results/f'seed{seed}.json'),'validation')
            mlflow.log_artifact(str(results/'expert_validation.json'),'validation')
            mlflow.log_artifact(str(results/'config.json'),'validation')
            mlflow.log_artifact(str(results/f'viewer_golden_seed{seed}.html'),'replay')
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
        ids.append(dict(iteration=iteration,seed=seed,run_id=run_id,verified_epoch_counts=checks))
        a.run_record.write_text(json.dumps(ids,indent=2)+'\n')
        print(ids[-1],flush=True)


if __name__=='__main__': main()
