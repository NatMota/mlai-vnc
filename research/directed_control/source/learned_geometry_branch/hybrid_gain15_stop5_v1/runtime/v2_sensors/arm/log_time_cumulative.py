"""Publish completed frozen tests with their original training histories."""
import argparse,json,math,re,time
from pathlib import Path
import mlflow
from mlflow.entities import Metric
HERE=Path(__file__).resolve().parent
ROOT=HERE/'results/time_cumulative_test'
def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--commit',required=True);p.add_argument('--notebook-html',type=Path,required=True)
    a=p.parse_args();assert re.fullmatch('[0-9a-f]{40}',a.commit)
    done=read(ROOT/'completion.json');assert done['attempts']==400 and done['all_accounted']
    scores=read(ROOT/'scores.json');assert len(scores)==4
    client=mlflow.MlflowClient();record=ROOT/'mlflow_runs.json';records=read(record) if record.exists() else []
    for score in scores:
        name=score['variant'];iteration=score['iteration'];out=ROOT/name;cfg=read(out/'config.json');md=out/'models'
        assert read(out/'completion.json')['attempts']==100
        saved=next((r for r in records if r['variant']==name),None)
        if saved and saved.get('verified'):continue
        if saved is None:
            assert not client.search_runs(['3'],f"tags.iteration = '{iteration}'",max_results=1),f'Iteration {iteration} occupied'
            run=client.create_run('3',tags={'mlflow.runName':f'v2-iter{iteration}-golden-time-park-{name}-seed7','iteration':str(iteration)})
            saved=dict(variant=name,iteration=iteration,run_id=run.info.run_id,verified=False);records.append(saved)
            record.write_text(json.dumps(records,indent=2)+'\n')
        run_id=saved['run_id'];anchor=f'iteration-{iteration}-cumulative-time-to-park-{name.replace("_","-")}'
        url=f'https://github.com/NatMota/mlai-vnc/blob/{a.commit}/v2_sensors/driving_v2/golden_dataset.md#{anchor}'
        description=(f'{score["change"]}. Frozen seed-7 checkpoints trained on 100 expert runs and selected on 100 validation trajectories. '
            f'Both networks evaluated on all 100 test targets: {score["parked"]}/100 PARK, {score["stopped"]}/100 STOP, '
            f'{score["success_10mm"]}/100 within 10 mm at STOP. Median parking error {score["median_park_error_cm"]:.1f} cm among parked attempts; '
            f'median STOP error {score["median_stop_mm"]:.1f} mm among stopped attempts. '
            'No retraining or threshold changes. Time-to-PARK: 3-second normalization, 0.25-second trigger. '
            'Camera/driving 30 Hz; arm torque/physics 1 kHz; native arm hold before PARK. '
            'Targets previously evaluated for baseline iterations 34 to 36: subsequent frozen comparison, not a new blind test. '
            f'[Notebook, plots and animation]({url}). Replay archive includes all recorded attempts, ghost expert and camera previews.')
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tags({'git.commit':a.commit,'mlflow.source.git.commit':a.commit,'mlflow.note.content':description,
                'evaluation_split':'test','run_role':'official_test_iteration','source':'golden300-camera200',
                'test_reuse':'previously evaluated baseline targets; subsequent frozen comparison'})
            mlflow.log_params(dict(seed=7,train_runs=100,val_runs=100,test_runs_evaluated=100,
                driving_inputs=cfg['drive_inputs'],arm_inputs=17,driving_hidden='16,16',arm_hidden=cfg['arm_hidden'],
                driving_epochs=300,arm_epochs=1200 if name in ('longer_arm','wider_arm') else 300,
                stop_epochs=600,lr=.01,dtype='float32',run_weighting='equal mean per run',
                park_target='time',park_clip_s=3,park_threshold_s=.25,stop_threshold_m=.01,
                renderer=cfg['renderer'],camera_hz=30,driving_hz=30,arm_hz=1000,split_sha256=cfg['split_sha256']))
            metrics=[];stamp=int(time.time()*1000)
            for kind in ('driving','torque','stop'):
                h=read(md/f'{kind}_seed7_history.json');m=read(md/f'{kind}_seed7_summary.json')
                mlflow.log_param(f'{kind}_selected_epoch',m['selected_epoch'])
                for key in ('train_command','train_aux','train_total','val_command','val_aux','val_total'):
                    existing={x.step for x in client.get_metric_history(run_id,f'{kind}/{key}')}
                    metrics.extend(Metric(f'{kind}/{key}',r[key],stamp,r['epoch']) for r in h if r['epoch'] not in existing)
            for i in range(0,len(metrics),900):client.log_batch(run_id,metrics=metrics[i:i+900])
            mlflow.log_metrics({'test/'+k:float(v) for k,v in score.items() if isinstance(v,(int,float)) and math.isfinite(float(v))})
            mlflow.log_metric('all_four_test_wall_seconds',done['elapsed_s']);mlflow.log_artifacts(str(md),'models')
            for f in ('config.json','completion.json','seed7.json','expert_test.json','seed7_metrics.csv'):mlflow.log_artifact(str(out/f),'test')
            mlflow.log_artifact(str(ROOT/'config.json'),'provenance')
            mlflow.log_artifact(str(ROOT/'checks.json'),'provenance')
            for f in ('evaluate_time_cumulative.py','report_time_cumulative.py','log_time_cumulative.py','check_time_cumulative.py','full_run.py','train_golden.py'):mlflow.log_artifact(str(HERE/f),'source')
            mlflow.log_artifact(str(out/'replay_test_seed7.zip'),'replay')
            mlflow.log_artifact(str(a.notebook_html),'notebook')
            mlflow.log_artifact(str(HERE.parent/'driving_v2/golden_dataset.ipynb'),'notebook')
        counts={}
        for kind in ('driving','torque','stop'):
            expected={r['epoch'] for r in read(md/f'{kind}_seed7_history.json')};got=client.get_metric_history(run_id,f'{kind}/val_total')
            assert {r.step for r in got}==expected;counts[kind]=len(got)
        assert client.get_run(run_id).info.status=='FINISHED'
        assert client.list_artifacts(run_id,'replay')
        assert len(client.list_artifacts(run_id,'models'))==8
        saved.update(verified=True,verified_epoch_counts=counts);record.write_text(json.dumps(records,indent=2)+'\n');print(saved,flush=True)

if __name__=='__main__':main()
