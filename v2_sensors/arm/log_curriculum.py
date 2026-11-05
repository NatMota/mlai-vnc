"""Publish a completed full-episode test, its validation results and training history."""
import argparse
import json
import math
from pathlib import Path
import re
import time
import hashlib
import mlflow
from mlflow.entities import Metric


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--iteration',type=int,required=True)
    p.add_argument('--commit',required=True);p.add_argument('--notebook',type=Path,required=True);a=p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',a.commit)
    import pandas as pd
    scores=pd.read_csv(a.run/'scores.csv').set_index('split')
    cfg=read(a.run/'test/config.json');model=a.run/'models'
    if not (model/'summary.json').exists():model=Path(cfg['models'])
    for filename,digest in cfg['model_hashes'].items():
        assert hashlib.sha256((model/filename).read_bytes()).hexdigest()==digest, filename
    for split in ('val','test'):
        c=read(a.run/split/'completion.json');assert c['attempts']==100 and c['all_accounted']
    client=mlflow.MlflowClient();record=a.run/'mlflow.json'
    audit=read(model/'data_audit.json')
    train_runs=sum(r['split']=='train' and 'excluded' not in r for r in audit['rows'])
    if (model/'extra_training_audits.json').exists():
        train_runs+=sum(r['split']=='train' and 'excluded' not in r for x in read(model/'extra_training_audits.json') for r in x['audit']['rows'])
    stop_training_runs=train_runs;train_runs+=len(audit.get('dagger',[]))
    training=read(model/'config.json')
    if record.exists():
        saved=read(record)
        if saved.get('verified'):print(saved);return
        run_id=saved['run_id']
    else:
        assert not client.search_runs(['3'],f"tags.iteration = '{a.iteration}'",max_results=1),'Iteration occupied'
        r=client.create_run('3',tags={'mlflow.runName':f'v2-iter{a.iteration}-full-episode-{a.run.name}-seed7','iteration':str(a.iteration)})
        run_id=r.info.run_id;record.write_text(json.dumps(dict(iteration=a.iteration,run_id=run_id,verified=False)))
    anchor=f'iteration-{a.iteration}-full-episode-{a.run.name.replace("_","-")}'
    url=f'https://github.com/NatMota/mlai-vnc/blob/{a.commit}/v2_sensors/driving_v2/golden_dataset.md#{anchor}'
    s=scores.loc['test']
    description=(f'Learned arm torque from reset with own PARK flag. No conventional arm hold or added gravity compensation. '
        f'{"Retrained driving and reaching" if training.get("train_driving") else "Frozen iteration-40 driving checkpoint"}. '
        f'Seed 7; {train_runs} command-training runs, 100 validation targets, all 100 test targets. '
        f'{int(s.parked)}/100 PARK, {int(s.stopped)}/100 STOP, {int(s.success_10mm)}/100 within 10 mm at STOP. '
        f'Condition: {cfg.get("condition",{})}. Previous test exposure disclosed; no fitting in evaluator. '
        f'[Notebook and complete replay]({url}).')
    if (a.run/'analysis.md').exists():
        description=(a.run/'analysis.md').read_text()+f'\n\n[Notebook and complete replay]({url}).'
    publication=read(a.run.parent/'publication_plan.json')[a.run.name]
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({'git.commit':a.commit,'mlflow.source.git.commit':a.commit,'mlflow.note.content':description,
            'evaluation_split':'test','run_role':'official_test_iteration','test_reuse':cfg['test_reuse'],
            'experiment_role':publication['role'],'parent_condition':publication['parent'] or 'iteration40',
            'scope_status':'outside_agreed_scope' if a.run.name=='imperfect_park' else 'agreed',
            'bursty_packets_tested':'false'})
        mlflow.log_params(dict(seed=7,train_runs=train_runs,stop_training_runs=stop_training_runs,val_runs=100,test_runs=100,arm_control='learned from reset',
            arm_inputs=18,driving_inputs=11,arm_hidden='128,128',driving_hidden='16,16',stop_hidden='32,16',
            camera_hz=30,driving_hz=30,arm_hz=1000,park_target='time',park_threshold_s=.25,
            stop_target='distance',stop_threshold_m=.01,condition=json.dumps(cfg.get('condition',{})),
            renderer=cfg['renderer'],split_sha256=cfg['split_sha256'],run_weighting='equal mean per run'))
        metrics=[];stamp=int(time.time()*1000)
        for kind in ('driving','torque','stop'):
            h=read(model/f'{kind}_seed7_history.json');chosen=read(model/f'{kind}_seed7_summary.json')
            mlflow.log_param(kind+'_selected_epoch',chosen['selected_epoch'])
            for key in ('train_command','train_aux','train_total','val_command','val_aux','val_total'):
                old={x.step for x in client.get_metric_history(run_id,kind+'/'+key)}
                metrics.extend(Metric(kind+'/'+key,r[key],stamp,r['epoch']) for r in h if r['epoch'] not in old)
        for i in range(0,len(metrics),900):client.log_batch(run_id,metrics=metrics[i:i+900])
        for split in ('val','test'):
            mlflow.log_metrics({split+'/'+k:float(v) for k,v in scores.loc[split].items() if math.isfinite(float(v))})
            for f in ('config.json','completion.json','summary.json'):mlflow.log_artifact(str(a.run/split/f),split)
            records=read(a.run/split/'seed7.json')
            fields=('dot','status','parked','park_s','stop_s','success','at_network_stop_mm','min_mm','end_mm','park_error_vs_expert_m','elapsed_s','timed_out')
            per_case=a.run/split/'publication_cases.csv'
            pd.DataFrame([{k:r.get(k) for k in fields} for r in records]).to_csv(per_case,index=False)
            mlflow.log_artifact(str(per_case),split)
        mlflow.log_artifacts(str(model),'models')
        mlflow.log_artifact(str(a.run/'analysis.md'),'analysis')
        for f in ('publication_plan.json','publication_scores.csv','recovery_status.json','EXPORT_REPAIR.md'):
            if (a.run.parent/f).exists():mlflow.log_artifact(str(a.run.parent/f),'analysis')
        for f in ('seed7.json','expert_test.json','seed7_metrics.csv','holding_metrics.csv'):mlflow.log_artifact(str(a.run/'test'/f),'test')
        mlflow.log_artifact(str(a.run/'test/replay_test_seed7.zip'),'replay')
        for pth in (a.notebook,a.notebook.with_suffix('.html')):mlflow.log_artifact(str(pth),'notebook')
        for name in ('full_run.py','train_golden.py','train_full_episode.py','evaluate_curriculum.py','curriculum_sensors.py','report_curriculum.py'):
            mlflow.log_artifact(str(Path(__file__).with_name(name)),'source')
        snapshot=a.run/'runtime'
        if not snapshot.exists():
            roots=[Path(p).parent.parent for p in cfg['native_hashes'] if 'runtime' in str(Path(p).parent.parent)]
            snapshot=roots[0] if roots else a.run.parent/'ladder_runtime'
        if snapshot.exists():mlflow.log_artifacts(str(snapshot),'execution_snapshot')
        if (a.run/'native').exists():mlflow.log_artifacts(str(a.run/'native'),'execution_native')
        mlflow.set_tag('source_provenance','git commit is publication state; execution_snapshot and config hashes identify the executed evaluator')
    counts={}
    for kind in ('driving','torque','stop'):
        expected={r['epoch'] for r in read(model/f'{kind}_seed7_history.json')}
        for key in ('train_command','train_aux','train_total','val_command','val_aux','val_total'):
            got=client.get_metric_history(run_id,kind+'/'+key)
            assert {r.step for r in got}==expected,(kind,key)
        counts[kind]=len(expected)
    assert client.get_run(run_id).info.status=='FINISHED'
    for folder in ('replay','notebook','models','test','val','analysis','execution_snapshot'):
        assert client.list_artifacts(run_id,folder),folder
    result=dict(iteration=a.iteration,run_id=run_id,verified=True,epoch_counts=counts,commit=a.commit)
    record.write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
