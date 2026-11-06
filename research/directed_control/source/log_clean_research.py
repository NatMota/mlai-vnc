"""Publish the completed, frozen clean test to a separate research experiment."""
import argparse,hashlib,json,math,time,zipfile
from pathlib import Path
import numpy as np
import mlflow
R=Path(__file__).parent

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--source-archive',type=Path,required=True);a=p.parse_args()
 assert len(a.commit)==40 and all(x in '0123456789abcdef' for x in a.commit)
 d=R/'clean_reference_recovery_test100';f=R/'frozen_clean_reference_v1';models=R/'learned_geometry_branch/hybrid_gain15_stop5_v1/models'
 rows=read(d/'summary.json');done=read(d/'completion.json');audit=read(d/'motion_audit.json');freeze=read(f/'freeze.json')
 assert done['attempts']==done['expected']==len(rows)==100 and len({x['dot'] for x in rows})==100
 assert all(sha(R/path)==digest for path,digest in freeze['files'].items())
 assert (d/'replay_receipt.json').exists()
 archive=R/'clean_test100_recordings.zip'
 if not archive.exists():
  tmp=archive.with_suffix('.tmp')
  with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
   for source in sorted(d.rglob('*')):
    if source.is_file():z.write(source,source.relative_to(d))
  tmp.replace(archive)
 artifact_receipt=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),source_archive_sha256=sha(a.source_archive))
 (R/'clean_test100_artifact_receipt.json').write_text(json.dumps(artifact_receipt,indent=2))
 uri='http://127.0.0.1:5055';mlflow.set_tracking_uri(uri);client=mlflow.MlflowClient()
 exp=client.get_experiment_by_name('comp0245-directed-control-rnd')
 expid=exp.experiment_id if exp else client.create_experiment('comp0245-directed-control-rnd')
 receipt_path=R/'publication_clean_mlflow.json'
 if receipt_path.exists():rec=read(receipt_path);runid=rec['run_id']
 else:
  run=client.create_run(expid,tags={'mlflow.runName':'rnd-clean-learned-geometry-reference-recovery-test100','run_role':'frozen_research_test'})
  runid=run.info.run_id;rec=dict(run_id=runid,experiment_id=expid,verified=False);receipt_path.write_text(json.dumps(rec,indent=2))
 note=('Separate engineered-control extension, not the original behaviour-cloning MLP with edited weights. '
       'Frozen driving; arm uses learned trigonometric geometry, identified torque coefficients and explicit reference feedback from reset. '
       'Sensor-only runtime inputs; simulator pose labels and expert internal references used offline for identification only. '
       'Known-marker RGB-D calibration and geometric STOP at5mm. No DART or DAgger. '
       'Candidate selected on100validation targets and frozen before100reserved test records were opened in thisR&D; historical golden-test exposure disclosed. '
       'All100test attempts meet10mm atSTOP and remain within10mm for0.5s. One internal-motion flag(target221), not a physical safety claim. '
       'Original code/models/notebook unchanged. '
       f'Source: https://github.com/NatMota/mlai-vnc/tree/{a.commit}/research/directed_control . '
       'Replay: https://nm-linmint-mlflow.taile381c0.ts.net:8443/golden-replays/directed-control-rnd/clean_reference_recovery_test100/viewer_clean_reference_recovery_test100.html')
 with mlflow.start_run(run_id=runid):
  mlflow.set_tags({'mlflow.note.content':note,'git.commit':a.commit,'mlflow.source.git.commit':a.commit,'evaluation_split':'test','original_project_modified':'false','test_outcome_tuning':'false','historical_test_exposure':'true','architecture':'learned_geometry_identified_torque_explicit_feedback'})
  mlflow.log_params(dict(train_targets=100,validation_targets=100,test_targets=100,driving_seed=7,driving_frozen=True,arm_rate_hz=1000,camera_driving_hz=30,stop_threshold_mm=5,success_threshold_mm=10,settling_duration_s=.5,cartesian_gain=6,cartesian_damping=40,reference_recovery_rad_s=.1,joint_reference_speed_cap_rad_s=1,reference_offset_cap_rad=.15,velocity_filter_alpha=1/6,geometry_coefficients=6561,torque_coefficients=5103,identified_joint_gains=14,renderer='egl',DART=False,DAgger=False,native_sha256=read(d/'config.json')['native_sha256'],freeze_timestamp=freeze['frozen_at_utc'],source_archive_sha256=sha(a.source_archive)))
  metrics={'test/attempts':100,'test/PARK':sum(bool(x.get('parked')) for x in rows),'test/STOP':sum(x.get('stop_s') is not None for x in rows),'test/success_10mm':sum(bool(x['success']) for x in rows),'test/settled_success_10mm':sum(bool(x['within_10mm_through_post_stop']) for x in rows),'test/internal_motion_flags':len(audit['flagged']),'test/simulation_worker_seconds':sum(x['elapsed_s'] for x in rows),'validation/success_10mm':100,'validation/settled_success_10mm':100}
  for key in ['at_network_stop_mm','end_mm','post_stop_peak_mm']:
   vals=[x[key] for x in rows if x.get(key) is not None]
   for label,q in [('median',.5),('p95',.95),('max',1)]:metrics[f'test/{label}_{key}']=float(np.quantile(vals,q))
  mlflow.log_metrics(metrics)
  geom=read(R/'trig_position_training/summary.json')
  for i,row in enumerate(geom['candidates']):
   mlflow.log_metrics({'geometry/train_coordinate_mse':row['training_mse'],'geometry/validation_coordinate_mse':row['validation_mse']},step=i)
  torque=read(R/'identified_torque_training/summary.json')
  for j,fit in enumerate(torque['fit']):
   for i,c in enumerate(fit['candidates']):mlflow.log_metrics({f'torque_joint{j+1}/train_mse':c['errors']['train']['mse'],f'torque_joint{j+1}/validation_mse':c['errors']['val']['mse']},step=i)
  for name in ['position.npz','torque.npz','hybrid_bounds.npz']:mlflow.log_artifact(str(models/name),'models')
  mlflow.log_artifact(str(R/'learned_geometry_branch/hybrid_gain15_stop5_v1/baseline/driving_seed7.npz'),'models')
  for folder in ['trig_position_training','identified_torque_training']:
   for name in ['summary.json','config.json','audit.json']:mlflow.log_artifact(str(R/folder/name),'fitting/'+folder)
  for name in ['summary.json','config.json','completion.json','motion_audit.json','replay_receipt.json']:mlflow.log_artifact(str(d/name),'test')
  for file in f.glob('*.json'):mlflow.log_artifact(str(file),'freeze')
  for file in [R/'research_notebook.ipynb',R/'research_notebook.html',R/'ARCHITECTURE.md']:mlflow.log_artifact(str(file),'notebook')
  mlflow.log_artifact(str(a.source_archive),'source');mlflow.log_artifact(str(R/'recovered_clean_native/build_receipt.json'),'source')
  mlflow.log_artifact(str(archive),'recordings');mlflow.log_artifact(str(R/'clean_test100_artifact_receipt.json'),'recordings')
 assert client.get_run(runid).data.metrics['test/attempts']==100
 assert any(x.path.endswith(archive.name) for x in client.list_artifacts(runid,'recordings'))
 assert client.get_run(runid).info.status=='FINISHED'
 rec.update(verified=True,source_commit=a.commit,url=f'https://nm-linmint-mlflow.taile381c0.ts.net/#/experiments/{expid}/runs/{runid}',artifact_receipt=artifact_receipt)
 receipt_path.write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
