"""Verify and summarize the fixed six-case torque-only validation pilot."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/media/nm/Fast Drive/comp0245-golden11109/overnight_20261105')
OUT=ROOT/'pilot_torque_only_20261105'
BASE=ROOT/'dagger_round3/models'
MODELS=OUT/'models'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    oldaudit=read(BASE/'data_audit.json'); newaudit=read(MODELS/'data_audit.json')
    for key in ('split_sha256','manifest_sha256','rows','dagger'):
        assert oldaudit[key]==newaudit[key], key
    for name in ('driving_seed7.npz','stop_seed7.npz'):
        assert sha(MODELS/name)==sha(BASE/name)
    baseline=dict(np.load(BASE/'reaching_seed7.npz'));new=dict(np.load(MODELS/'reaching_seed7.npz'))
    assert not any(k.startswith('stop.') for k in new)
    for k in baseline:
        if k.startswith('stopnet.'):np.testing.assert_array_equal(new[k],baseline[k])
    assert new['shared.0.weight'].shape==(128,18) and new['shared.2.weight'].shape==(128,128) and new['arm.weight'].shape==(7,128)
    history=read(MODELS/'torque_seed7_history.json');summary=read(MODELS/'torque_seed7_summary.json')
    assert len(history)==1201 and all(r['train_aux']==r['val_aux']==0 for r in history)
    best=min(history,key=lambda r:r['val_total']);assert best['epoch']==summary['selected_epoch']
    ckpt=torch.load(MODELS/'torque_seed7.pt',weights_only=False);ckpt['config']=summary;torch.save(ckpt,MODELS/'torque_seed7.pt')
    cfg=read(MODELS/'config.json');cfg.update(effective_train_driving=False,effective_train_stop=False,driving_retrained=False,stop_retrained=False,torque_auxiliary_head=False,torque_auxiliary_objective=False,reference_models=str(BASE),pilot_source_sha256=sha(Path(__file__).with_name('pilot_torque_only.py')));save(MODELS/'config.json',cfg)
    source_names=['pilot_torque_only.py','pilot_torque_runtime.py','evaluate_torque_pilot.py','train_golden.py','train_full_episode.py','train_arm.py','full_run.py','evaluate_curriculum.py','render_golden.py','curriculum_sensors.py']
    provenance=dict(driving_retrained=False,stop_retrained=False,source_hashes={n:sha(Path(__file__).with_name(n)) for n in source_names},baseline_models={p.name:sha(p) for p in BASE.glob('*.npz')},new_models={p.name:sha(p) for p in MODELS.glob('*.npz')},data_audit_sha256=sha(MODELS/'data_audit.json'),training_config_sha256=sha(MODELS/'config.json'),deletion_parity=read(MODELS/'deletion_parity.json'),test_files_opened=0)
    rows=[]
    for label,folder in [('reference',OUT/'reference_val'),('torque_only',OUT/'val')]:
        completion=read(folder/'completion.json');assert completion['all_accounted'] and completion['attempts']==6
        data=read(folder/'seed7.json');assert sorted(r['dot'] for r in data)==[6,16,23,45,55,80]
        for r in data:
            assert r['status']!='execution_failure'
            pre=[f for f in r['frames'] if not f['parked']]
            q=np.array([f['joints'][4:11] for f in pre]);tau=np.array([f['applied_arm_torque'] for f in pre])
            rows.append(dict(model=label,dot=r['dot'],status=r['status'],parked=r['parked'],stop_s=r['stop_s'],closest_mm=r['min_mm'],stop_mm=r['at_network_stop_mm'],final_mm=r['end_mm'],success=r['success'],prepark_frames=len(pre),max_arm_deviation_rad=float(abs(q-q[0]).max()),maximum_command_Nm=float(abs(tau).max()),torque_clipped_fraction=float((abs(tau)>=29.999).mean())))
    with (OUT/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(OUT/'provenance.json',provenance);save(OUT/'comparison.json',rows)
    result={}
    for label in ('reference','torque_only'):
        rs=[r for r in rows if r['model']==label];stops=[r['stop_mm'] for r in rs if r['stop_mm'] is not None]
        result[label]=dict(PARK=sum(r['parked'] for r in rs),STOP=len(stops),success=sum(r['success'] for r in rs),median_STOP_mm=float(np.median(stops)) if stops else None,median_closest_mm=float(np.median([r['closest_mm'] for r in rs])),maximum_prepark_joint_deviation_rad=max(r['max_arm_deviation_rad'] for r in rs))
    result.update(selected_epoch=summary['selected_epoch'],selected_val_torque_mse=summary['selected_val_loss'],reference_selected_epoch=read(BASE/'torque_seed7_summary.json')['selected_epoch'],scope='Six fixed validation targets only; no test evaluation, MLflow or animation')
    refepoch=result['reference_selected_epoch']
    refrow=next(r for r in read(BASE/'torque_seed7_history.json') if r['epoch']==refepoch)
    result.update(reference_selected_val_torque_mse=refrow['val_command'],reference_selected_val_aux_distance_mse=refrow['val_aux'])
    save(OUT/'pilot_summary.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
