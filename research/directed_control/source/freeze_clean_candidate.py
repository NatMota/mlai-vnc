"""Freeze the accepted validation candidate before opening reserved test records."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, shutil
R=Path(__file__).parent
F=R/'learned_geometry_branch/hybrid_gain15_stop5_v1'
SOURCE=Path('/media/nm/Fast Drive/comp0245-camera-11108/shared/COMP0245_Golden_300_2026-11-04')
OUT=R/'frozen_clean_reference_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rows=json.loads((R/'clean_reference_recovery_validation100/summary.json').read_text())
    audit=json.loads((R/'clean_reference_recovery_validation100/motion_audit.json').read_text())
    assert len(rows)==100 and len({r['dot'] for r in rows})==100
    assert all(r['success'] and r['within_10mm_through_post_stop'] for r in rows)
    assert audit['complete_cases']==100 and not audit['flagged']
    OUT.mkdir(exist_ok=False)
    paths=[R/'clean_reference_recovery_runtime.py',R/'evaluate_recorded_candidate.py',F/'evaluate_hybrid_geometry.py',
           F/'geometry_runtime.py',F/'audit_camera_fit_v2.py',F/'native/sensor_curriculum',F/'exec_native']
    paths+=list((F/'runtime').rglob('*.py'))+list((F/'models').glob('*.npz'))+list((F/'baseline').glob('*.npz'))
    hashes={str(p.relative_to(R)):sha(p) for p in paths}
    freeze=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),files=hashes,
                validation=dict(attempts=100,success=100,settled=100,internal_motion_flags=0),
                test_plan=dict(attempts=100,split='test',expected_success_at_least=95,
                               changes='None. Same model, camera, deadlines, physics and sensor/control rates.',
                               canonical='PARK and STOP with true distance<=10mm atSTOP; every attempt counted.',
                               secondary='Exact1kHz maximum distance during0.5s afterSTOP; internal velocity/clipping audit.',
                               selection='Candidate selected only on training/validation within this R&D. Golden test targets have historical project exposure.',
                               no_test_outcome_tuning=True),
                architecture='Learned geometry/identified torque with engineered feedback; separate from original BC results.')
    (OUT/'freeze.json').write_text(json.dumps(freeze,indent=2))
    # Only after the freeze receipt exists, open and verify the reserved recordings.
    data=OUT/'data';data.mkdir()
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    split=json.loads((SOURCE/'split.json').read_text())
    files=[f for f in manifest['files'] if f['path'].startswith('expert/test/')]
    assert len(files)==100 and {f['dot'] for f in files}==set(split['test'])
    receipts=[]
    for item in files:
        src=SOURCE/item['path'];dst=data/item['path'];dst.parent.mkdir(parents=True,exist_ok=True)
        assert sha(src)==item['sha256']
        shutil.copy2(src,dst)
        assert sha(dst)==item['sha256']
        receipts.append(dict(path=item['path'],sha256=item['sha256'],bytes=dst.stat().st_size))
    shutil.copy2(SOURCE/'split.json',data/'split.json')
    shutil.copy2(SOURCE/'manifest.json',data/'manifest.json')
    (OUT/'test_data_receipt.json').write_text(json.dumps(dict(copied_after_freeze=True,files=receipts),indent=2))
    assert all(sha(R/p)==h for p,h in hashes.items())
    print('Frozen candidate and100 hash-verified reserved expert records:',OUT)
if __name__=='__main__':main()
