"""Build a labelled, lazy replay from existing R&D recordings, without simulation."""
import argparse, hashlib, json, os, re, sys
from html import escape
from pathlib import Path
import numpy as np

R = Path(__file__).parent
sys.path.insert(0, str(R / 'replay_runtime/v2_sensors/arm'))
import report_golden

def ghost(path):
    with np.load(path, allow_pickle=False) as a:
        m = json.loads(str(a['manifest']))
        n, park = m['steps'], m['parking_step']
        steps = [round(i * 1000 / 30) for i in range((n * 30 + 999) // 1000)]
        valid = 0 <= park < n and 'timeout' not in m['termination'] and 'cap' not in m['termination']
        arrays = {k: a[k] for k in ['time_s', 'base_position', 'base_quaternion', 'q', 'hand_position', 'arm_torque_command', 'hand_distance']}
        frames = [dict(time_s=float(arrays['time_s'][k]), base=arrays['base_position'][k].tolist(),
                       quaternion=arrays['base_quaternion'][k].tolist(), joints=arrays['q'][k].tolist(),
                       ee=arrays['hand_position'][k].tolist(), tau=arrays['arm_torque_command'][k].tolist(),
                       parked=bool(valid and k >= park), ee_mm=float(1000 * arrays['hand_distance'][k]))
                  for k in steps if k < n]
        return dict(target=m['target_position_world_m'], park_s=park / 1000 if valid else None,
                    stop_s=(n - 1) / 1000, stop_mm=float(1000 * arrays['hand_distance'][-1]), frames=frames)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--title', required=True)
    p.add_argument('--native', type=Path, default=R / 'native')
    a = p.parse_args()
    cfg = json.loads((a.results / 'config.json').read_text())
    split = cfg['split']
    ids = cfg['ids']
    rows = [json.loads((a.results / f'run_{dot}.json').read_text()) for dot in ids]
    assert len(rows) == len(ids)
    # Presentation copies only; retain the recorded request and final sent torque.
    for row in rows:
        for frame in row['frames']:
            if 'sent_arm_torque' in frame:
                frame['tau'] = frame['sent_arm_torque']
    images = [a.results / f['camera_src'] for row in rows for f in row['frames'] if f.get('camera_src')]
    assert images and all(path.is_file() for path in images), 'Recorded camera previews must be present'
    ghosts = {str(dot): ghost(a.data / 'expert' / split / f'run_{dot}.npz') for dot in ids}
    display_split = 'test' if split == 'test' else 'validation'
    (a.results / 'seed7.json').write_text(json.dumps(rows))
    (a.results / f'expert_{display_split}.json').write_text(json.dumps(ghosts))
    os.environ['NATIVE_ROOT'] = str(a.native)
    path = report_golden.build_viewer(7, results=a.results, split=display_split, embed_camera=False)
    page = path.read_text()
    title = escape(a.title)
    page, n = re.subn(r'<title>.*?</title>', '<title>' + title + '</title>', page, count=1, flags=re.S)
    assert n == 1
    page, n = re.subn(r'(<div class="top"><h2>).*?(</h2>)', lambda m: m[1] + title + m[2], page, count=1, flags=re.S)
    assert n == 1
    page = page.replace('Learned arm torque from reset; no added PD hold or gravity compensation.',
                        'R&D controller from reset: learned geometry and torque coefficients with explicit reference feedback and damping. No expert reference at runtime.')
    page = page.replace('After STOP, the learned network runs for another 0.5 s.',
                        'After STOP, the same R&D controller continues for another 0.5 s.')
    page = page.replace('STOP prediction: ', 'Estimated gripper distance: ')
    page = page.replace('Learned driving and reaching.', 'Frozen learned driving; learned-geometry reaching with explicit feedback.')
    page = page.replace('Learned arm torques and physics: 1,000 Hz.', 'Arm control and physics: 1,000 Hz.')
    page = page.replace('No trajectory truncation.', 'No trajectory truncation. When the model finishes before the expert, its last pose and camera frame remain while the expert continues.')
    if 'condition' in cfg:
        noise_seed = cfg['condition'].get('sensor_seed')
        page = page.replace('Seed 7;', f'Driving training seed 7; sensor-noise seed {noise_seed};')
        page = page.replace('Camera inset: actual recorded frames,', 'Camera inset: current captured image; delayed control may use an earlier delivered frame. Actual recorded frames,')
    unique = a.results / f'viewer_{a.results.name}.html'
    path.rename(unique)
    path = unique
    path.write_text(page)
    receipt = dict(title=a.title, runs=len(rows), camera_previews=len(images),
                   player_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   source='Saved model trajectories and matching expert recordings; no rerendered images.')
    (a.results / 'replay_receipt.json').write_text(json.dumps(receipt, indent=2))
    print(path)

if __name__ == '__main__':
    main()
