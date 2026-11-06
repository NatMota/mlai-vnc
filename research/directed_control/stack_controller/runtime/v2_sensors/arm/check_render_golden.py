"""Focused schedule/loader checks and actual live expert vs recorded camera parity.

python check_render_golden.py --self-test
python check_render_golden.py --source .../run_0.npz --camera .../camera_0.npz
  --native /path/to/native --out parity_0.json
The live expert pauses while each frame is rendered; rendering never alters the physical world.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from render_golden import frame_steps, camera_inputs, replay_trackers, png, sha


def self_test():
    for n in [1, 33, 34, 66, 67, 68, 1000]:
        expected = []; i = 0
        while round(i * 1000 / 30) < n:
            expected.append(round(i * 1000 / 30)); i += 1
        np.testing.assert_array_equal(frame_steps(n), expected)
    np.testing.assert_array_equal(frame_steps(67), [0, 33])
    det = np.array([[np.nan]*3, [1., .2, .5], [np.nan]*3])
    q = np.array([[0.]*4, [0.]*4, [.1]*4])
    drive, arm, wheel, speed = replay_trackers(det, q)
    assert np.isnan(arm[0]).all() and np.all(drive[0] == 0)
    assert arm[2, 0] < arm[1, 0] and arm[2, 2] == arm[1, 2]  # hidden = dead reckoning
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / 'train'; dest.mkdir()
        np.savez(dest / 'camera_0.npz', step=[0, 33, 67], seen=[False, True, False],
                 drive=drive, arm=arm, wheel_features=wheel, n_steps=100)
        held = camera_inputs('train', 0, [0, 32, 33, 66, 67, 99], tmp)
        assert np.all(held['arm'][:2] == 0)
        np.testing.assert_equal(held['arm'][2], held['arm'][3])
        np.testing.assert_equal(held['arm'][4], held['arm'][5])
        assert held['drive'][4, 3] == 0 and held['arm'][4, 0] > 0
        try:
            camera_inputs('train', 0, [100], tmp)
        except ValueError:
            pass
        else:
            raise AssertionError('out-of-episode sample was accepted')
    print('PASS schedule endpoints, causal holding, missing frames, hidden dead reckoning')


def parity(source, camera, native, out):
    import pybullet as p
    from camera_loop import Scene, MOUNT, see_dot, camera_to_robot, Tracker
    from sensors import WheelEncoders, wheel_features
    from wrist_camera import RENDERER
    with np.load(source, allow_pickle=False) as z:
        m = json.loads(str(z['manifest'])); q, pos, quat = z['q'], z['base_position'], z['base_quaternion']
    with np.load(camera, allow_pickle=False) as z:
        saved = dict(z)
    assert json.loads(str(saved['provenance']))['renderer'] == RENDERER
    target = m['target_position_world_m']
    cmd = [str(native / 'build-clang19/golden_camera_stream'), str(native / 'assets'), *map(str, target), '30000']
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            env=dict(os.environ, EXPERT_SKIP_VISUALS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1'))
    scene = Scene(); steps=[]; detections=[]; live_q=[]; live_drive=[]; live_arm=[]; live_wheel=[]
    drive, arm, enc = Tracker(), Tracker(keep_height=True), WheelEncoders()
    pose_max=0.; done=None; ready=None
    try:
        for line in proc.stdout:
            if 'CAMERA {' not in line:
                continue
            f = json.loads(line.split('CAMERA ', 1)[1])
            if f['type'] == 'ready':
                ready=f; assert f['joint_names'] == m['joint_names']; continue
            if f['type'] == 'done':
                done=f; break
            k = f['step']; steps.append(k); joints = np.array(f['joints']); live_q.append(joints[:4])
            if k >= len(q):
                raise AssertionError('live simulation ran beyond recording')
            pose_max=max(pose_max, float(np.max(np.abs(joints-q[k]))),
                         float(np.max(np.abs(np.array(f['base'])-pos[k]))),
                         float(np.max(np.abs(np.array(f['quaternion'])-quat[k]))))
            B=np.array(p.getMatrixFromQuaternion(f['quaternion'])).reshape(3,3)
            scene.pose(np.r_[f['base'],f['quaternion']], m['joint_names'], joints, target)
            rgb,depth,_=scene.fisheye(np.array(f['base'])+B@MOUNT,np.c_[-B[:,1],-B[:,2],B[:,0]])
            hit=see_dot(rgb,depth); point=None if hit is None else camera_to_robot(hit)
            detections.append(np.full(3,np.nan) if point is None else point)
            if len(steps)==1: enc.reset(joints[:4])
            reading=enc.read(joints[:4]); xy=drive.update(point,reading['wheel_speed']); xyz=arm.update(point,reading['wheel_speed'])
            angle=0 if xy is None else np.arctan2(xy[1],xy[0])
            live_drive.append([0.,0.,0.] if xy is None else [np.linalg.norm(xy),np.sin(angle),np.cos(angle)])
            live_arm.append(np.full(3,np.nan) if xyz is None else xyz.copy())
            live_wheel.append(wheel_features(reading['wheel_angle'],reading['wheel_speed'])[0])
            proc.stdin.write('\n'); proc.stdin.flush()
        proc.wait(timeout=10)
    finally:
        p.disconnect(scene.cid)
        if proc.poll() is None: proc.kill(); proc.wait()
    assert done is not None and done['steps']==m['steps'] and done['parking_step']==m['parking_step'], (done,m['steps'],m['parking_step'])
    np.testing.assert_array_equal(steps,saved['step'])
    errors={}
    for key, actual in [('detection',detections),('arm',live_arm),('drive',live_drive),('wheel_features',live_wheel)]:
        actual=np.array(actual); expected=saved[key]
        np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
        errors[key]=float(np.nanmax(np.abs(actual-expected)))
    # Full vector millimetres, not merely a per-coordinate tolerance.
    err_mm={k:float(np.nanmax(np.linalg.norm(np.array(v)-saved[k],axis=1))*1000)
            for k,v in [('detection',detections),('arm',live_arm)]}
    report=dict(passed=all(v<1 for v in err_mm.values()),renderer=RENDERER,frames=len(steps),
                state_max_abs_difference=pose_max,max_abs_errors=errors,max_vector_error_mm=err_mm,
                source_sha256=sha(source),camera_sha256=sha(camera),engine=ready,termination=done)
    Path(out).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert report['passed'], 'Live/recorded camera mismatch; do not release this dataset'


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--source',type=Path); ap.add_argument('--camera',type=Path); ap.add_argument('--native',type=Path); ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    if a.self_test: self_test()
    else: parity(a.source,a.camera,a.native,a.out)
