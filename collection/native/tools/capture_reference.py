"""Capture one unchanged expert rollout, headless. No ROS or camera observations."""
import ast
import argparse
import shlex
from dataclasses import asdict
from datetime import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

os.environ['ROBOENV_HEADLESS'] = '1'
os.environ['ROBOENV_MAX_STEPS'] = '40000'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
import csv

HERE = Path(__file__).resolve().parents[1] / 'reference'
SOURCE = Path(os.environ.get('EXPERT_REFERENCE_SOURCE', '/Users/natanaelmota/github/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/robotnik_arm_task.py'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', default='run_001', help='New run folder, for example run_002')
parser.add_argument('--target', nargs=3, type=float, metavar=('X','Y','Z'), help='Target position in world metres')
args = parser.parse_args()
if args.target is not None and not np.isfinite(args.target).all():
    parser.error('--target requires finite coordinates')
if Path(args.run).name != args.run or not args.run.startswith('run_'):
    parser.error('--run must be a folder name starting with run_')
OUT = HERE/args.run
if OUT.exists():
    raise SystemExit(f'{OUT} already exists; preserve it or choose a new output path in this script.')
OUT.mkdir()
sys.path.insert(0, str(SOURCE.parent))
source = SOURCE.read_bytes()
tree = ast.parse(source, filename=str(SOURCE))
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
headless_return = next(n for n in main.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'headless')
headless_return.body[-1] = ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]))
ast.fix_missing_locations(tree)
ns = {'__file__': str(SOURCE), '__name__': 'captured_expert'}
exec(compile(tree, str(SOURCE), 'exec'), ns)
if args.target is not None:
    ns['TARGET_EE_POS'] = np.asarray(args.target, dtype=float)

rows = {name: [] for name in ['target', 'joint_state', 'base_state', 'kinematics', 'base_controller', 'arm_controller', 'commands', 'timing']}
tensors = {'mass_matrix': [], 'arm_jacobian': [], 'q_pin': []}
latest = {}
metadata = {}

def copied(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (bool, int, float, str, np.number)):
        return value
    return None

def controller_return(frame, event, result):
    name = frame.f_code.co_name
    if event == 'return' and frame.f_code.co_filename.endswith('robotnik_arm_control.py') and name in ('base_controller', 'arm_controller'):
        latest[name] = {k: copied(v) for k, v in frame.f_locals.items() if copied(v) is not None}


def vec(row, prefix, values, labels):
    for label, value in zip(labels, np.asarray(values).ravel()):
        row[f'{prefix}_{label}'] = float(value)

cls = ns['pb'].SimInterface
# The course's motor inertia term is commented out in ServoMotorModel.compute_torque. With REFERENCE_MOTOR_INERTIA_COEFF
# set, it is switched back on exactly as written there: minus the coefficient times the previous step's acceleration.
INERTIA = float(os.environ.get('REFERENCE_MOTOR_INERTIA_COEFF', '0'))
if INERTIA:
    from simulation_and_control.controllers.servo_motor import ServoMotorModel
    _compute = ServoMotorModel.compute_torque
    ServoMotorModel.compute_torque = lambda self, cmd, q, qd, qdd, M: _compute(self, cmd, q, qd, qdd, M) - INERTIA * np.asarray(qdd)
original_step, original_mass = cls.Step, cls.ComputeMassMatrix

def capture_mass(self, *args, **kwargs):
    value = original_mass(self, *args, **kwargs)
    latest['mass_matrix'] = value.copy()
    return value

def capture_step(self, *args, **kwargs):
    f = inspect.currentframe().f_back
    v = f.f_locals
    assert f.f_code.co_name == 'main'
    i, t, dt = len(rows['timing']), float(v['current_time']), self.GetTimeStep()
    key = dict(step=i, time_s=t)
    if not metadata:
        metadata.update(joint_names=list(self.getNameActiveJoints()), dt_s=dt,
                        arm_parameters=asdict(v['arm_params']), base_parameters=asdict(v['base_params']),
                        pin_nq=v['dyn_model'].pin_model.nq, pin_nv=v['dyn_model'].pin_model.nv,
                        pin_joint_names=list(v['dyn_model'].pin_model.names),
                        pin_velocity_names=['base_vx','base_vy','base_vz','base_wx','base_wy','base_wz'] + list(v['dyn_model'].pin_model.names)[2:],
                        fixed_finger_angles=v['target_finger_angles'].tolist())
    joint_labels = [f'j{j+1:02}' for j in range(17)]
    row = dict(key)
    vec(row, 'q_rad', v['q_mes'], joint_labels)
    vec(row, 'qd_rad_s', v['qd_mes'], joint_labels)
    vec(row, 'qdd_backward_rad_s2', self.bot[0].motor_accelerations, joint_labels)
    rows['joint_state'].append(row)
    row = dict(key)
    for prefix, value, labels in [('position_m',v['base_pos'],'xyz'), ('quaternion',v['base_ori'],['x','y','z','w']), ('linear_velocity_world_m_s',self.bot[0].base_lin_vel,'xyz'), ('angular_velocity_world_rad_s',self.bot[0].base_ang_vel,'xyz'), ('linear_velocity_body_m_s',self.bot[0].base_lin_vel_body_frame,'xyz'), ('angular_velocity_body_rad_s',self.bot[0].base_ang_vel_body_frame,'xyz')]:
        vec(row, prefix, value, labels)
    row['yaw_rad'] = float(v['base_bearing'])
    rows['base_state'].append(row)
    row = dict(key, target_yaw_not_specified=True)
    vec(row, 'ee_target_world_m', ns['TARGET_EE_POS'], 'xyz')
    vec(row, 'base_standoff_goal_world_m', v['base_goal_pos'], 'xy')
    row.update(base_goal_yaw_rad=float(v['base_goal_bearing']), standoff_m=float(v['standoff']))
    rows['target'].append(row)
    row = dict(key, ee_distance_error_m=float(v['ee_dist_error']))
    vec(row, 'ee_world_m',v['ee_pos'],'xyz')
    vec(row, 'ee_error_world_m',v['ee_error'],'xyz')
    vec(row, 'shoulder_world_m',v['dyn_model'].pin_data.oMf[v['shoulder_frame_id']].translation,'xyz')
    vec(row, 'ee_rotation',v['dyn_model'].pin_data.oMf[v['ee_frame_id']].rotation,[f'r{r+1}{c+1}' for r in range(3) for c in range(3)])
    rows['kinematics'].append(row)
    b = latest['base_controller']
    row = dict(key, phase_after_call=v['base_state'].phase, navigation_done=bool(v['base_state'].done))
    for name in ['dist_to_goal','heading_to_goal','v_cmd','w_cmd','err','lw','rw']:
        row[name] = float(b[name]) if name in b else np.nan
    vec(row, 'wheel_velocity_command_rad_s',v['wheel_cmds'],['front_right','front_left','back_left','back_right'])
    rows['base_controller'].append(row)
    a = latest['arm_controller']
    row = dict(key, reach_active=bool(a['reach_active']), sigma_min=float(a.get('sigma_min',np.nan)), ik_damping=float(a.get('damping',np.nan)))
    for prefix,value in [('q_input_rad',a['q_arm']),('qd_input_rad_s',a['qd_arm']),('q_des_rad',v['q_des_arm']),('qd_des_rad_s',v['qd_des_arm']),('tau_command_Nm',v['torque_arm']),('gravity_Nm',a['tau_g'][6:13]),('pd_term_Nm',v['arm_params'].kp*(v['q_des_arm']-v['q_mes'][4:11])+v['arm_params'].kd*(v['qd_des_arm']-v['qd_mes'][4:11]))]:
        vec(row,prefix,value,[f'arm{j+1}' for j in range(7)])
    for name in ['ik_error','v_cartesian']:
        vec(row,name,a.get(name,np.full(3,np.nan)),'xyz')
    rows['arm_controller'].append(row)
    tensors['arm_jacobian'].append(a.get('J_arm',np.full((3,7),np.nan)))
    tensors['q_pin'].append(v['q_pin'].copy())
    row = dict(key)
    vec(row,'mixed_command',v['cmd_all'],joint_labels)
    begin=time.perf_counter()
    original_step(self,*args,**kwargs)
    duration=time.perf_counter()-begin
    # Servo torques for first 11 joints are actually sent to TORQUE_CONTROL.
    # Finger entries are bypassed by the position-hold helper, so mark them missing.
    sent=self.bot[0].applied_motor_commands.copy()
    sent[11:]=np.nan
    vec(row,'sent_torque_Nm',sent,joint_labels)
    for j in range(17): row[f'mode_j{j+1:02}'] = 'velocity_to_torque' if j<4 else ('torque' if j<11 else 'position_hold')
    rows['commands'].append(row)
    tensors['mass_matrix'].append(latest['mass_matrix'])
    rows['timing'].append(dict(key,post_step_time_s=(i+1)*dt,physics_step_wall_s=duration,wall_elapsed_s=time.perf_counter()-start))
    if i%2000==0: print(f'CAPTURE step={i} simulated={t:.3f}s wall={time.perf_counter()-start:.2f}s',flush=True)

start=time.perf_counter()
previous_profiler=sys.getprofile()
cls.Step, cls.ComputeMassMatrix = capture_step, capture_mass
sys.setprofile(controller_return)
try:
    state=ns['main']()
finally:
    sys.setprofile(previous_profiler)
    cls.Step,cls.ComputeMassMatrix=original_step,original_mass
wall=time.perf_counter()-start
assert SOURCE.read_bytes()==source
count=len(rows['timing'])
assert count>0 and all(len(v)==count for v in rows.values())
assert np.isfinite(np.array(tensors['mass_matrix'])).all()
for name,records in rows.items():
    with (OUT/f'{name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
np.savez_compressed(OUT/'tensors.npz',**{k:np.asarray(v) for k,v in tensors.items()})
metadata.update(target_position_world_m=ns['TARGET_EE_POS'].tolist(),created_at=datetime.now().astimezone().isoformat(),steps=count,simulated_duration_s=count*metadata['dt_s'],wall_seconds=wall,final_pre_step_time_s=float(state['current_time']),final_pre_step_ee_error_m=float(state['ee_dist_error']),termination=state['stop_message'],headless=True,source=str(SOURCE),source_sha256=hashlib.sha256(source).hexdigest(),controller_sha256=hashlib.sha256((SOURCE.parent/'robotnik_arm_control.py').read_bytes()).hexdigest(),config=json.loads((SOURCE.parent/'configs/robotnik_arm.json').read_text()),python=sys.version,command=shlex.join([sys.executable, *sys.argv]),capture_note='Non-mutating function-return observer plus Step wrapper. All ordinary table rows describe pre-step state and commands. The terminal pre-step observation is retained. Capturing adds overhead; this is not a speed benchmark.',finger_torque_note='Finger torque commands are overridden by position hold; actual position-servo torques are not captured. Sent torques are commands, not torque-sensor measurements.')
(OUT/'manifest.json').write_text(json.dumps(metadata,indent=2))
state['sim'].pybullet_client.disconnect()
print('CAPTURE_COMPLETE',json.dumps({k:metadata[k] for k in ['steps','simulated_duration_s','wall_seconds','final_pre_step_ee_error_m','termination']}),flush=True)
