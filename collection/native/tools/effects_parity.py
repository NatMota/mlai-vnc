"""Check one course-simulator effect in the fast collector against the course's own Python simulator.

For a condition file in effects/ (course config names), this:
  1. copies the course's robotnik_arm_task folder and switches the same options on in its robotnik_arm.json
     (motor inertia, commented out in the course code, is switched on through capture_reference.py's hook),
  2. captures one Python expert rollout with tools/capture_reference.py,
  3. runs expert_compatible with --effects on the same target,
  4. compares the two: what the expert measured, the torques sent, the true joint angles, and how the run ended.

Effects without randomness (self-collision, delay, motor damping, elasticity, inertia) should match closely.
Noise uses a different random generator in each, so it is compared by the spread of the measurement error.

usage: python tools/effects_parity.py CONDITION [--target X Y Z] [--out FOLDER]
  run from collection/native; needs the RoboEnv pixi environment (ROBOENV) and the built expert_compatible.
"""
import argparse, csv, json, os, shutil, subprocess
from pathlib import Path
import numpy as np

NATIVE = Path(__file__).resolve().parents[1]
TASK = Path(os.environ.get('COURSE_TASK', Path.home() / 'github/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task'))
ROBOENV = Path(os.environ.get('ROBOENV', Path.home() / 'github/RoboEnv'))
BUILD = Path(os.environ.get('NATIVE_BUILD', Path.home() / 'github/comp0245-native-expert'))


def course_config(condition, folder):
    # Copy only the two scripts and the configs (the task folder is 1.6 GB); link everything else, such as the models.
    folder.mkdir(parents=True)
    for item in TASK.iterdir():
        if item.name in ('robotnik_arm_task.py', 'robotnik_arm_control.py'):
            shutil.copy2(item, folder / item.name)
        elif item.name == 'configs':
            shutil.copytree(item, folder / 'configs')
        elif item.name != '__pycache__':
            (folder / item.name).symlink_to(item)
    path = folder / 'configs/robotnik_arm.json'; conf = json.loads(path.read_text()); r = conf['robot_pybullet']
    for key, value in condition.items():
        if key in ('condition', 'seed'):
            continue
        if key == 'robot_noise':
            r['robot_noise'][0].update(value)
        elif key == 'motor_inertia_coeff':
            continue                      # switched on through the capture hook, not the (commented-out) config path
        else:
            r[key] = [value]
    path.write_text(json.dumps(conf, indent=1))


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    def num(x):
        try:
            return float(x)
        except ValueError:
            return float(x == 'True') if x in ('True', 'False') else np.nan     # text columns (control modes) are not compared
    return {k: np.array([num(r[k]) for r in rows]) for k in rows[0]}


def read_npy(folder, name):
    cols = json.loads((folder / f'{name}.schema.json').read_text())['columns']
    return dict(zip(cols, np.load(folder / f'{name}.npy').T))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('condition')
    ap.add_argument('--target', nargs=3, type=float, default=[1.0759623608071864, 0.2287028598995353, 0.5])
    ap.add_argument('--out', default='/tmp/effects_parity')
    a = ap.parse_args()
    cond = json.loads((NATIVE / f'effects/{a.condition}.json').read_text())
    out = Path(a.out) / a.condition
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    course_config(cond, out / 'robotnik_arm_task')
    run = f'run_fx_{a.condition}'
    shutil.rmtree(NATIVE / 'reference' / run, ignore_errors=True)
    env = dict(os.environ, EXPERT_REFERENCE_SOURCE=str(out / 'robotnik_arm_task/robotnik_arm_task.py'),
               REFERENCE_MOTOR_INERTIA_COEFF=str(cond.get('motor_inertia_coeff', 0) if cond.get('motor_inertia') else 0))
    subprocess.run(['pixi', 'run', 'python', str(NATIVE / 'tools/capture_reference.py'), '--run', run, '--target', *map(str, a.target)],
                   cwd=ROBOENV, env=env, check=True, capture_output=True)
    ref = NATIVE / 'reference' / run
    nat = out / 'native'
    r = subprocess.run(['pixi', 'run', 'build-clang19/expert_compatible', '--assets', 'assets', '--effects', str(NATIVE / f'effects/{a.condition}.json'),
                        '--target', *map(repr, a.target), '--out', str(nat), '--binary'], cwd=BUILD, env=dict(os.environ, EXPERT_SKIP_VISUALS='1'),
                       check=True, capture_output=True, text=True)
    line = next(l for l in r.stdout.splitlines() if 'RESULT {' in l)        # engine warnings can share the line
    result = json.loads(line[line.index('RESULT {') + 7:])

    py_meas, py_cmd = read_csv(ref / 'joint_state.csv'), read_csv(ref / 'commands.csv')
    measured = read_npy(nat, 'measured_state') if (nat / 'measured_state.npy').exists() else read_npy(nat, 'joint_state')
    truth, cmd = read_npy(nat, 'joint_state'), read_npy(nat, 'commands')
    n = min(len(py_meas['step']), len(measured['step']))
    q = [f'q_rad_j{j:02}' for j in range(1, 12)]
    tau = [f'sent_torque_Nm_j{j:02}' for j in range(1, 12)]
    report = dict(condition=a.condition, target=a.target,
                  python_steps=int(len(py_meas['step'])), native_steps=int(result['steps']), native_termination=result['termination'],
                  native_final_error_m=result['final_pre_step_ee_error_m'], native_parking_step=result['parking_step'])
    if cond.get('noise_flag'):
        # Different random generators: compare the spread of measured minus true, per joint group.
        err = np.stack([measured[k][:n] - truth[k][:n] for k in q], 1)
        report.update(native_joint_noise_sd=float(err.std()), course_joint_noise_sd=cond['robot_noise']['joint_cov'])
    else:
        dq = np.max(np.abs(np.stack([py_meas[k][:n] - measured[k][:n] for k in q], 1)), 0)
        dt = np.max(np.abs(np.stack([py_cmd[k][:n] - cmd[k][:n] for k in tau], 1)), 0)
        first = next((i for i in range(n) if any(abs(py_meas[k][i] - measured[k][i]) > 1e-9 for k in q)), None)
        D = np.max(np.abs(np.stack([py_cmd[k][:n] - cmd[k][:n] for k in tau], 1)), 1)       # torque difference per step
        over = lambda x: next((i for i in range(n) if D[i] > x), None)
        report.update(max_measured_angle_diff_rad=float(dq.max()), max_sent_torque_diff_Nm=float(dt.max()), first_step_differing=first,
                      first_torque_diff_Nm=float(D[D > 0][0]) if (D > 0).any() else 0.0, first_torque_diff_step=over(0),
                      step_torque_diff_over_1e_6=over(1e-6), step_torque_diff_over_1e_3=over(1e-3),
                      python_final_error_m=float(read_csv(ref / 'kinematics.csv')['ee_distance_error_m'][-1]))
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    shutil.rmtree(nat); shutil.rmtree(out / 'robotnik_arm_task'); shutil.rmtree(ref)      # keep only the report
    print(json.dumps(report))
