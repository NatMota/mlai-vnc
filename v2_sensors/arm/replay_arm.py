"""3D player for arm network tests (test_arm.py results) against the expert, using the driving player's template.

Solid robot: the arm network's run. Cyan ghost: the expert's own run for the same bearing. Both are drawn every 1/30 s.
The side table shows the 7 arm torques of the model and the expert. A model run is cut once the gripper is more
than 1.5 m from the dot after PARK, because the simulation has blown up by then and would make the view unreadable.

The expert frames are read once from data/expert_full (recorded every 1 ms) and saved in results/expert_frames.json,
so the player can be rebuilt without the raw recordings.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'driving_30'
sys.path.insert(0, str(OLD))
import replay as old_replay
from train_arm import DATA, SPLIT, table

EXPERT = HERE / 'results/expert_frames.json'
CUT_MM = 1500


def expert_frames():
    if EXPERT.exists():
        return json.loads(EXPERT.read_text())
    out = {}
    for b in sorted(SPLIT):
        run = DATA / f'run_{b // 12 - 1}'
        m = json.loads((run / 'manifest.json').read_text())
        js, base, arm, kin = (table(run, n) for n in ('joint_state', 'base_state', 'arm_controller', 'kinematics'))
        frames = []
        for d in [round(k * 1000 / 30) for k in range(10 ** 4) if round(k * 1000 / 30) < m['steps']]:
            frames.append(dict(time_s=d / 1000, base=[float(base[f'position_m_{a}'][d]) for a in 'xyz'],
                               quaternion=[float(base[f'quaternion_{a}'][d]) for a in 'xyzw'],
                               joints=[float(js[f'q_rad_j{j:02d}'][d]) for j in range(1, 18)],
                               ee=[float(kin[f'ee_world_m_{a}'][d]) for a in 'xyz'], parked=d >= m['parking_step'],
                               tau=[round(float(arm[f'tau_command_Nm_arm{j}'][d]), 3) for j in range(1, 8)],
                               ee_mm=1000 * float(kin['ee_distance_error_m'][d])))
        out[str(b)] = dict(target=m['target_position_world_m'], park_s=m['parking_step'] / 1000, stop_s=(m['steps'] - 1) / 1000,
                           stop_mm=1000 * m['final_pre_step_ee_error_m'], frames=frames)
    EXPERT.write_text(json.dumps(out, separators=(',', ':')))
    return out


def _frame(f):
    return dict(time_s=f['time_s'], base=f['base'], quaternion=f['quaternion'], joints=f['joints'], ee=f['ee'],
                command=[0.0] * 4, parked=f['parked'], park_probability=1.0 if f['parked'] else 0.0, speed_m_s=0.0,
                tau=f['tau'] if f['tau'] is not None else [0.0] * 7, ee_mm=f['ee_mm'])


def build_viewer(model, out_name=None, expert_file=None):
    """expert_file: expert runs for dots (export_dot_cases.py) instead of the 25 original bearings."""
    expert = json.loads(Path(expert_file).read_text()) if expert_file else expert_frames()
    runs = []
    for r in json.loads((HERE / f'results/{model}.json').read_text()):
        e = expert[str(r['bearing'])]
        frames = []
        for f in r['frames']:
            frames.append(_frame(f))
            if f['parked'] and f['ee_mm'] > CUT_MM:    # only after PARK: before it the dot can be more than 1.5 m away
                break
        cut = frames[-1]['parked'] and frames[-1]['ee_mm'] > CUT_MM
        metrics = dict(model_park_s=e['park_s'], expert_park_s=e['park_s'], closest_mm=r['min_mm'], expert_mm=e['stop_mm'],
                       stop_s=r['stop_s'], expert_stop_s=e['stop_s'], at_expert_stop_mm=r['at_expert_stop_mm'], cut=cut)
        runs.append(dict(bearing_deg=r['bearing'], split=r['split'], target=e['target'], metrics=metrics,
                         model=dict(frames=frames, final=dict(time_s=frames[-1]['time_s'])),
                         expert=dict(frames=[_frame(f) for f in e['frames']], final=dict(time_s=e['frames'][-1]['time_s']))))
    joint_names = old_replay_joint_names()
    payload = dict(joint_names=joint_names, runs=runs,
                   recording_description=f'Model: arm network {model}; the expert drives and parks, then the network moves the arm. Expert: its own full run.')
    payload['geometry'] = old_replay.geometry(old_replay.native_root(), joint_names)
    replace = [
        ('Park probability ≥ 0.5 latches zero wheel commands; no recovery. The supplied controller holds the arm.',
         'The expert drives and parks in both runs. After PARK the model arm gets only the network\'s torques. A model run is cut once the gripper is 1.5 m from the dot.'),
        ('\\nPark probability ${m.park_probability.toFixed(3)} · Speed ${m.speed_m_s.toFixed(2)} / ${e.speed_m_s.toFixed(2)} m/s',
         '\\nGripper to dot: model ${m.ee_mm.toFixed(1)} mm / expert ${e.ee_mm.toFixed(1)} mm'),
        ("for(const name of ['Front right','Front left','Back left','Back right'])",
         "for(const name of ['Joint 1','Joint 2','Joint 3','Joint 4','Joint 5','Joint 6','Joint 7'])"),
        ('r.children[1].textContent=m.command[i].toFixed(2);r.children[2].textContent=e.command[i].toFixed(2)',
         'r.children[1].textContent=m.tau[i].toFixed(2);r.children[2].textContent=e.tau[i].toFixed(2)'),
        ('<th>Command · rad/s</th>', '<th>Torque · N m</th>'),
        ('Park: model ${fmt(m.model_park_s,2)} s / expert ${fmt(m.expert_park_s,2)} s<br>At park: XY difference ${fmt(m.park_xy_error_m===null?null:m.park_xy_error_m*100,1)} cm<br>Heading difference ${fmt(m.park_yaw_error_deg,1)}°<br>Final XY difference ${fmt(m.final_xy_error_m*100,1)} cm',
         'PARK at ${fmt(m.expert_park_s,2)} s (expert driving)<br>Closest gripper to dot: ${fmt(m.closest_mm,1)} mm (expert at its stop ${fmt(m.expert_mm,1)} mm)<br>Stop: model ${fmt(m.stop_s,2)} s / expert ${fmt(m.expert_stop_s,2)} s<br>At the expert stop: ${fmt(m.at_expert_stop_mm,1)} mm${m.cut?"<br>Cut: gripper passed 1.5 m":""}'),
        ('<h2>Recorded robot motion · model + ghost expert</h2>', f'<h2>Arm network {model} · model + ghost expert</h2>'),
    ]
    page = (OLD / 'replay_template.html').read_text()
    for a, b in replace:
        assert page.count(a) == 1, a
        page = page.replace(a, b)
    page = (page.replace('__PLOTLY__', (OLD / 'replay_assets/plotly.min.js').read_text())
                .replace('__DATA__', json.dumps(payload, separators=(',', ':')).replace('</', '<\\/')))
    out = HERE / f'results/viewer_{out_name or model}.html'
    out.write_text(page)
    return out


def old_replay_joint_names():
    """The simulator's 17 joint names in state order (the same list the driving player uses)."""
    return json.loads((HERE.parent / 'driving_v2/closed_loop_native/result_wheels_live_dot.json').read_text())['joint_names']


if __name__ == '__main__':
    for m in sys.argv[1:]:
        print(build_viewer(m))
