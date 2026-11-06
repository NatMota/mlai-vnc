"""Read four frozen time-to-PARK tests and reuse the golden plots and replay player."""
import argparse
import json
import os
from pathlib import Path
import report_golden
import report_test_golden

ROOT=Path(__file__).resolve().parent/'results/time_cumulative_test'
LINEUP=json.loads((ROOT/'config.json').read_text())['lineup']
LABELS={'height':'Add camera height','no_speed':'Remove wheel-speed inputs',
        'longer_arm':'Longer arm training','wider_arm':'Wider arm layers'}


def select(name):
    if name not in LABELS:raise ValueError(name)
    report_test_golden.RESULTS=ROOT/name
    report_golden.MODELS=ROOT/name/'models'
    return report_test_golden


def scores():
    rows=[]
    for name,iteration,*_ in LINEUP:
        s=select(name).score(7);s.update(variant=name,iteration=iteration,change=LABELS[name]);rows.append(s)
    return rows


def losses(name):
    select(name)
    return report_golden.losses(7)


def reading(name):
    s=select(name).score(7)
    return (f"Iteration {dict((r[0],r[1]) for r in LINEUP)[name]}: {s['parked']}/100 PARK, "
        f"{s['stopped']}/100 STOP, {s['success_10mm']}/100 within 10 mm at STOP. "
        f"Median park error {s['median_park_error_cm']:.1f} cm among parked attempts; "
        f"median distance at STOP {s['median_stop_mm']:.1f} mm among stopped attempts. "
        f"{s['execution_failures']} execution/termination failures. "
        "Missing PARK or STOP remains a failed full task. Closest distance includes the 0.5-second observation after STOP. "
        "These are the same seed-7 checkpoints previously checked on six validation targets. "
        "The test targets were previously evaluated for baseline iterations 34 to 36, so this is a subsequent frozen comparison, not a new blind test.")


def build(name):
    report=select(name)
    page=report.build_viewer(7)
    text=page.read_text().replace('Golden data · seed 7 · model + ghost expert',f'{LABELS[name]} · seed 7 · model + ghost expert')
    page.write_text(text)
    # Package the modified page without calling the builder a second time.
    import zipfile
    archive=page.parent/'replay_test_seed7.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        z.write(page,page.name)
        for folder in sorted(page.parent.glob('seed7_dot*_camera')):
            for image in sorted(folder.glob('*.png')):z.write(image,image.relative_to(page.parent))
        z.writestr('READ_ME.txt','Extract the entire archive and open '+page.name+'. Keep the camera folders beside it. Includes all recorded test attempts, the ghost expert and camera previews. No simulator needed.\n')
    return page


def player_url(name):
    # Reuse the existing replay server. Its old root has a link to this result folder.
    base=os.environ.get('GOLDEN_REPLAY_BASE_URL','http://127.0.0.1:8899').rstrip('/')
    return base+'/time_cumulative_test/'+name+'/viewer_golden_seed7.html'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--build',action='store_true');a=p.parse_args()
    (ROOT/'scores.json').write_text(json.dumps(scores(),indent=2,allow_nan=False)+'\n')
    for name,*_ in LINEUP:
        report=select(name);report.table(7).to_csv(ROOT/name/'seed7_metrics.csv')
        if a.build:print(build(name),flush=True)
    print(json.dumps(scores(),indent=2,allow_nan=False))
