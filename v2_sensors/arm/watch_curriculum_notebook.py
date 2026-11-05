"""Append completed test comparisons as review-pending analysis, never as MLflow runs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--notebook',type=Path,required=True)
    p.add_argument('--hours',type=float,default=12);a=p.parse_args();deadline=time.monotonic()+a.hours*3600
    script=Path(__file__).with_name('append_curriculum.py');failed=set()
    while time.monotonic()<deadline:
        notebook=json.loads(a.notebook.read_text());ids={c.get('id') for c in notebook['cells']}
        for directory in sorted(a.root.iterdir()):
            name=directory.name
            if not directory.is_dir() or name in {'baseline','encoder_noise'} or name in failed:continue
            if f'pending-full-{name}-heading' in ids:continue
            if not all((directory/s/'completion.json').exists() for s in ('val','test')):continue
            if any(json.loads((directory/s/'completion.json').read_text())['attempts']!=100 for s in ('val','test')):continue
            try:
                with (directory/'notebook_append.log').open('w') as log:
                    subprocess.run([sys.executable,str(script),'--name',name,'--pending','--notebook',str(a.notebook)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    for kind in ('markdown','html'):
                        subprocess.run([sys.executable,'-m','jupyter','nbconvert','--to',kind,str(a.notebook)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    subprocess.run([sys.executable,'-m','jupyter','trust',str(a.notebook)],stdout=log,stderr=subprocess.STDOUT,check=True)
                (directory/'notebook_ready.json').write_text(json.dumps(dict(status='analysis appended; visual review and Git/MLflow publication pending',notebook=str(a.notebook))))
            except Exception as e:
                failed.add(name);(directory/'notebook_error.json').write_text(json.dumps(dict(error=repr(e))))
        time.sleep(20)


if __name__=='__main__':main()
