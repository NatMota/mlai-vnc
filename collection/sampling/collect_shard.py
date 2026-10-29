import argparse
from pathlib import Path
from dataset import collect,save_json
import json
p=argparse.ArgumentParser();p.add_argument('--host',choices=['mint','mac'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--native-root',type=Path,required=True)
a=p.parse_args()
try:
    collect(a.native_root,a.output,50,8 if a.host=='mint' else 4,host=a.host,repetition_offset=0 if a.host=='mint' else 50)
except BaseException as e:
    path=a.output/'status.json';s=json.loads(path.read_text()) if path.exists() else {}
    s.update(stage='error',error=str(e));save_json(path,s);raise
