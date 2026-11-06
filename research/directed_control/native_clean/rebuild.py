"""Rebuild the frozen clean simulator using the captured commands and supplied native dependencies."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--native-project',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('clean_native_build'));a=p.parse_args()
source=Path(__file__).resolve().parent;receipt=json.loads((source/'build_receipt.json').read_text());out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
old_project=receipt['commands'][0][0].split('/.pixi/')[0]
first=receipt['commands'][0];old_source=str(Path(first[first.index('-c')+1]).parent)
for name,digest in receipt['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
for command in receipt['commands']:
 args=[arg.replace(old_project,str(a.native_project.resolve())).replace(old_source,str(out)) for arg in command]
 if '-c' in args:args[args.index('-c')+1]=str(source/Path(args[args.index('-c')+1]).name)
 subprocess.run(args,check=True,cwd=out)
binary=out/'rebuilt_sensor_curriculum';digest=hashlib.sha256(binary.read_bytes()).hexdigest()
result=dict(binary_sha256=digest,frozen_binary_sha256=receipt['reference_binary_sha256'],byte_identical=digest==receipt['reference_binary_sha256'])
(out/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
