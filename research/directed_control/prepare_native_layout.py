"""Connect the isolated evaluation harness to a verified clean native build and course assets."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
r=Path(__file__).resolve().parent;f=r/'source/learned_geometry_branch/hybrid_gain15_stop5_v1';expected=json.loads((r/'native_clean/build_receipt.json').read_text())['reference_binary_sha256']
assert a.binary.is_file() and hashlib.sha256(a.binary.read_bytes()).hexdigest()==expected,'Use the byte-verified clean build. Different builds need a separate parity check.'
assert (a.assets/'summit_xl_arm_description/summit_xl_complete.urdf').is_file()
assets=json.loads((r/'frozen_clean_reference_v1/native_assets_receipt.json').read_text())['files']
for name,record in assets.items():
 if name.startswith('assets/'):
  path=a.assets/Path(name).relative_to('assets')
  assert path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],name
d=f/'native';d.mkdir(exist_ok=False);(d/'sensor_curriculum').symlink_to(a.binary.resolve());(d/'assets').symlink_to(a.assets.resolve(),target_is_directory=True)
print(d)
