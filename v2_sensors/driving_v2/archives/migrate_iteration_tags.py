"""Archive excluded run, then renumber existing runs without changing evidence."""
import json
from pathlib import Path
from mlflow import MlflowClient
client=MlflowClient(tracking_uri='http://127.0.0.1:5055')
base=Path('/media/nm/Fast Drive/comp0245-golden11109/overnight_20261105')
mapping=json.loads(Path('/tmp/comp0245_iteration_mapping.json').read_text())
items=[mapping['excluded']]+mapping['active']
before={}
for x in items:
 rid=x['immutable_run_id'];local=json.loads((base/x['name']/'mlflow.json').read_text())
 assert local['run_id']==rid and local['iteration']==x['original_iteration'],x
 r=client.get_run(rid);assert r.info.experiment_id=='3'
 assert r.data.tags.get('iteration')==str(x['original_iteration']),r.data.tags
 before[rid]={'tags':r.data.tags,'params':r.data.params,'metrics':r.data.metrics,'status':r.info.status,'artifacts':[a.path for a in client.list_artifacts(rid)]}
Path('/tmp/comp0245_mlflow_before_renumber.json').write_text(json.dumps(before,indent=2))
x=items[0];rid=x['immutable_run_id']
client.set_tag(rid,'mlflow.runName','archived-excluded-original-iteration51-imperfect-park')
client.set_tag(rid,'original_iteration','51');client.set_tag(rid,'publication_status','archived_excluded')
client.set_tag(rid,'archived_original_iteration','51');client.delete_tag(rid,'iteration')
for x in mapping['active']:
 rid=x['immutable_run_id'];new=x['display_iteration']
 occupied=client.search_runs(['3'],f"tags.iteration = '{new}'")
 assert not occupied,[(r.info.run_id,r.data.tags) for r in occupied]
 client.set_tag(rid,'original_iteration',str(x['original_iteration']))
 client.set_tag(rid,'publication_status','active')
 client.set_tag(rid,'iteration',str(new))
 client.set_tag(rid,'mlflow.runName',f'v2-iter{new}-full-episode-{x["name"]}-seed7')
after={}
for x in items:
 rid=x['immutable_run_id'];r=client.get_run(rid);old=before[rid]
 assert old['params']==r.data.params and old['metrics']==r.data.metrics and old['status']==r.info.status
 assert old['artifacts']==[a.path for a in client.list_artifacts(rid)]
 after[rid]=r.data.tags
for iteration in range(41,58):
 found=client.search_runs(['3'],f"tags.iteration = '{iteration}'")
 assert len(found)==1,(iteration,[r.info.run_id for r in found])
assert 'iteration' not in client.get_run(items[0]['immutable_run_id']).data.tags
Path('/tmp/comp0245_mlflow_after_renumber.json').write_text(json.dumps(after,indent=2))
print('Verified8immutableIDs; archived original51; active41..57unique; params/metrics/status/artifacts unchanged')
