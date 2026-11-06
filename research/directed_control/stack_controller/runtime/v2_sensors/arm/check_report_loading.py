"""Run with Python; Node checks fetch ordering, retry and bounded trajectory state."""
import json
from pathlib import Path
import subprocess
import tempfile
from report_loading import LOADER, lazy_replay

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    payload={'runs':[{'bearing_deg':i,'split':'test','model':{'frames':[{'camera_src':'seed7_dot11_camera/0.png'}]},'expert':{'frames':[{}]}} for i in (11,229)]}
    page=lazy_replay('async function select(){ } init().catch(fail);',payload,root,7)
    assert len(list((root/'viewer_seed7_runs').glob('*.json')))==2
    assert 'model' not in payload['runs'][0]
    assert json.loads((root/'viewer_seed7_runs/run_0.json').read_text())['model']['frames'][0]['camera_src']=='seed7_dot11_camera/0.png'
    assert 'loadInitial().catch(loadFailure)' in page
    js=r'''
const assert=require('node:assert/strict');global.window=global;
const DATA={runs:Array.from({length:100},(_,i)=>({bearing_deg:i===0?11:i===1?229:i,split:'test',url:'runs/'+i+'.json'}))};
const elements={};const $=id=>elements[id]??=( {disabled:false,hidden:false,value:'0',removeAttribute(k){delete this[k]}} );
let ready=false,playing=false,generation=0,rendering=Promise.resolve(),time=0,runIndex=0;
function frameAt(){return {}}function text(){}function fail(e){$('error').textContent=String(e)}
async function init(){ready=true;}
async function selectLoaded(index){runIndex=index;}
const pending=[];global.fetch=(url,opts)=>new Promise((resolve,reject)=>pending.push({url,opts,resolve,reject}));
const data=index=>({bearing_deg:index===0?11:229,model:{frames:[{}]},expert:{frames:[{}]}});
const finish=(p,index)=>p.resolve({ok:true,json:async()=>data(index)});
const flush=()=>new Promise(resolve=>setImmediate(resolve));
'''+LOADER+r'''
(async()=>{
 const first=loadInitial();assert.equal(pending.length,1);assert.equal(pending[0].url,'runs/0.json');
 finish(pending[0],0);await first;assert.equal(DATA.runs.filter(r=>r.model).length,1);
 $('sensorImage').src='previous.png';$('run').value='1';const older=select();
 assert.equal($('sensorImage').src,undefined);assert.equal($('sensorImage').hidden,true);assert.equal($('status').textContent,'Loading…');
 $('run').value='0';const newer=select();finish(pending[2],0);await newer;finish(pending[1],1);await older;
 assert.equal(runIndex,0);assert.equal(DATA.runs.filter(r=>r.model).length,1);assert.equal(DATA.runs[0].bearing_deg,11);
 $('run').value='1';const broken=select();pending[3].resolve({ok:false,status:404});await broken;assert.equal($('retry').hidden,false);assert.equal(ready,false);
 const retry=$('retry').onclick();finish(pending[4],1);await retry;assert.equal(runIndex,1);assert.equal(DATA.runs[1].bearing_deg,229);assert.equal(DATA.runs.filter(r=>r.model).length,1);assert.equal(ready,true);
 console.log('PASS first-only fetch, 11/229 switching, stale fetch ignored, camera cleared, failure/retry, one resident trajectory');
})().catch(e=>{console.error(e);process.exitCode=1});
'''
    subprocess.run(['node','-e',js],check=True)
