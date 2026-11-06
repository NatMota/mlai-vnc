"""Keep only the selected recorded trajectory in the browser."""
import json


def lazy_replay(page, payload, results, seed):
    folder = results / f'viewer_seed{seed}_runs'
    folder.mkdir(exist_ok=True)
    summaries = []
    for i, run in enumerate(payload['runs']):
        name = f'run_{i}.json'
        (folder / name).write_text(json.dumps(run, separators=(',', ':')))
        summaries.append({k: run[k] for k in ('bearing_deg', 'split')})
        summaries[-1]['url'] = f'{folder.name}/{name}'
    payload['runs'] = summaries
    page = page.replace('<pre id="error">', '<button id="retry" hidden>Retry loading run</button><pre id="error">')
    page = page.replace('async function select(){', "async function selectLoaded(selectedIndex=Number($('run').value)){")
    page = page.replace("runIndex=Number($('run').value);time=0;", "runIndex=selectedIndex;time=0;")
    page = page.replace('function render(t){const id=runIndex;', 'function render(t){const id=runIndex,renderGeneration=loadGeneration;')
    page = page.replace('text(m,e)});return rendering}', 'if(renderGeneration===loadGeneration)text(m,e)});return rendering}')
    page = page.replace('ready=true;await select();', 'ready=true;await selectLoaded();')
    page = page.replace("window.addEventListener('unhandledrejection'", LOADER + "\nwindow.addEventListener('unhandledrejection'")
    page = page.replace('init().catch(fail);', 'loadInitial().catch(loadFailure);')
    return page


LOADER = r'''
const runSummaries=DATA.runs.map(r=>({...r}));
let loadGeneration=0,loadAbort=null,selection=Promise.resolve();
function loading(){
  ready=false;playing=false;generation++;
  for(const id of ['play','pause','restart','scrub'])$(id).disabled=true;
  $('status').textContent='Loading…';$('error').textContent='';$('retry').hidden=true;
  window.REPLAY_STATE={ready:false,run:Number($('run').value),playing:false};
  if(typeof clearRecordedCamera==='function')clearRecordedCamera('Loading recorded run…');
  else {const image=$('sensorImage');if(image){image.removeAttribute('src');image.hidden=true;}}
}
function loadFailure(e){if(e.name==='AbortError')return;fail(e);$('status').textContent='Run failed to load';$('retry').hidden=false;}
async function fetchRun(index,signal){
  const response=await fetch(runSummaries[index].url,{signal});
  if(!response.ok)throw new Error('Run '+runSummaries[index].bearing_deg+': HTTP '+response.status);
  const r=await response.json();
  if(!r.model?.frames?.length||!r.expert?.frames?.length)throw new Error('Run has no recorded frames');
  return r;
}
async function loadInitial(){
  loading();DATA.runs[0]=await fetchRun(0);await init();
  $('scrub').disabled=false;$('retry').onclick=()=>select();
}
async function select(){
  const id=++loadGeneration,index=Number($('run').value);
  if(loadAbort)loadAbort.abort();loadAbort=new AbortController();
  const signal=loadAbort.signal;loading();
  try{
    const r=await fetchRun(index,signal);if(id!==loadGeneration)return;
    selection=selection.catch(()=>{}).then(async()=>{
      if(id!==loadGeneration)return;await rendering;if(id!==loadGeneration)return;
      DATA.runs=runSummaries.map(x=>({...x}));DATA.runs[index]=r;
      $('run').value=index;await selectLoaded(index);if(id!==loadGeneration)return;
      ready=true;for(const key of ['play','restart','scrub'])$(key).disabled=false;
      text(frameAt('model',time),frameAt('expert',time));
    });await selection;
  }catch(e){if(id===loadGeneration)loadFailure(e);}
}
$('retry').onclick=()=>loadInitial().catch(loadFailure);
'''
