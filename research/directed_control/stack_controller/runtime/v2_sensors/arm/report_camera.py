"""Recorded camera presentation; never substitutes or regenerates a capture."""
import base64
from pathlib import Path


def camera_frame(frame, results, embed):
    out = dict(frame)
    src = frame.get('camera_src')
    out['camera_image_time_s'] = frame.get('time_s') if src else None
    if src and embed:
        try:
            out['camera_src'] = 'data:image/png;base64,' + base64.b64encode((Path(results) / src).read_bytes()).decode('ascii')
        except OSError:
            out['camera_src'] = None
    return out


CAMERA_SECTION = '''<section><img id="sensorImage" hidden alt="Recorded front camera" style="width:100%;image-rendering:auto"><p id="cameraStatus" class="note" role="status">No recorded camera frame</p></section>'''

# A separate loader prevents an older request's load/error event painting a new run.
CAMERA_SCRIPT = '''
let cameraRequest=0, cameraKey=null;
function clearRecordedCamera(message){
  cameraRequest++;cameraKey=null;
  const image=$('sensorImage');image.hidden=true;image.removeAttribute('src');
  $('cameraStatus').textContent=message;
}
function showRecordedCamera(m){
  const src=m.camera_src||null;
  const key=runIndex+':'+(m.camera_image_time_s??m.time_s)+':'+src;
  if(key===cameraKey)return;
  clearRecordedCamera(src?'Loading recorded camera frame…':'No recorded camera frame for this state');
  cameraKey=key;
  if(!src)return;
  const request=cameraRequest, loader=new Image();
  loader.onload=()=>{if(request!==cameraRequest)return;
    const image=$('sensorImage');image.src=src;image.hidden=false;
    const capture=m.camera_image_time_s??m.time_s;
    $('cameraStatus').textContent='Recorded camera at '+Number(capture).toFixed(3)+' s'+
      (m.camera_capture_step==null?'':'; policy observation captured at '+Number(m.camera_capture_step/1000).toFixed(3)+' s');
  };
  loader.onerror=()=>{if(request!==cameraRequest)return;clearRecordedCamera('Recorded camera frame unavailable');};
  loader.src=src;
}
'''


def wire_camera(page):
    page = page.replace('<aside>', '<aside>' + CAMERA_SECTION, 1)
    page = page.replace('function text(m,e){', CAMERA_SCRIPT + '\nfunction text(m,e){showRecordedCamera(m);', 1)
    # Clear immediately, including while Plotly finishes the previous render.
    page = page.replace('async function select(){', "async function select(){clearRecordedCamera('Loading recorded run…');", 1)
    return page
