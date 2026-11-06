"""Small offline regression check; no simulation or image generation."""
import json
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from report_camera import CAMERA_SCRIPT, camera_frame, wire_camera

with TemporaryDirectory() as tmp:
    root=Path(tmp);(root/'capture.png').write_bytes(b'original')
    f=dict(time_s=1.25,camera_src='capture.png',camera_capture_step=1200)
    assert camera_frame(f,root,False)['camera_src']=='capture.png'
    assert camera_frame(f,root,True)['camera_src'].endswith('b3JpZ2luYWw=')
    assert camera_frame(dict(f,camera_src='missing.png'),root,True)['camera_src'] is None
    assert f['camera_src']=='capture.png'
page=wire_camera('<aside>\nfunction text(m,e){\nasync function select(){')
assert 'showRecordedCamera(m)' in page and 'clearRecordedCamera' in page
script='''
const assert=require('assert');let runIndex=0;
const elements={sensorImage:{hidden:false,src:'old-run',removeAttribute(){delete this.src}},cameraStatus:{}};
const $=id=>elements[id];const pending=[];
global.Image=class{set src(value){this.value=value;pending.push(this)}};
'''+CAMERA_SCRIPT+'''
showRecordedCamera({time_s:0,camera_src:'a.png'});
assert.equal(elements.sensorImage.src,undefined);assert.equal(elements.sensorImage.hidden,true);
runIndex=1;showRecordedCamera({time_s:0,camera_src:'b.png'});
pending[0].onload();assert.equal(elements.sensorImage.src,undefined);
pending[1].onload();assert.equal(elements.sensorImage.src,'b.png');assert.equal(elements.sensorImage.hidden,false);
showRecordedCamera({time_s:1,camera_src:null});assert.equal(elements.sensorImage.src,undefined);
assert.match(elements.cameraStatus.textContent,/No recorded/);
showRecordedCamera({time_s:2,camera_src:'absent.png'});pending[2].onerror();
assert.equal(elements.sensorImage.src,undefined);assert.match(elements.cameraStatus.textContent,/unavailable/);
console.log('camera regression checks passed');
'''
subprocess.run(['node','-e',script],check=True)
