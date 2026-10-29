"""Time the untouched headless expert. No profiler, observer or dataset export."""
import os
os.environ['ROBOENV_HEADLESS']='1'
os.environ['ROBOENV_MAX_STEPS']='40000'
os.environ['MPLBACKEND']='Agg'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import json,runpy,sys,time

source=Path(os.environ.get('EXPERT_REFERENCE_SOURCE','/Users/natanaelmota/github/LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/robotnik_arm_task.py'))
sys.path.insert(0,str(source.parent))
begin=time.perf_counter()
module=runpy.run_path(str(source),run_name='reference_timing')
loaded=time.perf_counter()
module['main'].__globals__['TARGET_EE_POS']=module['np'].array([1.1,0.,.5])
module['main']()
end=time.perf_counter()
print('TIMING '+json.dumps({'imports_s':loaded-begin,'main_s':end-loaded,'total_s':end-begin,'profiled':False,'exported_dataset':False}))
