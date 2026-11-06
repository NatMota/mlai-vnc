"""Forward-only activation, phase intervention and analytic Jacobian diagnostics."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent
w=dict(np.load(ROOT/'baseline/reaching_seed7.npz'));trace=np.load(ROOT/'trace_baseline_dot45.npz')
row=json.loads((ROOT/'trace_baseline_dot45.json').read_text());training=json.loads((ROOT/'activation_audit.json').read_text())
x=trace['x'];last=round(row['stop_s']*1000);x=x[:last+1];park=int(np.flatnonzero(x[:,-1])[0]);names=training['feature_order']
def forward(raw,prefix,head):
 h=raw;acts=[]
 for layer in [prefix+'.0',prefix+'.2']:
  z=h@w[layer+'.weight'].T+w[layer+'.bias'];h=np.maximum(z,0);acts.append(h)
 y=h@w[head+'.weight'].T+w[head+'.bias']
 return y*(float(w['ttp_clip']) if prefix=='stopnet' else 1),acts
out={'kind':'forward-only exact student-state diagnosis; causal claims require separate edited rollouts','case':45,'steps_through_STOP':len(x),'own_PARK_step':park,'heads':{}}
for prefix,head in [('shared','arm'),('stopnet','stopnet.4')]:
 y,acts=forward(x,prefix,head);audit={}
 for i,layer in enumerate([prefix+'.0',prefix+'.2']):
  freq=(acts[i]>0).mean(axis=0);tf=np.array(training['layers'][layer]['positive_fraction']);idx=np.argsort(np.abs(freq-tf))[-8:][::-1]
  dead=training['layers'][layer]['inactive_on_all_sampled_rows']
  audit[layer]={'largest_activation_frequency_shifts':[{'unit':int(j),'train':float(tf[j]),'student':float(freq[j])} for j in idx],
   'training_inactive_units_active_on_student':[j for j in dead if acts[i][:,j].max()>0]}
 indices=np.arange(0,len(x),10);jacs=[]
 for j in indices:
  jj=(((w[head+'.weight']*(acts[1][j]>0))@w[prefix+'.2.weight'])*(acts[0][j]>0))@w[prefix+'.0.weight']
  jacs.append(jj*(float(w['ttp_clip']) if prefix=='stopnet' else 1))
 J=np.array(jacs);std=np.array(training['training_feature_std'])
 scaled=np.median(np.abs(J)*std,axis=0)
 audit['median_absolute_local_derivative_per_training_SD']=scaled.tolist()
 audit['output_feature_ranking']=[[names[j] for j in np.argsort(v)[::-1][:6]] for v in scaled]
 before=x[park].copy();before[-1]=0;after=before.copy();after[-1]=1
 yp,ap=forward(before,prefix,head);yq,aq=forward(after,prefix,head)
 audit['PARK_flag_only_counterfactual_at_actual_PARK']={'before':yp.tolist(),'after':yq.tolist(),'delta':(yq-yp).tolist(),
  'hidden_mask_changes':[[int(j) for j in np.flatnonzero((a>0)!=(b>0))] for a,b in zip(ap,aq)]}
 out['heads'][head]=audit
examples=[]
for f in row['frames']:
 step=round(f['time_s']*1000)
 if step>=len(x) or not f['parked'] or f['ee_mm']>250:continue
 examples.append({'step':step,'true_distance_mm':f['ee_mm'],'STOP_predicted_mm':1000*float(trace['STOP_distance'][step]),
  'q7_training_z':float((x[step,6]-w['info_input_mean'][6])/w['info_input_std'][6])})
out['STOP_example_scope']='Only true distances <=250mm, respecting clipped STOP training target; through STOP only'
out['largest_STOP_underestimation_examples']=sorted(examples,key=lambda a:a['true_distance_mm']-a['STOP_predicted_mm'],reverse=True)[:8]
(ROOT/'mechanism_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'park':park,'head_rankings':{k:v['output_feature_ranking'] for k,v in out['heads'].items()},'phase_effects':{k:v['PARK_flag_only_counterfactual_at_actual_PARK']['delta'] for k,v in out['heads'].items()},'STOP_underestimation':out['largest_STOP_underestimation_examples'][:2]},indent=2))
