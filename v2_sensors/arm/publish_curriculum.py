"""Publish saved experiments without training, checkpoint selection or simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import nbformat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import report_curriculum as report

# Numbering follows completed official MLflow iterations 41 and 42.
RUNS = {
    'baseline': (41, None, 'New training', 'Learn arm torques from reset, including while driving. Removes the expert holding controller. Physical input scales; frozen iteration-40 driving.'),
    'encoder_noise': (42, 'baseline', 'Frozen-model stress test', 'Add Gaussian encoder-angle noise, standard deviation 0.0001 rad, before quantisation and velocity estimation. Networks unchanged.'),
    'basics_standard': (43, 'baseline', 'New training', 'Standardise arm inputs using training-set means and standard deviations. Keep the driving checkpoint and training budgets unchanged.'),
    'damping': (44, 'encoder_noise', 'Frozen-model stress test', 'Retain encoder noise and add the course motor damping term, c=1. Networks unchanged.'),
    'elasticity': (45, 'damping', 'Frozen-model stress test', 'Retain noise and damping; add the course elasticity term, k=1, towards zero joint angle. Networks unchanged.'),
    'inertia': (46, 'elasticity', 'Frozen-model stress test', 'Retain previous effects; add motor inertia J=0.001 kg m² using the previous acceleration. Networks unchanged.'),
    'delay': (47, 'inertia', 'Frozen-model stress test', 'Add a causal 3 ms observation delay to noise, damping, elasticity and inertia. Networks unchanged.'),
    'random_start_fixed': (48, 'delay', 'Frozen-model stress test', 'Retain all five disturbances and randomise starting x/y within ±0.15 m and heading within ±30 degrees. Networks unchanged.'),
    'random_start_retrained': (49, 'random_start_fixed', 'New training', 'Retrain driving and reaching on random-start expert data with physical input scales. Evaluate with random starts and all five disturbances. The expert retains its original world-origin parking convention.'),
    'random_start_retention': (50, 'random_start_retrained', 'Retention test', 'Use exactly the random-start-retrained networks, but restore original starts and clean physics. This changes the evaluation condition, not the weights.'),
    'imperfect_park': (51, 'random_start_retrained', 'Outside agreed scope', 'Retrain after omitting prepared training rows within ±0.25 s of PARK. This was an unapproved substitution for bursty packet loss. Preserve it as exploratory evidence only; it does not complete the requested packet-loss experiment.'),
    'standard_stack': (52, 'basics_standard', 'Frozen-model stress test', 'Test the clean-trained standardised model under all five disturbances and random starts. This is a separate comparison: the preceding noise ladder used the physical-scale baseline.'),
    'dart': (53, 'basics_standard', 'Recovery training', 'Add two batches of 100 perturbed expert episodes to the original data; 199 reached successfully and contributed training rows. Fixed action-noise standard deviations: wheels 0.025 rad/s, arm 0.5 N m, held for 33 ms. Labels are clean expert commands at visited states. DART-inspired fixed noise, not the full adaptive DART algorithm.'),
    'dart_stack': (54, 'dart', 'Frozen-model stress test', 'Test the DART-trained model under all five disturbances and random starts. Training used action perturbations, not the full evaluation disturbance stack.'),
    'dagger': (55, 'basics_standard', 'Recovery training', 'Three rounds of 100 student rollouts with stateful shadow-expert labels, retaining the original data and all earlier rounds. Round 3 was selected on six fixed validation cases before full testing. New DAgger rows train command outputs only; they do not train PARK, auxiliary distance or STOP targets.'),
    'dagger_stack': (56, 'dagger', 'Frozen-model stress test', 'Test the selected DAgger model under all five disturbances and random starts. This is the same checkpoint as its clean test.'),
    'dart_dagger': (57, 'dart', 'Recovery training', 'Retain original and both DART datasets, then aggregate three rounds of 100 student rollouts with shadow-expert command labels. Round 3 was selected on the six validation cases. New DAgger rows do not train PARK or STOP. Round 2 export was repaired and rerun without loosening the numerical tolerance; the selected model is round 3.'),
    'dart_dagger_stack': (58, 'dart_dagger', 'Frozen-model stress test', 'Test the selected combined model under all five disturbances and random starts. No bursty-packet loss, occlusion or direct weight perturbation is included.'),
}


def manifest(refresh=False):
    cached=report.ROOT/'publication_scores.csv'
    if cached.exists() and not refresh:
        return pd.read_csv(cached)
    rows=[]
    reference_ids={}
    for name,(iteration,parent,role,change) in RUNS.items():
        for split in ('val','test'):
            table=report.select(name,split).table(7)
            ids=set(table.index)
            assert len(ids)==100
            if split in reference_ids: assert ids==reference_ids[split]
            reference_ids[split]=ids
            score=report.select(name,split).score(7)
            assert score['execution_failures']==0, (name,split,'inspect failed execution before publication')
            rows.append(dict(iteration=iteration,name=name,split=split,parent=parent,role=role,**score))
    assert not reference_ids['val'] & reference_ids['test']
    return pd.DataFrame(rows)


def detail(name):
    iteration,parent,role,change=RUNS[name]
    score=report.scores(name);s=score.loc['test'];v=score.loc['val']
    model=report.models(name)
    training=json.loads((model/'config.json').read_text())
    selected={k:json.loads((model/f'{k}_seed7_summary.json').read_text())['selected_epoch'] for k in ('driving','torque','stop')}
    text=f'**What changed:** {change}\n\n**Type:** {role}. '
    text+='Seed 7 only; 100 validation and 100 test targets, with both learned networks controlling the complete episode. '
    text+='No conventional arm holding controller or extra gravity compensation is active.\n\n'
    text+=f'**Result:** test PARK {int(s.parked)}/100, STOP {int(s.stopped)}/100, full success {int(s.success_10mm)}/100; validation full success {int(v.success_10mm)}/100. '
    text+=f'Median parking error is {s.median_park_error_cm:.2f} cm among {int(s.parked)} parked attempts. Median STOP distance is {s.median_stop_mm:.1f} mm among {int(s.stopped)} stopped attempts. '
    text+=f'{100-int(s.parked)} attempts did not PARK; {int(s.parked-s.stopped)} parked without STOP. Neither is silently discarded from success.\n\n'
    if parent:
        p=report.scores(parent).loc['test']
        a=report.select(name).table(7);b=report.select(parent).table(7)
        both=a.join(b,lsuffix='_new',rsuffix='_parent')
        parked=both[both.parked_new & both.parked_parent]
        stopped=both[both.stop_s_new.notna() & both.stop_s_parent.notna()]
        text+=f'**Comparison with iteration {RUNS[parent][0]} ({parent}):** PARK count changes by {int(s.parked-p.parked):+d}; STOP by {int(s.stopped-p.stopped):+d}; successes by {int(s.success_10mm-p.success_10mm):+d}. '
        text+=f'On the same {len(parked)} targets parked by both, the median paired change in parking error is {(parked.park_error_cm_new-parked.park_error_cm_parent).median():+.2f} cm. '
        text+=f'On the same {len(stopped)} targets stopped by both, the median paired change at STOP is {(stopped.at_stop_mm_new-stopped.at_stop_mm_parent).median():+.1f} mm. Negative changes mean smaller errors. '
        text+='These paired subsets exclude failures, so read them alongside the counts, not as overall success rates.\n\n'
    text+=f'**Training provenance:** selected epochs: driving {selected["driving"]}, torque {selected["torque"]}, STOP {selected["stop"]}. '
    text+='The curves below belong to the actual checkpoint; frozen stress tests reuse their parent training histories and do not represent new training. '
    text+='Driving predicts four wheel speeds and time to PARK (trigger 0.25 s). The arm predicts seven torques; its STOP network predicts distance (trigger 10 mm). '
    text+='Command losses are MSE, with separate auxiliary MSE terms; run weights are equal. Saved configuration and histories give scaling and exact coefficients.\n\n'
    text+='**Limits:** these test targets have been viewed in earlier experiments. This is a reused-test comparison, not an untouched final estimate. One seed cannot establish seed robustness. '
    text+='Success requires PARK, STOP and distance at STOP ≤10 mm. Closest approach and later drift are diagnostics, not substitutes for that criterion. '
    text+='The complete per-target table, pre-PARK arm analysis, all loss terms, STOP/drift plots and 100-replay player follow below.'
    if name=='imperfect_park': text+='\n\n**Scope correction:** bursty packets remain untested. Do not describe this row deletion as packet loss or mark that roadmap task complete.'
    if name in ('encoder_noise','damping','elasticity','inertia','delay'):
        text+='\n\n**Interpretation:** this measures cumulative sensitivity of the original physical-scale model. It does not isolate the same effect on the standardised model. Differencing noisy angle readings can amplify velocity noise; this is a plausible mechanism, not a demonstrated sole cause.'
    if name in ('dart','dagger','dart_dagger'):
        text+='\n\n**Interpretation:** compare both approach accuracy and completion. A smaller STOP median can coexist with fewer STOPs. The six-case development scores are not estimates of performance across all 100 test targets.'
    return text


def overview_plot():
    data=manifest();test=data[data.split=='test'].sort_values('iteration')
    labels=[f'{r.iteration} {r.name}' for r in test.itertuples()]
    fig,axes=plt.subplots(1,3,figsize=(18,10),sharey=True)
    y=np.arange(len(test))
    axes[0].barh(y,test.parked,color='#4c78a8',label='PARK')
    axes[0].barh(y,test.stopped,color='#f58518',height=.5,label='STOP')
    axes[0].scatter(test.success_10mm,y,color='black',label='Within 10 mm at STOP',zorder=3)
    axes[0].set_yticks(y,labels);axes[0].invert_yaxis();axes[0].set_xlim(0,105);axes[0].legend(fontsize=8)
    axes[0].set_xlabel('Attempts out of 100')
    axes[1].scatter(test.median_park_error_cm,y);axes[1].set_xlabel('Median PARK error, cm (parked only)')
    axes[2].scatter(test.median_stop_mm,y);axes[2].set_xlabel('Median STOP distance, mm (stopped only)')
    fig.suptitle('One-seed full-run comparisons; frozen-model stress tests and retraining are different experiments')
    fig.tight_layout();return fig


def threshold_plot():
    fig,ax=plt.subplots(figsize=(10,5))
    thresholds=np.linspace(0,1000,201)
    for name in ('baseline','basics_standard','dart','dagger','dart_dagger','dart_dagger_stack'):
        d=report.select(name).table(7)
        rates=[100*(d.parked & d.stop_s.notna() & (d.at_stop_mm<=t)).mean() for t in thresholds]
        ax.plot(thresholds,rates,label=f'{RUNS[name][0]} {name}')
    ax.axvline(10,color='black',linestyle=':',label='Official 10 mm criterion')
    ax.set(xlabel='Distance threshold at STOP (mm)',ylabel='Successful attempts out of 100',ylim=(0,100),xlim=(0,1000),title='No PARK or no STOP remains a failure at every distance threshold')
    ax.legend(fontsize=8);fig.tight_layout();return fig


def smoke_table():
    rows=[]
    for name in ('dart_smoke','dagger_round1_smoke','dagger_round2_smoke','dagger_round3_smoke',
                 'dart_dagger_round1_smoke','dart_dagger_round2_smoke','dart_dagger_round3_smoke'):
        raw=report.plots.read(report.ROOT/name/'val/seed7.json')
        assert {r['dot'] for r in raw}=={6,16,23,45,55,80}
        stopped=[r['at_network_stop_mm'] for r in raw if r.get('stop_s') is not None]
        penalised=[(r['at_network_stop_mm'] if r.get('stop_s') is not None else r['end_mm'])+
                   (0 if r['parked'] and r.get('stop_s') is not None else 1000) for r in raw]
        rows.append(dict(check=name,attempts=6,parked=sum(r['parked'] for r in raw),stopped=len(stopped),
                         success_10mm=sum(r['success'] for r in raw),median_stop_mm=np.median(stopped),
                         selection_score_mm=np.median(penalised)))
    return pd.DataFrame(rows)


def publish(notebook):
    backup=report.ROOT/'notebook_before_publication_43_58.ipynb'
    if not backup.exists(): shutil.copy2(notebook,backup)
    n=nbformat.read(notebook,as_version=4)
    assert not any(c.get('id')=='overnight-publication-overview' for c in n.cells),'Already published'
    # Preserve every existing code cell and its output. Only provisional heading text is promoted.
    originals=[c for c in n.cells if c.cell_type=='code']
    new=[]
    for c in n.cells:
        name=next((name for name in RUNS if c.get('id') in (f'pending-full-{name}-heading',f'full-{RUNS[name][0]}-heading')),None)
        if name:
            iteration=RUNS[name][0]
            c.source=f'### Iteration {iteration}: full-episode {name.replace("_"," ")}\n\n'+detail(name)
            (report.ROOT/name/'analysis.md').write_text(c.source+'\n')
        new.append(c)
    assert [c for c in new if c.cell_type=='code']==originals
    n.cells=new
    def md(id,s):
        c=nbformat.v4.new_markdown_cell(s);c.id=id;return c
    def code(id,s):
        c=nbformat.v4.new_code_cell(s);c.id=id;return c
    n.cells.extend([
      md('overnight-publication-overview','### Overnight comparisons: publication and interpretation\n\nIterations 41–58 are completed 100-test-target evaluations. Each also has a separate completed 100-target validation evaluation. No additional validation simulation is needed for publication. Six-case recovery checks remain development records only.\n\nThe noise ladder starts from iteration 41, not the standardised iteration 43. The stack includes encoder noise, damping, elasticity, inertia, 3 ms delay and random starts. It does not include bursty packets or occlusion. Iteration 51 is outside the agreed scope. Recovery models were trained with added recovery data, then tested separately on clean conditions and on the full stack.\n\nAll results use seed 7. Test targets have been reused; subsequent design decisions cannot be presented as blind final-test selection.'),
      code('overnight-publication-table',"import publish_curriculum as publication\npublication_data=publication.manifest()\ndisplay(publication_data[['iteration','name','role','split','attempts','parked','stopped','success_10mm','median_park_error_cm','median_closest_mm','median_stop_mm','execution_failures']].round(2))"),
      code('overnight-publication-comparison',"publication.overview_plot();plt.show()"),
      code('overnight-publication-thresholds',"publication.threshold_plot();plt.show()"),
      code('overnight-publication-smokes',"display(publication.smoke_table().round(2))\nprint('Six fixed validation targets only. No official iteration or MLflow run. Selection score includes missing PARK/STOP with a 1 m penalty; STOP median alone excludes them.')"),
    ])
    nbformat.write(n,notebook)
    data=manifest(refresh=True);data.to_csv(report.ROOT/'publication_scores.csv',index=False)
    (report.ROOT/'publication_plan.json').write_text(json.dumps({k:dict(iteration=v[0],parent=v[1],role=v[2],change=v[3]) for k,v in RUNS.items()},indent=2)+'\n')
    print('Published headings and detailed readings; preserved',len(originals),'existing code cells. Total cells:',len(n.cells))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--notebook',type=Path,required=True)
    publish(p.parse_args().notebook)
