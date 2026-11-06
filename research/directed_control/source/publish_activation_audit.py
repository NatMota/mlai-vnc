"""Expose the existing training-only activation audit in the research notebook."""
from pathlib import Path
import nbformat
R = Path(__file__).parent
nb = nbformat.read(R/'research_notebook.ipynb', as_version=4)
description = nbformat.v4.new_markdown_cell('''## What inspecting the original weights revealed

The audit evaluated 26,867 sampled rows from all 100 training demonstrations. In the arm's second 128-unit hidden layer, 44 ReLU units never activated on those sampled rows; one unit in the independent STOP network was also inactive. This does not establish that they are permanently dead or that removing them would improve control. The first arm hidden layer had no always-inactive units in this sample.

Weights and biases are fully accessible numerical arrays, but an individual value rarely has an isolated behavioural meaning. The plots measure how often each unit participates, rather than assigning a label such as “parking neuron”. Changes to an activation function also change the fitted function and must be evaluated as new candidates. The activation and gain edits preserved in the early development tables did not establish reliable reaching, which motivated the later structural experiments.''')
description['id'] = 'rnd-original-activation-description'
plot = nbformat.v4.new_code_cell('''import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path.cwd()
activation=json.loads((ROOT/'activation_audit.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(12,6))
rows=[]
for ax,(name,layer) in zip(axes.flat,activation['layers'].items()):
    fraction=np.asarray(layer['positive_fraction'])
    ax.bar(np.arange(len(fraction)),fraction,width=1)
    ax.set(title=name,xlabel='Unit index',ylabel='Fraction active',ylim=(0,1))
    rows.append(dict(layer=name,width=layer['width'],inactive_on_sample=len(layer['inactive_on_all_sampled_rows']),
        median_fraction_active=float(np.median(fraction))))
fig.suptitle('Original ReLU activity on training demonstrations only')
fig.tight_layout();fig.savefig(ROOT/'original_activation_activity.png',dpi=150);plt.show()
display(pd.DataFrame(rows).round(3))
''')
plot['id'] = 'rnd-original-activation-plot'
ids = {description['id'], plot['id']}
nb.cells = [c for c in nb.cells if c.get('id') not in ids]
position = next(i for i,c in enumerate(nb.cells) if c.get('id') == 'research-interpretation') + 1
nb.cells[position:position] = [description, plot]
nbformat.write(nb, R/'research_notebook.ipynb')
