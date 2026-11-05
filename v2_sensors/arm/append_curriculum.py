"""Append one completed iteration and execute only its new analysis cells."""
import argparse
import copy
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True);p.add_argument('--iteration',type=int,required=True)
    p.add_argument('--notebook',type=Path,required=True);a=p.parse_args()
    assert a.name.replace('_','').isalnum()
    notebook=nbformat.read(a.notebook,as_version=4);before=copy.deepcopy(notebook.cells)
    prefix=f'full-{a.iteration}-'
    if any(c.get('id','').startswith(prefix) for c in notebook.cells):raise ValueError('Iteration already appended')
    def cell(kind,id,source):
        c=(nbformat.v4.new_code_cell if kind=='code' else nbformat.v4.new_markdown_cell)(source)
        c['id']=prefix+id;return c
    bootstrap='''from pathlib import Path
import sys,json,uuid
import matplotlib.pyplot as plt
from IPython.display import display,Markdown,HTML
sys.path.insert(0,str(Path('../arm').resolve()))
import report_curriculum as overnight
def lazy_player(url,height=850):
    k=uuid.uuid4().hex[:8]
    return HTML(f"<div><button onclick=\\"var f=document.getElementById('player{k}');f.src='{url}';f.style.display='block';this.style.display='none'\\" style=\\"padding:8px 14px\\">Load the 3D player</button><iframe id=\\"player{k}\\" style=\\"display:none;width:100%;height:{height}px;border:0\\"></iframe></div>")
'''
    cells=[cell('markdown','heading',f'### Iteration {a.iteration}: full-episode {a.name.replace("_"," ")}\n\nLearned arm torques from reset; frozen driving checkpoint; seed 7. This is the complete 100-target test comparison, with the separate 100-target validation results alongside it. The prior plan and the previous notebook results are retained above.'),
        cell('code','scores',bootstrap+f"\nfull_name={a.name!r}\ndisplay(overnight.build(full_name).round(3))\nprint('Validation and test are distinct. No fitting occurs in these analysis cells.')"),
        cell('code','configuration',"full_config=json.loads((overnight.ROOT/full_name/'test/config.json').read_text())\ndisplay(full_config)\nfor kind in ('driving','torque','stop'):\n    chosen=json.loads((overnight.models(full_name)/f'{kind}_seed7_summary.json').read_text())\n    print(kind, 'selected epoch',chosen['selected_epoch'],'validation loss',chosen['selected_val_loss'])"),
        cell('code','losses',"overnight.losses(full_name);plt.show()\nprint('All training and validation epochs. Test scores are not training losses.')"),
        cell('code','outcomes',"full_report=overnight.select(full_name)\nfull_report.outcomes(7);plt.show()\ndisplay(full_report.table(7).round(3))"),
        cell('code','holding',"overnight.holding(full_name);plt.show()\ndisplay(overnight.hold_table(full_name).round(4))\nprint('Learned torques are applied from reset. No expert holding or added gravity term.')"),
        cell('code','stopping',"full_report=overnight.select(full_name)\nfull_report.stopping(7);plt.show()\nfull_report.distances(7);plt.show()\nprint('Closest approach, exact STOP, and 0.5-second drift are different measurements.')"),
        cell('code','player',"full_url=overnight.player_url(full_name)\ndisplay(lazy_player(full_url))\ndisplay(HTML(f'<a href=\"{full_url}\" target=\"_blank\">Open all 100 test replays</a>'))\nprint('Wheel encoder speeds, wheel commands, PARK and STOP values, applied arm torques, camera and ghost.')"),
        cell('code','reading',"display(Markdown(overnight.reading(full_name)))")]
    work=nbformat.v4.new_notebook(cells=cells)
    NotebookClient(work,timeout=300,kernel_name='comp0245-robot-data',resources={'metadata':{'path':str(a.notebook.parent)}}).execute()
    assert not any(o.output_type=='error' for c in work.cells for o in c.get('outputs',[]))
    notebook.cells.extend(work.cells);assert notebook.cells[:len(before)]==before
    nbformat.write(notebook,a.notebook)
    print(f'Appended {len(cells)} executed cells; preserved {len(before)} existing cells')


if __name__=='__main__':main()
