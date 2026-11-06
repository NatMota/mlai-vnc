"""Render saved notebooks for GitHub without executing cells or changing originals.

Run from any directory: python3 docs/render_notebook_views.py
Only the two reading views, their image assets and manifest are written.
"""
import base64
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'v2_sensors/driving_v2'
OUT = ROOT / 'v2_sensors/review'
NAMES = ('driving_v2_sensors', 'golden_dataset')


def joined(value):
    return ''.join(value) if isinstance(value, list) else value


def rebase_links(text):
    # Both source notebooks share a directory. Leave web links and anchors alone.
    return re.sub(r'(?<=\]\()([^\s)]+)', lambda m: m[0] if
                  re.match(r'[a-zA-Z][\w+.-]*:|[/#]', m[0]) else '../driving_v2/' + m[0], text)


def render(name, start=0, end=None, page=None):
    raw = (SOURCE / (name + '.ipynb')).read_bytes()
    notebook = json.loads(raw)
    cells = list(enumerate(notebook['cells']))[start:end]
    parts = [f'# {name}: saved notebook reading view\n\n'
             '[V2 overview](../README.md) · [Iteration index](../ITERATIONS.md) · '
             f'[Original notebook](../driving_v2/{name}.ipynb)\n\n'
             'Generated from saved outputs without running cells. Code is collapsed. '
             'Interactive players remain in the original notebook; plots and tables appear below. '
             'Original notebook wording is preserved, including historical hypotheses and status notes.\n\n'
             f'Source SHA-256: `{hashlib.sha256(raw).hexdigest()}`\n\n## Contents\n']
    for i, cell in cells:
        if cell['cell_type'] == 'markdown':
            for line in joined(cell['source']).splitlines():
                if re.match(r'^#{1,3} ', line):
                    title = line.lstrip('# ').replace('[', '').replace(']', '')
                    parts.append(f'- [{title}](#cell-{i})\n')
    images = outputs = interactive = 0
    for i, cell in cells:
        parts.append(f'\n<a id="cell-{i}"></a>\n\n')
        source = joined(cell['source'])
        if cell['cell_type'] != 'code':
            parts.append(rebase_links(source) + '\n')
            continue
        parts.append(f'<details><summary>Code · original cell {i}</summary>\n\n```python\n{source}\n```\n\n</details>\n\n')
        for output in cell.get('outputs', []):
            outputs += 1
            data = output.get('data', {})
            if 'image/png' in data:
                png = base64.b64decode(joined(data['image/png']))
                assert png.startswith(b'\x89PNG\r\n\x1a\n')
                asset = f'assets/{hashlib.sha256(png).hexdigest()}.png'
                (OUT / asset).write_bytes(png)
                parts.append(f'![Saved plot from cell {i}]({asset})\n\n')
                images += 1
            elif 'text/markdown' in data:
                parts.append(rebase_links(joined(data['text/markdown'])) + '\n\n')
            elif 'text/html' in data and '<table' in joined(data['text/html']).lower():
                table = joined(data['text/html'])
                table = re.sub(r'<style\b[^>]*>.*?</style>', '', table, flags=re.S | re.I)
                parts.append(table + '\n\n')
            elif 'text/html' in data:
                parts.append(f'[Interactive output: open original notebook, cell {i}](../driving_v2/{name}.ipynb).\n\n')
                interactive += 1
            elif 'text/plain' in data or 'text' in output:
                text = joined(data.get('text/plain', output.get('text', '')))
                parts.append(f'```text\n{text}\n```\n\n')
            elif output.get('output_type') == 'error':
                parts.append(f"```text\n{output.get('ename')}: {output.get('evalue')}\n```\n\n")
    result = ''.join(parts).rstrip() + '\n'
    assert result.count('<a id="cell-') == len(cells)
    assert images == sum('image/png' in o.get('data', {}) for _, c in cells for o in c.get('outputs', []))
    assert (SOURCE / (name + '.ipynb')).read_bytes() == raw
    (OUT / ((page or name) + '.md')).write_text(result)
    return dict(source=f'v2_sensors/driving_v2/{name}.ipynb', sha256=hashlib.sha256(raw).hexdigest(),
                view=(page or name) + '.md', cells=len(cells), original_cell_range=[cells[0][0], cells[-1][0]],
                outputs=outputs, plots=images, interactive_outputs=interactive)


if __name__ == '__main__':
    (OUT / 'assets').mkdir(parents=True, exist_ok=True)
    records = [render('driving_v2_sensors'),
               render('golden_dataset', 0, 233, 'golden_dataset_34_40'),
               render('golden_dataset', 233, 323, 'golden_dataset_41_50'),
               render('golden_dataset', 323, None, 'golden_dataset_51_57')]
    assert sum(row['cells'] for row in records[1:]) == len(json.loads((SOURCE / 'golden_dataset.ipynb').read_text())['cells'])
    (OUT / 'golden_dataset.md').write_text('''# Golden dataset: saved notebook reading view

[V2 overview](../README.md) · [Iteration index](../ITERATIONS.md) · [Original notebook](../driving_v2/golden_dataset.ipynb)

Read the notebook in three parts. Saved prose, code, tables and plots are retained, with no cells executed. Interactive players remain in the original notebook.

1. [Iterations 34–40: golden data, preliminary design checks and cumulative comparisons](golden_dataset_34_40.md)
2. [Iterations 41–50: learned holding and cumulative disturbances](golden_dataset_41_50.md)
3. [Iterations 51–57: recovery comparisons and later validation pilots](golden_dataset_51_57.md)

[Source hashes and output counts](manifest.json) · Regenerate with `python3 docs/render_notebook_views.py` from the repository.
''')
    (OUT / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))
