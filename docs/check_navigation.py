"""Check review links, iteration identities and frozen evidence without simulation."""
import hashlib
import json
from pathlib import Path
import re
import runpy
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'v2_sensors'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check():
    rows = json.loads((V2 / 'iterations.json').read_text())['iterations']
    assert [r['iteration'] for r in rows] == list(range(1, 58))
    labels = runpy.run_path(str(V2 / 'arm/experiment_labels.py'))['ITERATIONS']
    for row in rows:
        for field in ('source', 'models', 'evidence'):
            assert (V2 / row[field]).exists(), (row['iteration'], field)
        page, anchor = row['reading_view'].split('#')
        assert f'id="{anchor}"' in (V2 / page).read_text()
        if row['iteration'] >= 41:
            assert labels[row['condition']] == row['iteration']
            assert row['archive_iteration'] == row['iteration'] + (row['iteration'] >= 51)
    for ref in json.loads((V2 / 'REFERENCE.json').read_text())['references']:
        for name, digest in ref['sha256'].items():
            assert sha(ROOT / name) == digest, name
    for view in json.loads((V2 / 'review/manifest.json').read_text()):
        assert sha(ROOT / view['source']) == view['sha256'], view['source']
        rendered = (V2 / 'review' / view['view']).read_text()
        assert rendered.count('<a id="cell-') == view['cells']
        assert rendered.count('![Saved plot from cell ') == view['plots']
        assert len(rendered.encode()) < 500_000, view['view']
    documents = [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md'), V2 / 'README.md',
                 V2 / 'ITERATIONS.md', V2 / 'driving_v2/HISTORICAL_BASELINE.md',
                 V2 / 'arm/results/overnight_20261105/README.md', *(V2 / 'review').glob('*.md')]
    links = 0
    for page in documents:
        content = re.sub(r'```.*?```', '', page.read_text(), flags=re.S)
        for link in re.findall(r'\]\(([^\s)]+)\)', content):
            if re.match(r'[a-zA-Z][\w+.-]*:', link):
                continue
            path, _, anchor = unquote(link).partition('#')
            target = page.parent / path if path else page
            assert target.exists(), (str(page.relative_to(ROOT)), link)
            if anchor and target.suffix == '.md':
                text = re.sub(r'```.*?```', '', target.read_text(), flags=re.S)
                headings = re.findall(r'^#+ (.+)$', text, flags=re.M)
                slugs = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-') for h in headings}
                assert anchor in slugs or f'id="{anchor}"' in text, (str(page.relative_to(ROOT)), link)
            links += 1
    print(f'PASS: 57 iterations, 2 frozen references, notebook hashes and {links} local links.')


if __name__ == '__main__':
    check()
