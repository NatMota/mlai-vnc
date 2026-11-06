"""Verify selected downloaded archives against the committed SHA-256 manifest."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dir', type=Path, required=True)
parser.add_argument('archives', nargs='+', help='Archive filenames to check')
args = parser.parse_args()
manifest = json.loads(Path(__file__).with_name('data-manifest.json').read_text())
entries = {entry['archive']: entry for entry in manifest['archives']}
for name in args.archives:
    entry = entries[name]
    path = args.dir / name
    assert path.stat().st_size == entry['bytes'], f'Size mismatch: {name}'
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == entry['sha256'], f'SHA-256 mismatch: {name}'
    print(f'PASS {name}')
