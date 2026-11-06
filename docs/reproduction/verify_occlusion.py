"""Check all 1,200 raw occlusion episodes against their hashes and published table."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import median
import zipfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--zip', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True, help='New output JSON file')
args = parser.parse_args()
assert not args.out.exists(), args.out
groups = []
with zipfile.ZipFile(args.zip) as archive:
    table = list(csv.DictReader(io.StringIO(archive.read(
        'lighthouse-review/analysis/episodes.csv').decode())))
    assert len(table) == 1200
    index = {(row['condition'], row['split'], row['variant'], int(row['dot'])): row
             for row in table}
    assert len(index) == 1200
    for condition in ('baseline', 'recovery', 'disturbed'):
        cases = json.loads(archive.read(f'lighthouse/{condition}/cases.json'))
        for split in ('val', 'test'):
            for variant in ('control', 'lighthouse'):
                folder = f'lighthouse/full/{condition}/{split}/{variant}'
                completion = json.loads(archive.read(folder + '/completion.json'))
                expected = set(map(int, cases[split]))
                assert len(expected) == completion['attempts'] == 100
                rows = []
                for dot in sorted(expected):
                    name = f'dot{dot}.json'
                    raw = archive.read(folder + '/' + name)
                    assert hashlib.sha256(raw).hexdigest() == completion['result_sha256'][name]
                    row = json.loads(raw)
                    saved = index[(condition, split, variant, dot)]
                    success = bool(row['parked'] and row['stop_s'] is not None
                                   and row['at_network_stop_mm'] <= 10)
                    assert success == row['success'] == (saved['success'] == 'True')
                    assert row['parked'] == (saved['parked'] == 'True')
                    assert (row['stop_s'] is not None) == (saved['stopped'] == 'True')
                    for actual, column in ((row['at_network_stop_mm'], 'stop_mm'),
                                           (row['min_mm'], 'closest_mm'),
                                           (row['end_mm'], 'end_mm')):
                        value = float(saved[column]) if saved[column] else None
                        assert (actual is None and value is None) or (
                            actual is not None and value is not None
                            and math.isclose(actual, value, rel_tol=1e-12, abs_tol=1e-10))
                    rows.append(row)
                stopped = [row['at_network_stop_mm'] for row in rows if row['stop_s'] is not None]
                groups.append(dict(condition=condition, split=split, variant=variant,
                                   attempts=len(rows), parked=sum(row['parked'] for row in rows),
                                   stopped=len(stopped), success=sum(row['success'] for row in rows),
                                   stop_median_mm=median(stopped) if stopped else None))
                print(f'PASS {condition}/{split}/{variant}: 100 original records', flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open('x') as handle:
    json.dump(dict(episodes=1200, hashes_verified=1200, groups=groups), handle, indent=2)
print('PASS: all 1,200 original hashes, result flags and distances match the analysis table.')
