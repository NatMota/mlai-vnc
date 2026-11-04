"""Package 300 golden expert runs and 200 camera traces, without code or robot assets.

Copies the existing archives byte-for-byte; never resamples, relabels or trains.
Run from the repository root with Python/NumPy. See v2_sensors/arm/GOLDEN_DATA.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def package(source, camera, out, revision):
    selection = json.loads((ROOT / 'collection/native/effects/dots_spread.json').read_text())['100']
    full_split = json.loads((ROOT / 'collection/native/effects/split_1000.json').read_text())
    splits = dict(train=selection, val=full_split['val'], test=full_split['test'])
    assert all(len(v) == len(set(v)) == 100 for v in splits.values())
    assert len(set().union(*map(set, splits.values()))) == 300
    assert set(selection) <= set(full_split['train'])
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    out.mkdir(parents=True)
    entries = []
    for split, ids in splits.items():
        (out / 'expert' / split).mkdir(parents=True)
        if split != 'test':
            (out / 'camera' / split).mkdir(parents=True)
        for dot in ids:
            src = source / f'run_{dot}.npz'
            h = digest(src)
            with np.load(src, allow_pickle=False) as a:
                m = json.loads(str(a['manifest']))
                n = int(m['steps'])
                assert m['condition'] == 'baseline' and m['dt_s'] == .001
                np.testing.assert_allclose(a['time_s'], np.arange(n) * .001, atol=1e-8, rtol=0)
                assert all(a[k].shape[0] == n for k in a.files if k not in ('manifest', 'columns'))
                assert a['q'].shape == (n, 17) and a['arm_torque_command'].shape == (n, 7)
            dest = out / 'expert' / split / src.name
            shutil.copyfile(src, dest)
            assert digest(dest) == h
            entries.append(dict(path=str(dest.relative_to(out)), sha256=h, bytes=dest.stat().st_size,
                                dot=dot, split=split, steps=n, termination=m['termination']))
            if split != 'test':
                src_cam = camera / split / f'camera_{dot}.npz'
                with np.load(src_cam, allow_pickle=False) as c:
                    prov = json.loads(str(c['provenance']))
                    assert prov['source_sha256'] == h
                    assert int(c['dot']) == dot and int(c['n_steps']) == n
                    assert int(c['parking_step']) == m['parking_step']
                    assert np.all(np.diff(c['step']) > 0) and c['step'][0] == 0 and c['step'][-1] < n
                    frames = len(c['step'])
                dest_cam = out / 'camera' / split / src_cam.name
                shutil.copyfile(src_cam, dest_cam)
                hc = digest(src_cam)
                assert digest(dest_cam) == hc
                entries.append(dict(path=str(dest_cam.relative_to(out)), sha256=hc,
                                    bytes=dest_cam.stat().st_size, dot=dot, split=split, frames=frames))
        print(f'{split}: 100 expert runs; {0 if split == "test" else 100} camera traces verified', flush=True)
    (out / 'split.json').write_text(json.dumps(splits, indent=2) + '\n')
    (out / 'README.md').write_text((ROOT / 'v2_sensors/arm/GOLDEN_DATA.md').read_text().replace('CODE_REVISION', revision))
    (out / 'manifest.json').write_text(json.dumps(dict(schema_version=1, code_revision=revision,
        repository='https://github.com/NatMota/mlai-vnc', branch='v2-sensor',
        expert_runs=300, camera_runs=200, camera_frames=sum(e.get('frames', 0) for e in entries),
        files=entries), indent=2) + '\n')
    files = sorted(p for p in out.rglob('*') if p.is_file())
    (out / 'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.relative_to(out).as_posix()}\n' for p in files))
    archive = out.with_suffix('.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, (Path(out.name) / p.relative_to(out)).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    info = dict(archive=archive.name, bytes=archive.stat().st_size, sha256=digest(archive),
                expert_bytes=sum(e['bytes'] for e in entries if e['path'].startswith('expert/')),
                camera_bytes=sum(e['bytes'] for e in entries if e['path'].startswith('camera/')))
    archive.with_suffix('.zip.sha256').write_text(f'{info["sha256"]}  {archive.name}\n')
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--camera', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--code-revision', required=True)
    a = ap.parse_args()
    package(a.source, a.camera, a.out, a.code_revision)
