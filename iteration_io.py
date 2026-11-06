"""Small, local helpers for immutable iteration identities and single-writer output."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


@contextmanager
def output_lock(path):
    """A crash leaves a lock: inspect the process before removing a stale lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def relative_path(path, root):
    """Inputs belong to the checkout so recorded names are portable."""
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
