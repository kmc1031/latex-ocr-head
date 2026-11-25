"""Relative paths, atomic writes, and data/code identities independent of drive."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve(root, relative):
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Path escapes its portable root: '+str(relative))
    return path


def code_hashes():
    root = Path(__file__).parent
    return {p.name: digest(p) for p in root.glob('*.py')}
