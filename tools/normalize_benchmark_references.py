"""Explicit, reviewed text-only overrides; never infer speech from digits.

No model/network operations. Atomic per-file publication with rollback on errors;
not a crash-safe multi-file transaction. Do not run concurrent manifest writers.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import long_audio_benchmark as b

VERSION = 'fleurs60-spoken-v1'
DEFAULT = b.ROOT / 'benchmarks/annotations/fleurs60-spoken-references.json'


def transform(rows, annotations):
    if annotations['version'] != VERSION or len(rows) != annotations['expected_samples']:
        raise ValueError('version/sample count mismatch')
    expanded = []
    for original in annotations['entries']:
        e = copy.deepcopy(original)
        if 'spans' in e:
            e['current'] = e['raw']
            e['replacements'] = []
            for old, new, rule in e['spans']:
                if e['raw'].count(old) != 1:
                    raise ValueError('ambiguous curated span')
                start = e['raw'].index(old)
                e['replacements'].append(dict(start=start, end=start+len(old), old=old, new=new, rule=rule))
            e['new'] = e['raw']
            for span in reversed(e['replacements']):
                e['new'] = e['new'][:span['start']] + span['new'] + e['new'][span['end']:]
        expanded.append(e)
    entries = {(e['source'], e['id']): e for e in expanded}
    identities = {(r['source'], r['id']) for r in rows}
    if len(entries) != len(annotations['entries']) or len(identities) != len(rows) or not entries.keys() <= identities:
        raise ValueError('annotation/sample identity mismatch')
    result = copy.deepcopy(rows)
    for r in result:
        raw = r.get('reference_raw', r['reference'])
        e = entries.get((r['source'], r['id']))
        expected = raw
        if e:
            if any(r[k] != e[k] for k in ('original_id', 'source_revision')) or raw != e['raw'] or e['current'] != raw or e['audio_validated'] is not False:
                raise ValueError('raw/provenance mismatch')
            end = 0
            parts = []
            for span in e['replacements']:
                start, stop = span['start'], span['end']
                if not (type(start) is int and type(stop) is int and end <= start < stop <= len(raw)) or raw[start:stop] != span['old'] or not span['rule'] or not span['new']:
                    raise ValueError('invalid replacement span')
                parts.extend([raw[end:start], span['new']])
                end = stop
            expected = ''.join(parts) + raw[end:]
            if expected != e['new']:
                raise ValueError('replacement result mismatch')
        if 'reference_convention' in r:
            if r['reference_convention'] != VERSION or 'reference_raw' not in r or r['reference'] != expected:
                raise ValueError('existing convention/raw mismatch')
        elif 'reference_raw' in r or r['reference'] != raw:
            raise ValueError('unexpected raw state')
        r.update(reference=expected, reference_raw=raw, reference_convention=VERSION)
    return result


def atomic(path, data):
    fd, name = tempfile.mkstemp(prefix='.reference-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def apply(manifest, annotations, validate=b.validate):
    manifest = Path(manifest)
    rows = b.load(manifest)
    proposed = transform(rows, annotations)
    validation = validate(proposed)  # Complete audio/hash/leak check before any write.
    paths = [manifest] + [manifest.parent/'splits'/f'{s}.json' for s in ('validation', 'test')]
    values = [proposed] + [[r for r in proposed if r['split'] == s] for s in ('validation', 'test')]
    old = [p.read_bytes() for p in paths]
    # Refuse pre-existing incoherence, rather than hiding an unrelated edit.
    for p, split in zip(paths[1:], ('validation', 'test')):
        if b.load(p) != [r for r in rows if r['split'] == split]:
            raise ValueError('partition mismatch')
    if proposed == rows:
        return dict(changed=0, validation=validation)
    backup = manifest.parent/'reference-backups'/str(time.time_ns())
    backup.mkdir(parents=True)
    for p, content in zip(paths, old):
        (backup/p.name).write_bytes(content)
    encoded = [(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode() for v in values]
    written = []
    try:
        for p, content in zip(paths, encoded):
            atomic(p, content)
            written.append(p)
    except BaseException:
        for p in reversed(written):
            atomic(p, old[paths.index(p)])
        raise
    return dict(changed=sum(a['reference'] != z['reference'] for a, z in zip(rows, proposed)),
                backup=str(backup), validation=validation)


def inventory(rows):
    return [dict(id=r['id'], original_id=r['original_id'], reference=r['reference'],
                 expressions=re.findall(r'\d+(?:[.,]\d+)*(?:\s*(?:km/h|kilomet|dặm|m)\b)?', r['reference']))
            for r in rows if re.search(r'\d', r['reference'])]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('audit', 'apply'))
    p.add_argument('--manifest', type=Path, default=b.ROOT/'data/benchmark/long-audio/manifest.json')
    p.add_argument('--annotations', type=Path, default=DEFAULT)
    args = p.parse_args()
    annotations = b.load(args.annotations)
    if args.command == 'apply':
        result = apply(args.manifest, annotations)
    else:
        rows = b.load(args.manifest)
        proposed = transform(rows, annotations)
        result = dict(samples=len(rows), inventory=inventory(rows),
                      changed=sum(a['reference'] != z['reference'] for a,z in zip(rows, proposed)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
