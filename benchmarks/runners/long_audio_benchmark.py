"""Bounded Vietnamese acquisition and staged, shared-candidate benchmark.

Acquisition/validation use Python's standard library and ffmpeg only.
No model is loaded except by decode/score. See docs/long-audio-benchmark.md.
"""
import argparse
import csv
import io
import tarfile
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCES = ['capleaf/viVoice', 'google/fleurs', 'NhutP/VietSpeech']


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != 'https':
            raise ValueError('refusing non-HTTPS redirect')
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected and urllib.parse.urlsplit(req.full_url).netloc != urllib.parse.urlsplit(newurl).netloc:
            redirected.remove_header('Authorization')
        return redirected


def safe_urlopen(request, timeout):
    if urllib.parse.urlsplit(request.full_url).scheme != 'https':
        raise ValueError('refusing non-HTTPS request')
    return urllib.request.build_opener(SafeRedirect()).open(request, timeout=timeout)


def normalize(text):
    text = unicodedata.normalize('NFC', text).lower()
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def check_leaks(rows):
    seen = {}
    for row in rows:
        keys = [('id', row['source'], row['id']), ('audio', row['sha256']),
                ('text', normalize(row['reference']))]
        if row.get('pcm_sha256'):
            keys.append(('decoded_audio', row['pcm_sha256']))
        if row.get('group'):
            keys.append(('group', row['source'], row['group']))
        for key in keys:
            if key in seen and seen[key] != row['split']:
                raise ValueError(f'cross-split leak: {key[0]}')
            seen[key] = row['split']


def pcm_info(path):
    # Decode the complete file, not just its header. No silent 30-second trim.
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 's16le',
                        '-ac', '1', '-ar', '16000', 'pipe:1'], capture_output=True,
                       check=True, timeout=20)
    return len(p.stdout) / 32000, digest(p.stdout)


def validate(rows):
    if not rows:
        raise ValueError('empty manifest')
    ids, hashes, pcm_hashes = set(), set(), set()
    for r in rows:
        for key in ('source_revision', 'source_url', 'license', 'original_split',
                    'selection', 'audio_path', 'reference', 'sha256'):
            if not r.get(key):
                raise ValueError(f'missing {key}')
        if r['split'] not in ('validation', 'test') or not normalize(r['reference']):
            raise ValueError('invalid split/reference')
        key = (r['source'], r['id'])
        if key in ids or r['sha256'] in hashes:
            raise ValueError('duplicate ID/audio')
        ids.add(key)
        hashes.add(r['sha256'])
        if file_hash(r['audio_path']) != r['sha256']:
            raise ValueError('audio hash mismatch')
        duration, pcm_hash = pcm_info(r['audio_path'])
        if pcm_hash in pcm_hashes:
            raise ValueError('duplicate decoded audio')
        pcm_hashes.add(pcm_hash)
        if not 0 < duration < 30:
            raise ValueError('benchmark supports only 0 < duration < 30 seconds')
        if abs(duration - r['duration']) > 1 / 16000 or pcm_hash != r['pcm_sha256']:
            raise ValueError('decoded audio mismatch')
    check_leaks(rows)
    return {'samples': len(rows), 'counts': counts(rows),
            'duration_min': min(r['duration'] for r in rows),
            'duration_max': max(r['duration'] for r in rows),
            'duration_total': sum(r['duration'] for r in rows)}


def counts(rows):
    return {s: {sp: sum(r['source'] == s and r['split'] == sp for r in rows)
                for sp in ('validation', 'test')} for s in SOURCES}


class BudgetHTTP:
    def __init__(self, byte_limit, seconds):
        self.remaining = byte_limit
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.requests = 0

    def get(self, url, token=None):
        if self.remaining <= 0 or time.monotonic() >= self.deadline:
            raise RuntimeError('network budget exhausted')
        headers = {'User-Agent': 'phowhisper-bounded-benchmark/1'}
        if token and urllib.parse.urlsplit(url).hostname in ('huggingface.co', 'datasets-server.huggingface.co'):
            headers['Authorization'] = 'Bearer ' + token
        request = urllib.request.Request(url, headers=headers)
        self.requests += 1
        with safe_urlopen(request, timeout=min(20, max(1, self.deadline - time.monotonic()))) as f:
            result = bytearray()
            while True:
                if time.monotonic() >= self.deadline:
                    raise RuntimeError('time budget exhausted')
                chunk = f.read(min(65536, self.remaining + 1))
                if not chunk:
                    return bytes(result)
                self.remaining -= len(chunk)
                if self.remaining < 0:
                    raise RuntimeError('byte budget exhausted')
                result.extend(chunk)

    def json(self, url, token=None):
        return json.loads(self.get(url, token))

    def head(self, url, token=None):
        if self.remaining <= 0 or time.monotonic() >= self.deadline:
            raise RuntimeError('network budget exhausted')
        headers = {'Authorization': 'Bearer ' + token} if token and urllib.parse.urlsplit(url).hostname in ('huggingface.co', 'datasets-server.huggingface.co') else {}
        self.requests += 1
        request = urllib.request.Request(url, headers=headers, method='HEAD')
        with safe_urlopen(request, timeout=min(15, max(1, self.deadline - time.monotonic()))):
            pass


def env_file_token(path):
    # Parse data only: never source, interpolate, execute, or echo file contents.
    try:
        lines = Path(path).read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError):
        raise ValueError('credential file unavailable') from None
    values = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        key, sep, value = line.partition('=')
        if sep and key.strip() in ('HF_TOKEN', 'HUGGING_FACE_API_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACEHUB_API_TOKEN'):
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if value and not any(c in value for c in '\r\n'):
                values[key.strip()] = value
    if not values:
        raise ValueError('credential file has no supported HF credential')
    return next(iter(values.values()))


def advertised_duration(row):
    value = row.get('num_samples', 0) / 16000 if 'num_samples' in row else row.get('duration', 0)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else 0


def existing_token():
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        return token.strip()
    path = Path(os.environ.get('HF_TOKEN_PATH',
                str(Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface'))) / 'token')))
    return path.read_text().strip() if path.is_file() else None


def acquisition_targets(sources, per_split):
    if per_split <= 0 or not sources or any(s not in SOURCES for s in sources):
        raise ValueError('invalid acquisition targets')
    return dict.fromkeys(sources, per_split)


def parse_fleurs_tsv(text):
    """Physical seven-column official FLEURS TSV; quotes are literal data.

    Native sentence IDs may repeat across recordings; filenames must be unique.
    Never let malformed source quotes merge transcription with character tokens.
    """
    rows = list(csv.reader(io.StringIO(text, newline=''), delimiter='\t',
                           quoting=csv.QUOTE_NONE, strict=True))
    seen = set()
    if not rows:
        raise ValueError('empty official FLEURS TSV')
    for row in rows:
        if (len(row) != 7 or not re.fullmatch(r'\d+', row[0]) or
                not re.fullmatch(r'[A-Za-z0-9_-]+\.wav', row[1]) or row[1] in seen or
                not normalize(row[2]) or not normalize(row[3]) or not row[4] or
                not re.fullmatch(r'[1-9]\d*', row[5]) or row[6] not in ('MALE', 'FEMALE', 'OTHER')):
            raise ValueError('invalid official FLEURS TSV schema/filename')
        seen.add(row[1])
    return rows


def fleurs_archive(http, out, rows, revision, target, status, scan_limit=300, seed=42,
                   min_duration=0, blocked=()):
    """Read only a bounded gzip/tar prefix from the official immutable archive.

    No archive extraction or whole-shard buffering; stop immediately at quota.
    TSV: native sentence ID, filename, raw text, transcription, characters,
    16kHz sample count, gender. Quotes are literal, not TSV field delimiters.
    """
    base = f'https://huggingface.co/datasets/google/fleurs/resolve/{revision}/data/vi_vn/'
    for split, native in [('validation', 'dev'), ('test', 'test')]:
        if sum(r['source'] == 'google/fleurs' and r['split'] == split for r in rows) >= target:
            continue
        metadata = parse_fleurs_tsv(http.get(base + native + '.tsv').decode())
        mapping = {r[1]: (i, r) for i, r in enumerate(metadata)}
        request = urllib.request.Request(base + 'audio/' + native + '.tar.gz')
        http.requests += 1
        class Limited:
            def read(self, size=-1):
                if time.monotonic() >= http.deadline or http.remaining <= 0:
                    raise RuntimeError('archive budget exhausted')
                data = response.read(min(65536, http.remaining, size if size >= 0 else 65536))
                http.remaining -= len(data)
                return data
        scanned = 0
        with safe_urlopen(request, timeout=min(20, max(1, http.deadline - time.monotonic()))) as response:
            with tarfile.open(fileobj=Limited(), mode='r|gz') as archive:
                for member in archive:
                    if not member.isfile() or member.size > 4000000:
                        continue
                    entry = mapping.get(Path(member.name).name)
                    if not entry:
                        continue
                    scanned += 1
                    status[split + '_archive_members_scanned'] = scanned
                    if scanned > scan_limit:
                        break
                    index, record = entry
                    stable_id = f'{split}:{index}'
                    if any(r['source'] == 'google/fleurs' and (r['id'] == stable_id or
                           (r['original_split'] == split and str(r.get('original_id')) == record[0])) for r in list(rows) + list(blocked)):
                        continue
                    data = archive.extractfile(member).read()
                    local = out / f'google_fleurs_{split}_official_{index}.wav'
                    local.write_bytes(data)
                    duration, pcm = pcm_info(local)
                    if abs(duration - int(record[5]) / 16000) > 1 / 16000:
                        raise ValueError('official TSV/audio sample-count mismatch')
                    if not 0 < duration < 30 or duration < min_duration:
                        local.unlink()
                        continue
                    row = dict(source='google/fleurs', source_revision=revision, id=stable_id,
                               original_id=int(record[0]), original_split=split, split=split,
                               source_url=base + 'audio/' + native + '.tar.gz', license='cc-by-4.0',
                               reference=record[3], audio_path=str(local), duration=duration,
                               sha256=digest(data), pcm_sha256=pcm, group=None,
                               group_limit='speaker/recording identity unavailable',
                               selection='preferred' if duration >= 20 else 'fallback: bounded archive prefix',
                               selection_seed=seed, transcript_audit='pending',
                               acquisition_method='official TSV + streaming tar prefix', archive_member=member.name)
                    try:
                        check_leaks(rows + [row])
                        if any(r['sha256'] == row['sha256'] or r['pcm_sha256'] == pcm for r in rows):
                            continue
                    except ValueError:
                        continue
                    rows.append(row)
                    save(out / 'manifest.json', rows)
                    if sum(r['source'] == 'google/fleurs' and r['split'] == split for r in rows) >= target:
                        break
        status[split + '_archive_members_scanned'] = scanned


def replacement_plan(rows, candidates):
    result, changes = list(rows), []
    for index, old in enumerate(rows):
        if old['source'] != 'google/fleurs' or old['duration'] >= 15:
            continue
        for new in sorted(candidates, key=lambda r: (r['duration'] < 20, -r['duration'])):
            if new['split'] != old['split'] or not 15 <= new['duration'] < 30:
                continue
            trial = result[:index] + [new] + result[index + 1:]
            if len({(r['source'], r['id']) for r in trial}) != len(trial):
                continue
            if any(len({r[k] for r in trial}) != len(trial) for k in ('sha256', 'pcm_sha256')):
                continue
            try:
                check_leaks(trial)
            except ValueError:
                continue
            result = trial
            changes.append(dict(split=old['split'], old_id=old['id'], new_id=new['id'],
                                old_duration=old['duration'], new_duration=new['duration'],
                                old_audio_path=old['audio_path'] if 'audio_path' in old else None,
                                new_audio_path=new.get('audio_path')))
            break
    return result, changes


def duration_summary(rows):
    result = {}
    for split in ('validation', 'test'):
        durations = [r['duration'] for r in rows if r['split'] == split]
        result[split] = dict(samples=len(durations), under15=sum(d < 15 for d in durations),
                            from15to20=sum(15 <= d < 20 for d in durations),
                            at_least20=sum(d >= 20 for d in durations),
                            min=min(durations), max=max(durations),
                            mean=sum(durations) / len(durations), total=sum(durations))
    return result


def replace_short(args):
    out = args.output.resolve()
    manifest = out / 'manifest.json'
    original = load(manifest)
    validate(original)
    if len(original) != 60 or any(sum(r['split'] == sp for r in original) != 30
                                  for sp in ('validation', 'test')):
        raise ValueError('expected native 30/30 manifest')
    if any(r['source'] != 'google/fleurs' or r['original_split'] != r['split'] for r in original):
        raise ValueError('expected native FLEURS rows')
    revisions = {r['source_revision'] for r in original}
    if len(revisions) != 1:
        raise ValueError('mixed revisions')
    revision = next(iter(revisions))
    stamp = str(time.time_ns())
    backup = out / 'replacement-backups' / stamp
    backup.mkdir(parents=True)
    for name in ('manifest.json', 'acquisition-report.json', 'retry-provenance.json',
                 'replacement-provenance.json', 'splits/validation.json', 'splits/test.json'):
        source = out / name
        if source.exists():
            destination = backup / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    stage = out / 'replacement-staging'
    stage.mkdir(exist_ok=True)
    staged_manifest = stage / 'manifest.json'
    candidates = load(staged_manifest) if staged_manifest.exists() else []
    if candidates:
        validate(candidates)
        if any(r['source_revision'] != revision for r in candidates):
            raise ValueError('staging revision mismatch')
    protected = [r for r in original if r['duration'] >= 15]
    pool = protected + [r for r in candidates if (r['source'], r['id']) not in
                        {(x['source'], x['id']) for x in original}]
    http = BudgetHTTP(args.max_bytes, args.seconds)
    status = {}
    try:
        fleurs_archive(http, stage, pool, revision, 30, status, args.scan_rows,
                       min_duration=args.min_duration, blocked=original)
    except (OSError, ValueError, RuntimeError, tarfile.TarError) as exc:
        status['blocker'] = type(exc).__name__
    # Persist even an interrupted bounded scan for a later resume.
    candidates = [r for r in pool if r not in protected]
    save(staged_manifest, candidates)
    result, changes = replacement_plan(original, candidates)
    validation = validate(result)
    if counts(result) != counts(original):
        raise ValueError('replacement changed split counts')
    report = dict(before=duration_summary(original), after=duration_summary(result),
                  changes=changes, replaced=len(changes),
                  unreplaced=sum(r['duration'] < 15 for r in result),
                  source_revision=revision, backup=str(backup), status=status,
                  network_bytes=args.max_bytes - http.remaining, requests=http.requests,
                  elapsed_seconds=time.monotonic() - http.started,
                  max_bytes=args.max_bytes, max_seconds=args.seconds, scan_rows=args.scan_rows,
                  min_duration=args.min_duration, validation=validation,
                  policy='20+ first; bounded prefix, not global maximum; superseded files retained')
    save(backup / 'replacement-plan.json', report)
    # Commit point: all audio and the complete proposed manifest validated first.
    save(manifest, result)
    for split in ('validation', 'test'):
        save(out / 'splits' / (split + '.json'), [r for r in result if r['split'] == split])
    save(out / 'replacement-provenance.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def acquire(args):
    targets = acquisition_targets(args.sources, args.per_split)
    out = args.output.resolve()
    if shutil.disk_usage(out.parent if out.parent.exists() else ROOT).free < args.max_bytes * 2:
        raise ValueError('insufficient disk for configured budget')
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    rows = load(manifest) if manifest.exists() else []
    if rows:
        validate(rows)
    http = BudgetHTTP(args.max_bytes, args.seconds)
    token = env_file_token(args.env_file) if args.env_file else existing_token()
    report = {'seed': args.seed, 'max_scan_per_split': args.scan_rows,
              'max_network_bytes': args.max_bytes, 'max_seconds': args.seconds,
              'transcript_audit': 'pending: not human validated',
              'training_independence': 'unknown; cannot assume training-data independence',
              'duration_policy': 'prefer 20 <= duration < 30; otherwise 0 < duration < 30; no concatenation',
              'sources': {}}
    for source in targets:
        status = report['sources'][source] = {}
        try:
            meta = http.json('https://huggingface.co/api/datasets/' + source, token)
            revision = meta['sha']
            if any(r['source'] == source and r['source_revision'] != revision for r in rows):
                raise ValueError('source revision changed; use a new output directory')
            license_name = meta.get('cardData', {}).get('license', 'unknown')
            status.update(revision=revision, license=license_name, gated=meta.get('gated', False))
            if source == 'google/fleurs' and args.official_fleurs:
                report['duration_policy'] = 'official archive order bounded prefix; 0 < duration < 30; label >=20 preferred; no global preferred exhaustion claim'
                fleurs_archive(http, out, rows, revision, targets[source], status, args.scan_rows, args.seed)
                status['status'] = 'bounded official archive acquisition completed'
                continue
            if meta.get('gated'):
                # A small range GET is unsafe if servers ignore Range. HEAD verifies access
                # without downloading a multi-GB shard. Never accept gated terms here.
                shard = next(x['rfilename'] for x in meta['siblings'] if x['rfilename'].endswith('.parquet'))
                try:
                    http.head(f'https://huggingface.co/datasets/{source}/resolve/{revision}/{shard}', token)
                    status['access'] = 'authorized'
                except urllib.error.HTTPError as exc:
                    status['access'] = f'blocked HTTP {exc.code}; request access manually at official source'
                    continue
            config = 'vi_vn' if source == 'google/fleurs' else 'default'
            splits = ['validation', 'test'] if source == 'google/fleurs' else ['train']
            for original_split in splits:
                candidates = []
                for offset in range(0, args.scan_rows, 100):
                    query = urllib.parse.urlencode(dict(dataset=source, config=config, split=original_split,
                                                        offset=offset, length=min(100, args.scan_rows - offset)))
                    payload = http.json('https://datasets-server.huggingface.co/rows?' + query, token)
                    for item in payload['rows']:
                        r = item['row']
                        d = advertised_duration(r)
                        if source == 'google/fleurs' and not 0 < d < 30:
                            continue
                        candidates.append((item['row_idx'], r, d))
                    if offset + 100 >= payload['num_rows_total']:
                        break
                # Fixed bounded pool; prefer long clips, seeded order within each tier.
                random.Random(args.seed).shuffle(candidates)
                candidates.sort(key=lambda x: not 20 <= x[2] < 30)
                status[original_split + '_scanned_candidates'] = len(candidates)
                durations = [c[2] for c in candidates if c[2] > 0]
                status[original_split + '_bounded_duration_seconds'] = dict(count=len(durations), min=min(durations, default=None), max=max(durations, default=None), global_cap=False)
                for index, r, advertised_seconds in candidates:
                    if source == 'google/fleurs':
                        split = original_split
                    else:
                        group = r.get('channel')
                        split = 'validation' if int(digest(str(group or index).encode()), 16) % 2 == 0 else 'test'
                    if sum(x['source'] == source and x['split'] == split for x in rows) >= targets[source]:
                        if source == 'google/fleurs':
                            break
                        continue
                    stable_id = f'{original_split}:{index}'
                    if any(x['source'] == source and x['id'] == stable_id for x in rows):
                        continue
                    audio = r.get('audio')
                    if not isinstance(audio, list) or not audio:
                        raise ValueError('viewer audio schema unsupported; no unbounded fallback')
                    url = audio[0]['src']
                    # Persist source revision from viewer cache URL, not mutable main metadata.
                    parts = urllib.parse.urlsplit(url).path.split('/--/')
                    audio_revision = parts[1] if len(parts) > 2 else revision
                    if source == 'google/fleurs' and len(audio_revision) != 40:
                        raise ValueError('missing immutable viewer revision')
                    if audio_revision != revision:
                        raise ValueError('viewer revision differs from source metadata; refusing stale selection')
                    local = out / (source.replace('/', '_') + f'_{original_split}_{index}.wav')
                    data = http.get(url)  # never forward credentials to asset URLs
                    local.write_bytes(data)
                    duration, pcm_hash = pcm_info(local)
                    if not 0 < duration < 30:
                        local.unlink()
                        continue
                    row = dict(source=source, source_revision=audio_revision, id=stable_id,
                               original_id=r.get('id', index), original_split=original_split, split=split,
                               source_url=f'https://huggingface.co/datasets/{source}/tree/{audio_revision}',
                               license=license_name, reference=r.get('transcription') or r.get('text'),
                               audio_path=str(local), duration=duration, sha256=digest(data), pcm_sha256=pcm_hash,
                               group=r.get('channel'), group_limit='speaker/recording identity unavailable unless channel supplied',
                               selection='preferred' if 20 <= duration < 30 else 'fallback: insufficient preferred clips in bounded pool',
                               selection_seed=args.seed, transcript_audit='pending')
                    try:
                        check_leaks(rows + [row])
                        if any(x['sha256'] == row['sha256'] or x['pcm_sha256'] == pcm_hash for x in rows):
                            continue
                    except ValueError:
                        continue
                    rows.append(row)
                    save(manifest, rows)
            status['status'] = 'bounded acquisition completed'
        except urllib.error.HTTPError as exc:
            status['blocker'] = f'HTTP {exc.code}'
        except Exception as exc:
            # Avoid logging signed URLs, request headers, tokens, or response bodies.
            status['blocker'] = type(exc).__name__
    save(manifest, rows)
    report.update(counts=counts(rows), network_bytes=args.max_bytes - http.remaining,
                  requests=http.requests, elapsed_seconds=time.monotonic() - http.started,
                   targets=targets, complete=all(sum(r['source'] == s and r['split'] == sp for r in rows) == n
                                                for s, n in targets.items() for sp in ('validation', 'test')))
    if rows:
        report['validation'] = validate(rows)
    save(out / 'acquisition-report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def choose(candidates, alpha):
    return max(candidates, key=lambda c: c['avg_logprob'] + alpha * c['lm'])


def require_validation(rows):
    if not rows or any(r['split'] != 'validation' for r in rows):
        raise ValueError('tuning requires nonempty validation-only rows')


def artifact(path, kind, manifest_hash, records, **metadata):
    metadata['records_sha256'] = digest(json.dumps(records, sort_keys=True, ensure_ascii=False).encode())
    metadata['configuration_sha256'] = digest(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode())
    save(path, dict(kind=kind, manifest_sha256=manifest_hash, records=records, **metadata))


REGENERATE = 'Invalid/legacy protocol: regenerate decode, score and validation tuning artifacts (never tune test)'


def validate_protocol(protocol, *, scored):
    """Validate the complete invariant protocol, not only its self-reported hash.

    Unknown fields are rejected so a producer cannot silently add semantics that
    an older consumer ignores. Version strings are opaque numeric-leading package
    versions; comparison is exact, including local/build suffixes.
    """
    def require(condition):
        if not condition:
            raise ValueError(REGENERATE)

    def fields(value, names):
        require(type(value) is dict and set(value) == set(names.split()))

    def text(value):
        return type(value) is str and bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./:+-]*', value))

    def sha(value, length):
        return type(value) is str and bool(re.fullmatch('[0-9a-f]{%d}' % length, value))

    def runtime(value, names):
        fields(value, 'python ' + names)
        for version in value.values():
            require(type(version) is str and bool(re.fullmatch(r'[0-9][A-Za-z0-9.!+_-]*(?:\.[A-Za-z0-9!+_-]+)*', version)))
        require(bool(re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', value['python'])))

    if scored:
        fields(protocol, 'acoustic lm')
        acoustic = protocol['acoustic']
        lm = protocol['lm']
        fields(lm, 'model_id revision tokenizer_id tokenizer_revision scoring segmentation runtime prefix_policy prefix_id vocabulary')
        require(text(lm['model_id']) and '/' in lm['model_id'])
        require(text(lm['tokenizer_id']) and '/' in lm['tokenizer_id'])
        require(sha(lm['revision'], 40) and sha(lm['tokenizer_revision'], 40))
        require(lm['scoring'] == 'normalized-mean-v3-model-prefix-or-special-excluded-pll')
        vocab = lm['vocabulary']
        fields(vocab, 'embedding_size config_size tokenizer_size tokenizer_sha256')
        require(all(type(vocab[k]) is int and vocab[k] > 0 for k in ('embedding_size', 'config_size', 'tokenizer_size')))
        require(vocab['embedding_size'] == vocab['config_size'])
        require(sha(vocab['tokenizer_sha256'], 64))
        if lm['segmentation'] == 'none':
            require(lm['prefix_policy'] in ('model-config-bos', 'model-config-eos'))
            require(type(lm['prefix_id']) is int and 0 <= lm['prefix_id'] < vocab['embedding_size'])
        else:
            require(lm['prefix_policy'] == 'tokenizer-specials-pll' and lm['prefix_id'] is None)
        require(lm['segmentation'] in ('none', 'pyvi'))
        runtime(lm['runtime'], 'torch transformers tokenizers pyvi numpy')
    else:
        acoustic = protocol
    fields(acoustic, 'checkpoint_sha256 decoding runtime')
    require(sha(acoustic['checkpoint_sha256'], 64))
    options = acoustic['decoding']
    fields(options, 'beam_size language task temperature without_timestamps fp16 ranker mel')
    require(type(options['beam_size']) is int and options['beam_size'] >= 2)
    require(options['language'] == 'vi' and options['task'] == 'transcribe')
    require(type(options['temperature']) in (int, float) and options['temperature'] == 0)
    require(options['without_timestamps'] is True and options['fp16'] is False)
    require(options['ranker'] == 'MaximumLikelihoodRanker:length')
    require(options['mel'] == 'standard-fp32-v1')
    runtime(acoustic['runtime'], 'torch openai-whisper numpy')


def verify_configuration(data):
    if type(data) is not dict:
        raise ValueError(REGENERATE)
    validate_protocol(data.get('protocol'), scored=data.get('kind') != 'nbest')
    metadata = {k: v for k, v in data.items()
                if k not in ('kind', 'manifest_sha256', 'records', 'configuration_sha256')}
    expected = digest(json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode())
    if data.get('configuration_sha256') != expected or not data.get('protocol'):
        raise ValueError(REGENERATE)
    if 'records' in data and data.get('records_sha256') != digest(
            json.dumps(data['records'], sort_keys=True, ensure_ascii=False).encode()):
        raise ValueError('candidate identity/content mismatch; regenerate artifacts')


def runtime_versions(names):
    import importlib.metadata
    import platform
    return dict(python=platform.python_version(), **{
        name: importlib.metadata.version(name) for name in names})


def partition(args):
    rows = load(args.manifest)
    validate(rows)
    for split in ('validation', 'test'):
        save(args.output / (split + '.json'), [r for r in rows if r['split'] == split])


def decode(args):
    rows = load(args.manifest)
    validate(rows)
    import torch
    import whisper
    sys.path.insert(0, str(ROOT))
    from src.inference import get_model, model_path
    from benchmarks.runners.generate_dev_nbest import NBestDecodingTask, extract_standard_mel
    model = get_model()
    beam_options = whisper.DecodingOptions(language='vi', task='transcribe', temperature=0,
                        beam_size=args.beam_size, without_timestamps=True, fp16=False)
    greedy_options = whisper.DecodingOptions(language='vi', task='transcribe', temperature=0,
                        without_timestamps=True, fp16=False)
    task = NBestDecodingTask(model, beam_options)
    records = []
    for r in rows:
        start = time.perf_counter()
        with torch.no_grad():
            mel = extract_standard_mel(r['audio_path'], model, fp16=False)
            candidates = task.run_nbest(mel.unsqueeze(0))[0]
            greedy = whisper.decode(model, mel, greedy_options).text
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not candidates:
            raise ValueError('empty N-best')
        for i, c in enumerate(candidates):
            c['id'] = digest((r['source'] + r['id'] + str(i) + json.dumps(c['tokens'])).encode())
        records.append(dict(source=r['source'], id=r['id'], split=r['split'],
                            greedy=greedy, candidates=candidates, seconds=time.perf_counter() - start))
    artifact(args.output, 'nbest', file_hash(args.manifest), records,
             protocol=dict(checkpoint_sha256=file_hash(model_path()),
                 decoding=dict(beam_size=args.beam_size, language='vi', task='transcribe',
                     temperature=0, without_timestamps=True, fp16=False,
                     ranker='MaximumLikelihoodRanker:length', mel='standard-fp32-v1'),
                 runtime=runtime_versions(['torch', 'openai-whisper', 'numpy'])),
             checkpoint_sha256=file_hash(model_path()), model_id='vinai/PhoWhisper-base',
             model_revision='local-checkpoint-sha256', whisper_version=whisper.__version__,
             config=dict(beam_size=args.beam_size, language='vi', temperature=0, fp16=False,
                          without_timestamps=True, acoustic='sum_logprob/(tokens+1); rank=sum_logprob/tokens', max_duration_exclusive=30))


def lm_context_limit(config, tokenizer, lm):
    limit = config.max_position_embeddings
    if lm == 'phobert':
        # RoBERTa non-padding positions start at padding_idx + 1.
        limit -= config.pad_token_id + 1
    return min(limit, tokenizer.model_max_length)


def check_context_length(length, limit):
    if length > limit:
        raise ValueError('LM context exceeded; refusing silent truncation')


def pll_positions(ids, tokenizer):
    special = set(tokenizer.all_special_ids)
    return [pos for pos, token in enumerate(ids) if int(token) not in special]


def gpt2_prefix(config, vocabulary_size):
    for name in ('bos', 'eos'):
        token = getattr(config, name + '_token_id', None)
        if type(token) is int and 0 <= token < vocabulary_size:
            return token, 'model-config-' + name
    raise ValueError('No valid model-config prefix')


def validate_candidate_ids(ids, vocabulary_size):
    if any(type(token) is not int or not 0 <= token < vocabulary_size for token in ids):
        raise ValueError('Candidate token outside model vocabulary')


def score(args):
    data = load(args.input)
    verify_configuration(data)
    if data['kind'] != 'nbest':
        raise ValueError('score requires persisted nbest artifact')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
    model_name = 'NlpHUST/gpt2-vietnamese' if args.lm == 'gpt2' else 'vinai/phobert-base-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=args.revision)
    cls = AutoModelForCausalLM if args.lm == 'gpt2' else AutoModelForMaskedLM
    model = cls.from_pretrained(model_name, revision=args.revision).eval()
    vocabulary_size = model.get_input_embeddings().num_embeddings
    if vocabulary_size != model.config.vocab_size:
        raise ValueError('Model vocabulary mismatch')
    prefix_id, prefix_policy = (gpt2_prefix(model.config, vocabulary_size) if args.lm == 'gpt2'
                                else (None, 'tokenizer-specials-pll'))
    vocabulary = dict(embedding_size=vocabulary_size, config_size=model.config.vocab_size,
                      tokenizer_size=len(tokenizer), tokenizer_sha256=digest(json.dumps(
                          tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()))
    if args.lm == 'phobert':
        from pyvi import ViTokenizer
    records = data['records']
    for r in records:
        start = time.perf_counter()
        for c in r['candidates']:
            text = normalize(c['text'])
            if args.lm == 'phobert':
                text = ViTokenizer.tokenize(text)
            ids = tokenizer.encode(text, return_tensors='pt', add_special_tokens=args.lm != 'gpt2')
            validate_candidate_ids(ids[0].tolist(), vocabulary_size)
            limit = lm_context_limit(model.config, tokenizer, args.lm)
            check_context_length(ids.shape[1], limit)
            with torch.no_grad():
                if args.lm == 'gpt2':
                    # Fixed model-config BOS then EOS fallback; never resize embeddings.
                    ids = torch.cat([torch.tensor([[prefix_id]]), ids], dim=1)
                    if ids.shape[1] > limit:
                        raise ValueError('GPT2 context exceeded')
                    c['lm'] = -float(model(ids, labels=ids).loss) if ids.shape[1] > 1 else -20.0
                else:
                    scores = []
                    for pos in pll_positions(ids[0], tokenizer):
                        masked = ids.clone()
                        masked[0, pos] = tokenizer.mask_token_id
                        lp = model(masked).logits[0, pos].log_softmax(-1)[ids[0, pos]]
                        scores.append(float(lp))
                    c['lm'] = sum(scores) / len(scores) if scores else -20.0
            if not math.isfinite(c['lm']):
                raise ValueError('nonfinite LM score')
        r['lm_seconds'] = time.perf_counter() - start
    artifact(args.output, 'scored', data['manifest_sha256'], records,
             protocol=dict(acoustic=data['protocol'], lm=dict(model_id=model_name,
                 revision=args.revision, tokenizer_id=model_name, tokenizer_revision=args.revision,
                  scoring='normalized-mean-v3-model-prefix-or-special-excluded-pll',
                  prefix_id=prefix_id, prefix_policy=prefix_policy, vocabulary=vocabulary,
                 segmentation='pyvi' if args.lm == 'phobert' else 'none',
                 runtime=runtime_versions(['torch', 'transformers', 'tokenizers', 'pyvi', 'numpy']))),
             nbest_sha256=file_hash(args.input), lm=args.lm, model_id=model_name,
             revision=args.revision, resolved_revision=getattr(model.config, '_commit_hash', None),
              scoring='mean token logprob; GPT2 model-config prefix, no automatic specials / PhoBERT PyVi segmented single-token PLL')


def join(manifest, scored):
    if scored['manifest_sha256'] != file_hash(manifest):
        raise ValueError('manifest hash mismatch')
    rows = load(manifest)
    mapping = {(r['source'], r['id']): r for r in rows}
    if len(mapping) != len(rows) or len(scored['records']) != len(rows):
        raise ValueError('sample count/identity mismatch')
    seen = set()
    for r in scored['records']:
        key = (r['source'], r['id'])
        if key in seen or key not in mapping or mapping[key]['split'] != r['split']:
            raise ValueError('sample identity/split mismatch')
        seen.add(key)
    return mapping


def tune(args):
    data = load(args.input)
    verify_configuration(data)
    if data['kind'] != 'scored':
        raise ValueError('tune requires scored artifact')
    mapping = join(args.manifest, data)
    records = data['records']
    require_validation(records)
    def errors(alpha):
        return sum(distance(normalize(mapping[(r['source'], r['id'])]['reference']).split(),
                            normalize(choose(r['candidates'], alpha)['text']).split()) for r in records)
    alpha = min([0, .05, .1, .2, .4], key=errors)
    frozen = dict(alpha=alpha, protocol=data['protocol'], selection='validation-only grid minimum word errors; ties smallest alpha',
                          validation_manifest_sha256=file_hash(args.manifest),
                          validation_scores_sha256=file_hash(args.input), lm=data['lm'],
                          model_id=data['model_id'], resolved_revision=data['resolved_revision'])
    frozen['configuration_sha256'] = digest(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode())
    save(args.output, frozen)


def evaluate(args):
    manifest = load(args.manifest)
    validate(manifest)
    g, p = load(args.gpt2), load(args.phobert)
    verify_configuration(g)
    verify_configuration(p)
    if g['protocol']['acoustic'] != p['protocol']['acoustic']:
        raise ValueError('shared acoustic protocol mismatch; regenerate artifacts')
    if g.get('lm') != 'gpt2' or p.get('lm') != 'phobert' or g.get('kind') != 'scored' or p.get('kind') != 'scored':
        raise ValueError('expected GPT2 and PhoBERT scored artifacts')
    mapping = join(args.manifest, g)
    join(args.manifest, p)
    if g['nbest_sha256'] != p['nbest_sha256']:
        raise ValueError('LMs must use identical persisted N-best')
    configs = [load(args.gpt2_config), load(args.phobert_config)]
    for data, config in zip([g, p], configs):
        verify_configuration(config)
        if config['protocol'] != data['protocol']:
            raise ValueError('validation/test protocol mismatch; regenerate artifacts')
        if any(config[k] != data[k] for k in ('lm', 'model_id', 'resolved_revision')):
            raise ValueError('frozen model configuration mismatch')
        if any(r['split'] == 'test' for r in manifest) and config['validation_manifest_sha256'] == file_hash(args.manifest):
            raise ValueError('test cannot be tuning manifest')
    second = {(r['source'], r['id']): r for r in p['records']}
    output = []
    for a in g['records']:
        b = second[(a['source'], a['id'])]
        if [(c['id'], c['text'], c['avg_logprob']) for c in a['candidates']] != [(c['id'], c['text'], c['avg_logprob']) for c in b['candidates']]:
            raise ValueError('candidate identity mismatch')
        row = mapping[(a['source'], a['id'])]
        ref = normalize(row['reference'])
        oracle = min(a['candidates'], key=lambda c: distance(ref.split(), normalize(c['text']).split()))
        texts = dict(greedy=a['greedy'], beam_top1=a['candidates'][0]['text'],
                     gpt2=choose(a['candidates'], configs[0]['alpha'])['text'],
                     phobert=choose(b['candidates'], configs[1]['alpha'])['text'], oracle=oracle['text'])
        output.append(dict(source=a['source'], id=a['id'], split=a['split'], reference=ref,
                           hypotheses={k: normalize(v) for k, v in texts.items()},
                           decode_seconds=a['seconds'], gpt2_seconds=a['lm_seconds'], phobert_seconds=b['lm_seconds']))
    summary = {}
    for split in sorted({r['split'] for r in output}):
        for source in ['pooled'] + SOURCES:
            subset = [r for r in output if r['split'] == split and (source == 'pooled' or r['source'] == source)]
            if not subset:
                continue
            metrics = {}
            for method in subset[0]['hypotheses']:
                we = ce = nw = nc = 0
                for r in subset:
                    ref, hyp = r['reference'], r['hypotheses'][method]
                    we += distance(ref.split(), hyp.split())
                    ce += distance(ref, hyp)
                    nw += len(ref.split())
                    nc += len(ref)
                metrics[method] = dict(wer=we / nw, cer=ce / nc, word_errors=we, words=nw)
            summary[split + ':' + source] = dict(samples=len(subset), metrics=metrics)
    save(args.output, dict(summary=summary, records=output,
                          manifest_sha256=file_hash(args.manifest),
                          inputs_sha256={str(x): file_hash(x) for x in [args.gpt2, args.phobert, args.gpt2_config, args.phobert_config]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    replacement = sub.add_parser('replace-short', help='bounded resumable native FLEURS replacement')
    replacement.add_argument('--output', type=Path, default=ROOT / 'data/benchmark/long-audio')
    replacement.add_argument('--max-bytes', type=int, default=256 * 1024 * 1024)
    replacement.add_argument('--seconds', type=int, default=240)
    replacement.add_argument('--scan-rows', type=int, default=500)
    replacement.add_argument('--min-duration', type=float, choices=(15, 20), default=20)
    replacement.set_defaults(func=replace_short)
    a = sub.add_parser('acquire', help='resume bounded acquisition; never accepts gated terms')
    a.add_argument('--output', type=Path, default=ROOT / 'data/benchmark/long-audio')
    a.add_argument('--scan-rows', type=int, default=300)
    a.add_argument('--max-bytes', type=int, default=64 * 1024 * 1024)
    a.add_argument('--seconds', type=int, default=180)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--env-file', type=Path, help='literal KEY=value HF credential file; never executed')
    a.add_argument('--sources', nargs='+', choices=SOURCES, default=SOURCES)
    a.add_argument('--per-split', type=int, default=10)
    a.add_argument('--official-fleurs', action='store_true', help='bounded official TSV/tar prefix instead of viewer')
    a.set_defaults(func=acquire)
    part = sub.add_parser('partition', help='validate combined manifest then emit validation/test manifests')
    part.add_argument('--manifest', type=Path, required=True)
    part.add_argument('--output', type=Path, required=True)
    part.set_defaults(func=partition)
    v = sub.add_parser('validate')
    v.add_argument('--manifest', type=Path, required=True)
    v.set_defaults(func=lambda a: print(json.dumps(validate(load(a.manifest)), indent=2)))
    d = sub.add_parser('decode')
    d.add_argument('--manifest', type=Path, required=True)
    d.add_argument('--output', type=Path, required=True)
    d.add_argument('--beam-size', type=int, default=5)
    d.set_defaults(func=decode)
    s = sub.add_parser('score')
    s.add_argument('--input', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--lm', choices=['gpt2', 'phobert'], required=True)
    s.add_argument('--revision', required=True, help='immutable HF commit SHA')
    s.set_defaults(func=score)
    t = sub.add_parser('tune')
    t.add_argument('--manifest', type=Path, required=True)
    t.add_argument('--input', type=Path, required=True)
    t.add_argument('--output', type=Path, required=True)
    t.set_defaults(func=tune)
    e = sub.add_parser('evaluate')
    for field in ['manifest', 'gpt2', 'phobert', 'gpt2-config', 'phobert-config', 'output']:
        e.add_argument('--' + field, type=Path, required=True)
    e.set_defaults(func=evaluate)
    args = parser.parse_args()
    if args.command in ('acquire', 'replace-short') and (args.scan_rows <= 0 or args.max_bytes <= 0 or args.seconds <= 0):
        parser.error('acquisition budgets must be positive')
    if args.command == 'decode' and args.beam_size < 2:
        parser.error('--beam-size must be at least 2')
    if args.command == 'score' and (len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision)):
        parser.error('--revision must be an immutable 40-character lowercase commit SHA')
    try:
        args.func(args)
    except ImportError as exc:
        parser.exit(2, f'Missing optional dependency {exc.name}; install requirements-benchmark.txt in a compatible Python environment.\n')
    except ValueError:
        # Never render arbitrary exception text (it can contain credentials/URLs).
        parser.exit(2, 'Invalid input, protocol or credential file. Check manifest/configuration; for legacy or mismatched benchmark artifacts regenerate decode, score and validation tuning outputs (never tune test).\n')


if __name__ == '__main__':
    main()
