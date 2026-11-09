"""Metadata-only selection; audio audit is restricted to the selected ten files."""
import csv
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def metadata(root, split):
    root = Path(root).resolve()
    rows, seen = [], set()
    with (root / f'metadata_{split}.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='|')
        if not {'audio_file', 'text', 'speaker_name'} <= set(reader.fieldnames or []):
            raise ValueError('Required columns: audio_file|text|speaker_name')
        for row in reader:
            name = row['audio_file']
            if not re.fullmatch(r'(?:wavs/)?audio\d+\.wav', name):
                raise ValueError('Unsafe metadata path')
            name = Path(name).name
            if name in seen:
                raise ValueError('Duplicate metadata ID')
            seen.add(name)
            speaker = ' '.join(unicodedata.normalize('NFC', row['speaker_name']).casefold().split())
            if not speaker or not row['text'].strip():
                raise ValueError('Empty speaker/text')
            rows.append(dict(id=name, speaker=speaker, text=row['text'],
                             audio_path=str(root / 'wavs' / name)))
    return rows


def select(rows, test_ids, seed=42):
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for row in rows:
        if int(Path(row['id']).stem[5:]) > 1250 and row['id'] not in test_ids:
            buckets[row['speaker']].append(row)
    speakers = sorted(buckets)
    rng.shuffle(speakers)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    reserved, count = [], 0
    for speaker in speakers:
        reserved.append(speaker)
        count += len(buckets[speaker])
        if count >= 2:
            break
    def balanced(names, size):
        result = []
        for index in range(max((len(buckets[s]) for s in names), default=0)):
            for speaker in names:
                if index < len(buckets[speaker]):
                    result.append(buckets[speaker][index])
                    if len(result) == size:
                        return result
        raise ValueError('Insufficient speaker-disjoint data')
    return balanced([s for s in speakers if s not in reserved], 8), balanced(reserved, 2)


def audit(rows, root, processor):
    import numpy as np
    import soundfile as sf
    import hashlib
    hashes = set()
    root = Path(root).resolve()
    for row in rows:
        path = Path(row['audio_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Missing/unsafe selected audio')
        info = sf.info(path)
        if info.frames <= 0 or info.samplerate <= 0 or not 0 < info.duration <= 30:
            raise ValueError('Invalid selected audio duration')
        audio, _ = sf.read(path, dtype='float32', always_2d=True)
        if not np.isfinite(audio).all():
            raise ValueError('Nonfinite selected audio')
        if not 0 < len(processor.tokenizer(row['text']).input_ids) <= 448:
            raise ValueError('Invalid label length')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        row['sha256'] = digest.hexdigest()
        if row['sha256'] in hashes:
            raise ValueError('Duplicate selected audio bytes')
        hashes.add(row['sha256'])
