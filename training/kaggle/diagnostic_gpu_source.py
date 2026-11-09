import os
os.environ['KAGGLE_DIAGNOSTIC_STAGE'] = 'gpu'
"""Standalone runtime ladder; no installation, model, or network operations."""
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat() + ' diagnostic startup', flush=True)

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from collections import Counter


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    os.replace(temporary, path)


class Reporter:
    def __init__(self, output):
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.lock = threading.Lock()
        self.stage = 'startup'
        self.count = 0

    def emit(self, status, **values):
        with self.lock:
            self.count += 1
            record = dict(utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.perf_counter() - self.started,
                          stage=self.stage, status=status, event_count=self.count, **values)
            print(json.dumps(record), flush=True)
            with (self.output / 'diagnostic_events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
                stream.flush()
            atomic(self.output / 'diagnostic_status.json', record)
            return record


def safe_audio(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'(?:wavs/)?audio\d+\.wav', name):
        raise ValueError('Invalid audio path syntax')
    root = root.resolve()
    path = (root / 'wavs' / Path(name).name).resolve()
    if not path.is_relative_to(root / 'wavs') or not path.is_file():
        raise ValueError('Missing or unsafe audio sample')
    return path


def metadata(root, reporter=None):
    root = root.resolve()
    if not (root / 'wavs').is_dir() or (root / 'wavs').is_symlink():
        raise ValueError('Required safe wavs directory missing')
    summaries, sample = {}, None
    for split in ('train', 'test'):
        path = root / f'metadata_{split}.csv'
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Required safe metadata file missing')
        speakers, rows, missing = Counter(), 0, 0
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='|')
            required = {'audio_file', 'text', 'speaker_name'}
            if not required <= set(reader.fieldnames or []):
                raise ValueError('Required metadata columns missing')
            for row in reader:
                if None in row or any(row.get(key) is None for key in required):
                    raise ValueError('Malformed metadata row')
                name = row['audio_file']
                if not re.fullmatch(r'(?:wavs/)?audio\d+\.wav', name):
                    raise ValueError('Invalid metadata audio path syntax')
                rows += 1
                speaker = ' '.join(unicodedata.normalize('NFC', row['speaker_name']).casefold().split())
                if speaker:
                    speakers[speaker] += 1
                else:
                    missing += 1
                if sample is None:
                    sample = safe_audio(root, name)
                if reporter and rows % 10000 == 0:
                    reporter.emit('progress', split=split, processed_rows=rows)
        if not rows:
            raise ValueError('Empty metadata split')
        summaries[split] = dict(rows=rows, columns=sorted(required), speakers=len(speakers),
                                missing_speaker_rows=missing,
                                min_rows_per_speaker=min(speakers.values(), default=0),
                                max_rows_per_speaker=max(speakers.values(), default=0))
    return summaries, sample


def run_child(code, arguments=(), timeout=120):
    # Children emit only fixed diagnostic aggregates, never metadata content.
    try:
        result = subprocess.run([sys.executable, '-u', '-c', code, *map(str, arguments)],
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        for value, stream in ((exc.stdout, sys.stdout), (exc.stderr, sys.stderr)):
            if value:
                print(value.decode(errors='replace') if isinstance(value, bytes) else value,
                      file=stream, flush=True)
        raise
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', file=sys.stderr, flush=True)
    summary = dict(returncode=result.returncode, stdout=result.stdout[-8000:], stderr=result.stderr[-8000:])
    if result.returncode:
        raise RuntimeError('Diagnostic subprocess failed: ' + json.dumps(summary))
    return summary


GPU_CODE = '''
import json
print('torch import start', flush=True)
import torch
print(json.dumps(dict(torch_version=torch.__version__, cuda_version=torch.version.cuda,
    cuda_available=torch.cuda.is_available(), devices=torch.cuda.device_count())), flush=True)
assert torch.cuda.is_available(), 'CUDA unavailable'
print(json.dumps(dict(device_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])), flush=True)
x = torch.ones(8, device='cuda')
y = (x + 1).sum()
torch.cuda.synchronize()
assert y.item() == 16
print('tiny CUDA tensor/synchronize success', flush=True)
'''

AUDIO_CODE = '''
import json, sys
print('soundfile import start', flush=True)
import soundfile as sf
import numpy as np
info = sf.info(sys.argv[1])
assert 0 < info.frames and 0 < info.samplerate <= 384000
assert 0 < info.channels <= 32 and 0 < info.duration <= 30
assert info.frames * info.channels <= 12000000, 'sample allocation bound'
audio, rate = sf.read(sys.argv[1], dtype='float32', always_2d=True)
assert audio.shape == (info.frames, info.channels) and np.isfinite(audio).all()
print(json.dumps(dict(shape=list(audio.shape), rate=rate, duration=info.duration, finite=True)), flush=True)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cpu', 'gpu', 'data'),
                        default=os.environ.get('KAGGLE_DIAGNOSTIC_STAGE', 'cpu'))
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/kaggle-diagnostic'))
    parser.add_argument('--root', type=Path,
                        default=Path('/kaggle/input/datasets/tuandaodev/viet-bud500-formatted'))
    parser.add_argument('--heartbeat-test-seconds', type=float, default=20)
    args = parser.parse_args(argv)
    if not 0 <= args.heartbeat_test_seconds <= 60:
        parser.error('heartbeat test duration must be between 0 and 60 seconds')
    reporter = Reporter(args.output)
    reporter.stage = args.stage
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(5):
            reporter.emit('heartbeat')

    worker = threading.Thread(target=heartbeat, name='diagnostic-heartbeat')
    worker.start()
    success, values = False, {}
    try:
        reporter.emit('start')
        if args.stage == 'cpu':
            stop.wait(args.heartbeat_test_seconds)
        else:
            values['gpu'] = run_child(GPU_CODE)
            reporter.emit('gpu_done', subprocess_summary=values['gpu'])
            if args.stage == 'data':
                values['metadata'], sample = metadata(args.root, reporter)
                reporter.emit('metadata_done', aggregates=values['metadata'])
                values['audio'] = run_child(AUDIO_CODE, [sample])
        success = True
    except BaseException as exc:
        values.update(error_type=type(exc).__name__, error=str(exc)[:8000],
                      traceback=traceback.format_exc()[-16000:])
        print(values['traceback'], file=sys.stderr, flush=True)
    finally:
        stop.set()
        worker.join()
        result = reporter.emit('success' if success else 'failure', **values)
        atomic(args.output / 'diagnostic_result.json', result)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
