"""Build a small self-contained candidate; never uploads or launches Kaggle."""
import hashlib
import json
from pathlib import Path


def build(root):
    trainer = (root / 'kaggle_training/train.py').read_text()
    helper = (root / 'kaggle_training/smoke10.py').read_text()
    cells = [
'''import os, sys, json, time, threading, subprocess, hashlib
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
def event(stage, status):
    print(datetime.now(timezone.utc).isoformat(), stage, status, flush=True)
@contextmanager
def stage(name):
    event(name, 'start')
    stop = threading.Event()
    def pulse():
        while not stop.wait(5): event(name, 'heartbeat')
    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    except BaseException:
        event(name, 'failed')
        raise
    else:
        event(name, 'done')
    finally:
        stop.set()
        thread.join()
event('startup/config', 'start')
os.environ.update(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', HF_HUB_DOWNLOAD_TIMEOUT='60', HF_HUB_ETAG_TIMEOUT='60')
OUT = Path('/kaggle/working/outputs/benchmarks') / ('smoke10-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
OUT.mkdir(parents=True, exist_ok=False)
MODEL = 'vinai/PhoWhisper-base'
def command(argv, log, timeout=None):
    with stage(log), (OUT / log).open('w') as stream:
        child = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
        timer = threading.Timer(timeout, child.kill) if timeout else None
        if timer: timer.start()
        try:
            for line in child.stdout:
                print(line, end='', flush=True)
                stream.write(line); stream.flush()
            if child.wait(): raise RuntimeError('Command failed: ' + log)
        finally:
            if timer: timer.cancel()
            if child.poll() is None: child.kill(); child.wait()
event('startup/config', 'done')
''',
'''with stage('dependencies/versions'):
    import importlib, importlib.util
    if importlib.util.find_spec('jiwer') is None:
        command([sys.executable, '-u', '-m', 'pip', 'install', '--retries', '2', '--timeout', '30', 'jiwer==3.0.5'], 'pip.log', 600)
    from importlib.metadata import version
    versions = {}
    for name in ('torch', 'transformers', 'accelerate', 'huggingface_hub', 'numpy', 'scipy', 'soundfile', 'jiwer'):
        try: importlib.import_module(name)
        except ImportError as exc: raise RuntimeError('Fix image dependency: ' + name) from exc
        versions[name] = version(name)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one visible CUDA GPU')
    print(versions, flush=True)
    (OUT / 'dependencies.json').write_text(json.dumps(versions, indent=2))
''',
helper + '''
with stage('metadata/subset/audio-validation'):
    roots = [Path('/kaggle/input/datasets/tuandaodev/viet-bud500-formatted'), Path('/kaggle/input/viet-bud500-formatted')]
    ROOT = next((p for p in roots if all((p / ('metadata_' + s + '.csv')).is_file() for s in ('train', 'test'))), None)
    if ROOT is None: raise ValueError('Expected exact dataset metadata layout')
    rows, official = metadata(ROOT, 'train'), metadata(ROOT, 'test')
    train_rows, val_rows = select(rows, {r['id'] for r in official})
    from huggingface_hub import model_info
    from transformers import WhisperProcessor
    REVISION = model_info(MODEL, timeout=60).sha
    processor = WhisperProcessor.from_pretrained(MODEL, revision=REVISION, language='vi', task='transcribe')
    audit(train_rows + val_rows, ROOT, processor)
    for name, subset in [('train', train_rows), ('validation', val_rows)]:
        (OUT / (name + '.json')).write_text(json.dumps(subset, ensure_ascii=False, indent=2))
    COUNTS = dict(train=8, validation=2, corpus_speakers=len({r['speaker'] for r in rows}),
                  train_speakers=len({r['speaker'] for r in train_rows}), validation_speakers=len({r['speaker'] for r in val_rows}), speaker_disjoint=not bool({r['speaker'] for r in train_rows} & {r['speaker'] for r in val_rows}))
    print(COUNTS, flush=True)
''',
f'''with stage('emit-trainer'):
    TRAINER_SOURCE = {trainer!r}
    TRAINER = OUT / 'train.py'
    TRAINER.write_text(TRAINER_SOURCE)
    PROVENANCE = dict(base_git_revision='b15d7fb0141f385fd4bc6014df915e1c37212467', executed_git_revision=None,
        trainer_sha256=hashlib.sha256(TRAINER_SOURCE.encode()).hexdigest(), resolved_revision=REVISION,
        notebook_source_sha256=NOTEBOOK_SOURCE_SHA256)
    (OUT / 'provenance.json').write_text(json.dumps(PROVENANCE, indent=2))
    COMMON = ['--model', MODEL, '--revision', REVISION, '--precision', 'fp32', '--batch', '1', '--accumulation', '1', '--learning-rate', '1e-5', '--workers', '0', '--seed', '42']
''',
'''with stage('baseline'):
    command([sys.executable, '-u', str(TRAINER), 'evaluate', '--manifest', str(OUT / 'validation.json'), '--output', str(OUT / 'baseline')] + COMMON, 'baseline.log')
    baseline_run = json.loads((OUT / 'baseline/run.json').read_text())
    if baseline_run['resolved_revision'] != REVISION: raise ValueError('Baseline revision mismatch')
''',
'''with stage('training-eight-steps/post'):
    command([sys.executable, '-u', str(TRAINER), 'train', '--manifest', str(OUT / 'train.json'), '--validation', str(OUT / 'validation.json'), '--output', str(OUT / 'trained'), '--max-steps', '8', '--save-steps', '8', '--require-finite-training'] + COMMON, 'training.log')
    state = json.loads((OUT / 'trained/trainer_state.json').read_text())
    if state['global_step'] != 8: raise ValueError('Expected exactly eight optimizer steps')
''',
'''with stage('smoke-report'):
    report = dict(counts=COUNTS, provenance=PROVENANCE,
        baseline=json.loads((OUT / 'baseline/metrics.json').read_text()),
        post=json.loads((OUT / 'trained/metrics.json').read_text()),
        caveat='Smoke only, not quality evidence. Subset byte audit only; no global byte, near-duplicate, official-test byte or pretraining exclusion guarantee.',
        logs=['baseline.log', 'training.log'], output=str(OUT))
    (OUT / 'smoke-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
''']
    # Hash canonical executable cells before adding the hash declaration (no self-reference).
    digest = hashlib.sha256(json.dumps(cells, ensure_ascii=False).encode()).hexdigest()
    cells[0] = f'NOTEBOOK_SOURCE_SHA256 = {digest!r}\n' + cells[0]
    return dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
                cells=[dict(cell_type='code', id=f'smoke10-{i}', metadata={}, execution_count=None, outputs=[], source=s.splitlines(True)) for i, s in enumerate(cells)])


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / 'temp_session_kaggle/phowhisper2-smoke10'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'phowhisper2.ipynb').write_text(json.dumps(build(root), ensure_ascii=False, indent=1))
    metadata = json.loads((root / 'temp_session_kaggle/phowhisper2-before-smoke10/kernel-metadata.json').read_text())
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
