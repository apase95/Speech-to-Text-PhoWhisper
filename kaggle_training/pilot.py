"""Additive, selected-subset-only HF pilot. No corpus audio walk or Kaggle API."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from kaggle_training.train import AudioDataset, digest, Progress

PROGRESS = Progress()

EXCLUSIONS = ('viet_bud500_1000.csv', 'viet_bud500_dev_250.csv')
DECODE = dict(language='vi', task='transcribe', return_timestamps=False,
              num_beams=1, do_sample=False, max_length=448, use_cache=True)


def validate_layout(root):
    if not (root / 'wavs').is_dir() or not all(
            (root / f'metadata_{split}.csv').is_file() for split in ('train', 'test')):
        raise ValueError(f'Invalid dataset layout at {root}: require metadata_train.csv, metadata_test.csv and wavs/')
    for split in ('train', 'test'):
        metadata(root, split)


def source_provenance():
    root = Path(__file__).resolve().parents[1]
    names = ('kaggle_training/train.py', 'kaggle_training/pilot.py',
             'kaggle_training/build_pilot_source.py', 'kaggle_training/test_smoke.py',
             'kaggle_training/test_pilot.py', 'benchmarks/metrics.py')
    hashes = {name: digest(root / name) for name in names}
    aggregate = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(git_revision=None, base_git_revision='b15d7fb0141f385fd4bc6014df915e1c37212467',
                executed_source_files=hashes, executed_source_sha256=aggregate,
                implementation_id='openai/gpt-6.1-sol:scoped-follow-up')


def dependency_versions():
    import importlib
    from importlib.metadata import version
    versions = {}
    for name in ('torch', 'transformers', 'accelerate', 'huggingface_hub',
                 'numpy', 'scipy', 'soundfile', 'jiwer'):
        try:
            importlib.import_module(name)
            versions[name] = version(name)
        except Exception as exc:
            raise RuntimeError(f'Cannot import {name}: {exc}. Fix the Kaggle image dependency; '
                               'do not blindly install legacy requirements on Python 3.13.') from exc
    return versions


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def metadata(root, split):
    """Read metadata only; never stat unselected audio."""
    rows = {}
    with (root / f'metadata_{split}.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='|')
        if not {'audio_file', 'text', 'speaker_name'} <= set(reader.fieldnames or []):
            raise ValueError('Required audio_file|text|speaker_name columns')
        for raw in reader:
            name = raw['audio_file']
            if not re.fullmatch(r'(?:wavs/)?audio\d+\.wav', name):
                raise ValueError(f'Unsafe metadata path: {name}')
            name = Path(name).name
            speaker = ' '.join(unicodedata.normalize('NFC', raw['speaker_name']).casefold().split())
            text = unicodedata.normalize('NFC', raw['text']).strip()
            row = dict(id=name, audio_path=str(root / 'wavs' / name), text=text, speaker=speaker)
            if name in rows and rows[name] != row:
                raise ValueError(f'Conflicting duplicate metadata: {name}')
            rows[name] = row
    return list(rows.values())


def epoch_steps(count, batch, accumulation, world):
    if min(count, batch, accumulation, world) <= 0:
        raise ValueError('Positive count/batch/accumulation/world required')
    return math.ceil(math.ceil(count / (batch * world)) / accumulation)


def balanced(rows, cap, rng):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row['speaker']].append(row)
    speakers = sorted(buckets)
    rng.shuffle(speakers)
    for speaker in speakers:
        rng.shuffle(buckets[speaker])
    queue = deque(speakers)
    result = []
    while queue and len(result) < cap:
        speaker = queue.popleft()
        result.append(buckets[speaker].pop())
        if buckets[speaker]:
            queue.append(speaker)
    return result


def select(rows, excluded, train_cap=1000, validation_cap=200, seed=42):
    eligible = [r for r in rows if r['id'] not in excluded and
                int(Path(r['id']).stem[5:]) > 1250 and r['speaker'] and r.get('text', 'x').strip()]
    buckets = Counter(r['speaker'] for r in eligible)
    speakers = sorted(buckets)
    rng = random.Random(seed)
    rng.shuffle(speakers)
    validation_speakers, count = set(), 0
    for speaker in speakers:
        if count >= validation_cap:
            break
        validation_speakers.add(speaker)
        count += buckets[speaker]
    train = balanced([r for r in eligible if r['speaker'] not in validation_speakers], train_cap, rng)
    validation = balanced([r for r in eligible if r['speaker'] in validation_speakers], validation_cap, rng)
    if len(train) != train_cap or len(validation) != validation_cap:
        raise ValueError('Insufficient examples in deterministic speaker split; no clip-random fallback')
    return train, validation, dict(eligible_examples=len(eligible), eligible_speakers=len(buckets),
        excluded_metadata_examples=len(rows) - len(eligible), validation_speakers=sorted(validation_speakers))


def prepare(root, output, exclusions, processor):
    root = root.resolve()
    rows, official = metadata(root, 'train'), metadata(root, 'test')
    official_ids = {r['id'] for r in official}
    excluded, hashes, provenance = set(official_ids), set(), []
    for path in exclusions:
        with path.open(encoding='utf-8', newline='') as stream:
            entries = list(csv.DictReader(stream))
        for entry in entries:
            excluded.add(Path(entry['id']).stem + '.wav')
            hashes.add(entry['sha256'])
        provenance.append(dict(path=str(path), sha256=digest(path), rows=len(entries)))
    train, validation, stats = select(rows, excluded)
    selected = train + validation
    seen = set(hashes)
    paths = set()
    import soundfile as sf
    dataset = AudioDataset(selected, processor)
    for index, row in enumerate(selected):
        path = Path(row['audio_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing/unsafe selected file: {path}')
        if str(path) in paths:
            raise ValueError('Selected path overlap')
        paths.add(str(path))
        dataset[index]  # Actual decode, finite audio, duration and tokenizer limit checks.
        row['sha256'] = digest(path)
        if row['sha256'] in seen:
            raise ValueError(f'Selected duplicate/excluded audio hash: {row["id"]}')
        seen.add(row['sha256'])
        row['audio_seconds'] = sf.info(path).duration
        PROGRESS.update('selected_audio_audit', processed=index + 1, total=len(selected))
    for key in ('id', 'speaker', 'audio_path', 'sha256'):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f'Train/validation {key} overlap')
    if {r['id'] for r in selected} & official_ids:
        raise ValueError('Official test ID overlap')
    for name, group in (('train', train), ('validation', validation)):
        write(output / f'{name}.json', group)
        counts = Counter(r['speaker'] for r in group)
        stats[name] = dict(examples=len(group), speakers=len(counts), speaker_counts=dict(counts),
            maximum_speaker_share=max(counts.values()) / len(group),
            audio_seconds=sum(r['audio_seconds'] for r in group),
            manifest_sha256=digest(output / f'{name}.json'))
    stats.update(seed=42, exclusions=provenance, official_test_ids=len(official_ids),
        metadata_hashes={s: digest(root / f'metadata_{s}.csv') for s in ('train', 'test')},
        audit_scope='Selected 1200 audio bytes versus each other and supplied exclusion hashes only. '
            'Official test IDs excluded; official test audio hashes not scanned. Global byte/near '
            'duplicates, speaker aliases and pretraining exposure are not guaranteed excluded.')
    write(output / 'prepare.json', stats)
    return train, validation


def environment():
    dependencies = dependency_versions()
    import torch
    from importlib.metadata import version
    return dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
        transformers=version('transformers'), accelerate=version('accelerate'), jiwer=version('jiwer'),
        cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        **source_provenance(), dependencies=dependencies, runtime='HF Transformers, not .pt')


def checkpoint_hash(directory):
    """Deterministic relative-name/content hash of final HF artifacts, not Trainer state."""
    files = sorted(p for p in directory.iterdir() if p.is_file() and
                   (p.suffix in ('.safetensors', '.bin') or p.name in
                    ('config.json', 'generation_config.json', 'preprocessor_config.json',
                     'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json')))
    if not any(p.suffix in ('.safetensors', '.bin') for p in files):
        raise ValueError('No final model weights')
    return hashlib.sha256(''.join(f'{p.name}\0{digest(p)}\n' for p in files).encode()).hexdigest()


def benchmark(source, revision, processor, rows, manifest, output, identity):
    import torch
    from transformers import WhisperForConditionalGeneration
    from benchmarks.metrics import corpus_error_rates, normalize_vietnamese
    output.mkdir(exist_ok=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with PROGRESS.stage(output.name + '_model_load'):
        model = WhisperForConditionalGeneration.from_pretrained(source, revision=revision).float().cuda().eval()
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    predictions = []
    dataset = AudioDataset(rows, processor)
    with (output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            torch.cuda.synchronize()
            started = time.perf_counter()
            error, hypothesis = None, ''
            try:
                features = torch.as_tensor(dataset[index]['input_features']).unsqueeze(0).float().cuda()
                with torch.inference_mode():
                    tokens = model.generate(features, **DECODE)
                hypothesis = processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            prediction = dict(id=row['id'], audio_path=row['audio_path'], split='validation',
                reference=row['text'], hypothesis=hypothesis,
                normalized_reference=normalize_vietnamese(row['text']),
                normalized_hypothesis=normalize_vietnamese(hypothesis),
                audio_seconds=row['audio_seconds'], inference_seconds=elapsed,
                rtf=elapsed / row['audio_seconds'], error=error)
            stream.write(json.dumps(prediction, ensure_ascii=False) + '\n')
            stream.flush()
            predictions.append(prediction)
            PROGRESS.update(output.name, processed=index + 1, total=len(rows))
    rates = corpus_error_rates([r['reference'] for r in predictions], [r['hypothesis'] for r in predictions])
    audio = sum(r['audio_seconds'] for r in predictions)
    elapsed = sum(r['inference_seconds'] for r in predictions)
    summary = dict(run_id=output.parent.name, name=output.name, samples=len(rows),
        failures=sum(r['error'] is not None for r in predictions), **rates,
        total_audio_seconds=audio, total_inference_seconds=elapsed, rtf=elapsed / audio,
        model_load_seconds=load_seconds, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_process_rss_scope='Parent process lifetime high-water mark, including preparation and prior benchmarks; not per-run or comparable baseline/post.',
        manifest=str(manifest), manifest_sha256=digest(manifest), checkpoint=source,
        checkpoint_sha256=identity if revision is None else None,
        checkpoint_sha256_explanation=None if revision is None else 'HF immutable commit recorded instead of local artifact hash',
        resolved_revision=revision, decode_options=DECODE, environment=environment(),
        runtime_note='HF FP32 greedy; no .pt deployment parity implied. Timing includes audio/features/decode; no warmup.')
    write(output / 'summary.json', summary)
    del model
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('/kaggle/working/outputs/benchmarks'))
    parser.add_argument('--model', default='vinai/PhoWhisper-base')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    run = args.output / ('pilot1000-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run.mkdir(parents=True, exist_ok=False)
    global PROGRESS
    PROGRESS = Progress(run / 'progress.json', args.output / 'current-progress.json')
    os.environ['PILOT_PROGRESS'] = str(run / 'progress.json')
    os.environ['PILOT_CURRENT_PROGRESS'] = str(args.output / 'current-progress.json')
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    try:
        with PROGRESS.stage('pilot'):
            execute(args, run)
    except Exception as exc:
        write(run / 'FAILED.json', dict(error=f'{type(exc).__name__}: {exc}', success=False))
        raise


def execute(args, run):
    with PROGRESS.stage('layout_validation'):
        validate_layout(args.root)
    with PROGRESS.stage('dependency_imports'):
        dependency_versions()
    import inspect
    import torch
    from huggingface_hub import model_info
    from transformers import WhisperProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed
    env = environment()
    write(run / 'preflight.json', env)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise RuntimeError('Pilot requires exactly one visible CUDA GPU and one process')
    if 'processing_class' not in inspect.signature(Seq2SeqTrainer.__init__).parameters or \
            'eval_strategy' not in inspect.signature(Seq2SeqTrainingArguments.__init__).parameters:
        raise RuntimeError('Incompatible Transformers Trainer API; select compatible stack, do not blindly downgrade Python')
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    with PROGRESS.stage('hf_metadata'):
        revision = model_info(args.model, revision=args.revision, timeout=60).sha
    with PROGRESS.stage('processor_load'):
        processor = WhisperProcessor.from_pretrained(args.model, revision=revision, language='vi', task='transcribe')
    processor.tokenizer.set_prefix_tokens(language='vi', task='transcribe', predict_timestamps=False)
    exclusions = [Path(__file__).parent / 'exclusions' / name for name in EXCLUSIONS]
    with PROGRESS.stage('selected_audio_audit'):
        train, validation = prepare(args.root, run, exclusions, processor)
    write(run / 'smoke_train.json', train[:2])
    write(run / 'smoke_validation.json', validation[:2])
    def launch(output, train_manifest, val_manifest, steps, save, resume=None, batch=2, accumulation=4):
        command = [sys.executable, '-u', '-m', 'kaggle_training.train', 'train', '--model', args.model,
            '--revision', revision, '--precision', 'fp32', '--require-finite-training',
            '--seed', '42', '--learning-rate', '1e-5',
            '--manifest', str(train_manifest), '--validation', str(val_manifest), '--output', str(output),
            '--max-steps', str(steps), '--save-steps', str(save), '--batch', str(batch),
            '--accumulation', str(accumulation)]
        if resume:
            command.extend(['--resume', str(resume)])
        with PROGRESS.stage(output.name):
            subprocess.run(command, check=True, env=dict(os.environ, PYTHONUNBUFFERED='1'))
        state = json.loads((output / 'trainer_state.json').read_text())
        if state['global_step'] != steps:
            raise RuntimeError('Trainer did not reach requested steps')
        metrics = json.loads((output / 'metrics.json').read_text())
        if not all(math.isfinite(v) for v in metrics.values() if isinstance(v, (float, int))):
            raise RuntimeError('Nonfinite training/evaluation metrics')
        losses = [r['loss'] for r in state['log_history'] if 'loss' in r]
        if not losses or not all(math.isfinite(v) and v > 0 for v in losses):
            raise RuntimeError('Missing/invalid training loss')
        return state
    smoke = run / 'smoke'
    launch(smoke, run / 'smoke_train.json', run / 'smoke_validation.json', 1, 1, batch=1, accumulation=1)
    checkpoint = smoke / 'checkpoint-1'
    for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt'):
        if not (checkpoint / name).is_file():
            raise RuntimeError(f'Missing resume artifact: {name}')
    resumed = launch(run / 'smoke-resume', run / 'smoke_train.json', run / 'smoke_validation.json',
                     2, 1, checkpoint, batch=1, accumulation=1)
    write(run / 'smoke_verified.json', dict(initial_step=1, resumed_step=resumed['global_step'],
        verification_scope='Advancement-only: scheduler horizon changes from 1 to 2; optimizer/scheduler continuity not verified.',
        optimizer_continuity_verified=False,
        checkpoint_state_sha256=digest(checkpoint / 'trainer_state.json')))
    with PROGRESS.stage('baseline'):
        baseline = benchmark(args.model, revision, processor, validation, run / 'validation.json', run / 'baseline', revision)
    steps = epoch_steps(len(train), 2, 4, 1)
    write(run / 'setup.json', dict(model=args.model, resolved_revision=revision, seed=42, precision='fp32',
        batch=2, accumulation=4, world_size=1, train_examples=1000, validation_examples=200,
        epochs=1, calculated_steps=steps, learning_rate=1e-5,
        checkpoint_policy='Compare final one-epoch checkpoint, no best-checkpoint selection; official test never evaluated.',
        source_sha256=digest(Path(__file__)), decode_options=DECODE))
    final = run / 'train-final'
    launch(final, run / 'train.json', run / 'validation.json', steps, steps)
    with PROGRESS.stage('post'):
        post = benchmark(str(final), None, processor, validation, run / 'validation.json', run / 'post', checkpoint_hash(final))
    PROGRESS.update('reporting', 'start')
    comparison = dict(baseline=baseline, post=post,
        delta_wer=post['wer'] - baseline['wer'], delta_cer=post['cer'] - baseline['cer'],
        inference_seconds_delta=post['total_inference_seconds'] - baseline['total_inference_seconds'],
        validation_only=True, success=baseline['failures'] == post['failures'] == 0)
    write(run / 'comparison.json', comparison)
    (run / 'comparison.md').write_text(
        f'# HF pilot1000 validation comparison\n\n1000 train / 200 speaker-disjoint validation; seed 42.\n'
        f'Final checkpoint, one epoch, FP32 greedy; not .pt runtime parity.\n\n'
        f'|Metric|Baseline|Post|\n|---|---:|---:|\n|WER|{baseline["wer"]}|{post["wer"]}|\n'
        f'|CER|{baseline["cer"]}|{post["cer"]}|\n|Failures|{baseline["failures"]}|{post["failures"]}|\n', encoding='utf-8')
    if not comparison['success']:
        raise RuntimeError('Inference failures: reports preserved, run not successful')
    write(run / 'SUCCESS.json', dict(success=True, model_revision=revision, final_hash=post['checkpoint_sha256']))
    PROGRESS.update('reporting', 'done', success=True)


if __name__ == '__main__':
    main()
