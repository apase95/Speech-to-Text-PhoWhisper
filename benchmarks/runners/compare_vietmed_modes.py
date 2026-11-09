"""Offline public-API VietMed100 four-cell workflow comparison."""
import argparse
from contextlib import ExitStack
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from unittest.mock import patch
from benchmarks.checkpoints import legacy_base_file

from benchmarks.metrics import corpus_error_rates, normalize_vietnamese, paired_bootstrap_test
from benchmarks.runners.vietmed100_experiment import sha, REVISION

CELLS = ('base-fast', 'base-rescore', 'small-fast', 'small-rescore')
REVISIONS = dict(base='7ebdb9e88f5cc5271fb88f4d642c82ff9388650e', small='a86b604c346caf7148c37512eafe783a16420adb')
CHECKPOINTS = {k: Path(f'models/official/acquisition-20261109/vinai/PhoWhisper-{k}/{v}/native.pt') for k, v in REVISIONS.items()}
LM = 'NlpHUST/gpt2-vietnamese'
LM_REVISION = '65818d14816b42be09e2201933bf07106d9a2647'
LM_PATH = Path.home()/f'.cache/huggingface/hub/models--NlpHUST--gpt2-vietnamese/snapshots/{LM_REVISION}'
SOURCE = ('benchmarks/runners/compare_vietmed_modes.py', 'benchmarks/metrics.py',
          'benchmarks/runners/vietmed100_experiment.py', 'src/phowhisper/pipeline.py',
          'src/phowhisper/rescorer.py', 'src/phowhisper/text/normalization.py')
PROTOCOL = dict(seed=42, threads=4, batch_size=1, time_stretch_rate=1.0,
    asr_precision='FP16 on CUDA', warmup='one excluded dev[0] call per cell',
    fast='public pipeline defaults, including Whisper temperature fallback',
    rescoring=dict(beam_size=5, alpha=.2, beta=-.02, gamma=-.2,
        confidence_threshold=-.020, lm=LM, revision=LM_REVISION,
        lm_device='cuda', lm_precision='FP32'),
    normalization='NFC, casefold, punctuation to spaces, collapse whitespace; accents retained',
    scope='official checkpoints, all100 descriptive and split results; no local finetuned weights',
    timing='audio IO and inference; loads and one warmup reported separately')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def preflight():
    import soundfile as sf
    historical = json.loads(Path('outputs/benchmarks/vietmed100-trained-20261014T200306Z/report.json').read_text())
    rows, hashes, ids = [], {}, set()
    used = {k: set() for k in ('speaker_name', 'seq_name', 'audio_name', 'utterance_id', 'source_sha256')}
    for split, count in (('train', 70), ('dev', 15), ('test', 15)):
        path = Path(f'data/benchmark/vietmed100/{split}.json')
        for suffix in ('json', 'csv'):
            p = path.with_suffix('.'+suffix)
            hashes[str(p)] = sha(p)
            if hashes[str(p)] != historical['data'][split][suffix+'_sha256']:
                raise ValueError('Frozen manifest hash mismatch')
        group = json.loads(path.read_text())
        if len(group) != count:
            raise ValueError('Wrong split count')
        for key in used:
            values = {str(r[key]).strip().casefold() for r in group}
            if '' in values or values & used[key]:
                raise ValueError('Cross-split overlap: '+key)
            used[key].update(values)
        for row in group:
            info = sf.info(row['audio_path'])
            if (row['id'] in ids or row['split'] != split or not normalize_vietnamese(row['reference'])
                    or sha(row['audio_path']) != row['sha256']):
                raise ValueError('Invalid ID/reference/audio hash')
            if (info.samplerate != 16000 or info.channels != 1 or not 0 < info.duration <= 30
                    or abs(info.duration-row['actual_duration']) > 1e-6):
                raise ValueError('Invalid audio format/duration; no truncation allowed')
            ids.add(row['id'])
            hashes[row['audio_path']] = row['sha256']
        rows.extend(group)
    return rows, hashes


def edits(refs, hyps, characters=False):
    import jiwer
    result = (jiwer.process_characters if characters else jiwer.process_words)(
        [normalize_vietnamese(x) for x in refs], [normalize_vietnamese(x) for x in hyps])
    return {k: getattr(result, k) for k in ('substitutions', 'deletions', 'insertions', 'hits')}


def summarize(rows):
    refs, hyps = [r['reference'] for r in rows], [r['hypothesis'] for r in rows]
    rates = corpus_error_rates(refs, hyps)
    latencies = sorted(r['inference_seconds'] for r in rows)
    audio = sum(r['audio_seconds'] for r in rows)
    return dict(rates, samples=len(rows), word_edits=edits(refs, hyps), character_edits=edits(refs, hyps, True),
                exact_match=sum(normalize_vietnamese(a) == normalize_vietnamese(b) for a, b in zip(refs, hyps))/len(rows),
                total_audio_seconds=audio, total_inference_seconds=sum(latencies), rtf=sum(latencies)/audio,
                latency_mean_seconds=statistics.mean(latencies), latency_median_seconds=statistics.median(latencies),
                latency_p95_seconds=latencies[math.ceil(.95*len(rows))-1])


def validate(rows, expected):
    if len(rows) != len(expected) or not rows:
        raise ValueError('Incomplete predictions')
    for row, source in zip(rows, expected, strict=True):
        if row.get('error'):
            raise ValueError('Failed sample')
        for key in ('id', 'split', 'reference', 'audio_path'):
            if row[key] != source[key]:
                raise ValueError('Pair mismatch: '+key)
        if row['audio_sha256'] != source['sha256']:
            raise ValueError('Pair audio hash mismatch')


def compare(cells, expected):
    if set(cells) != set(CELLS):
        raise ValueError('Four complete cells required')
    for rows in cells.values():
        validate(rows, expected)
    result = {}
    for a, b in (('base-fast', 'small-fast'), ('base-rescore', 'small-rescore'),
                 ('base-fast', 'base-rescore'), ('small-fast', 'small-rescore')):
        result[a+'->'+b] = {}
        for split in ('all100', 'train', 'dev', 'test'):
            left = [r for r in cells[a] if split == 'all100' or r['split'] == split]
            right = [r for r in cells[b] if split == 'all100' or r['split'] == split]
            if not left:
                continue
            result[a+'->'+b][split] = dict(
                a=summarize(left), b=summarize(right),
                paired_wer_bootstrap_pp=paired_bootstrap_test([r['reference'] for r in left],
                    [r['hypothesis'] for r in left], [r['hypothesis'] for r in right], n_bootstraps=2000, seed=42))
    return result


def pinned_loader():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LM, revision=LM_REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(LM, revision=LM_REVISION, local_files_only=True)
    return model.float().to('cuda').eval(), tokenizer


class FailWarnings(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.WARNING:
            raise RuntimeError(record.getMessage())


class Diagnostics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.candidates = None
        self.lm_calls = 0

    def finish(self, hypothesis):
        if not self.candidates:
            raise ValueError('Missing/empty N-best diagnostics')
        top = self.candidates[0]
        single = len(self.candidates) == 1
        gate = not single and top['avg_logprob'] >= -.020
        scored = any('lm_logprob_per_token' in c for c in self.candidates)
        if scored != bool(self.lm_calls) or (not single and not gate and not scored):
            raise ValueError('Incomplete LM scoring diagnostics')
        return dict(single_candidate=single, acoustic_gate=gate, actual_lm_scored=scored,
                    winner_changed=hypothesis != top['text'].strip(), beam_top1=top['text'])


def worker(cell, output, limit):
    import random
    import numpy as np
    import torch
    from phowhisper import pipeline
    rows, _ = preflight()
    warm = next(r for r in rows if r['split'] == 'dev')
    rows = rows[:limit] if limit else rows
    size, mode = cell.split('-')
    os.environ['PHOWHISPER_MODEL_PATH'] = str(CHECKPOINTS[size].resolve())
    report = dict(status='blocked', cell=cell, pilot=limit is not None)
    diagnostic = Diagnostics()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; no CPU fallback')
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        report['environment'] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            packages={name: version(name) for name in ('openai-whisper', 'transformers', 'tokenizers', 'jiwer', 'soundfile', 'numpy')})
        def peaks():
            return dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        def timed(call):
            torch.cuda.synchronize(); start = time.perf_counter()
            value = call()
            torch.cuda.synchronize()
            return value, time.perf_counter()-start
        with ExitStack() as stack:
            torch.cuda.reset_peak_memory_stats()
            pipeline.get_model.cache_clear()
            _, report['model_load_seconds'] = timed(pipeline.get_model)
            report['lm_load_seconds'] = 0
            if mode == 'rescore':
                from phowhisper import rescorer
                handler = FailWarnings()
                rescorer.logger.addHandler(handler)
                stack.callback(rescorer.logger.removeHandler, handler)
                cached = lru_cache(maxsize=1)(pinned_loader)
                _, report['lm_load_seconds'] = timed(cached)
                original = rescorer.NBestDecodingTask.run_nbest
                def observe(task, mel):
                    result = original(task, mel)
                    diagnostic.candidates = result[0]
                    return result
                def lm(name=LM):
                    if name != LM:
                        raise ValueError('Unexpected LM')
                    diagnostic.lm_calls += 1
                    return cached()
                stack.enter_context(patch.object(rescorer.NBestDecodingTask, 'run_nbest', observe))
                stack.enter_context(patch.object(rescorer, 'get_lm_model', lm))
            report['load_cuda_peak_bytes'] = peaks()
            def transcribe(row):
                diagnostic.reset()
                text = pipeline.transcribe(row['audio_path'], rescore=mode == 'rescore')
                return text, diagnostic.finish(text) if mode == 'rescore' else {}
            _, report['warmup_seconds'] = timed(lambda: transcribe(warm))
            torch.cuda.reset_peak_memory_stats()
            with (output/(cell+'.jsonl')).open('x', encoding='utf-8') as stream:
                for row in rows:
                    prediction = dict(id=row['id'], split=row['split'], reference=row['reference'],
                        audio_path=row['audio_path'], audio_sha256=row['sha256'], audio_seconds=row['actual_duration'])
                    try:
                        (text, details), elapsed = timed(lambda: transcribe(row))
                        prediction.update(hypothesis=text, inference_seconds=elapsed, diagnostics=details)
                    except Exception as exc:
                        prediction.update(error=f'{type(exc).__name__}: {exc}')
                    stream.write(json.dumps(prediction, ensure_ascii=False)+'\n'); stream.flush()
                    if prediction.get('error'):
                        raise RuntimeError(prediction['error'])
            report.update(status='completed', measured_cuda_peak_bytes=peaks(),
                cuda_capacity_bytes=torch.cuda.get_device_properties(0).total_memory,
                peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except Exception as exc:
        report['blocker'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save(output/(cell+'.json'), report)


def snapshot(hashes):
    paths = dict(hashes)
    for name in ('phowhisper-base.pt', 'pytorch_model.bin', 'config.json'):
        path = str(legacy_base_file(name))
        paths[path] = sha(path)
    for size, path in CHECKPOINTS.items():
        metadata = json.loads(Path(str(path)+'.provenance.json').read_text())
        if metadata['resolved_revision'] != REVISIONS[size] or sha(path) != metadata['native_sha256']:
            raise ValueError('Checkpoint provenance mismatch')
        paths[str(path)] = sha(path)
        paths[str(path)+'.provenance.json'] = sha(str(path)+'.provenance.json')
    for path in SOURCE:
        paths[path] = sha(path)
    # Enumerating this one pinned snapshot is deliberate provenance collection, not data selection.
    for path in sorted(LM_PATH.iterdir()):
        if path.is_file():
            paths[str(path)] = sha(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--worker', choices=CELLS)
    args = parser.parse_args()
    if args.limit is not None and not 0 < args.limit <= 100:
        parser.error('limit must be 1..100 (pilot only)')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    if args.worker:
        worker(args.worker, args.output_dir, args.limit)
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='blocked', pilot=args.limit is not None, valid4cellcomplete=False,
                  dataset_revision=REVISION, lm_revision=LM_REVISION, automatic_promotion=False,
                  protocol=PROTOCOL)
    before = None
    try:
        rows, hashes = preflight()
        before = snapshot(hashes)
        save(args.output_dir/'manifest.json', rows)
        report['before'] = before
        report['checkpoint_metadata'] = {k: json.loads(Path(str(p)+'.provenance.json').read_text()) for k, p in CHECKPOINTS.items()}
        report['head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        report['dirty'] = subprocess.check_output(['git', 'status', '--porcelain'], text=True)
        save(args.output_dir/'report.json', report)
        cells, resources = {}, {}
        for cell in CELLS:
            command = [sys.executable, '-m', 'benchmarks.runners.compare_vietmed_modes',
                       '--output-dir', str(args.output_dir), '--worker', cell]
            if args.limit:
                command += ['--limit', str(args.limit)]
            subprocess.run(command, check=True)
            metadata = json.loads((args.output_dir/(cell+'.json')).read_text())
            if metadata['status'] != 'completed':
                raise ValueError('Incomplete cell')
            resources[cell] = metadata
            cells[cell] = [json.loads(line) for line in (args.output_dir/(cell+'.jsonl')).read_text().splitlines()]
        if any(r['environment'] != resources[CELLS[0]]['environment'] for r in resources.values()):
            raise ValueError('Environment differs between cells')
        expected = rows[:args.limit] if args.limit else rows
        comparison = compare(cells, expected)
        save(args.output_dir/'comparison.json', comparison)
        report.update(status='completed', valid4cellcomplete=args.limit is None,
            resources=resources, summaries={c: dict(all100=summarize(v), by_split={s: summarize([r for r in v if r['split']==s])
                for s in ('train', 'dev', 'test') if any(r['split']==s for r in v)}) for c, v in cells.items()})
    except Exception as exc:
        report.update(status='blocked', valid4cellcomplete=False, blocker=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            if before is not None:
                report['after'] = {p: sha(p) for p in before}
                if report['after'] != before:
                    report.update(status='blocked', valid4cellcomplete=False, blocker='Input/source mutation detected')
                    raise ValueError(report['blocker'])
        finally:
            save(args.output_dir/'report.json', report)


if __name__ == '__main__':
    main()
