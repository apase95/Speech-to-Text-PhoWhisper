"""Offline section 2.3 benchmark; never rewrites frozen assets."""
import csv
import hashlib
import json
import os
import time
from datetime import datetime, UTC
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.metrics import corpus_error_rates, normalize_vietnamese


def sha(path):
    return hashlib.file_digest(open(path, 'rb'), 'sha256').hexdigest()


def summarize(rows):
    audio = sum(r['audio_seconds'] for r in rows)
    if audio <= 0:
        raise ValueError('Positive audio duration required')
    elapsed = sum(r['inference_seconds'] for r in rows)
    return dict(corpus_error_rates([r['reference'] for r in rows], [r['hypothesis'] for r in rows]),
                samples=len(rows), exact_matches=sum(normalize_vietnamese(r['reference']) == normalize_vietnamese(r['hypothesis']) for r in rows),
                total_audio_seconds=audio, total_inference_seconds=elapsed, rtf=elapsed/audio)


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import logging
    import platform
    import soundfile as sf
    import torch
    from src import inference, rescorer
    from importlib.metadata import version
    torch.set_num_threads(4)
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    output = Path(os.environ.get('SECTION23_OUTPUT', 'outputs/benchmarks/section23-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')))
    output.mkdir(exist_ok=True)
    report = dict(status='running', provider_model='openai/gpt-6.1-sol', agent_id='not supplied',
                  environment=dict(python=platform.python_version(), gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                                   packages={p: version(p) for p in ['torch', 'openai-whisper', 'transformers', 'jiwer', 'soundfile']}),
                  checkpoint_sha256=sha('models/phowhisper-base.pt'), results={}, manifests={})
    def persist():
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    class FailWarnings(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.WARNING:
                raise RuntimeError(record.getMessage())
    rescorer.logger.addHandler(FailWarnings())
    model = inference.get_model()
    rescorer.get_lm_model()
    warmup = next(csv.DictReader(open('benchmarks/manifests/viet_bud500_dev_250.csv')))['audio_path']
    def decode(path, mode):
        if mode == 'fast':
            return inference.transcribe(path)
        if mode == 'gpt2-gated':
            return inference.transcribe(path, rescore=True)
        # Same N-best decoder and top-1 ordering as the production rescorer,
        # but return before LM; threshold -inf always takes its acoustic gate.
        return rescorer.rescore_audio(path, model, confidence_threshold=float('-inf'))
    started = time.perf_counter()
    try:
        for split, filename, count in [('test1000','viet_bud500_1000.csv',1000), ('holdout500','viet_bud500_holdout_500.csv',500)]:
            manifest = Path('benchmarks/manifests')/filename
            rows = list(csv.DictReader(manifest.open()))
            assert len(rows) == count
            report['manifests'][split] = dict(path=str(manifest), sha256=sha(manifest))
            for row in rows:
                assert sha(row['audio_path']) == row['sha256']
            for mode in ['fast','beam5-control','gpt2-gated']:
                decode(warmup, mode)
                torch.cuda.synchronize()
                path = output/f'{split}-{mode}.jsonl'
                predictions = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
                assert [p['id'] for p in predictions] == [r['id'] for r in rows[:len(predictions)]]
                with path.open('a') as stream:
                    for row in rows[len(predictions):]:
                        duration = sf.info(row['audio_path']).duration
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        hypothesis = decode(row['audio_path'], mode)
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-tick
                        pred = dict(id=row['id'], reference=row['reference'], hypothesis=hypothesis,
                                    audio_seconds=duration, inference_seconds=elapsed)
                        predictions.append(pred)
                        stream.write(json.dumps(pred, ensure_ascii=False)+'\n')
                        stream.flush()
                saved = [json.loads(line) for line in path.read_text().splitlines()]
                metrics = summarize(saved)
                assert metrics == summarize(predictions)
                metrics.update(predictions_sha256=sha(path), recomputed_from_saved_predictions=True)
                report['results'][f'{split}-{mode}'] = metrics
                persist()
                print(json.dumps(dict(mode=f'{split}-{mode}', **metrics)), flush=True)
        report['status'] = 'completed'
        assert sha('models/phowhisper-base.pt') == report['checkpoint_sha256']
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        persist()
        print(output, flush=True)


if __name__ == '__main__':
    main()
