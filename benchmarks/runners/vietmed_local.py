"""Offline, read-only fresh evaluation of frozen VietMed100 assets."""
import json
import os
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.metrics import corpus_error_rates, normalize_vietnamese
from benchmarks.runners.vietmed100_experiment import sha
from benchmarks.checkpoints import legacy_base_file


def summarize(rows):
    rates = corpus_error_rates([r['reference'] for r in rows], [r['hypothesis'] for r in rows])
    audio = sum(r['audio_seconds'] for r in rows)
    if audio <= 0:
        raise ValueError('Positive audio duration required')
    latencies = sorted(r['inference_seconds'] for r in rows)
    return dict(rates, samples=len(rows), total_audio_seconds=audio,
                total_inference_seconds=sum(latencies), rtf=sum(latencies)/audio,
                latency_mean_seconds=statistics.mean(latencies),
                latency_median_seconds=statistics.median(latencies),
                latency_p95_seconds=latencies[max(0, __import__('math').ceil(.95*len(rows))-1)],
                exact_match=sum(normalize_vietnamese(r['reference']) == normalize_vietnamese(r['hypothesis']) for r in rows)/len(rows))


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import logging
    import soundfile as sf
    import torch
    import whisper
    from src import inference, rescorer

    output = Path('outputs/benchmarks') / ('vietmed-local-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(exist_ok=False)
    report = dict(status='running', implementer_id='ses_e8a99fa75ffeXvTh006SYXk76h',
                  provider_model='openai/gpt-6.1-sol', command=' '.join(sys.argv),
                  environment=dict(python=sys.version, platform=platform.platform(), processor=platform.processor(),
                      packages={p: version(p) for p in ['torch', 'openai-whisper', 'transformers', 'jiwer', 'soundfile', 'numpy']},
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
                  policy='One excluded warmup per mode on dev[0]; preload models; CUDA synchronize around each complete public API call including audio IO; batch=1; threads=4; seed=42; no stretching; default public decode options.',
                  normalization='NFC, casefold, punctuation replaced by spaces, whitespace collapsed; accents retained; no tone harmonization; WER whitespace tokens; CER includes spaces.',
                  results={}, integrity={})
    started = time.perf_counter()
    def persist():
        report['elapsed_seconds'] = time.perf_counter()-started
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(4)
        torch.manual_seed(42)
        historical = json.loads(Path('outputs/benchmarks/vietmed100-trained-20261014T200306Z/report.json').read_text())
        splits = {}
        used = {k: set() for k in ['speaker_name', 'seq_name', 'audio_name', 'utterance_id', 'source_sha256']}
        for split, count in [('train',70), ('dev',15), ('test',15)]:
            path = Path('data/benchmark/vietmed100')/f'{split}.json'
            assert sha(path) == historical['data'][split]['json_sha256']
            assert sha(path.with_suffix('.csv')) == historical['data'][split]['csv_sha256']
            rows = json.loads(path.read_text())
            assert len(rows) == count
            for key in used:
                values = {str(r[key]).strip().casefold() for r in rows}
                assert not values & used[key], (key, split)
                used[key].update(values)
            for row in rows:
                assert sha(row['audio_path']) == row['sha256']
                info = sf.info(row['audio_path'])
                assert info.samplerate == 16000 and info.channels == 1
                assert abs(info.duration-row['actual_duration']) < 1e-6
            splits[split] = rows
            report['integrity'][split] = dict(json_sha256=sha(path), csv_sha256=sha(path.with_suffix('.csv')), count=count)
        base = legacy_base_file()
        assert sha(base) == historical['base_sha256_before']
        report['base_sha256'] = sha(base)
        report['decode_settings'] = dict(fast='src.inference.transcribe defaults including Whisper temperature fallback',
             rescoring=dict(beam_size=5, alpha=.2, beta=-.02, gamma=-.2, confidence_threshold=-.020,
                            lm=rescorer.DEFAULT_LM_MODEL, fp16_asr=True, lm_precision='FP32'))
        class Warnings(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.WARNING:
                    raise RuntimeError(record.getMessage())
        rescorer.logger.addHandler(Warnings())
        all_rows = sum(splits.values(), [])
        for name, checkpoint, rescore, rows in [
            ('base-fast-all100', base, False, all_rows),
            ('base-gpt2-all100', base, True, all_rows),
            ('finetuned-fast-test15', Path(historical['trained_checkpoint']), False, splits['test'])]:
            os.environ['PHOWHISPER_MODEL_PATH'] = str(checkpoint)
            inference.get_model.cache_clear()
            torch.cuda.empty_cache()
            inference.get_model()
            if rescore:
                rescorer.get_lm_model(rescorer.DEFAULT_LM_MODEL)
            tick = time.perf_counter()
            inference.transcribe(splits['dev'][0]['audio_path'], rescore=rescore)
            torch.cuda.synchronize()
            warmup = time.perf_counter()-tick
            torch.cuda.reset_peak_memory_stats()
            predictions = []
            path = output/(name+'.jsonl')
            with path.open('x') as stream:
                for row in rows:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    hypothesis = inference.transcribe(row['audio_path'], rescore=rescore)
                    torch.cuda.synchronize()
                    prediction = dict(id=row['id'], split=row['split'], reference=row['reference'], hypothesis=hypothesis,
                        normalized_reference=normalize_vietnamese(row['reference']), normalized_hypothesis=normalize_vietnamese(hypothesis),
                        audio_seconds=row['actual_duration'], inference_seconds=time.perf_counter()-tick)
                    predictions.append(prediction)
                    stream.write(json.dumps(prediction, ensure_ascii=False)+'\n')
                    stream.flush()
            saved = [json.loads(line) for line in path.read_text().splitlines()]
            summary = summarize(saved)
            assert summary == summarize(predictions)
            summary.update(checkpoint_sha256=sha(checkpoint), warmup_seconds=warmup,
                           peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                           predictions_sha256=sha(path), recomputed_from_saved_predictions=True)
            summary['by_split'] = {s: summarize([r for r in saved if r['split']==s]) for s in splits if any(r['split']==s for r in saved)}
            report['results'][name] = summary
            persist()
            print(json.dumps(dict(mode=name, **summary)), flush=True)
        assert sha(base) == report['base_sha256']
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'blocked'
        report['blocker'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        persist()
        print(str(output), flush=True)


if __name__ == '__main__':
    main()
