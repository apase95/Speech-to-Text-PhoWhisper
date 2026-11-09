"""Offline small-model comparison on identical independent 28-second pieces."""
import argparse
from contextlib import ExitStack
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
from unittest.mock import patch


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def validate_checkpoint(checkpoint):
    sidecar = Path(str(checkpoint)+'.provenance.json')
    metadata = json.loads(sidecar.read_text())
    expected = dict(origin='huggingface', model_id='vinai/PhoWhisper-small',
        resolved_revision='a86b604c346caf7148c37512eafe783a16420adb')
    if (any(metadata.get(key) != value for key, value in expected.items())
            or metadata.get('native_sha256') != sha(checkpoint)):
        raise ValueError('Checkpoint provenance mismatch')
    return sidecar


def validate_audio(audio):
    import numpy as np
    if audio.ndim != 1 or not audio.size or audio.dtype != np.float32 or not np.isfinite(audio).all():
        raise ValueError('Expected nonempty finite mono float32 decoded audio')


def write_segments(audio, output):
    import numpy as np
    import soundfile as sf
    validate_audio(audio)
    rows = []
    for start in range(0, len(audio), 448000):
        end = min(start+448000, len(audio))
        path = output/f'segment-{len(rows):04d}.wav'
        if path.exists():
            raise FileExistsError(path)
        sf.write(path, audio[start:end], 16000, subtype='FLOAT')
        restored, rate = sf.read(path, dtype='float32')
        if rate != 16000 or not np.array_equal(restored, audio[start:end]):
            raise ValueError('FLOAT WAV changed decoded samples')
        rows.append(dict(audio_path=str(path.resolve()), sha256=sha(path), frames=end-start,
                         start_sample=start, end_sample=end, start_seconds=start/16000, end_seconds=end/16000))
    return rows


def join_transcript(rows):
    return ' '.join(' '.join(row['hypothesis'].split()) for row in rows if row['hypothesis'].strip())


def compare(left, right):
    keys = ('audio_path', 'sha256', 'frames', 'start_sample', 'end_sample')
    if not left or len(left) != len(right) or any(any(a[k] != b[k] for k in keys) for a, b in zip(left, right)):
        raise ValueError('Segment identity mismatch or incomplete modes')
    return dict(wer=None, cer=None, correctness='Unknown: no reference; transcript differences do not prove correctness')


def worker(mode, output, audio, checkpoint):
    from benchmarks.runners import compare_vietmed_modes as shared
    name = 'small-fast' if mode == 'fast' else 'small-rescoring'
    report_path = output/(name+'.summary.json')
    if report_path.exists():
        raise FileExistsError(report_path)
    report = dict(status='running', mode=mode, wer=None, cer=None)
    rows = []
    save(report_path, report)
    try:
        import random
        import numpy as np
        import torch
        validate_checkpoint(checkpoint)
        os.environ['PHOWHISPER_MODEL_PATH'] = str(checkpoint.resolve())
        from phowhisper import pipeline
        manifest = json.loads((output/'manifest.json').read_text())
        if str(audio.resolve()) != manifest['audio_path']:
            raise ValueError('Worker input differs from manifest')
        segments = manifest['segments']
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        report['environment'] = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
            packages={p: version(p) for p in ('openai-whisper', 'transformers', 'tokenizers', 'soundfile', 'numpy')})
        report['cuda_capacity_bytes'] = torch.cuda.get_device_properties(0).total_memory
        def peaks():
            return dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved(),
                        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        def timed(fn):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            return result, time.perf_counter()-start
        diagnostic = shared.Diagnostics()
        with ExitStack() as stack:
            torch.cuda.reset_peak_memory_stats()
            pipeline.get_model.cache_clear()
            _, report['model_load_seconds'] = timed(pipeline.get_model)
            report['lm_load_seconds'] = 0
            if mode == 'rescore':
                from phowhisper import rescorer
                if (rescorer.DEFAULT_ALPHA_GPT2, rescorer.DEFAULT_BETA_GPT2, rescorer.DEFAULT_GAMMA_GPT2) != (.2, -.02, -.2):
                    raise ValueError('Rescoring policy changed')
                handler = shared.FailWarnings()
                rescorer.logger.addHandler(handler)
                stack.callback(rescorer.logger.removeHandler, handler)
                cached = lru_cache(maxsize=1)(shared.pinned_loader)
                _, report['lm_load_seconds'] = timed(cached)
                original = rescorer.NBestDecodingTask.run_nbest
                def observe(task, mel):
                    result = original(task, mel)
                    diagnostic.candidates = result[0]
                    return result
                def lm(name=shared.LM):
                    if name != shared.LM:
                        raise ValueError('Unexpected LM')
                    diagnostic.lm_calls += 1
                    return cached()
                stack.enter_context(patch.object(rescorer.NBestDecodingTask, 'run_nbest', observe))
                stack.enter_context(patch.object(rescorer, 'get_lm_model', lm))
            report['load_peaks'] = peaks()
            def transcribe(row):
                if sha(row['audio_path']) != row['sha256']:
                    raise ValueError('Segment mutation')
                diagnostic.reset()
                text = pipeline.transcribe(row['audio_path'], rescore=mode == 'rescore', beam_size=5, confidence_threshold=-.020)
                return text, diagnostic.finish(text) if mode == 'rescore' else {}
            _, report['warmup_seconds'] = timed(lambda: transcribe(segments[0]))
            report['warmup'] = 'First chunk duplicated once, excluded from transcript and measured inference'
            torch.cuda.reset_peak_memory_stats()
            for segment in segments:
                (text, details), elapsed = timed(lambda: transcribe(segment))
                rows.append(dict(segment, hypothesis=text, diagnostics=details, inference_seconds=elapsed))
                save(output/(name+'.segments.json'), rows)
            compare(segments, rows)
            report['inference_peaks_including_resident_weights'] = peaks()
            elapsed = sum(r['inference_seconds'] for r in rows)
            report.update(status='completed', inference_seconds=elapsed, rtf=elapsed/manifest['decoded_seconds'],
                          transcript=join_transcript(rows), decoded_seconds=manifest['decoded_seconds'])
            (output/(name+'.txt')).write_text(report['transcript']+'\n\n'+ '\n'.join(
                f"[{r['start_seconds']:.6f} --> {r['end_seconds']:.6f}] {r['hypothesis']}" for r in rows)+'\n', encoding='utf-8')
    except Exception as exc:
        report.update(status='blocked', blocker=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(output/(name+'.segments.json'), rows)
        save(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/phowhisper-small.pt'))
    parser.add_argument('--worker', choices=('fast', 'rescore'))
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', OMP_NUM_THREADS='4')
    output = args.output_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if args.worker:
        worker(args.worker, output, args.audio, checkpoint)
        return
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', wer=None, cer=None, scope='Both modes chunked independently; not full-recording fast evaluation',
                  shared_overhead='AAC decode and FLOAT WAV canonicalization excluded from per-mode inference',
                  seed=42, threads=4, asr='CUDA FP16', beam_size=5, alpha=.2, beta=-.02, gamma=-.2, gate=-.020)
    save(output/'report.json', report)
    before = {}
    try:
        from benchmarks.runners import compare_vietmed_modes as shared
        report.update(lm=shared.LM, lm_revision=shared.LM_REVISION, lm_policy='local_files_only CUDA FP32')
        sidecar = validate_checkpoint(checkpoint)
        paths = list(shared.SOURCE)+[__file__, str(args.audio.resolve()), str(checkpoint), str(sidecar)]
        paths.extend(str(p) for p in shared.LM_PATH.iterdir() if p.is_file())
        before = {p: sha(p) for p in paths}
        report['before'] = before
        save(output/'report.json', report)
        report['ffprobe'] = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration:stream=codec_name,sample_rate,channels', '-of', 'json', str(args.audio)],
            check=True, capture_output=True, text=True, timeout=20).stdout)
        import whisper
        start = time.perf_counter()
        audio = whisper.load_audio(str(args.audio))
        report['aac_decode_seconds'] = time.perf_counter()-start
        start = time.perf_counter()
        segments = write_segments(audio, output)
        report['canonicalization_seconds'] = time.perf_counter()-start
        manifest = dict(audio_path=str(args.audio.resolve()), input_sha256=before[str(args.audio.resolve())],
            decoded_samples=len(audio), decoded_seconds=len(audio)/16000, samplerate=16000,
            decoded_float_sha256=hashlib.sha256(audio.tobytes()).hexdigest(), segments=segments)
        save(output/'manifest.json', manifest)
        before.update({r['audio_path']: r['sha256'] for r in segments})
        before[str(output/'manifest.json')] = sha(output/'manifest.json')
        report['before'] = before
        save(output/'report.json', report)
        results = {}
        for mode, name in (('fast', 'small-fast'), ('rescore', 'small-rescoring')):
            subprocess.run([sys.executable, '-m', 'benchmarks.runners.compare_personal_small', '--audio',
                str(args.audio.resolve()), '--output-dir', str(output), '--checkpoint', str(checkpoint),
                '--worker', mode], check=True)
            results[name] = json.loads((output/(name+'.summary.json')).read_text())
            if results[name]['status'] != 'completed':
                raise ValueError('Incomplete worker')
        rows = [json.loads((output/(name+'.segments.json')).read_text()) for name in results]
        for group in rows:
            compare(segments, group)
        comparison = compare(*rows)
        comparison['modes'] = results
        save(output/'comparison.json', comparison)
        report.update(status='completed', results=results)
    except Exception as exc:
        report.update(status='blocked', blocker=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            report['after'] = {p: sha(p) for p in before}
            if report['after'] != before:
                report.update(status='blocked', blocker='Input/source/checkpoint/segment mutation')
                raise ValueError(report['blocker'])
        finally:
            save(output/'report.json', report)
    print(output)


if __name__ == '__main__':
    main()
