"""Read-only offline comparison of two frozen workflows; no search or training."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, UTC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REV = '65818d14816b42be09e2201933bf07106d9a2647'
FROZEN = ROOT/'outputs/benchmarks/long/gpt2-optimization-20261104'


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def chunks(n):
    if n <= 0:
        raise ValueError('empty audio')
    return [(a, min(a+448000, n)) for a in range(0, n, 448000)]


def rtf(seconds, duration):
    if duration <= 0 or seconds < 0:
        raise ValueError('invalid timing')
    return seconds/duration


def identity(expected, actual):
    keys = lambda rows: [(r['split'], r['id']) for r in rows]
    if keys(expected) != keys(actual) or len(set(keys(actual))) != len(actual):
        raise ValueError('sample identity mismatch')


def decision(base, long):
    if long['wer'] > base['wer'] or long['cer'] > base['cer']:
        return 'keep-production; regression in WER or CER; long opt-in'
    return 'keep-production; long opt-in (no automatic promotion)'


def main():
    os.chdir(ROOT)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT/'outputs/benchmarks'/('personal-vietmed-compare-'+stamp)
    out.mkdir(exist_ok=False)
    snapshot = ROOT/'outputs/config-snapshots'/stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', provider_model='openai/gpt-6.1-sol', agent_id='not exposed',
                  primary='pooled WER; CER secondary; retain production if either degrades', results={})
    started = time.perf_counter()
    sources = ['src/inference.py', 'src/rescorer.py', 'benchmarks/runners/long_audio_benchmark.py',
               'benchmarks/runners/generate_dev_nbest.py', 'benchmarks/runners/gpt2_optimization.py']
    hashes = {p: sha(p) for p in sources}
    for p in sources:
        shutil.copyfile(p, snapshot/Path(p).name)
    checkpoint = ROOT/'models/phowhisper-base.pt'
    ck = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size, sha256=sha(checkpoint))
    freeze = json.loads((FROZEN/'beam15-freeze.json').read_text())
    winner = json.loads((FROZEN/'winner.json').read_text())
    assert winner['beam'] == 15 and winner['freeze_sha256'] == sha(FROZEN/'beam15-freeze.json')
    assert freeze['protocol']['acoustic']['checkpoint_sha256'] == ck['sha256']
    profile = dict(frozen=freeze, winner=winner, checkpoint=ck,
                   normalization='NFC lowercase; non-alphanumeric to space; collapse whitespace',
                   candidate_ranking='installed MaximumLikelihoodRanker length; stable original order ties',
                   gate=None, context='independent chunks, no cross-chunk LM context')
    original = dict(checkpoint=ck, beam_size=5, alpha=.2, beta=-.02, gamma=-.2,
                    confidence_threshold=-.020, fp16=True, ranking='descending sum_logprob',
                    lm_prefix='tokenizer.encode default specials; NO explicit model-config prefix',
                    lm='NlpHUST/gpt2-vietnamese', revision=REV,
                    scoring='avg_logprob + .2 * negative mean causal loss - .02 * normalized word count',
                    source_hashes=hashes, production_status='unchanged; fast remains default, rescoring opt-in')
    configs = ROOT/'configs/experiments'/('personal-vietmed-'+stamp)
    configs.mkdir(parents=True, exist_ok=False)
    save(configs/'current-long-beam15.json', profile)
    save(configs/'original-production.json', original)
    save(snapshot/'snapshot.json', dict(source_hashes=hashes, checkpoint=ck, configs=str(configs)))
    report.update(snapshot=str(snapshot), configs=str(configs), checkpoint=ck, source_hashes=hashes)
    def persist():
        report['wall_seconds'] = time.perf_counter()-started
        save(out/'report.json', report)
    persist()
    try:
        import torch
        import whisper
        import soundfile as sf
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from src import rescorer
        from benchmarks.runners.generate_dev_nbest import NBestDecodingTask
        from benchmarks.runners.long_audio_benchmark import normalize, gpt2_prefix, runtime_versions
        from benchmarks.runners.gpt2_optimization import verify, choose
        from benchmarks.metrics import normalize_vietnamese, corpus_error_rates
        from jiwer import process_words, process_characters
        verify(freeze)
        verify(winner)
        assert torch.cuda.is_available(), 'CUDA required; no silent CPU acoustic fallback'
        torch.set_num_threads(2)
        torch.manual_seed(42)
        report['environment'] = dict(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            runtime=runtime_versions(['torch', 'openai-whisper', 'transformers', 'tokenizers', 'numpy']), threads=2)
        def timed(fn):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            result = fn()
            torch.cuda.synchronize()
            return result, time.perf_counter()-tick
        model, whisper_load = timed(lambda: whisper.load_model(str(checkpoint), device='cuda'))
        tokenizer, tokenizer_load = timed(lambda: AutoTokenizer.from_pretrained(original['lm'], revision=REV, local_files_only=True))
        lm, lm_load = timed(lambda: AutoModelForCausalLM.from_pretrained(original['lm'], revision=REV, local_files_only=True).eval())
        report['loads'] = dict(whisper_seconds=whisper_load, tokenizer_seconds=tokenizer_load, gpt2_cpu_seconds=lm_load)
        prefix, policy = gpt2_prefix(lm.config, lm.get_input_embeddings().num_embeddings)
        assert prefix == freeze['protocol']['lm']['prefix_id'] and policy == freeze['protocol']['lm']['prefix_policy']
        vocab_sha = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        assert vocab_sha == freeze['protocol']['lm']['vocabulary']['tokenizer_sha256']
        report['lm_device'] = 'cpu FP32, same as frozen long experiment; public baseline loss formula unchanged'
        def decode(audio, long):
            dtype = torch.float32 if long else torch.float16
            def mel_fn():
                full = whisper.log_mel_spectrogram(audio, model.dims.n_mels, padding=whisper.audio.N_SAMPLES)
                content = min(whisper.audio.N_FRAMES, max(0, full.shape[-1]-whisper.audio.N_FRAMES))
                return whisper.pad_or_trim(full[:, :content], whisper.audio.N_FRAMES).to(model.device).to(dtype)
            mel, prep = timed(mel_fn)
            options = whisper.DecodingOptions(language='vi', task='transcribe', beam_size=15 if long else 5,
                temperature=0, without_timestamps=True, fp16=not long)
            task = (NBestDecodingTask if long else rescorer.NBestDecodingTask)(model, options)
            with torch.no_grad():
                cs, acoustic = timed(lambda: task.run_nbest(mel.unsqueeze(0))[0])
            if not cs:
                raise ValueError('empty candidates')
            def score():
                if not long and (len(cs) == 1 or cs[0]['avg_logprob'] >= -.020):
                    return cs[0]
                for c in cs:
                    text = normalize(c['text']) if long else normalize_vietnamese(c['text'])
                    ids = tokenizer.encode(text, return_tensors='pt', **({'add_special_tokens': False} if long else {}))
                    if long:
                        ids = torch.cat([torch.tensor([[prefix]]), ids], dim=1)
                    if ids.shape[1] > min(lm.config.max_position_embeddings, tokenizer.model_max_length):
                        raise ValueError('LM context exceeded')
                    with torch.no_grad():
                        c['lm'] = -float(lm(ids, labels=ids).loss) if text and ids.shape[1] > 1 else -20.
                    if not math.isfinite(c['lm']):
                        raise ValueError('nonfinite LM loss; refusing fallback')
                    c['normalized_text'] = text
                if long:
                    return choose(cs, freeze['alpha'], freeze['beta'])
                top = best = cs[0]
                value = lambda c: c['avg_logprob']+.2*c['lm']-.02*len(c['normalized_text'].split())
                for c in cs[1:]:
                    if value(c) > value(best) and c['avg_logprob']-top['avg_logprob'] >= -.2:
                        best = c
                return best
            selected, scoring = timed(score)
            return dict(hypothesis=selected['text'], candidates=cs, selected_rank=selected['rank'],
                preprocessing_seconds=prep, decode_seconds=acoustic, lm_seconds=scoring,
                inference_seconds=prep+acoustic+scoring)
        personal = ROOT/'data/benchmark/my-audio/ThuanLoi5.m4a'
        if personal.is_file():
            probe = json.loads(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,sample_rate,channels','-of','json',str(personal)], capture_output=True, text=True, check=True, timeout=20).stdout)
            audio, load_audio = timed(lambda: whisper.load_audio(str(personal)))
            segments = []
            wall = time.perf_counter()
            for a,b in chunks(len(audio)):
                result = decode(audio[a:b], True)
                result.update(start_sample=a, end_sample=b, start_seconds=a/16000, end_seconds=b/16000)
                segments.append(result)
                save(out/'personal-segments.json', segments)
            duration = float(probe['format']['duration'])
            total = load_audio+sum(s['inference_seconds'] for s in segments)
            (out/'personal-transcript.txt').write_text('\n'.join(s['hypothesis'] for s in segments)+'\n', encoding='utf-8')
            personal_result = dict(probe=probe, input_sha256=sha(personal), decoded_samples=len(audio),
                decoded_seconds=len(audio)/16000, covered_samples=sum(s['end_sample']-s['start_sample'] for s in segments),
                chunks=len(segments), audio_preprocessing_seconds=load_audio, processing_seconds=total,
                warm_rtf=rtf(total,duration), cold_rtf=rtf(total+sum(report['loads'].values()),duration),
                processing_wall_seconds=time.perf_counter()-wall+load_audio,
                transcript=str(out/'personal-transcript.txt'), segments=str(out/'personal-segments.json'),
                correctness='unknown; no reference; no WER', boundary_policy='contiguous <=28s, no overlap, independent LM')
            save(out/'personal-timing.json', personal_result)
            report['personal'] = personal_result
            persist()
        else:
            report['personal'] = dict(status='blocked', reason='exact input path absent')
            persist()
        historical = json.loads((ROOT/'outputs/benchmarks/vietmed100-trained-20261014T200306Z/report.json').read_text())
        rows = []
        integrity = {}
        groups = {key: set() for key in ['speaker_name','seq_name','audio_name','utterance_id','source_sha256']}
        for split, count in [('train',70),('dev',15),('test',15)]:
            path = ROOT/f'data/benchmark/vietmed100/{split}.json'
            assert sha(path) == historical['data'][split]['json_sha256']
            assert sha(path.with_suffix('.csv')) == historical['data'][split]['csv_sha256']
            batch = json.loads(path.read_text())
            assert len(batch) == count
            for key, seen in groups.items():
                values = {str(r[key]).strip().casefold() for r in batch}
                assert not seen & values
                seen.update(values)
            for r in batch:
                assert sha(r['audio_path']) == r['sha256']
                info = sf.info(r['audio_path'])
                assert info.samplerate == 16000 and info.channels == 1
                assert abs(info.duration-r['actual_duration']) < 1e-6 and info.duration < 30
            rows += batch
            integrity[split] = dict(count=count, json_sha256=sha(path), csv_sha256=sha(path.with_suffix('.csv')))
        assert len(rows) == 100 and ck['sha256'] == historical['base_sha256_before']
        report['integrity'] = integrity
        archived = ROOT/'outputs/benchmarks/vietmed-local-20261025T233412Z/report.json'
        report['archived_baseline'] = json.loads(archived.read_text())['results'] if archived.is_file() else 'unavailable'
        all_predictions = {}
        for name, long in [('original-production',False),('current-long',True)]:
            stage = time.perf_counter()
            warm_audio = whisper.load_audio(rows[70]['audio_path'])
            _, warmup = timed(lambda: decode(warm_audio,long))
            predictions = []
            with (out/(name+'.jsonl')).open('x', encoding='utf-8') as stream:
                for row in rows:
                    if time.perf_counter()-stage > 1800:
                        raise TimeoutError('profile stage exceeded 1800s')
                    audio, io = timed(lambda: whisper.load_audio(row['audio_path']))
                    pred = decode(audio,long)
                    pred.update(id=row['id'],split=row['split'],reference=row['reference'],audio_seconds=row['actual_duration'],audio_io_seconds=io)
                    pred['inference_seconds'] += io
                    predictions.append(pred)
                    stream.write(json.dumps(pred,ensure_ascii=False)+'\n')
                    stream.flush()
            identity(rows,predictions)
            refs = [normalize_vietnamese(r['reference']) for r in predictions]
            hyps = [normalize_vietnamese(r['hypothesis']) for r in predictions]
            w,c = process_words(refs,hyps),process_characters(refs,hyps)
            duration = sum(r['audio_seconds'] for r in predictions)
            elapsed = sum(r['inference_seconds'] for r in predictions)
            report['results'][name] = dict(corpus_error_rates([r['reference'] for r in predictions],[r['hypothesis'] for r in predictions]),
                samples=len(predictions),word_edits=dict(S=w.substitutions,D=w.deletions,I=w.insertions),
                character_edits=dict(S=c.substitutions,D=c.deletions,I=c.insertions),
                total_audio_seconds=duration,inference_seconds=elapsed,rtf=rtf(elapsed,duration),
                warmup_seconds=warmup, stage_wall_seconds=time.perf_counter()-stage,
                predictions_sha256=sha(out/(name+'.jsonl')))
            all_predictions[name] = predictions
            persist()
        base,long = all_predictions.values()
        identity(base,long)
        comparisons = []
        for a,b in zip(base,long):
            ref = normalize_vietnamese(a['reference']).split()
            def errors(r):
                w = process_words(ref and ' '.join(ref) or '',normalize_vietnamese(r['hypothesis']))
                return w.substitutions+w.deletions+w.insertions
            x,y = errors(a),errors(b)
            comparisons.append(dict(id=a['id'],split=a['split'],baseline_errors=x,long_errors=y,outcome='win' if y<x else 'loss' if y>x else 'tie'))
        save(out/'paired-comparison.json',comparisons)
        report['paired'] = {k:sum(r['outcome']==k for r in comparisons) for k in ['win','tie','loss']}
        report['decision'] = decision(report['results']['original-production'], report['results']['current-long'])
        report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'blocked'
        report['blocker'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['source_unchanged'] = hashes == {p:sha(p) for p in sources}
        report['checkpoint_unchanged'] = sha(checkpoint) == ck['sha256']
        persist()
        print(out, flush=True)


if __name__ == '__main__':
    main()
