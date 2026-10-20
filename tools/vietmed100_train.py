"""Exact-checkpoint, single-GPU VietMed100 experiment; no downloads or base writes."""
import json
import math
import random
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from tools.vietmed100_experiment import sha


def teacher_tokens(prefix, text, eot):
    if not prefix or not text:
        raise ValueError('Nonempty decoder prefix and transcript required')
    sequence = list(prefix) + list(text) + [eot]
    return sequence[:-1], [-100] * (len(prefix) - 1) + sequence[len(prefix):]


def accumulation_size(index, total, accumulation):
    return min(accumulation, total - (index // accumulation) * accumulation)


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import whisper
    from torch.utils.checkpoint import checkpoint
    from benchmarks.metrics import corpus_error_rates, normalize_vietnamese

    start = time.perf_counter()
    output = Path('outputs/benchmarks') / ('vietmed100-trained-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    base = Path('models/phowhisper-base.pt')
    prep = json.loads(Path('outputs/benchmarks/vietmed100-20261014T195830Z/report.json').read_text())
    report = dict(status='blocked', provider_model='openai/gpt-6.1-sol',
                  scout_id=prep['scout_id'], implementer_id='not injected',
                  preparation_report='outputs/benchmarks/vietmed100-20261014T195830Z/report.json',
                  base_sha256_before=sha(base), training_steps=0,
                  config=dict(seed=42, epochs=1, batch=1, accumulation=4, lr=1e-5,
                              precision='FP32', checkpointing_use_reentrant=False,
                              selection='predeclared final epoch; no test-based selection'),
                  limitations=prep['limitations'], benchmarks={}, steps=[])

    def persist():
        report['elapsed_seconds'] = time.perf_counter() - start
        report['base_sha256_after'] = sha(base)
        report['base_unchanged'] = report['base_sha256_after'] == report['base_sha256_before']
        (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))

    def log(event, **values):
        record = dict(utc=datetime.now(UTC).isoformat(), event=event,
                      elapsed_seconds=time.perf_counter()-start, **values)
        with (output / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)

    try:
        assert report['base_sha256_before'] == prep['base_sha256_before']
        assert torch.cuda.is_available(), 'CUDA required; no silent CPU fallback'
        torch.set_num_threads(4)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        report['environment'] = dict(torch=torch.__version__, whisper=whisper.__version__,
                                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
        manifests = {}
        report['data'] = {}
        for split, count in [('train', 70), ('dev', 15), ('test', 15)]:
            path = Path('data/benchmark/vietmed100') / f'{split}.json'
            manifests[split] = json.loads(path.read_text())
            assert len(manifests[split]) == count
            csv_path = path.with_suffix('.csv')
            assert sha(csv_path) == prep['stages'][split]['manifest_sha256']
            for row in manifests[split]:
                assert sha(row['audio_path']) == row['sha256']
            report['data'][split] = dict(count=count, duration_seconds=sum(r['actual_duration'] for r in manifests[split]),
                                       json_sha256=sha(path), csv_sha256=sha(csv_path))
        model = whisper.load_model(str(base), device='cuda').float()
        tokenizer = whisper.tokenizer.get_tokenizer(True, language='vi', task='transcribe')
        prefix = list(tokenizer.sot_sequence_including_notimestamps)
        report['decoder_prefix'] = prefix
        report['initial_parity'] = 'Same OpenAI model, tokenizer and runtime object; no mapping/conversion involved.'

        def benchmark(stage, split, weight_path):
            model.eval()
            directory = output / f'{stage}-{split}'
            directory.mkdir()
            torch.cuda.reset_peak_memory_stats()
            refs, hyps, durations, latencies = [], [], [], []
            options = dict(language='vi', task='transcribe', temperature=0.0, beam_size=None,
                           best_of=None, without_timestamps=True, condition_on_previous_text=False, fp16=True)
            with (directory / 'predictions.jsonl').open('w') as stream:
                for row in manifests[split]:
                    torch.cuda.synchronize(); tick = time.perf_counter()
                    with torch.no_grad():
                        hypothesis = model.transcribe(row['audio_path'], **options)['text'].strip()
                    torch.cuda.synchronize(); elapsed = time.perf_counter()-tick
                    refs.append(row['reference']); hyps.append(hypothesis)
                    durations.append(row['actual_duration']); latencies.append(elapsed)
                    stream.write(json.dumps(dict(id=row['id'], reference=row['reference'], hypothesis=hypothesis,
                        normalized_reference=normalize_vietnamese(row['reference']),
                        normalized_hypothesis=normalize_vietnamese(hypothesis),
                        inference_seconds=elapsed, audio_seconds=row['actual_duration'], error=None), ensure_ascii=False)+'\n')
            summary = dict(corpus_error_rates(refs, hyps), samples=len(refs), failures=0,
                           total_audio_seconds=sum(durations), total_inference_seconds=sum(latencies),
                           rtf=sum(latencies)/sum(durations), decode_options=options,
                           checkpoint_sha256=sha(weight_path), manifest_sha256=report['data'][split]['csv_sha256'],
                           peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(), environment=report['environment'])
            (directory / 'summary.json').write_text(json.dumps(summary, indent=2))
            report['benchmarks'][f'{stage}-{split}'] = summary
            log('benchmark', stage=stage, split=split, wer=summary['wer'], cer=summary['cer'])
            persist()

        benchmark('baseline', 'dev', base)
        benchmark('baseline', 'test', base)
        # Non-reentrant activation checkpointing preserves trainable encoder/decoder gradients.
        originals = []
        for block in list(model.encoder.blocks) + list(model.decoder.blocks):
            original = block.forward
            originals.append((block, original))
            def wrapped(*args, _original=original, **kwargs):
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            block.forward = wrapped
        model.train()
        model.float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        rows = list(manifests['train']); random.shuffle(rows)
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        training_start = time.perf_counter()
        for index, row in enumerate(rows):
            audio, rate = sf.read(row['audio_path'], dtype='float32')
            assert rate == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio)).unsqueeze(0).cuda()
            inputs, targets = teacher_tokens(prefix, tokenizer.encode(row['reference']), tokenizer.eot)
            assert len(inputs) <= model.dims.n_text_ctx
            x = torch.tensor([inputs], device='cuda')
            y = torch.tensor([targets], device='cuda')
            logits = model(mel, x)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
            assert torch.isfinite(loss), 'Nonfinite loss'
            (loss / accumulation_size(index, len(rows), 4)).backward()
            accumulated_loss += loss.item()
            del logits, loss, mel, x, y
            if index == 0:
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                log('finite_forward_backward_gate', passed=True, peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
            if (index + 1) % 4 == 0 or index + 1 == len(rows):
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()), 'Missing/nonfinite full-model gradients'
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameter'
                optimizer.zero_grad(set_to_none=True)
                report['training_steps'] += 1
                record = dict(step=report['training_steps'], examples=index+1,
                              loss=accumulated_loss/accumulation_size(index, len(rows), 4), grad_norm=float(norm),
                              peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
                report['steps'].append(record); log('optimizer_step', **record)
                accumulated_loss = 0.0
        assert report['training_steps'] == 18
        report['training_seconds'] = time.perf_counter()-training_start
        for block, original in originals:
            block.forward = original
        trained = output / 'phowhisper-vietmed100.pt'
        torch.save(dict(dims=vars(model.dims), model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}), trained)
        report['trained_checkpoint'] = str(trained)
        report['trained_sha256'] = sha(trained)
        del optimizer
        # Reload saved checkpoint to benchmark the artifact, rather than just in-memory weights.
        del model
        torch.cuda.empty_cache()
        model = whisper.load_model(str(trained), device='cuda')
        benchmark('post', 'dev', trained)
        benchmark('post', 'test', trained)
        report['comparison'] = {}
        for split in ('dev', 'test'):
            report['comparison'][split] = {}
            for metric in ('wer', 'cer'):
                before = report['benchmarks'][f'baseline-{split}'][metric]
                after = report['benchmarks'][f'post-{split}'][metric]
                report['comparison'][split][metric] = dict(before=before, after=after,
                    delta_percentage_points=(after-before)*100,
                    relative_delta_percent=(after-before)/before*100 if before else None)
        assert sha(base) == report['base_sha256_before']
        report['status'] = 'completed'
    except Exception as exc:
        report['blocker'] = f'{type(exc).__name__}: {exc}'
        (output / 'failure.txt').write_text(traceback.format_exc())
        log('blocked', reason=report['blocker'])
    finally:
        persist()
        (output / 'report.md').write_text('# VietMed100 — '+report['status']+'\n\n'
            + 'Exact local checkpoint, frozen 70/15/15 clips (625 seconds). Direct OpenAI FP32 training; identical greedy FP16 benchmark settings.\n\n'
            + '```json\n'+json.dumps(report.get('comparison', {}), indent=2)+'\n```\n\n'
            + '\n'.join(report['limitations'])+'\n\n'+report.get('blocker', '')+'\n\nSee report.json and progress.jsonl for configuration, hashes, timing, steps and memory.\n')
        print(json.dumps(dict(output=str(output), status=report['status'], steps=report['training_steps'])), flush=True)
    if report['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
