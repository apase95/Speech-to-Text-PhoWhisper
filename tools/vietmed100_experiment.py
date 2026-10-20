"""Bounded VietMed100 preparation; never trains if execution gates are incomplete."""
import csv
import hashlib
import io
import json
import math
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

REVISION = 'dc1a2ebef3ffa772a955f2c9b4412ad8722c0b74'


def normalized(value):
    return unicodedata.normalize('NFC', str(value or '')).strip().casefold()


def select_rows(rows, counts=(70, 15, 15)):
    used = {k: set() for k in ('speaker_name', 'seq_name', 'audio_name', 'utterance_id')}
    selected = {}
    for split, count in zip(('train', 'dev', 'test'), counts, strict=True):
        groups = {}
        for row in rows[split]:
            if not all(normalized(row.get(k)) for k in used):
                continue
            duration = float(row['duration'])
            if not math.isfinite(duration) or not 0 < duration <= 30 or not normalized(row['text']):
                continue
            if any(normalized(row[k]) in used[k] for k in used):
                continue
            groups.setdefault(normalized(row.get('accent')), []).append(row)
        chosen = []
        while groups and len(chosen) < count:
            for accent in sorted(list(groups)):
                chosen.append(groups[accent].pop(0))
                if not groups[accent]:
                    del groups[accent]
                if len(chosen) == count:
                    break
        if len(chosen) != count:
            raise ValueError(f'{split}: only {len(chosen)} disjoint eligible rows; need {count}')
        selected[split] = chosen
        for row in chosen:
            for k in used:
                used[k].add(normalized(row[k]))
    return selected


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    from huggingface_hub import hf_hub_download
    from tools.convert_phowhisper import convert_key, whisper_dimensions

    started = time.perf_counter()
    output = Path('outputs/benchmarks') / ('vietmed100-' + datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ'))
    output.mkdir(parents=True, exist_ok=False)
    data = Path('data/benchmark/vietmed100')
    report = dict(status='blocked', provider_model='openai/gpt-6.1-sol',
                  scout_id='ses_ec405f9f4ffeYW2P8PXJMsqSav', implementer_id='not injected',
                  dataset_revision=REVISION, exported_audio=0, training_steps=0,
                  wer=None, cer=None, stages={}, limitations=[
                      '15 test clips cannot establish general ASR quality or fast-speech improvement.',
                      'VietMed medical-domain speech; label boundaries may omit 1–2 words.',
                      'Near duplicates and pretraining exposure are not excluded.'])
    base = Path('models/phowhisper-base.pt')
    report['base_sha256_before'] = sha(base)
    try:
        checkpoint = torch.load(base, map_location='cpu', weights_only=True)
        state = torch.load('models/pytorch_model.bin', map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        mapped = {}
        for key, tensor in state.items():
            target = convert_key(key)
            if target is None:
                assert key == 'proj_out.weight' and torch.equal(tensor, state['model.decoder.embed_tokens.weight'])
                continue
            assert target not in mapped, target
            mapped[target] = tensor
        original = checkpoint['model_state_dict']
        assert mapped.keys() == original.keys()
        assert all(v.dtype == original[k].dtype and v.shape == original[k].shape and torch.equal(v, original[k]) for k, v in mapped.items())
        assert checkpoint['dims'] == whisper_dimensions(json.loads(Path('models/config.json').read_text()))
        report['checkpoint_parity'] = dict(passed=True, pt_keys=len(original), hf_keys=len(state),
                                           mapped_keys=len(mapped), tied_projection=True,
                                           duplicates=0, missing=0, unmapped=0, mismatches=0,
                                           dimensions_equal=True, equality='exact dtype/shape/tensor')
        del checkpoint, state, original, mapped
        report['hardware'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
        paths, metadata = {}, {}
        for split in ('train', 'dev', 'test'):
            paths[split] = hf_hub_download('leduckhai/VietMed', f'data/{split}-00000-of-00001.parquet',
                                          repo_type='dataset', revision=REVISION)
            metadata[split] = pq.read_table(paths[split], columns=['speaker_name', 'seq_name', 'audio_name',
                'duration', 'text', 'utterance_id', 'accent']).to_pylist()
        report['shards'] = {s: dict(path=p, bytes=Path(p).stat().st_size, sha256=sha(p), rows=len(metadata[s])) for s, p in paths.items()}
        report['downloaded_shard_bytes'] = sum(v['bytes'] for v in report['shards'].values())
        selected = select_rows(metadata)
        data.mkdir(parents=True, exist_ok=True)
        assert not (data / 'train.json').exists(), 'Existing frozen manifest; refusing overwrite'
        hashes = set()
        for split, chosen in selected.items():
            wanted = {r['utterance_id'] for r in chosen}
            audio_rows = pq.read_table(paths[split], columns=['utterance_id', 'audio']).to_pylist()
            audio = {r['utterance_id']: r['audio'] for r in audio_rows if r['utterance_id'] in wanted}
            manifest = []
            for index, row in enumerate(chosen):
                encoded = audio[row['utterance_id']]['bytes']
                source_hash = hashlib.sha256(encoded).hexdigest()
                assert source_hash not in hashes, 'duplicate source audio'
                hashes.add(source_hash)
                waveform, rate = sf.read(io.BytesIO(encoded), dtype='float32', always_2d=True)
                waveform = waveform.mean(axis=1)
                if rate != 16000:
                    from scipy.signal import resample_poly
                    divisor = math.gcd(rate, 16000)
                    waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
                    rate = 16000
                assert waveform.size and np.isfinite(waveform).all() and 0 < len(waveform)/rate <= 30
                path = data / f'{split}-{index:03d}.wav'
                assert not path.exists(), 'Refusing audio overwrite'
                sf.write(path, waveform, rate, subtype='FLOAT')
                manifest.append(dict(row, id=row['utterance_id'], audio_path=str(path.resolve()),
                                     reference=row['text'], split=split, source_sha256=source_hash,
                                     sha256=sha(path), actual_duration=len(waveform)/rate))
                report['exported_audio'] += 1
            (data / f'{split}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            with (data / f'{split}.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['id', 'audio_path', 'reference', 'split'])
                writer.writeheader()
                writer.writerows({k: r[k] for k in writer.fieldnames} for r in manifest)
            report['stages'][split] = dict(count=len(manifest), duration_seconds=sum(r['actual_duration'] for r in manifest),
                accents={a: sum(r['accent'] == a for r in manifest) for a in sorted(set(r['accent'] for r in manifest))},
                manifest_sha256=sha(data / f'{split}.csv'))
        report['blocker'] = 'Preparation completed only. HF/OpenAI two-clip parity, training dependency setup, baseline, finite GPU batch, pilot training and post benchmark remain unimplemented/unexecuted in this bounded turn.'
    except Exception as exc:
        report['blocker'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['base_sha256_after'] = sha(base)
        report['base_unchanged'] = report['base_sha256_before'] == report['base_sha256_after']
        report['elapsed_seconds'] = time.perf_counter() - started
        (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (output / 'report.md').write_text('# VietMed100 — BLOCKED / partial preparation\n\n' + report['blocker'] + '\n\nSee report.json for authentic checkpoint, dataset and timing evidence. No trained checkpoint or quality metrics were produced.\n')
        print(json.dumps(dict(report_path=str(output.resolve()), status=report['status'],
                              exported_audio=report['exported_audio'], blocker=report['blocker'],
                              elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
