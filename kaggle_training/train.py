"""Standalone, lazy-audio PhoWhisper training. No inference-runtime imports."""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import unicodedata
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize('NFC', text).lower()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())


def read_metadata(root, split):
    root = Path(root).resolve()
    rows, seen = [], set()
    with (root / f'metadata_{split}.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='|')
        if not {'audio_file', 'text', 'speaker_name'} <= set(reader.fieldnames or []):
            raise ValueError('Required columns: audio_file|text|speaker_name')
        for row in reader:
            name = row['audio_file']
            if not re.fullmatch(r'(?:wavs/)?audio\d+\.wav', name):
                raise ValueError(f'Unexpected path: {name}')
            name = Path(name).name
            if name in seen or not row['text'].strip():
                raise ValueError(f'Duplicate ID or empty text: {name}')
            seen.add(name)
            path = root / 'wavs' / name
            if not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f'Missing/unsafe audio: {path}')
            rows.append(dict(id=name, audio_path=str(path), text=row['text'], speaker=row['speaker_name'].strip()))
    return rows


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    train, test = read_metadata(args.root, 'train'), read_metadata(args.root, 'test')
    if {r['id'] for r in train} & {r['id'] for r in test}:
        raise ValueError('Official train/test ID overlap; resolve provenance before training')
    excluded_ids, excluded_hashes = set(), set()
    for manifest in args.exclude_manifest:
        with open(manifest, encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                excluded_ids.add(Path(row.get('audio_path', row.get('id', ''))).stem)
                if row.get('sha256'):
                    excluded_hashes.add(row['sha256'])
    kept, seen_hashes, excluded = [], set(), 0
    for row in train:
        number = int(Path(row['id']).stem[5:])
        if number <= 1250 or Path(row['id']).stem in excluded_ids:
            excluded += 1
            continue
        row['sha256'] = digest(row['audio_path'])
        if row['sha256'] in excluded_hashes:
            excluded += 1
            continue
        if row['sha256'] in seen_hashes:
            raise ValueError(f'Duplicate audio bytes: {row["id"]}')
        seen_hashes.add(row['sha256'])
        kept.append(row)
    rng = random.Random(args.seed)
    speakers = sorted({r['speaker'] for r in kept})
    if '' in speakers or len(speakers) < 2:
        raise ValueError('Speaker-group split requires at least two nonempty speaker IDs')
    rng.shuffle(speakers)
    validation_speakers = set(speakers[:max(1, int(len(speakers) * args.validation_fraction))])
    groups = {'train': [r for r in kept if r['speaker'] not in validation_speakers],
              'validation': [r for r in kept if r['speaker'] in validation_speakers], 'test': test}
    for name, rows in groups.items():
        if name != 'test':
            rng.shuffle(rows)
            cap = args.train_cap if name == 'train' else args.validation_cap
            rows = rows[:cap] if cap else rows
        if not rows:
            raise ValueError(f'Empty {name} split')
        (args.output / f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    (args.output / 'prepare.json').write_text(json.dumps(dict(seed=args.seed, root=str(args.root),
        excluded=excluded, train_cap=args.train_cap, validation_cap=args.validation_cap,
        validation_speakers=sorted(validation_speakers), exclude_manifest=args.exclude_manifest), indent=2))


class AudioDataset:
    def __init__(self, rows, processor):
        self.rows, self.processor = rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        row = self.rows[index]
        info = sf.info(row['audio_path'])
        if info.frames <= 0 or info.samplerate <= 0 or info.duration > 30:
            raise ValueError(f'Empty/invalid/>30s audio: {row["id"]}')
        audio, rate = sf.read(row['audio_path'], dtype='float32', always_2d=True)
        audio = audio.mean(axis=1)
        if not np.isfinite(audio).all():
            raise ValueError(f'Nonfinite audio: {row["id"]}')
        factor = gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // factor, rate // factor)
        labels = self.processor.tokenizer(row['text']).input_ids
        if len(labels) > 448:
            raise ValueError(f'Transcript exceeds decoder capacity: {row["id"]}')
        return {'input_features': self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0],
                'labels': labels}


class Collator:
    def __init__(self, processor, start):
        self.processor, self.start = processor, start

    def __call__(self, features):
        batch = self.processor.feature_extractor.pad([{'input_features': f['input_features']} for f in features], return_tensors='pt')
        # Strip per example, never remove a real first text token.
        labels = []
        for feature in features:
            tokens = feature['labels']
            tokens = tokens[1:] if tokens and tokens[0] == self.start else tokens
            if not tokens:
                raise ValueError('Empty decoder labels after decoder-start stripping')
            labels.append(tokens)
        padded = self.processor.tokenizer.pad([{'input_ids': x} for x in labels], return_tensors='pt')
        batch['labels'] = padded.input_ids.masked_fill(padded.attention_mask.ne(1), -100)
        return batch


def run(args):
    import numpy as np
    import torch
    import jiwer
    from huggingface_hub import model_info
    from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
    set_seed(args.seed)
    source = args.model
    revision = args.revision if Path(source).is_dir() else model_info(source, revision=args.revision).sha
    processor = WhisperProcessor.from_pretrained(source, revision=revision, language='vi', task='transcribe')
    processor.tokenizer.set_prefix_tokens(language='vi', task='transcribe', predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(source, revision=revision)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = 'vi'
    model.generation_config.task = 'transcribe'
    model.generation_config.return_timestamps = False
    model.generation_config.num_beams = 1
    model.generation_config.do_sample = False
    def metrics(prediction):
        labels = np.where(prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids)
        refs = [normalize(x) for x in processor.batch_decode(labels, skip_special_tokens=True)]
        hyps = [normalize(x) for x in processor.batch_decode(prediction.predictions, skip_special_tokens=True)]
        words = jiwer.process_words(refs, hyps)
        return dict(wer=words.wer, cer=jiwer.cer(refs, hyps), substitutions=words.substitutions,
                    deletions=words.deletions, insertions=words.insertions)
    rows = json.loads(args.manifest.read_text())
    dataset = AudioDataset(rows, processor)
    training = args.mode in ('train', 'smoke')
    validation = AudioDataset(json.loads(args.validation.read_text()), processor) if training else dataset
    options = Seq2SeqTrainingArguments(output_dir=str(args.output), seed=args.seed,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation, learning_rate=args.learning_rate,
        max_steps=args.max_steps, fp16=torch.cuda.is_available(), gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        eval_strategy='steps' if training else 'no', eval_steps=args.save_steps,
        save_steps=args.save_steps, save_total_limit=2, logging_steps=1,
        predict_with_generate=True, generation_max_length=448, generation_num_beams=1,
        dataloader_num_workers=args.workers, report_to=[], remove_unused_columns=False,
        ddp_find_unused_parameters=False)
    trainer = Seq2SeqTrainer(model=model, args=options, train_dataset=dataset if training else None,
        eval_dataset=validation, data_collator=Collator(processor, model.config.decoder_start_token_id),
        processing_class=processor, compute_metrics=metrics)
    if trainer.is_world_process_zero():
        processor.save_pretrained(args.output)
        (args.output / 'run.json').write_text(json.dumps(dict(arguments={k: str(v) for k, v in vars(args).items()},
            resolved_revision=revision, world_size=int(os.environ.get('WORLD_SIZE', 1)),
            manifest_sha256=digest(args.manifest), validation_sha256=digest(args.validation) if args.validation else None,
            torch=torch.__version__), indent=2))
    if training:
        trainer.train(resume_from_checkpoint=args.resume or None)
        trainer.save_model()
        trainer.save_state()
    prediction = trainer.predict(validation)
    if trainer.is_world_process_zero():
        (args.output / 'metrics.json').write_text(json.dumps(prediction.metrics, indent=2))
        hyp = processor.batch_decode(prediction.predictions, skip_special_tokens=True)
        with (args.output / 'errors.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['id', 'reference', 'hypothesis', 'substitutions', 'deletions', 'insertions'])
            for row, text in zip(validation.rows, hyp):
                result = jiwer.process_words(normalize(row['text']), normalize(text))
                writer.writerow([row['id'], row['text'], text, result.substitutions, result.deletions, result.insertions])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['inspect', 'prepare', 'smoke', 'train', 'evaluate'])
    p.add_argument('--root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--exclude-manifest', action='append', default=[])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--validation-fraction', type=float, default=.05)
    p.add_argument('--train-cap', type=int, default=2000)
    p.add_argument('--validation-cap', type=int, default=100)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--validation', type=Path)
    p.add_argument('--model', default='vinai/PhoWhisper-base')
    p.add_argument('--revision', default='main')
    p.add_argument('--resume')
    p.add_argument('--max-steps', type=int, default=10)
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    args = p.parse_args()
    if min(args.train_cap, args.validation_cap, args.workers) < 0 or min(args.max_steps, args.save_steps, args.batch, args.accumulation) <= 0:
        p.error('caps/workers must be nonnegative; steps/batch/accumulation positive')
    if args.mode in ('inspect', 'prepare'):
        if not args.root or not 0 < args.validation_fraction < 1:
            p.error('root and validation fraction in (0,1) required')
        if args.mode == 'inspect':
            print(json.dumps({s: len(read_metadata(args.root, s)) for s in ('train', 'test')}))
            return
    elif not args.manifest or (args.mode in ('train', 'smoke') and not args.validation):
        p.error('manifest required; training also needs validation')
    if args.mode in ('inspect', 'prepare') and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        p.error('prepare/inspect must run on one process')
    args.output.mkdir(parents=True, exist_ok=True)
    prepare(args) if args.mode == 'prepare' else run(args)


if __name__ == '__main__':
    main()
