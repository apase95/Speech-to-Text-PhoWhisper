"""Guarded user span audit, fixed cached evaluation and offline copied-WAV packages.

No decoder, scorer, network operation or candidate selection is invoked.
"""
import copy
import argparse
import csv
import html
import io
import json
from pathlib import Path
import re
import shutil
import sys
import time
import wave
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import long_audio_benchmark as b
from scripts.data import normalize_benchmark_references as n
from tools import reevaluate_numeric_metrics as m
from tools import gpt2_optimization as o

VERSION = 'fleurs60-human-v3'
CORRECTIONS = {
    'test:266': [('hai nghìn không trăm linh hai', 'hai không không hai')],
    'test:153': [('bốn mươi lăm', 'bốn lăm')],
    'test:472': [('hai nghìn không trăm linh hai', 'hai ngàn lẻ hai')],
    'test:813': [('một nghìn chín trăm tám mươi tám', 'một ngàn chín trăm tám tám')],
    'validation:126': [('bốn trăm nghìn', 'bốn trăm ngàn')],
    'validation:151': [('bơi một trăm mét và hai trăm mét', 'hơn một trăm mét và hai trăm mét')],
    'validation:231': [('bốn nghìn tám trăm chín mươi hai mét', 'bốn ngàn tám trăm chín mươi hai mét')],
    'validation:237': [('một nghìn bảy trăm tám mươi chín', 'một ngàn bảy trăm tám mươi chín')],
    'validation:260': [('một nghìn', 'một ngàn'), ('hai nghìn', 'hai ngàn')],
    'validation:73': [('một trăm linh tám', 'một trăm lẻ tám')],
}
EVIDENCE = [
    'test:266 2002 -> hai không không hai',
    'test:153 45 -> bốn lăm',
    'test:472 2002 -> hai ngàn lẻ hai',
    'test:813 1988 -> một ngàn chín trăm tám tám',
    'validation:126 400.000 -> bốn trăm ngàn',
    'validation:151 bơi 100m và 200m -> hơn một trăm mét và hai trăm mét (explicitly user clearly heard hơn, not bơi)',
    'validation:231 4892 [m] -> bốn ngàn tám trăm chín mươi hai mét (avoid duplicating mét if existing)',
    'validation:237 1789 -> một ngàn bảy trăm tám mươi chín',
    'validation:260 1.000 -> một ngàn; 2000 -> hai ngàn',
    'validation:73 108 -> một trăm lẻ tám',
]
MANIFEST = b.ROOT/'data/benchmark/long-audio/manifest.json'
ANNOTATION = b.ROOT/'benchmarks/annotations/fleurs60-human-reviewed-v3.json'
QUEUE = b.ROOT/'outputs/benchmarks/long/error-audit-20261104/review-queue.json'
LEGACY_FIXED = b.ROOT/'outputs/benchmarks/long/numeric-metric-v2-20261104-final/fixed-hypotheses.json'
FIXED = b.ROOT/'outputs/benchmarks/long/human-review-v3-cache-verification-20261104/fixed-hypotheses.json'
OUT = b.ROOT/'outputs/benchmarks/long/human-review-v3-20261104'


def replace_spans(text, spans):
    positions = []
    for old, new in spans:
        if not old or not new or text.count(old) != 1:
            raise ValueError('missing/ambiguous user span')
        start = text.index(old)
        positions.append((start, start+len(old), new))
    positions.sort()
    if any(a[1] > z[0] for a, z in zip(positions, positions[1:])):
        raise ValueError('overlapping user spans')
    for start, end, new in reversed(positions):
        text = text[:start]+new+text[end:]
    return text


def transform(rows, annotation):
    if annotation['version'] != VERSION or len(rows) != annotation['expected_samples']:
        raise ValueError('version/count')
    entries = {(e['source'], e['id']): e for e in annotation['entries']}
    keys = [(r['source'], r['id']) for r in rows]
    if len(set(keys)) != len(keys) or len(entries) != len(annotation['entries']) or not entries.keys() <= set(keys):
        raise ValueError('identity')
    result = copy.deepcopy(rows)
    for r in result:
        e = entries.get((r['source'], r['id']))
        old = r.get('reference_v2', r['reference'])
        expected = old
        if e:
            if any(r[k] != e[k] for k in ('original_id', 'source_revision')) or r['sha256'] != e['audio_sha256'] or r['reference_raw'] != e['raw'] or old != e['old_v2']:
                raise ValueError('source/raw/v2 guard')
            expected = replace_spans(old, e['spans'])
            if expected != e['reviewed']:
                raise ValueError('reviewed span guard')
        convention = r['reference_convention']
        if convention == VERSION:
            if 'reference_v2' not in r or r['reference'] != expected:
                raise ValueError('existing v3 guard')
        elif convention != 'fleurs60-spoken-v2' or 'reference_v2' in r or r['reference'] != old:
            raise ValueError('existing v2 guard')
        r.update(reference=expected, reference_v2=old, reference_convention=VERSION)
    return result


def v2_rows(rows):
    result = copy.deepcopy(rows)
    for r in result:
        if r['reference_convention'] == VERSION:
            r['reference'] = r.pop('reference_v2')
            r['reference_convention'] = 'fleurs60-spoken-v2'
    if n.transform(result, b.load(n.DEFAULT_V2)) != result:
        raise ValueError('old v2 annotation binding')
    return result


def make_annotation(old, queue):
    ids = {r['id'] for r in queue['records']}
    if len(ids) != 13 or not CORRECTIONS.keys() <= ids:
        raise ValueError('corrections not in original13')
    entries = []
    for r in old:
        if r['id'] in CORRECTIONS:
            spans = [list(span) for span in CORRECTIONS[r['id']]]
            entries.append(dict(source=r['source'], id=r['id'], original_id=r['original_id'],
                source_revision=r['source_revision'], audio_sha256=r['sha256'],
                raw=r['reference_raw'], old_v2=r['reference'],
                reviewed=replace_spans(r['reference'], spans), spans=spans,
                span_verification='user-confirmed', whole_sentence_verification='limited, not independent audit'))
    return dict(version=VERSION, expected_samples=60,
        base_annotation_sha256=b.file_hash(n.DEFAULT_V2),
        v1_annotation_sha256=b.file_hash(n.DEFAULT), review_queue_sha256=b.file_hash(QUEUE),
        old_v2_manifest_sha256=b.digest((json.dumps(old, ensure_ascii=False, indent=2)+'\n').encode()),
        evidence_verbatim=EVIDENCE, entries=entries,
        unchanged_v2_user_assertion_ids=sorted(ids-CORRECTIONS.keys()),
        scope='User-confirmed listed spans; other original13 cases asserted correct v2. No all60/fresh30 human audit; remaining spans pending.')


def check_fixed(old):
    fixed = b.load(FIXED)
    o.verify(fixed)
    legacy = b.load(LEGACY_FIXED)
    o.verify(legacy)
    if fixed['records'] != legacy['records']:
        raise ValueError('historical fixed candidate/text changed')
    # Verify every bound source, including exact reconstructed OLD manifests.
    reconstructed = {'data/benchmark/long-audio/manifest.json': old}
    for split in ('validation', 'test'):
        reconstructed[f'data/benchmark/long-audio/splits/{split}.json'] = [r for r in old if r['split'] == split]
    for path, digest in fixed['inputs_sha256'].items():
        actual = b.digest((json.dumps(reconstructed[path], ensure_ascii=False, indent=2)+'\n').encode()) if path in reconstructed else b.file_hash(b.ROOT/path)
        if actual != digest:
            raise ValueError('fixed source hash mismatch: '+path)
    m.identity_join(old, fixed['records'])
    raw = m.original_rows(old)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for split in ('validation', 'test'):
            p = Path(tmp)/(split+'.json')
            b.save(p, [r for r in raw if r['split']==split])
            m.verify_original_cache(p, b.load(o.CACHE/split/'results.json'))
    winner = b.load(o.OUT/'winner.json')
    o.verify(winner)
    if winner['beam'] != 15 or winner['freeze_sha256'] != b.file_hash(o.OUT/'beam15-freeze.json'):
        raise ValueError('frozen winner identity')
    return fixed


def evaluate(old, reviewed, fixed):
    summaries, records = [], []
    conventions = {'A': m.original_rows(old), 'C': old, 'human-v3': reviewed}
    for convention, rows in conventions.items():
        mapping = {r['id']: r for r in rows}
        for split in ('validation', 'test'):
            items = [r for r in fixed['records'] if r['split']==split]
            for method in items[0]['hypotheses']:
                refs, hyps = [], []
                for item in items:
                    ref, hyp = m.normalized_pair(mapping[item['id']]['reference'], item['hypotheses'][method]['text'])
                    refs.append(ref); hyps.append(hyp)
                    records.append(dict(id=item['id'], split=split, convention=convention, method=method,
                        candidate_id=item['hypotheses'][method]['candidate_id'], reference=ref, hypothesis=hyp,
                        metrics=m.measure([ref], [hyp]), edit_spans=m.edit_spans(ref, hyp)))
                summaries.append(dict(split=split, convention=convention, method=method, **m.measure(refs, hyps)))
    return dict(summary=summaries, records=records, fixed_hypotheses_sha256=b.file_hash(FIXED),
                policy='Reference-only label audit, not independent model improvement. Literal NFC/lower/alphanumeric/space normalization; no dialect collapse; CER includes spaces. No reselection.')


def safe_id(value):
    if not re.fullmatch(r'(test|validation):[0-9]+', value):
        raise ValueError('unsafe sample id')
    return value.replace(':', '_')


def package(root, rows, hypotheses, provenance, resume=False):
    root = Path(root)
    def write(path, text):
        if path.exists() and path.read_bytes() != text.encode('utf-8'):
            raise ValueError('refusing to overwrite edited package file: '+str(path))
        if not path.exists():
            path.write_text(text, encoding='utf-8')
    def save(path, value):
        write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if root.exists() and not resume:
        raise ValueError('existing package')
    allowed_root = {'README.md', 'review.html', 'manifest.json', 'manifest.csv', 'SHA256.json'}
    allowed_sample = {'audio.wav', 'transcript_raw.txt', 'transcript_v2.txt', 'transcript_reviewed.txt', 'model_gpt2.txt', 'alignment.json', 'decision.json', 'annotated.html'}
    if root.exists():
        for p in root.rglob('*'):
            parts = p.relative_to(root).parts
            if p.is_symlink() or not ((len(parts)==1 and (parts[0] in allowed_root or parts[0] in {safe_id(r['id']) for r in rows})) or (len(parts)==2 and parts[0] in {safe_id(r['id']) for r in rows} and parts[1] in allowed_sample)):
                raise ValueError('unexpected resume package member')
    root.mkdir(parents=True, exist_ok=resume)
    records, index = [], []
    for r in rows:
        name = safe_id(r['id'])
        folder = root/name
        folder.mkdir(exist_ok=resume)
        audio = folder/'audio.wav'
        if not audio.exists():
            shutil.copyfile(r['audio_path'], audio)
        if audio.is_symlink() or b.file_hash(audio) != r['sha256']:
            raise ValueError('copied audio hash')
        try:
            with wave.open(str(audio), 'rb') as wav:
                frames = wav.readframes(wav.getnframes())
                if len(frames) != wav.getnframes()*wav.getnchannels()*wav.getsampwidth():
                    raise ValueError('truncated WAV')
        except wave.Error:
            # Official source WAVs include IEEE float, unsupported by stdlib wave.
            duration, _ = b.pcm_info(audio)
            if duration <= 0:
                raise ValueError('empty decoded WAV')
        hyp = hypotheses[r['id']]
        for filename, text in [('transcript_raw.txt', r['reference_raw']), ('transcript_v2.txt', r['reference_v2']),
                ('transcript_reviewed.txt', r['reference']), ('model_gpt2.txt', hyp['text'])]:
            write(folder/filename, text+'\n')
        spans = m.edit_spans(*m.normalized_pair(r['reference'], hyp['text']))
        save(folder/'alignment.json', spans)
        save(folder/'decision.json', dict(id=r['id'], span_status='user-confirmed' if r['id'] in CORRECTIONS else 'pending',
            whole_sentence_status='pending independent listening', decision='pending', notes=''))
        highlights = ''.join('<p><mark>'+html.escape(s['type'])+'</mark> REF: '+html.escape(s['reference'])+' / ASR: '+html.escape(s['hypothesis'])+'</p>' for s in spans)
        page = '<!doctype html><meta charset="utf-8"><title>'+html.escape(r['id'])+'</title><h1>'+html.escape(r['id'])+'</h1><audio controls src="audio.wav"></audio>'
        page += '<p>Mismatch needs listening; ASR is untrusted. Whole sentence not independently certified.</p>'
        for label, text in [('Raw',r['reference_raw']), ('Old v2',r['reference_v2']), ('Current human-v3',r['reference']), ('Untrusted GPT2',hyp['text'])]:
            page += '<h2>'+label+'</h2><p>'+html.escape(text)+'</p>'
        write(folder/'annotated.html', page+'<h2>Normalized word edits</h2>'+highlights)
        index.append('<li><a href="'+name+'/annotated.html">'+html.escape(r['id'])+'</a><audio controls src="'+name+'/audio.wav"></audio></li>')
        records.append(dict(id=r['id'], source=r['source'], original_id=r['original_id'], source_revision=r['source_revision'],
            audio_path=name+'/audio.wav', audio_sha256=r['sha256'], candidate_id=hyp['candidate_id'],
            raw=name+'/transcript_raw.txt', old_v2=name+'/transcript_v2.txt', reviewed=name+'/transcript_reviewed.txt',
            hypothesis=name+'/model_gpt2.txt', alignment=name+'/alignment.json',
            status='mismatch needs listening', confirmed_spans=CORRECTIONS.get(r['id'], [])))
    write(root/'review.html', '<!doctype html><meta charset="utf-8"><h1>FLEURS — cần nghe kiểm tra</h1><ul>'+''.join(index)+'</ul>')
    write(root/'README.md', '# Gói nghe kiểm tra FLEURS\n\nMở review.html trực tiếp; tất cả WAV được sao chép, không cần mạng. Có thể dùng VLC mở audio.wav.\n'
        'Hoặc chạy: python -m http.server 8080 --bind 127.0.0.1 --directory PACKAGE_DIR\nChỉ phục vụ thư mục gói, không phục vụ dự án.\n'
        'Raw là nguồn gốc; v2 là bản cũ; reviewed là human-v3 hiện tại; GPT2 chưa được xác nhận.\n'
        'Chỉ các đoạn được ghi user-confirmed đã được người dùng xác nhận. Những đoạn khác cần nghe; không phải kiểm định độc lập toàn câu.\n'
        'Alignment gắn nhãn substitute/delete/insert theo văn bản chuẩn hóa chữ thường, bỏ dấu câu, giữ khác biệt ngàn/nghìn và lẻ/linh. CER tính khoảng trắng.\n'
        'Có thể sửa transcript_reviewed.txt và decision.json để ghi đề xuất; chưa tự động áp dụng vào dự án. Áp dụng sau này phải kiểm tra nguồn và đoạn thay thế.\n')
    save(root/'manifest.json', dict(records=records, provenance=provenance))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=['id','audio_path','audio_sha256','raw','old_v2','reviewed','hypothesis','candidate_id'])
    writer.writeheader()
    writer.writerows({k:r[k] for k in writer.fieldnames} for r in records)
    write(root/'manifest.csv', stream.getvalue())
    paths = sorted(p for p in root.rglob('*') if p.is_file())
    for p in paths:
        if p.suffix == '.html':
            for target in re.findall(r'(?:src|href)="([^"]+)"', p.read_text(encoding='utf-8')):
                if target.startswith('/') or ':' in target or '..' in Path(target).parts or not (p.parent/target).is_file():
                    raise ValueError('nonportable HTML link')
    paths = [p for p in paths if p.name != 'SHA256.json']
    hashes = {p.relative_to(root).as_posix(): b.file_hash(p) for p in paths}
    save(root/'SHA256.json', hashes)
    paths.append(root/'SHA256.json')
    archive = root.with_suffix('.zip')
    if archive.exists() and not resume:
        raise ValueError('existing zip')
    if not archive.exists():
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in paths:
                relative = p.relative_to(root).as_posix()
                if p.is_symlink() or relative.startswith('/') or '..' in Path(relative).parts:
                    raise ValueError('unsafe zip member')
                z.write(p, relative)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or len(z.namelist()) != len(paths) or set(z.namelist()) != {p.relative_to(root).as_posix() for p in paths} or z.read('SHA256.json') != (root/'SHA256.json').read_bytes():
            raise ValueError('zip verification')
        for name, digest in hashes.items():
            if b.digest(z.read(name)) != digest:
                raise ValueError('zip hash')
    return dict(path=str(root), zip=str(archive), zip_bytes=archive.stat().st_size,
                samples=len(rows), zip_entries=len(paths), audio_hashes_verified=len(rows), files_hashed=len(hashes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, help='Existing pre-v3 backup for resuming an interrupted first publication')
    args = parser.parse_args()
    started = time.monotonic()
    active = b.load(MANIFEST)
    old = v2_rows(active)
    queue = b.load(QUEUE)
    annotation = make_annotation(old, queue)
    if ANNOTATION.exists() and b.load(ANNOTATION) != annotation:
        raise ValueError('existing v3 annotation mismatch')
    fixed = check_fixed(old)
    reviewed = transform(active, annotation)
    if transform(reviewed, annotation) != reviewed:
        raise ValueError('non-idempotent')
    # Mandatory real audio validation, no inference. Preflight before publication.
    validation = b.validate(reviewed)
    paths = [MANIFEST]+[MANIFEST.parent/'splits'/(s+'.json') for s in ('validation','test')]
    for path, split in zip(paths[1:], ('validation','test')):
        if b.load(path) != [r for r in active if r['split']==split]:
            raise ValueError('active partitions')
    metrics = evaluate(old, reviewed, fixed)
    if OUT.exists() and b.load(OUT/'metrics.json') != metrics:
        raise ValueError('existing execution metrics mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    b.save(ANNOTATION, annotation)
    b.save(OUT/'metrics.json', metrics)
    if active == reviewed:
        state = OUT/'apply-state.json'
        backup = Path(b.load(state)['backup']) if state.exists() else args.backup
        if backup is None or b.load(backup/'manifest.json') != old:
            raise ValueError('resume requires exact pre-v3 backup')
    else:
        backup = MANIFEST.parent/'reference-backups'/('human-v3-'+str(time.time_ns()))
    backup.mkdir(parents=True, exist_ok=active == reviewed)
    before = [p.read_bytes() for p in paths]
    for path, content in zip(paths, before):
        if active != reviewed:
            (backup/path.name).write_bytes(content)
    b.save(OUT/'apply-state.json', dict(backup=str(backup)))
    values = [reviewed]+[[r for r in reviewed if r['split']==s] for s in ('validation','test')]
    try:
        for path, value in zip(paths, values):
            n.atomic(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
    except BaseException:
        for path, content in zip(paths, before):
            n.atomic(path, content)
        raise
    if transform(b.load(MANIFEST), annotation) != b.load(MANIFEST):
        raise ValueError('actual reapply mismatch')
    hypotheses = {r['id']:r['hypotheses']['gpt2_tuned_beam15'] for r in fixed['records']}
    current = [r for r in metrics['records'] if r['convention']=='human-v3' and r['method']=='gpt2_tuned_beam15']
    mismatch_ids = {r['id'] for r in current if r['metrics']['errors']}
    char_only = [r for r in current if not r['metrics']['errors'] and r['metrics']['chars']]
    b.save(OUT/'char-only-differences.json', char_only)
    provenance = dict(annotation_sha256=b.file_hash(ANNOTATION), fixed_hypotheses_sha256=b.file_hash(FIXED),
        convention=VERSION, human_scope=annotation['scope'], evidence_verbatim=EVIDENCE,
        unchanged_v2_user_assertion_ids=annotation['unchanged_v2_user_assertion_ids'])
    package_root = b.ROOT/'outputs/review-packages'
    original_ids = {r['id'] for r in queue['records']}
    packages = [package(package_root/'fleurs13-human-reviewed-v3', [r for r in reviewed if r['id'] in original_ids], hypotheses, provenance, resume=True),
                package(package_root/'fleurs60-mismatches-human-v3', [r for r in reviewed if r['id'] in mismatch_ids], hypotheses, provenance, resume=True)]
    result = dict(changed_sentences=sum(a['reference'] != z['reference'] for a,z in zip(old,reviewed)),
        corrected_spans=sum(map(len,CORRECTIONS.values())), actual_reapply_changed=0, samples=60,
        mismatches=len(mismatch_ids), char_only=len(char_only), backup=str(backup), validation=validation,
        packages=packages, seconds=time.monotonic()-started)
    b.save(OUT/'execution.json', result)
    lines = ['# Human review / cached FLEURS60 benchmark', '', metrics['policy'], '',
        'Fresh30 and its test:836 parser correction are unchanged; not human-reviewed here.',
        'Only 10 sentences / 11 supplied spans corrected. Other original13 assertion covers 3 cases only. Whole-sentence independent verification remains pending.', '',
        '| Split | Ref | Method | Words | Characters | S | D | I | Errors | WER % | Char errors | CER % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in metrics['summary']:
        lines.append(f"| {r['split']} | {r['convention']} | {r['method']} | {r['words']} | {r['characters']} | {r['S']} | {r['D']} | {r['I']} | {r['errors']} | {r['wer']*100:.4f} | {r['chars']} | {r['cer']*100:.4f} |")
    lines += ['', 'S/D/I use existing deterministic backtrace; independent jiwer counts and all per-sample edits are in metrics.json. Equivalent optimal jiwer alignments may have different S/D/I, but word/character errors and denominators are independently checked.',
        'The sealed fixed-hypothesis artifact and every bound input hash were verified against reconstructed old-v2 manifests. Historical raw split hashes/references and frozen beam15 winner were checked. No scoring or reselection.',
        '', '## Execution', '', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```']
    text = '\n'.join(lines)+'\n'
    (OUT/'report.md').write_text(text, encoding='utf-8')
    (b.ROOT/'docs/human-review-benchmark-fleurs60.md').write_text(text, encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
