"""Read-only cached deletion diagnostic; audio preprocessing only, never inference.

Official metadata and a bounded archive prefix are independently fetched without
credentials. Original manifest/caches/results remain immutable.
"""
import csv
import io
import json
from pathlib import Path
import sys
import tarfile
import time
from types import SimpleNamespace as NS
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.runners import long_audio_benchmark as b
from benchmarks.runners import gpt2_optimization as o
from benchmarks.audits import reevaluate_numeric_metrics as m
from benchmarks.runners import run_fleurs_holdout as h


def concentration(rows):
    ranked=sorted(rows,key=lambda r:(-r['metrics']['D'],r['id']))
    total=sum(r['metrics']['D'] for r in rows)
    return dict(deletions=total,nonzero_samples=sum(r['metrics']['D']>0 for r in rows),
                top_ids=[r['id'] for r in ranked[:3]],
                top1_deletions=sum(r['metrics']['D'] for r in ranked[:1]),
                top3_deletions=sum(r['metrics']['D'] for r in ranked[:3]),
                top1_share=sum(r['metrics']['D'] for r in ranked[:1])/total if total else 0,
                top3_share=sum(r['metrics']['D'] for r in ranked[:3])/total if total else 0)


def verify_tsv(rows, text):
    metadata=b.parse_fleurs_tsv(text)
    evidence=[]
    for r in rows:
        index=int(r['id'].split(':')[1]); record=metadata[index]
        if str(r['original_id'])!=record[0] or Path(r['archive_member']).name!=record[1] or r['reference']!=record[3]:
            raise ValueError('official TSV mapping mismatch')
        evidence.append(dict(id=r['id'],tsv_index=index,original_id=record[0],filename=record[1],
                             raw_transcription=record[2],transcription=record[3],transcription_column=3,
                             raw_differs=record[2]!=record[3]))
    return evidence


def verify_cache_ids(records):
    for r in records:
        for i,c in enumerate(r['candidates']):
            if c['id']!=b.digest((r['source']+r['id']+str(i)+json.dumps(c['tokens'])).encode()):
                raise ValueError('candidate token/index identity mismatch')


def verify_bindings(result, paths):
    o.verify(result)
    if any(result[k]!=b.file_hash(p) for k,p in paths.items()):
        raise ValueError('sealed artifact binding mismatch')


def physical_tsv_audit(rows, text):
    """Literal physical TSV fields; do not let CSV quotes consume a delimiter.

    Diagnostic only: never rewrite references or re-evaluate a cleaned reference.
    Seven physical fields are independently constrained by ID/file/sample count.
    """
    lines=text.splitlines()
    # Deliberate legacy-bug reproduction ONLY; never an ingestion path.
    legacy=list(csv.reader(io.StringIO(text),delimiter='\t'))
    evidence=[]
    for r in rows:
        index=int(r['id'].split(':')[1]); fields=lines[index].split('\t')
        if len(fields)!=7 or str(r['original_id'])!=fields[0] or Path(r['archive_member']).name!=fields[1] or int(fields[5])!=round(r['duration']*16000):
            raise ValueError('physical official TSV identity/schema mismatch')
        evidence.append(dict(id=r['id'],native_id=fields[0],filename=fields[1],physical_fields=len(fields),
            legacy_csv_fields=len(legacy[index]),source_samples=int(fields[5]),
            physical_raw_transcription=fields[2],physical_transcription=fields[3],
            adjacent_character_field=fields[4],
            stored_reference_equals_legacy_csv=r['reference']==legacy[index][3],
            stored_normalized_words=len(b.normalize(r['reference']).split()),
            physical_transcription_normalized_words=len(b.normalize(fields[3]).split()),
            character_field_normalized_words=len(b.normalize(fields[4]).split()),
            reference_contaminated=b.normalize(r['reference'])!=b.normalize(fields[3])))
    return evidence


def schema_detail():
    started=time.monotonic()
    output=h.OUT/'deletion-diagnostic'
    if (output/'schema-evidence.json').exists():
        raise ValueError('refusing existing schema evidence')
    audit=b.load(output/'diagnostic.json'); o.verify(audit)
    if any(b.file_hash(p)!=sha for p,sha in audit['original_sha256'].items()):
        raise ValueError('original artifacts changed')
    tsv=(output/'official-test.tsv').read_text()
    if b.file_hash(output/'official-test.tsv')!=audit['source_identity']['tsv_sha256']:
        raise ValueError('official TSV changed')
    evidence=physical_tsv_audit(b.load(h.DATA/'manifest.json'),tsv)
    contaminated=[r for r in evidence if r['reference_contaminated']]
    preprocessing=audit['preprocessing']
    compact=dict(content_frames_min=min(r['content_frames'] for r in preprocessing),
                 content_frames_max=max(r['content_frames'] for r in preprocessing),
                 trimmed_content_frames=sum(r['trimmed_content_frames'] for r in preprocessing),
                 output_shapes=sorted({tuple(r['output_shape']) for r in preprocessing}),
                 candidate_min=min(x for r in preprocessing for x in r['candidate_lengths']),
                 candidate_max=max(x for r in preprocessing for x in r['candidate_lengths']),
                 reference_reencoded_max=max(r['reference_reencoded_tokens'] for r in preprocessing),
                 greedy_reencoded_max=max(r['greedy_reencoded_tokens'] for r in preprocessing),
                 top3=[r for r in preprocessing if r['id'] in audit['concentration']['top_ids']])
    b.save(output/'schema-evidence.json',o.seal(dict(rows=evidence,contaminated=contaminated,
        preprocessing_summary=compact,diagnostic_sha256=b.file_hash(output/'diagnostic.json'),
        tsv_sha256=b.file_hash(output/'official-test.tsv'),script_sha256=b.file_hash(Path(__file__)),
        seconds=time.monotonic()-started,references_modified=False,new_cleaned_reference_metrics=False)))
    c=audit['concentration']; old=audit['old_test_concentration']
    lines=['## Targeted deletion follow-up — cached only, original headline unchanged','',
        f"Fresh frozen GPT2 D has {c['deletions']} deletions across {c['nonzero_samples']}/30 samples. Top1 contributes {c['top1_deletions']} ({100*c['top1_share']:.4f}%); top3 contribute {c['top3_deletions']} ({100*c['top3_share']:.4f}%). Old test D has {old['deletions']} deletions across {old['nonzero_samples']}/30, top1 {old['top1_deletions']} / top3 {old['top3_deletions']}.",
        '', '| Sample | Duration s | Ref words D | GPT2 S/D/I | Word-oracle D | Selected / oracle tokens | Candidate token range |',
        '|---|---:|---:|---|---:|---|---|']
    for r in audit['per_sample'][:3]:
        x=r['metrics']
        lines.append(f"| {r['id']} | {r['duration']:.2f} | {r['reference_words']} | {x['S']}/{x['D']}/{x['I']} | {r['method_metrics']['oracle']['D']} | {r['selected_tokens']} / {r['oracle_tokens']} | {r['candidate_token_min']}–{r['candidate_token_max']} |")
    lines += ['', '### Confirmed reference-ingestion defect, not a listening conclusion',
        'The top outlier test:836 (native ID1898) has a sentence followed by a character-spelled duplicate. This is not merely an unusually long transcript: the official physical TSV line has seven tab-separated fields, while the existing `csv.reader(..., delimiter="\\t")` quote handling returns six, merging transcription field3 with adjacent character field4 because field3 has an unmatched opening quote. The local manifest exactly reproduces this legacy parser output. The diagnostic first reproduced that mapping, then independently inspected literal tab boundaries; reproducing the legacy parser alone was insufficient validation.',
        f"Across all30, {len(contaminated)} normalized stored references differ from physical transcription column3. No manifest, reference, selected candidate, score or original result has been edited. No cleaned-reference score is published here."]
    for r in contaminated:
        lines.append(f"- {r['id']}: native ID {r['native_id']}, filename `{r['filename']}`, source samples {r['source_samples']}; physical TSV fields {r['physical_fields']} versus CSV fields {r['legacy_csv_fields']}; stored ref {r['stored_normalized_words']} words versus physical transcription {r['physical_transcription_normalized_words']} words, adjacent character field {r['character_field_normalized_words']} normalized tokens.")
    lines += ['The 148-deletion result is therefore dominated by contaminated reference text, not evidence of 148 missing spoken words. The exact true spoken transcript, punctuation/quote readings and remaining lexical mismatches still require human listening. Fixing the shared acquisition parser, auditing its historical output and publishing a separately versioned reference correction would require a new explicit change; this diagnostic does not silently perform that change or replace any holdout row.',
        '', '### Top3 saved text (D; no listening performed)']
    for r in audit['per_sample'][:3]:
        lines += [f"- **{r['id']} reference:** `{r['reference']}`",f"  **Frozen GPT2:** `{r['hypotheses']['gpt2_frozen']}`",f"  **Word oracle:** `{r['hypotheses']['oracle']}`"]
    lines += ['', '### Source and cache identity',
        'All30 sample/native IDs and archive filenames match the independently fetched pinned official TEST TSV. All30 physical rows have seven fields, and their official sample-count column5 equals the decoded duration ×16000. Column2 is raw transcription, column3 transcription, column4 the character-spelled field; the preserved TSV makes this distinction directly inspectable. The attempted pinned `fleurs.py` builder fetch returned HTTPError and is not claimed as schema evidence. No missing builder content was inferred.',
        'All three top-deletion local WAV files independently match their named members in the pinned official TEST archive byte-for-byte (SHA256), not just filename. This rules out an observed top3 file swap; it does not certify the speech/transcript match by listening.',
        'Strict integrity passed: original results/frozen seals, result→manifest/N-best/GPT2/protocol SHA bindings; full strict N-best/scored schemas; sample/source/split joins; exact frozen validation protocol/revision; unchanged old/backups and normalizer hashes; candidate IDs recomputed from source+sample+index+token arrays; candidate text independently decoded from stored tokens; greedy/candidate source equality between N-best and scored; and exact cached reconstruction of every saved metric/result row. Fresh-vs-old/superseded disjointness was rechecked. Fresh30 full FFmpeg/file/PCM/duration validation passed again.',
        '', '### Preprocessing and stopping evidence',
        f"All30 waveforms were decoded and actual production mel preprocessing was recomputed without a model forward. Content frames {compact['content_frames_min']}–{compact['content_frames_max']} (10ms/frame), retained content frames exactly equal the corresponding full spectrogram prefix; zero content frames trimmed. All output mels are {compact['output_shapes']}, right-padded to3000. Thus the 15.06–28.38s source audio fits the30s window; there is no observed audio-content truncation in this preprocessing path.",
        f"Metadata-only checkpoint read used `torch.load(map_location='meta', weights_only=True)` (no model instantiation, materialized weights, or encoder/decoder inference), after matching its frozen SHA. n_text_ctx={audit['decoder']['n_ctx']}, default sample_len={audit['decoder']['default_sample_len']} and4 initial tokens. Installed Whisper DecodingTask uses sample_len or n_text_ctx//2; main loop stops on completion/context or sample iteration cap. Across450 cached candidates, generated lengths are {compact['candidate_min']}–{compact['candidate_max']}; none reaches224. Even max141+4 is below448. Top outlier has78–82 generated tokens, selected79, oracle80, far below either hard length cap.",
        'The cache strips EOT and does not store generation-loop stop reasons. Beam finalize can append EOT to unfinished sequences, so actual per-candidate natural-EOT events cannot be independently reconstructed from cache alone. No candidate shows a token-length-cap signature, and the dominant deletion anomaly has direct reference-parser evidence; normal/early EOT for remaining errors is unconfirmed. Greedy lengths are only text re-encodings, not retained greedy token sequences. No silent LM truncation is used: the scorer checks context and fails on overflow.',
        '', '### Leave-one-out sensitivity ONLY — not a revised headline or exclusion rule',
        'Omitting test:836 solely to quantify concentration gives the following fixed-GPT2 diagnostic. The original30-sample A/D headline results remain unchanged; this is not tuning, a replacement holdout, reference cleaning or an adopted exclusion.']
    for conv,x in audit['sensitivity_diagnostic_only']['metrics'].items():
        lines.append(f"- {conv}, remaining29: S/D/I={x['S']}/{x['D']}/{x['I']}, {x['errors']}/{x['words']} word errors, WER {100*x['wer']:.4f}%, CER {100*x['cer']:.4f}%.")
    lines += ['', '### Commands, tests and artifacts',
        '```sh',
        '.venv-benchmark/bin/python -m unittest discover -s tests -p test_holdout_deletion_audit.py -v',
         'HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python benchmarks/audits/audit_holdout_deletions.py',
         '.venv-benchmark/bin/python benchmarks/audits/audit_holdout_deletions.py --schema-detail',
        '```',
        f"Actual first diagnostic wall time {audit['seconds']:.3f}s including metadata-only checkpoint read, pure audio preprocessing, validation and bounded network. Independent source transfer `{json.dumps(audit['source_identity']['network'])}`; total data bytes including original acquisition {402066784+audit['source_identity']['network']['bytes']:,} (<1GiB). No credentials/signed URLs accessed/output. Existing source artifacts were checked unchanged at end.",
        'New isolated tests demonstrated RED (missing module/helper), then GREEN: five portable synthetic tests cover concentration/zero denominator, swapped official filename/transcription, candidate-token tampering, sealed manifest binding/tampering, and the actual malformed-quote contamination pattern. No model reruns, decoding, rescoring, tuning, downloads of weights, reference changes or holdout replacements. No full-suite rerun; unrelated passing checks are not repeated.',
        f"Artifact directory: `{output.relative_to(b.ROOT)}/`: sealed `diagnostic.json`, `schema-evidence.json`, `per-sample.csv`, independently fetched `official-test.tsv`, and `report.md`. Original frozen evaluation remains at `{h.OUT.relative_to(b.ROOT)}/results.json`; manifest `{h.DATA.relative_to(b.ROOT)}/manifest.json`."]
    report='\n\n'.join(lines)+'\n'
    (output/'report.md').write_text(report)
    with (b.ROOT/'docs/gpt2-error-audit-holdout.md').open('a') as f:
        f.write('\n'+report)
    print(json.dumps(dict(contaminated=contaminated,preprocessing_summary=compact,seconds=time.monotonic()-started),ensure_ascii=False,indent=2))


def official_identity(rows, top_ids, output):
    budget=256*1024*1024
    http=b.BudgetHTTP(budget,240)
    rev=rows[0]['source_revision']
    base=f'https://huggingface.co/datasets/google/fleurs/resolve/{rev}/'
    tsv=http.get(base+'data/vi_vn/test.tsv')
    (output/'official-test.tsv').write_bytes(tsv)
    mapping=verify_tsv(rows,tsv.decode())
    # Keep the pinned official builder as independently inspectable schema evidence.
    source_status={}
    try:
        source=http.get(base+'fleurs.py')
        (output/'official-fleurs.py').write_bytes(source)
        source_status=dict(passed=True,sha256=b.digest(source))
    except Exception as exc:
        source_status=dict(passed=False,error_type=type(exc).__name__)
    targets={r['archive_member']:r for r in rows if r['id'] in top_ids}
    checked=[]; status='complete'
    request=urllib.request.Request(base+'data/vi_vn/audio/test.tar.gz')
    http.requests+=1
    try:
        with b.safe_urlopen(request,timeout=20) as response:
            class Limited:
                def read(self,size=-1):
                    if time.monotonic()>=http.deadline or http.remaining<=0:
                        raise RuntimeError('bounded independent archive check exhausted')
                    data=response.read(min(65536,http.remaining,size if size>=0 else 65536))
                    http.remaining-=len(data)
                    return data
            with tarfile.open(fileobj=Limited(),mode='r|gz') as archive:
                for member in archive:
                    r=targets.get(member.name)
                    if not r or not member.isfile():
                        continue
                    data=archive.extractfile(member).read()
                    if b.digest(data)!=r['sha256']:
                        raise ValueError('independent official archive audio mismatch')
                    checked.append(dict(id=r['id'],archive_member=member.name,sha256=b.digest(data),
                                        bytes=len(data),exact_local_file_match=True))
                    if len(checked)==len(targets):
                        break
    except (OSError,RuntimeError,tarfile.TarError) as exc:
        status=type(exc).__name__
    return dict(tsv_sha256=b.digest(tsv),mapping=mapping,builder=source_status,
                archive_top3=checked,archive_status=status,archive_complete=len(checked)==len(targets),
                network=dict(bytes=budget-http.remaining,requests=http.requests,seconds=time.monotonic()-http.started,
                             max_bytes=budget,max_seconds=240))


def main():
    started=time.monotonic()
    output=h.OUT/'deletion-diagnostic'
    if output.exists():
        raise ValueError('refusing existing diagnostic')
    manifest=h.DATA/'manifest.json'
    paths=dict(manifest_sha256=manifest,nbest_sha256=h.OUT/'nbest.json',scores_sha256=h.OUT/'gpt2.json',
               frozen_protocol_sha256=h.OUT/'frozen-protocol.json')
    before={str(p):b.file_hash(p) for p in [*paths.values(),h.OUT/'results.json']}
    result=b.load(h.OUT/'results.json'); verify_bindings(result,paths)
    persisted=b.load(paths['frozen_protocol_sha256']); o.verify(persisted)
    freeze=b.load(o.OUT/'beam15-freeze.json'); o.verify(freeze)
    if persisted['freeze']!=freeze or persisted['freeze_source_sha256']!=b.file_hash(o.OUT/'beam15-freeze.json'):
        raise ValueError('frozen source mismatch')
    if persisted['normalizer_sha256']!=b.file_hash(Path(m.__file__)) or any(b.file_hash(b.ROOT/p)!=sha for p,sha in persisted['source_sha256'].items()):
        raise ValueError('normalizer/old source changed')
    scored,mapping=o.checked(manifest,paths['nbest_sha256'],paths['scores_sha256'])
    o.verify_test(freeze,scored,b.file_hash(manifest))
    nbest=b.load(paths['nbest_sha256']); verify_cache_ids(nbest['records'])
    if any(n['greedy']!=s['greedy'] for n,s in zip(nbest['records'],scored['records'],strict=True)):
        raise ValueError('greedy cache mismatch')
    rows=b.load(manifest)
    if len(rows)!=30 or any(r['role']!='fresh_holdout' or r['split']!='test' for r in rows):
        raise ValueError('holdout identity')
    blocked=[b.load(b.ROOT/p) for p in persisted['source_sha256']]
    blocked=[r for group in blocked for r in group]
    if not all(h.disjoint(r,blocked+rows[:i]) for i,r in enumerate(rows)):
        raise ValueError('holdout disjointness')
    validation=b.validate(rows)
    summary,details=h.evaluate(rows,scored['records'],freeze['alpha'],freeze['beta'])
    if summary!=result['summary'] or details!=result['records']:
        raise ValueError('persisted results differ from cached reconstruction')
    per_sample=[]
    cache={r['id']:r for r in scored['records']}
    for d in details:
        if d['convention']!='D':
            continue
        r=mapping['google/fleurs',d['id']]
        metrics={k:m.measure([d['reference']],[text]) for k,text in d['hypotheses'].items()}
        c=cache[d['id']]
        per_sample.append(dict(id=d['id'],duration=r['duration'],reference_words=len(d['reference'].split()),
            raw_reference=r['reference'],reference=d['reference'],hypotheses=d['hypotheses'],
            metrics=metrics['gpt2_frozen'],method_metrics=metrics,
            selected_candidate_id=d['selected_candidate_id'],oracle_candidate_id=d['oracle_candidate_id'],
            candidate_count=len(c['candidates']),unique_candidates=d['unique_candidates'],
            selected_tokens=len(next(x for x in c['candidates'] if x['id']==d['selected_candidate_id'])['tokens']),
            oracle_tokens=len(next(x for x in c['candidates'] if x['id']==d['oracle_candidate_id'])['tokens']),
            candidate_token_min=min(len(x['tokens']) for x in c['candidates']),
            candidate_token_max=max(len(x['tokens']) for x in c['candidates']),
            deleted_spans=[x for x in m.edit_spans(d['reference'],d['hypotheses']['gpt2_frozen']) if x['type']=='delete']))
    conc=concentration(per_sample)
    ranked=sorted(per_sample,key=lambda r:(-r['metrics']['D'],r['id']))
    omit=ranked[0]['id']
    sensitivity={}
    for convention in ('A','D'):
        subset=[r for r in details if r['convention']==convention and r['id']!=omit]
        sensitivity[convention]=m.measure([r['reference'] for r in subset],[r['hypotheses']['gpt2_frozen'] for r in subset])
    # Metadata-only checkpoint read on meta device: no model is instantiated,
    # no weights are materialized on CPU/GPU, no encoder/decoder forward occurs.
    import torch
    import whisper
    from whisper.tokenizer import get_tokenizer
    from src.inference import model_path
    from benchmarks.runners.generate_dev_nbest import extract_standard_mel
    checkpoint=model_path()
    if b.file_hash(checkpoint)!=freeze['protocol']['acoustic']['checkpoint_sha256']:
        raise ValueError('local checkpoint identity changed')
    meta=torch.load(checkpoint,map_location='meta',weights_only=True)
    dims=meta['dims']; del meta
    tokenizer=get_tokenizer(dims['n_vocab']>=51865,num_languages=dims['n_vocab']-51765-int(dims['n_vocab']>=51865),language='vi',task='transcribe')
    sample_limit=dims['n_text_ctx']//2
    initial=len(tokenizer.sot_sequence_including_notimestamps)
    model_stub=NS(dims=NS(n_mels=dims['n_mels']),device=torch.device('cpu'))
    preprocessing=[]
    for r in rows:
        audio=whisper.load_audio(r['audio_path'])
        full=whisper.log_mel_spectrogram(audio,dims['n_mels'],padding=whisper.audio.N_SAMPLES)
        frames=full.shape[-1]-whisper.audio.N_FRAMES
        actual=extract_standard_mel(r['audio_path'],model_stub,fp16=False)
        if len(audio)/16000!=r['duration'] or frames>3000 or not torch.equal(actual[:,:frames],full[:,:frames]):
            raise ValueError('unexpected audio truncation or frame mismatch')
        c=cache[r['id']]
        for candidate in c['candidates']:
            if tokenizer.decode(candidate['tokens']).strip()!=candidate['text']:
                raise ValueError('cached token/text mismatch')
        preprocessing.append(dict(id=r['id'],samples=len(audio),duration=len(audio)/16000,content_frames=frames,
                                  full_frames=full.shape[-1],output_shape=list(actual.shape),trimmed_content_frames=0,
                                  selected_tokens=len(o.choose(c['candidates'],.3,-.02)['tokens']),
                                  reference_reencoded_tokens=len(tokenizer.encode(r['reference'])),
                                  greedy_reencoded_tokens=len(tokenizer.encode(c['greedy'])),
                                  candidate_lengths=[len(x['tokens']) for x in c['candidates']]))
    output.mkdir()
    identity=official_identity(rows,conc['top_ids'],output)
    old_metrics=b.load(b.ROOT/'outputs/benchmarks/long/numeric-metric-v2-20261104-final/metrics.json')
    old=[r for r in old_metrics['records'] if r['split']=='test' and r['convention']=='D' and r['method']=='gpt2_tuned_beam15']
    old_conc=concentration(old)
    if any(b.file_hash(p)!=sha for p,sha in before.items()):
        raise ValueError('original cache changed during audit')
    audit=dict(concentration=conc,old_test_concentration=old_conc,per_sample=ranked,
               sensitivity_diagnostic_only=dict(omitted_id=omit,remaining=29,metrics=sensitivity),
               validation=validation,strict_integrity='passed: source seals, manifest/cache hashes, protocols, candidate identities, exact persisted metric reconstruction',
               source_identity=identity,preprocessing=preprocessing,
               decoder=dict(checkpoint_sha256=b.file_hash(checkpoint),dims=dims,default_sample_len=sample_limit,
                            n_ctx=dims['n_text_ctx'],initial_tokens=initial,eot=tokenizer.eot,
                            candidate_count=sum(len(r['candidates']) for r in scored['records']),
                            max_candidate_tokens=max(x for r in preprocessing for x in r['candidate_lengths']),
                            candidates_at_sample_limit=sum(x>=sample_limit for r in preprocessing for x in r['candidate_lengths']),
                            stopping_reasons_persisted=False,
                            limitation='EOT and loop stop reasons are not persisted; output tokens strip EOT and finalize may append EOT. Re-encoded greedy length is not a recorded greedy token sequence.'),
               original_sha256=before,source_sha256={str(p):b.file_hash(p) for p in
                    (Path(__file__),Path(h.__file__),b.ROOT/'benchmarks/runners/generate_dev_nbest.py',Path(whisper.decoding.__file__),Path(whisper.audio.__file__))},
               seconds=time.monotonic()-started,human_listening='not performed; pending')
    b.save(output/'diagnostic.json',o.seal(audit))
    with (output/'per-sample.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['id','duration','ref_words','S','D','I','selected_tokens','oracle_D','oracle_tokens','candidate_min','candidate_max'])
        for r in ranked:
            writer.writerow([r['id'],r['duration'],r['reference_words'],r['metrics']['S'],r['metrics']['D'],r['metrics']['I'],r['selected_tokens'],r['method_metrics']['oracle']['D'],r['oracle_tokens'],r['candidate_token_min'],r['candidate_token_max']])
    print(json.dumps(dict(concentration=conc,old_test_concentration=old_conc,top3=ranked[:3],
                         sensitivity=audit['sensitivity_diagnostic_only'],decoder=audit['decoder'],
                         validation=validation,identity=dict(archive_complete=identity['archive_complete'],builder=identity['builder'],network=identity['network']),
                         seconds=audit['seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    if sys.argv[1:]==['--schema-detail']:
        schema_detail()
    else:
        main()
