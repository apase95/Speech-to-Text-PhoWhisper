"""Versioned physical-TSV source correction of current90; cached inference only.

Never rewrite historical cache hashes. Bind old caches to their exact backed-up
manifest, verify a reference-only bridge, then score unchanged frozen hypotheses
against the separately hashed corrected source manifest. No model imports.
"""
import ast
import copy
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools import long_audio_benchmark as b
from tools import gpt2_optimization as o
from tools import reevaluate_numeric_metrics as m
from tools import run_fleurs_holdout as h
from tools.audit_holdout_deletions import verify_bindings, verify_cache_ids

VERSION='fleurs-source-tsv-v2'
EXTRA={'source_reference_v2','source_reference_raw_v2','source_reference_version',
       'reference_previous','reference_raw_previous','source_fields_previous','source_correction_evidence'}


def correct_rows(rows, metadata, metadata_hashes, activate=True):
    corrected=[]; evidence=[]
    for old in rows:
        if any(k in old for k in EXTRA):
            raise ValueError('source correction already present')
        split=old['original_split']; index=int(old['id'].split(':')[1])
        if split!=old['split'] or old['source']!='google/fleurs':
            raise ValueError('expected native FLEURS split')
        if old.get('archive_member'):
            candidates=[(index,metadata[split][index])]
        else:
            # Historical viewer ordinals are NOT necessarily physical TSV ordinals.
            # They do retain native sentence ID and decoded sample count. Require
            # all matching physical rows to agree on the source transcription;
            # retain the filename ambiguity instead of inventing a file identity.
            candidates=[(i,s) for i,s in enumerate(metadata[split]) if
                        str(old['original_id'])==s[0] and abs(old['duration']-int(s[5])/16000)<=1/16000]
        if not candidates or len({(s[2],s[3]) for i,s in candidates})!=1:
            raise ValueError('official source identity/sample-count ambiguity')
        index,source=candidates[0]
        if str(old['original_id'])!=source[0] or abs(old['duration']-int(source[5])/16000)>1/16000:
            raise ValueError('official source identity/sample-count mismatch')
        if old.get('archive_member') and old['archive_member']!=('dev' if split=='validation' else 'test')+'/'+source[1]:
            raise ValueError('official archive filename mismatch')
        previous_raw=old.get('reference_raw',old['reference'])
        item=dict(source=old['source'],id=old['id'],split=split,tsv_index=index,native_id=source[0],
                  filename=source[1],metadata_sha256=metadata_hashes[split],
                  matching_tsv_indices=[i for i,s in candidates],matching_filenames=[s[1] for i,s in candidates],
                  filename_identity='persisted archive member' if old.get('archive_member') else
                      'native sentence ID + exact samples; filename not independently audio-verified',
                  previous_source_reference=previous_raw,source_reference_v2=source[3],
                  physical_raw_transcription=source[2],official_samples=int(source[5]),
                  archive_filename_persisted=bool(old.get('archive_member')),
                  exact_difference=previous_raw!=source[3],
                  normalized_difference=b.normalize(previous_raw)!=b.normalize(source[3]))
        new=copy.deepcopy(old)
        new.update(source_reference_v2=source[3],source_reference_raw_v2=source[2],
                   source_reference_version=VERSION,reference_previous=old['reference'],
                   reference_raw_previous=old.get('reference_raw'),
                   source_fields_previous=copy.deepcopy(old),source_correction_evidence=item)
        if activate:
            if 'reference_convention' in old:
                raise ValueError('do not overwrite annotated convention with source text')
            new['reference']=source[3]
        corrected.append(new); evidence.append(item)
    verify_reference_only(rows,corrected)
    return corrected,evidence


def verify_reference_only(original, corrected):
    if len(original)!=len(corrected):
        raise ValueError('sample count changed')
    for old,new in zip(original,corrected,strict=True):
        restored={k:v for k,v in new.items() if k not in EXTRA}
        restored['reference']=old['reference']
        if (restored!=old or new.get('source_fields_previous')!=old or new.get('reference_previous')!=old['reference'] or
                new['reference'] not in (old['reference'],new['source_reference_v2'])):
            raise ValueError('not a reference-only source overlay')


def reader_inventory():
    paths=['tools/long_audio_benchmark.py','tools/audit_holdout_deletions.py',
           'tools/run_fleurs_holdout.py','tools/audit_gpt2_errors.py','tools/reevaluate_numeric_metrics.py',
            'scripts/data/normalize_benchmark_references.py','tools/gpt2_optimization.py',
            'scripts/data/correct_fleurs_sources.py']
    result=[]
    for name in paths:
        tree=ast.parse((b.ROOT/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                calls=[x for x in ast.walk(node) if isinstance(x,ast.Call) and
                       isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and
                       x.func.value.id=='csv' and x.func.attr=='reader']
                for call in calls:
                    result.append(dict(file=name,function=node.name,line=call.lineno,
                                       call=ast.unparse(call),
                                       role='intentional historical bug reproduction, not ingestion'
                                       if node.name=='physical_tsv_audit' else 'strict production literal TSV parser'))
    return result


def main():
    started=time.monotonic()
    output=h.OUT/'source-correction-v2'
    resume=sys.argv[1:]==['--resume-metadata']
    if output.exists() and not resume:
        raise ValueError('refusing existing source correction')
    if resume and (output/'corrected-results.json').exists():
        raise ValueError('already published')
    active=h.DATA/'manifest.json'; oldpath=b.ROOT/'data/benchmark/long-audio/manifest.json'
    originals=[active,oldpath,h.OUT/'nbest.json',h.OUT/'gpt2.json',h.OUT/'results.json',h.OUT/'frozen-protocol.json',
               oldpath.parent/'splits/validation.json',oldpath.parent/'splits/test.json']
    hashes={str(p):b.file_hash(p) for p in originals}
    historical=b.load(h.OUT/'results.json')
    verify_bindings(historical,dict(manifest_sha256=active,nbest_sha256=h.OUT/'nbest.json',
        scores_sha256=h.OUT/'gpt2.json',frozen_protocol_sha256=h.OUT/'frozen-protocol.json'))
    sealed=b.load(h.OUT/'frozen-protocol.json'); o.verify(sealed)
    freeze=b.load(o.OUT/'beam15-freeze.json'); o.verify(freeze)
    if sealed['freeze']!=freeze or sealed['freeze_source_sha256']!=b.file_hash(o.OUT/'beam15-freeze.json') or sealed['normalizer_sha256']!=b.file_hash(Path(m.__file__)):
        raise ValueError('freeze/normalizer changed')
    if any(b.file_hash(b.ROOT/p)!=sha for p,sha in sealed['source_sha256'].items()):
        raise ValueError('historical data source changed')
    scored,_=o.checked(active,h.OUT/'nbest.json',h.OUT/'gpt2.json')
    o.verify_test(freeze,scored,b.file_hash(active)); verify_cache_ids(scored['records'])
    nbest=b.load(h.OUT/'nbest.json')
    if any(n['greedy']!=s['greedy'] for n,s in zip(nbest['records'],scored['records'],strict=True)):
        raise ValueError('greedy source changed')
    fresh,old=b.load(active),b.load(oldpath)
    if len(fresh)!=30 or len(old)!=60:
        raise ValueError('expected current60 plus fresh30')
    for split in ('validation','test'):
        if b.load(oldpath.parent/f'splits/{split}.json')!=[r for r in old if r['split']==split]:
            raise ValueError('current60 partition mismatch')
    original_summary,original_details=h.evaluate(fresh,scored['records'],freeze['alpha'],freeze['beta'])
    if original_summary!=historical['summary'] or original_details!=historical['records']:
        raise ValueError('historical exact metric reconstruction failed')
    output.mkdir(exist_ok=resume)
    metadata={}; metadata_hashes={}; http=b.BudgetHTTP(4*1024*1024,60)
    revisions={r['source_revision'] for r in old+fresh}
    if len(revisions)!=1:
        raise ValueError('mixed official revisions')
    revision=next(iter(revisions))
    for split,native in [('validation','dev'),('test','test')]:
        if split=='test':
            cached=h.OUT/'deletion-diagnostic/official-test.tsv'
            prior=b.load(h.OUT/'deletion-diagnostic/schema-evidence.json'); o.verify(prior)
            if b.file_hash(cached)!=prior['tsv_sha256']:
                raise ValueError('cached pinned official metadata changed')
            data=cached.read_bytes()
        elif resume and (output/'official-dev.tsv').exists():
            data=(output/'official-dev.tsv').read_bytes()
        else:
            data=http.get(f'https://huggingface.co/datasets/google/fleurs/resolve/{revision}/data/vi_vn/{native}.tsv')
        (output/f'official-{native}.tsv').write_bytes(data)
        metadata[split]=b.parse_fleurs_tsv(data.decode()); metadata_hashes[split]=b.digest(data)
    corrected,evidence_fresh=correct_rows(fresh,metadata,metadata_hashes)
    old_overlay,evidence_old=correct_rows(old,metadata,metadata_hashes,activate=False)
    source_old=[dict(r,reference=r['source_reference_v2']) for r in old_overlay]
    if not all(h.disjoint(r,source_old+corrected[:i]) for i,r in enumerate(corrected)):
        raise ValueError('corrected physical-source transcript disjointness failed')
    validation=dict(current60=b.validate(old),corrected_fresh30=b.validate(corrected),combined90=b.validate(old+corrected))
    summary,details=h.evaluate(corrected,scored['records'],freeze['alpha'],freeze['beta'])
    old_choices={(r['id'],r['convention']):r['selected_candidate_id'] for r in historical['records']}
    if any(r['selected_candidate_id']!=old_choices[r['id'],r['convention']] for r in details):
        raise ValueError('frozen selection changed')
    per_sample=[]
    for row in details:
        per_sample.append(dict(id=row['id'],convention=row['convention'],
            metrics={k:m.measure([row['reference']],[v]) for k,v in row['hypotheses'].items()}))
    b.save(output/'current60-source-reference-v2.json',old_overlay)
    b.save(output/'corrected-holdout30-manifest.json',corrected)
    b.save(output/'source-reference-v2-overlay.json',o.seal(dict(version=VERSION,revision=revision,
        parser='csv.reader(delimiter=TAB, quoting=csv.QUOTE_NONE, strict=True); exactly7 physical fields',
        metadata_sha256=metadata_hashes,all90=evidence_old+evidence_fresh,reader_inventory=reader_inventory(),
        counts=dict(current60=60,fresh30=30,current60_exact_differences=sum(r['exact_difference'] for r in evidence_old),
                    current60_normalized_differences=sum(r['normalized_difference'] for r in evidence_old),
                    fresh30_exact_differences=sum(r['exact_difference'] for r in evidence_fresh),
                    fresh30_normalized_differences=sum(r['normalized_difference'] for r in evidence_fresh)),
        human_confirmation='pending: source schema correction is not listening validation')))
    backup=h.DATA/'source-correction-backups'/VERSION
    if backup.exists():
        raise ValueError('refusing existing historical backup')
    backup.mkdir(parents=True)
    shutil.copy2(active,backup/'manifest.json')
    if b.file_hash(backup/'manifest.json')!=hashes[str(active)]:
        raise ValueError('historical backup hash mismatch')
    # Strict old-cache join remains against the exact ORIGINAL backed-up manifest.
    # A reference-only bridge maps those unchanged audio/candidate identities to
    # the new manifest. No cache manifest hash or fingerprint is rewritten.
    o.checked(backup/'manifest.json',h.OUT/'nbest.json',h.OUT/'gpt2.json')
    verify_reference_only(b.load(backup/'manifest.json'),corrected)
    independent=dict(passed=True,engine='jiwer',corpus_rows=8,sample_method_rows=len(per_sample)*4,
                     corpus={c:{k:v['independent'] for k,v in methods.items() if k!='comparisons_vs_beam15'} for c,methods in summary.items()})
    result=o.seal(dict(version=VERSION,summary=summary,records=details,per_sample_metrics=per_sample,
        before_summary=historical['summary'],historical_results_sha256=b.file_hash(h.OUT/'results.json'),
        historical_manifest_sha256=b.file_hash(backup/'manifest.json'),historical_manifest=str(backup/'manifest.json'),
        corrected_manifest_sha256=b.file_hash(output/'corrected-holdout30-manifest.json'),
        source_overlay_sha256=b.file_hash(output/'source-reference-v2-overlay.json'),
        nbest_sha256=b.file_hash(h.OUT/'nbest.json'),scores_sha256=b.file_hash(h.OUT/'gpt2.json'),
        frozen_protocol_sha256=b.file_hash(h.OUT/'frozen-protocol.json'),independent=independent,
        validation=validation,source_code_sha256={str(p.relative_to(b.ROOT)):b.file_hash(p) for p in
            (Path(__file__),Path(b.__file__),Path(h.__file__),Path(m.__file__),Path(o.__file__))},
        no_sample_exclusions_or_replacements=True,frozen_selected_candidate_ids_unchanged=True,
        oracle='diagnostic word oracle recomputed for corrected references, decoder-order ties',
        network=dict(bytes=4*1024*1024-http.remaining,requests=http.requests,seconds=time.monotonic()-http.started,
                     max_bytes=4*1024*1024,max_seconds=60,test_tsv='reused independently fetched pinned/hash-verified file'),
        seconds=time.monotonic()-started,human_confirmation='pending'))
    b.save(output/'corrected-results.json',result)
    b.save(output/'independent-audit.json',independent)
    # Publish active holdout only after all checks and durable versioned outputs.
    if any(b.file_hash(p)!=sha for p,sha in hashes.items()):
        raise ValueError('historical inputs changed before publication')
    b.save(active,corrected)
    if b.file_hash(active)!=result['corrected_manifest_sha256']:
        raise ValueError('published corrected manifest hash mismatch')
    if any(b.file_hash(p)!=sha for p,sha in hashes.items() if p!=str(active)):
        raise ValueError('non-manifest historical input changed')
    report(output,result,evidence_old,evidence_fresh)
    print(json.dumps(dict(counts=b.load(output/'source-reference-v2-overlay.json')['counts'],
                         summary=summary,network=result['network'],seconds=result['seconds'],output=str(output)),ensure_ascii=False,indent=2))


def verify_published():
    started=time.monotonic()
    output=h.OUT/'source-correction-v2'
    result=b.load(output/'corrected-results.json'); o.verify(result)
    overlay=b.load(output/'source-reference-v2-overlay.json'); o.verify(overlay)
    active=h.DATA/'manifest.json'; backup=Path(result['historical_manifest'])
    if (b.file_hash(active)!=result['corrected_manifest_sha256'] or
            b.file_hash(output/'corrected-holdout30-manifest.json')!=b.file_hash(active) or
            b.file_hash(backup)!=result['historical_manifest_sha256'] or
            b.file_hash(output/'source-reference-v2-overlay.json')!=result['source_overlay_sha256']):
        raise ValueError('published correction binding mismatch')
    historical=b.load(h.OUT/'results.json')
    verify_bindings(historical,dict(manifest_sha256=backup,nbest_sha256=h.OUT/'nbest.json',
                                  scores_sha256=h.OUT/'gpt2.json',frozen_protocol_sha256=h.OUT/'frozen-protocol.json'))
    if b.file_hash(h.OUT/'results.json')!=result['historical_results_sha256']:
        raise ValueError('historical results changed')
    scores,_=o.checked(backup,h.OUT/'nbest.json',h.OUT/'gpt2.json')
    freeze=b.load(o.OUT/'beam15-freeze.json'); o.verify(freeze)
    protocol=b.load(h.OUT/'frozen-protocol.json'); o.verify(protocol)
    if protocol['freeze']!=freeze or protocol['freeze_source_sha256']!=b.file_hash(o.OUT/'beam15-freeze.json') or protocol['normalizer_sha256']!=b.file_hash(Path(m.__file__)):
        raise ValueError('frozen normalization/protocol changed')
    if any(b.file_hash(b.ROOT/p)!=sha for p,sha in protocol['source_sha256'].items()):
        raise ValueError('current60/backup manifests changed')
    o.verify_test(freeze,scores,b.file_hash(backup)); verify_cache_ids(scores['records'])
    original=b.load(backup); corrected=b.load(active)
    verify_reference_only(original,corrected)
    summaries,details=h.evaluate(corrected,scores['records'],freeze['alpha'],freeze['beta'])
    if summaries!=result['summary'] or details!=result['records']:
        raise ValueError('corrected metric/choice reconstruction mismatch')
    old=b.load(b.ROOT/'data/benchmark/long-audio/manifest.json')
    metadata={sp:b.parse_fleurs_tsv((output/f'official-{native}.tsv').read_text())
              for sp,native in [('validation','dev'),('test','test')]}
    hashes={sp:b.file_hash(output/f'official-{native}.tsv') for sp,native in [('validation','dev'),('test','test')]}
    if hashes!=overlay['metadata_sha256']:
        raise ValueError('physical TSV hashes changed')
    expected,ev=correct_rows(original,metadata,hashes)
    old_expected,old_ev=correct_rows(old,metadata,hashes,activate=False)
    if expected!=corrected or old_expected!=b.load(output/'current60-source-reference-v2.json') or old_ev+ev!=overlay['all90']:
        raise ValueError('all90 source correction reconstruction mismatch')
    validation=b.validate(corrected)
    verification=dict(passed=True,strict_old_cache_join='exact backed-up historical manifest',
        reference_only_bridge='all30 original rows retained, source correction reconstructed',
        all90_metadata_audit='reconstructed from pinned literal TSV and matched sealed evidence',
        validation=validation,current60_normalized_source_differences=sum(x['normalized_difference'] for x in old_ev),
        viewer_filename_unverified_rows=sum(not x['archive_filename_persisted'] for x in old_ev),
        viewer_ambiguous_filename_rows=[x['id'] for x in old_ev if len(x['matching_filenames'])>1],
        test836_corrected_metrics=[r for r in result['per_sample_metrics'] if r['id']=='test:836'],
        metadata_attempts=dict(initial_attempt='stopped before active publication: viewer ordinal differs from physical TSV ordinal; corrected to native ID + sample count mapping',
            initial_dev_requests=1,initial_dev_response_bytes=(output/'official-dev.tsv').stat().st_size,
            initial_wall_seconds='not instrumented',resumed_new_requests=0,
            metadata_bound_bytes=4*1024*1024,metadata_bound_seconds=60),
        seconds=time.monotonic()-started)
    b.save(output/'postpublication-verification.json',o.seal(verification))
    print(json.dumps(verification,ensure_ascii=False,indent=2))


def report(output,result,old,fresh):
    lines=['# Corrected source-v2 headline — all original30 holdout samples retained','',
        'This supersedes the contaminated-reference headline only. No audio/sample exclusions or replacements, model inference, LM rescoring, tuning, normalization changes or listening claims. Frozen greedy/top1/GPT2 texts and selected IDs are unchanged; the word oracle is reference-dependent and remains diagnostic.',
        '', '| Policy | Method | Words | S | D | I | Errors | WER % | CER % |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for convention,methods in result['summary'].items():
        for method,x in methods.items():
            if method!='comparisons_vs_beam15':
                lines.append(f"| {convention} | {method} | {x['words']} | {x['S']} | {x['D']} | {x['I']} | {x['errors']} | {100*x['wer']:.4f} | {100*x['cer']:.4f} |")
    lines+=['','A uses corrected literal source transcription; D applies the unchanged conservative numeric grammar to refs and hyps symmetrically. Both remain separate policies; source-schema validation is not human audio validation.',
            '', '## All current90 source audit']
    for label,items in [('current60',old),('fresh30',fresh)]:
        differences=[r for r in items if r['normalized_difference']]
        lines.append(f"- {label}: {len(items)} rows audited; {sum(r['exact_difference'] for r in items)} exact raw-source text differences; {len(differences)} normalized-source differences. IDs with normalized differences: {[r['id'] for r in differences]}.")
    lines+=['All rows checked against pinned official dev/test literal seven-field TSV, native ID, filenames and source sample count. Archive-backed rows require their persisted archive-member filename and physical ordinal to match. Historical viewer ordinals are not physical TSV ordinals: those rows map by native split/native ID/exact samples, requiring all matches to agree on source text. Physical filename candidates are retained, not claimed independently audio-verified in this metadata-only pass.',
            'Current60 active annotations/reference_raw and partitions were left untouched; the full enriched source_reference_v2 overlay preserves every prior field. Any old-source difference is explicitly represented there without forging original cache hashes. The only activated references are in the corrected fresh30 manifest, with complete previous-row snapshots and evidence.']
    for conv in ('A','D'):
        before=result['before_summary'][conv]['gpt2_frozen']; after=result['summary'][conv]['gpt2_frozen']
        lines.append(f"- Frozen GPT2 {conv}, all30 before→after: errors {before['errors']}→{after['errors']}, words {before['words']}→{after['words']}, S/D/I {before['S']}/{before['D']}/{before['I']}→{after['S']}/{after['D']}/{after['I']}, WER {100*before['wer']:.4f}%→{100*after['wer']:.4f}%, CER {100*before['cer']:.4f}%→{100*after['cer']:.4f}%.")
        lines.append(f"- {conv} comparisons versus beam15: `{json.dumps(result['summary'][conv]['comparisons_vs_beam15'])}`.")
    lines+=['','The exact diagnosis is a147-token character-spelled field appended to test:836 by CSV quote handling (193 stored words versus46 source transcription words), not128 words. Quotes remain literal source data; ordinary metric punctuation normalization removes them, not a per-sample manual override. The all30 corrected result is not the earlier29-sample sensitivity diagnostic. The old 148-deletion outlier is re-aligned against its real physical transcription here; per-sample counts for every method/policy are saved.',
            '', '## Integrity, publication and artifacts',
            'The original results.json/N-best/GPT2/frozen files are unchanged. The exact original manifest is backed up byte-for-byte and still passes strict o.checked joins against the old caches. New results bind both old and corrected manifest hashes, source-overlay hashes and original cache hashes. A verified reference-only identity bridge prohibits audio, IDs, split, source provenance or candidate changes. Every corrected sample/corpus word and character total was independently checked with jiwer. No protocol check was disabled, and no old hash was rewritten.',
            f"Versioned directory: `{output.relative_to(b.ROOT)}/`; corrected-results.json, independent-audit.json, source-reference-v2-overlay.json, current60-source-reference-v2.json, corrected-holdout30-manifest.json, official-dev.tsv, official-test.tsv. Active manifest: `data/benchmark/fleurs-holdout30/manifest.json`; historical original: `data/benchmark/fleurs-holdout30/source-correction-backups/{VERSION}/manifest.json`.",
            f"Wall time {result['seconds']:.3f}s; metadata-only network `{json.dumps(result['network'])}`. All90 full audio/hash/duration validations passed before publication. Human listening remains pending for numeric/dialect and lexical readings; no audio was certified."]
    text='\n'.join(lines)+'\n'
    (output/'report.md').write_text(text)
    path=b.ROOT/'docs/gpt2-error-audit-holdout.md'
    previous=path.read_text()
    path.write_text(text+'\n---\n\n# Historical execution and diagnostic record (superseded headline)\n\n'+previous)


if __name__=='__main__':
    if sys.argv[1:]==['--verify']:
        verify_published()
    else:
        main()
