"""Supplementary reference-only curation; explicit staged seals before inference."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys
import time
import tarfile
import urllib.request
import os
import html
import shutil
import zipfile
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools import long_audio_benchmark as b
from tools import gpt2_optimization as o
from tools import reevaluate_numeric_metrics as m
from scripts.data import human_reference_review as hr
from tools import run_fleurs_holdout as h

DATA = b.ROOT/'data/benchmark/fleurs-general60'
OUT = b.ROOT/'outputs/benchmarks/long/fleurs-general60-20261104'
SOURCE = b.ROOT/'outputs/benchmarks/long/fleurs-holdout30-20261104/source-correction-v2'
OLD = b.ROOT/'data/benchmark/long-audio/manifest.json'
HOLD = b.ROOT/'data/benchmark/fleurs-holdout30/manifest.json'
POLICY = dict(version='general-content-reference-only-v1',
    purpose='Supplementary FLEURS general-content curated; not unbiased or independent',
    screening='Raw physical TSV column2 plus transcription column3; manual decisions on all duration-eligible texts and current60 before inference',
    exclusions=['any explicit digits', 'explicit spoken numeric counts, dates, ages, years, ordinals or units; simple indefinite một allowed',
                'clearly foreign proper people/places/organizations, titles, acronyms, code, citations or transliteration-heavy content'],
    inclusion='Plain Vietnamese general science/social/everyday explanations, no novel named entities; Vietnamese generic words are not foreign names',
    uncertainty='Conservative manual text decision with exact spans; not exhaustive automatic NER',
    forbidden=['model errors or hypotheses as screening inputs', 'accent/dialect guesses', 'post-inference exclusions', 'new reference rewrites'],
    duration='15 <= seconds < 30; prefer20+ replacements then15+; no sub15 fallback',
    selection='retain eligible current60 in original order, deduplicate text; replacements ascending physical TSV index within20+ then15+ tiers; native splits',
    holdout='fresh30 excluded by native sentence ID/audio/PCM/text; no speaker/training independence claim',
    accent='Unknown, human verification pending; no accent restriction')


def automatic_flags(text):
    patterns = [('digits', r'\d+'), ('spoken_numeric', r'\b(?:mười|mươi|trăm|nghìn|ngàn|triệu|tỷ)\b')]
    return [dict(reason=reason, span=match.group()) for reason, pattern in patterns
            for match in re.finditer(pattern, text.lower())]


def unique(rows):
    seen = set()
    for r in rows:
        if r['split'] != r['original_split']:
            raise ValueError('native split changed')
        keys = [('id', r['source'], r['id']), ('native', r['source'], str(r['original_id'])),
                ('audio', r['sha256']), ('pcm', r['pcm_sha256']), ('text', b.normalize(r['reference']))]
        if any(k in seen for k in keys):
            raise ValueError('duplicate identity/audio/text')
        seen.update(keys)


def inventory():
    if OUT.exists() or DATA.exists():
        raise ValueError('existing curation')
    OUT.mkdir(parents=True); DATA.mkdir(parents=True)
    paths = [OLD, OLD.parent/'splits/validation.json', OLD.parent/'splits/test.json', HOLD]
    freeze = b.load(o.OUT/'beam15-freeze.json'); o.verify(freeze)
    winner = b.load(o.OUT/'winner.json'); o.verify(winner)
    if (freeze['alpha'], freeze['beta'], freeze['revision'], winner['beam']) != (.3, -.02, o.REV, 15) or winner['freeze_sha256'] != b.file_hash(o.OUT/'beam15-freeze.json'):
        raise ValueError('freeze mismatch')
    b.save(OUT/'policy.json', o.seal(dict(policy=POLICY, created_unix=time.time(),
        original_sha256={str(p.relative_to(b.ROOT)): b.file_hash(p) for p in paths},
        freeze=freeze, winner=winner, normalizer_sha256=b.file_hash(Path(m.__file__)),
        code_sha256=b.file_hash(Path(__file__)))))
    pool=[]
    for split,native in [('validation','dev'),('test','test')]:
        path=SOURCE/f'official-{native}.tsv'
        for i,s in enumerate(b.parse_fleurs_tsv(path.read_text())):
            duration=int(s[5])/16000
            pool.append(dict(key=f'{split}:{i}', split=split, index=i, native_id=s[0], filename=s[1],
                raw=s[2], reference=s[3], duration=duration, metadata_sha256=b.file_hash(path),
                automatic=automatic_flags(s[2]+' '+s[3])))
    b.save(OUT/'full-pool.json', pool)
    old=b.load(OLD)
    lines=['CURRENT60']+[f"{r['id']} | {r['duration']} | {r.get('reference_raw',r['reference'])}" for r in old]
    lines+=['DURATION-ELIGIBLE POOL (automatic-clear only; flagged rows remain audited in full-pool.json)']
    lines += [f"{r['key']} | {r['duration']} | {r['raw']}" for r in pool if 15<=r['duration']<30 and not r['automatic']]
    (OUT/'screening-text.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(pool=len(pool), duration_eligible=sum(15<=r['duration']<30 for r in pool),
        automatic_clear=sum(15<=r['duration']<30 and not r['automatic'] for r in pool), output=str(OUT))))


# Manual source-only screening of the bounded175 automatic-clear long recordings.
# Tokens below are literal exclusion spans, not ASR-derived spelling rules.
FOREIGN = ('bhajan|shri shyam|point marion|majorca|java|npws|armand versace|kruger|catalan|bishkek|'
           'bush|romania|newt gingrich|iraq|ski bums|plitvice|alpes|thẻ wild|barcelona|catalunya|giza|'
           'toginet|tây âu|do thái|paris|saint petersburg|greenland|muhammad|schengen|tiền anh|'
           'kundalini|kitô|tupua|california|katrina|princeton|danielle|scaffolds|singapore|sundarbans|'
           'usoc|nyiragongo|olympic|hồng kông|sintra|pamela|goma|ottawa|cochamó|deity|oliver sacks|'
           'falkland|berlin|bennet|mosasaur|martelly|huhne|glen cushing|wales|trung hoa|của anh').split('|')
NUMERIC = ['một năm', 'ba phần', 'một nửa', 'thứ bảy', 'hai lối', 'thứ nhất', 'hai thế kỷ', 'hai ý tưởng']


def manual_flags(raw, reference):
    flags=automatic_flags(raw+' '+reference)
    text=(raw+' '+reference).lower()
    for reason, spans in [('foreign_entity_or_term', FOREIGN), ('explicit_numeric_construct', NUMERIC)]:
        for span in spans:
            if re.search(r'(?<!\w)'+re.escape(span)+r'(?!\w)', text):
                flags.append(dict(reason=reason, span=span))
    return flags


def unchanged(policy):
    o.verify(policy)
    if any(b.file_hash(b.ROOT/p)!=sha for p,sha in policy['original_sha256'].items()):
        raise ValueError('original changed')
    if b.file_hash(Path(m.__file__))!=policy['normalizer_sha256']:
        raise ValueError('normalizer changed')


def screen():
    policy=b.load(OUT/'policy.json'); unchanged(policy)
    if (OUT/'selection.json').exists():
        raise ValueError('already screened')
    pool=b.load(OUT/'full-pool.json'); old=b.load(OLD); hold=b.load(HOLD)
    logs=[]; eligible=[]
    for r in pool:
        flags=manual_flags(r['raw'],r['reference'])
        if not 15<=r['duration']<30:
            flags.append(dict(reason='duration', span=str(r['duration'])))
        logs.append(dict(r, flags=flags, content_eligible=not flags))
        if not flags:
            eligible.append(r)
    # Overlay supplies physical TSV identity for historical viewer ordinals.
    overlay=b.load(SOURCE/'current60-source-reference-v2.json')
    previous=hr.v2_rows(old)
    oldlog=[]; retained=[]
    for r, prior, ov in zip(old,previous,overlay,strict=True):
        if prior!=ov['source_fields_previous']:
            raise ValueError('current overlay mismatch')
        ev=ov['source_correction_evidence']
        flags=manual_flags(ev['physical_raw_transcription'],ev['source_reference_v2'])
        item=copy.deepcopy(r)
        item.update(official_filename=ev['filename'],physical_tsv_index=ev['tsv_index'],
                    curation_role='retained-current60', reference_source_raw=ev['physical_raw_transcription'])
        if not flags:
            try:
                unique(retained+[item])
                retained.append(item)
            except ValueError:
                flags.append(dict(reason='duplicate_identity_or_text',span=r['id']))
        oldlog.append(dict(id=r['id'],split=r['split'],flags=flags,retained=not flags,raw=ev['physical_raw_transcription']))
    selected=list(retained)
    exclusions=[]
    for r in sorted(eligible,key=lambda x:(x['duration']<20,x['index'],x['split'])):
        reasons=[]
        if any(str(x['original_id'])==r['native_id'] or b.normalize(x.get('source_reference_v2',x.get('reference_raw',x['reference'])))==b.normalize(r['reference']) for x in hold):
            reasons.append('fresh_holdout_native_or_text')
        if any(x['split']==r['split'] and (str(x['original_id'])==r['native_id'] or b.normalize(x.get('reference_raw',x['reference']))==b.normalize(r['reference'])) for x in selected):
            reasons.append('already_selected_native_or_text')
        if sum(x['split']==r['split'] for x in selected)>=30:
            reasons.append('quota')
        if reasons:
            exclusions.append(dict(key=r['key'],reasons=reasons)); continue
        # New namespace avoids collisions with historical viewer ordinal IDs.
        selected.append(dict(source='google/fleurs', id=f"{r['split']}:{10000+r['index']}",
            original_id=int(r['native_id']), original_split=r['split'],split=r['split'],
            source_revision=old[0]['source_revision'],source_url=f"https://huggingface.co/datasets/google/fleurs/resolve/{old[0]['source_revision']}/data/vi_vn/",
            license='cc-by-4.0',reference=r['reference'],reference_raw=r['reference'],reference_source_raw=r['raw'],
            duration=r['duration'],official_filename=r['filename'],physical_tsv_index=r['index'],
            selection='reference-only manual screened;20+ tier then15+;physical index',curation_role='replacement',
            transcript_audit='pending independent listening',previous_usage='not audited against superseded pools; may have appeared previously'))
    b.save(OUT/'screening.json',o.seal(dict(created_unix=time.time(),policy_sha256=b.file_hash(OUT/'policy.json'),
        full_pool_sha256=b.file_hash(OUT/'full-pool.json'),rules=dict(foreign=FOREIGN,numeric=NUMERIC),pool=logs,current60=oldlog,
        exclusions=exclusions,manual_scope='175 automatic-clear duration-eligible recordings reviewed from raw source before any inference; remaining recordings have automatic/duration exclusions')))
    b.save(OUT/'selection.json',o.seal(dict(created_unix=time.time(),screening_sha256=b.file_hash(OUT/'screening.json'),
        rows=selected,counts={s:sum(r['split']==s for r in selected) for s in ('validation','test')},
        retained=len(retained),target=60,complete=len(selected)==60,
        shortage='Full pinned TSV15–30s eligible unique content pool, fresh30 excluded; no duration/content relaxation')))
    print(json.dumps({k:v for k,v in b.load(OUT/'selection.json').items() if k!='rows'},indent=2))


def acquire():
    policy=b.load(OUT/'policy.json'); unchanged(policy)
    selection=b.load(OUT/'selection.json'); o.verify(selection)
    screening=b.load(OUT/'screening.json'); o.verify(screening)
    if selection['screening_sha256']!=b.file_hash(OUT/'screening.json'):
        raise ValueError('screen seal mismatch')
    if (DATA/'manifest.json').exists():
        raise ValueError('already acquired')
    rows=selection['rows']; network=[]
    for split,native in [('validation','dev'),('test','test')]:
        targets={r['official_filename']:r for r in rows if r['split']==split and r['curation_role']=='replacement'}
        if not targets:
            continue
        limit=1024**3-sum(x['bytes'] for x in network)
        http=b.BudgetHTTP(limit,360)
        url=next(iter(targets.values()))['source_url']+'audio/'+native+'.tar.gz'
        class Limited:
            def read(self,size=-1):
                if time.monotonic()>=http.deadline or http.remaining<=0:
                    raise RuntimeError('bounded archive budget exhausted')
                data=response.read(min(65536,http.remaining,size if size>=0 else 65536))
                http.remaining-=len(data)
                return data
        try:
            with b.safe_urlopen(urllib.request.Request(url),timeout=20) as response:
                with tarfile.open(fileobj=Limited(),mode='r|gz') as archive:
                    for member in archive:
                        name=Path(member.name).name
                        if name not in targets or not member.isfile():
                            continue
                        if member.size>4000000:
                            raise ValueError('oversized audio')
                        r=targets.pop(name); data=archive.extractfile(member).read()
                        local=DATA/(split+'_'+name); local.write_bytes(data)
                        duration,pcm=b.pcm_info(local)
                        if abs(duration-r['duration'])>1/16000:
                            raise ValueError('official sample count mismatch')
                        r.update(audio_path=str(local),sha256=b.digest(data),pcm_sha256=pcm,archive_member=member.name)
                        if not targets:
                            break
        except Exception as exc:
            b.save(OUT/'acquisition-failure.json',dict(type=type(exc).__name__,split=split,remaining=list(targets),bytes=limit-http.remaining))
            raise RuntimeError('bounded acquisition failed; no inference; details sanitized') from None
        network.append(dict(split=split,bytes=limit-http.remaining,seconds=time.monotonic()-http.started,max_seconds=360))
        b.save(OUT/'network.json',network)
        if targets:
            raise RuntimeError('archive ended before selected targets')
    unique(rows)
    hold=b.load(HOLD)
    if any(any(r['sha256']==x['sha256'] or r['pcm_sha256']==x['pcm_sha256'] for x in hold) for r in rows):
        raise ValueError('holdout audio overlap')
    validation=b.validate(rows)
    b.save(DATA/'manifest.json',rows)
    for split in ('validation','test'):
        b.save(DATA/f'splits/{split}.json',[r for r in rows if r['split']==split])
    unchanged(policy)
    b.save(OUT/'acquired-seal.json',o.seal(dict(created_unix=time.time(),manifest_sha256=b.file_hash(DATA/'manifest.json'),
        selection_sha256=b.file_hash(OUT/'selection.json'),policy_sha256=b.file_hash(OUT/'policy.json'),
        validation=validation,network=network,complete=len(rows)==60,code_sha256=b.file_hash(Path(__file__)))))
    print(json.dumps(dict(validation=validation,network=network),indent=2))


def review_package(rows, records):
    root=b.ROOT/'outputs/review-packages/fleurs-general60-feasible23'
    if root.exists() or root.with_suffix('.zip').exists():
        raise ValueError('existing review package')
    root.mkdir(parents=True)
    index=[]; entries=[]
    mapping={r['id']:r for r in records}
    for row in rows:
        folder=root/hr.safe_id(row['id']); folder.mkdir()
        shutil.copyfile(row['audio_path'],folder/'audio.wav')
        if b.file_hash(folder/'audio.wav')!=row['sha256']:
            raise ValueError('copy hash')
        record=mapping[row['id']]; pick=o.choose(record['candidates'],.3,-.02)
        texts=dict(source_raw=row['reference_source_raw'],source_transcription=row['reference_raw'],
                   current_reference=row['reference'],untrusted_gpt2=pick['text'])
        for name,text in texts.items():
            (folder/(name+'.txt')).write_text(text+'\n')
        edits=m.edit_spans(*m.normalized_pair(row['reference'],pick['text']))
        b.save(folder/'alignment.json',edits)
        b.save(folder/'decision.json',dict(decision='pending',reviewer='',notes='',accent='unknown; no text-based inference'))
        page='<!doctype html><meta charset="utf-8"><h1>'+html.escape(row['id'])+'</h1><audio controls src="audio.wav"></audio><p>All listening pending. GPT2 is untrusted. No accent assumptions.</p>'
        page+=''.join('<h2>'+html.escape(name)+'</h2><p>'+html.escape(text)+'</p>' for name,text in texts.items())
        page+='<h2>Normalized word edits (not causal diagnoses)</h2>'+''.join('<p><mark>'+html.escape(e['type'])+'</mark> REF '+html.escape(e['reference'])+' / ASR '+html.escape(e['hypothesis'])+'</p>' for e in edits)
        (folder/'review.html').write_text(page)
        index.append('<li><a href="'+folder.name+'/review.html">'+html.escape(row['id'])+'</a><audio controls src="'+folder.name+'/audio.wav"></audio></li>')
        entries.append(dict(id=row['id'],audio=folder.name+'/audio.wav',sha256=row['sha256'],candidate_id=pick['id'],role=row['curation_role']))
    (root/'review.html').write_text('<!doctype html><meta charset="utf-8"><h1>General-content feasible23 / requested60 incomplete</h1><ul>'+''.join(index)+'</ul>')
    (root/'README.md').write_text('# FLEURS general-content feasible23\n\nOpen review.html; copied WAVs and relative links work offline. Requested60 is incomplete (12 validation/11 test). Source raw/current labels and untrusted GPT2 are separate. Whole-sentence and accent listening pending. No edits automatically applied. License cc-by-4.0, google/fleurs revision '+rows[0]['source_revision']+'.\n')
    b.save(root/'manifest.json',dict(records=entries,manifest_sha256=b.file_hash(DATA/'manifest.json')))
    files=sorted(p for p in root.rglob('*') if p.is_file())
    for p in files:
        if p.suffix=='.html':
            for target in re.findall(r'(?:href|src)="([^"]+)"',p.read_text()):
                if ':' in target or target.startswith('/') or '..' in Path(target).parts or not (p.parent/target).is_file():
                    raise ValueError('nonportable link')
    hashes={p.relative_to(root).as_posix():b.file_hash(p) for p in files}
    b.save(root/'SHA256.json',hashes); files.append(root/'SHA256.json')
    archive=root.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or set(z.namelist())!={p.relative_to(root).as_posix() for p in files}:
            raise ValueError('ZIP integrity')
        for name,sha in hashes.items():
            if b.digest(z.read(name))!=sha:
                raise ValueError('ZIP hash')
    return dict(path=str(root.relative_to(b.ROOT)),zip=str(archive.relative_to(b.ROOT)),bytes=archive.stat().st_size,samples=len(rows),entries=len(files),verified=True)


def benchmark():
    started=time.monotonic()
    policy=b.load(OUT/'policy.json'); unchanged(policy)
    seal=b.load(OUT/'acquired-seal.json'); o.verify(seal)
    selection=b.load(OUT/'selection.json'); o.verify(selection)
    screening=b.load(OUT/'screening.json'); o.verify(screening)
    bindings=[('manifest_sha256',DATA/'manifest.json'),('selection_sha256',OUT/'selection.json'),('policy_sha256',OUT/'policy.json')]
    if any(seal[k]!=b.file_hash(p) for k,p in bindings) or selection['screening_sha256']!=b.file_hash(OUT/'screening.json'):
        raise ValueError('preinference bindings')
    if (OUT/'inference-freeze.json').exists():
        raise ValueError('refusing repeated inference')
    freeze=policy['freeze']; o.verify(freeze)
    rows=b.load(DATA/'manifest.json'); unique(rows)
    validations=dict(new=b.validate(rows),original=b.validate(b.load(OLD)))
    for sp in ('validation','test'):
        if b.load(DATA/f'splits/{sp}.json')!=[r for r in rows if r['split']==sp]:
            raise ValueError('partition mismatch')
    b.save(OUT/'inference-freeze.json',o.seal(dict(created_unix=time.time(),acquired_seal_sha256=b.file_hash(OUT/'acquired-seal.json'),
        freeze=freeze,code_sha256=b.file_hash(Path(__file__)),screening_sha256=b.file_hash(OUT/'screening.json'),
        policies=['A-source-transcription','D-symmetric-conservative-v1','current-preserved-reference'],no_tuning=True)))
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    timings={}; summaries={}; details=[]; all_records=[]
    for sp in ('validation','test'):
        manifest=DATA/f'splits/{sp}.json'; folder=OUT/sp
        for name,fn in [('decode',lambda:b.decode(NS(manifest=manifest,output=folder/'nbest.json',beam_size=15))),
                        ('score',lambda:b.score(NS(input=folder/'nbest.json',output=folder/'gpt2.json',lm='gpt2',revision=o.REV)))]:
            t=time.monotonic(); fn(); timings[sp+'_'+name]=time.monotonic()-t
            b.save(OUT/'timings.json',timings)
            if name=='decode' and b.load(folder/'nbest.json')['protocol']!=freeze['protocol']['acoustic']:
                raise ValueError('acoustic freeze mismatch')
        scored,_=o.checked(manifest,folder/'nbest.json',folder/'gpt2.json')
        o.verify_test(freeze,scored,b.file_hash(manifest))
        nbest=b.load(folder/'nbest.json')
        for nr,r in zip(nbest['records'],scored['records'],strict=True):
            if nr['greedy']!=r['greedy']:
                raise ValueError('greedy mismatch')
            for i,c in enumerate(r['candidates']):
                if c['id']!=b.digest((r['source']+r['id']+str(i)+json.dumps(c['tokens'])).encode()):
                    raise ValueError('candidate fingerprint')
        items=[r for r in rows if r['split']==sp]
        raw=[dict(r,reference=r['reference_raw']) for r in items]
        summary,ds=h.evaluate(raw,scored['records'],.3,-.02)
        current,cd=h.evaluate(items,scored['records'],.3,-.02)
        summary['current']=current['A']
        summaries[sp]=summary
        details += [dict(r,split=sp) for r in ds]+[dict(r,split=sp,convention='current') for r in cd if r['convention']=='A']
        all_records+=scored['records']
    # Same frozen historical hypotheses; exact raw historical manifests for joins.
    old=b.load(OLD); oldraw=m.original_rows(hr.v2_rows(old)); originals={}
    for sp in ('validation','test'):
        manifest=OUT/'original-raw-manifests'/(sp+'.json')
        subset=[r for r in oldraw if r['split']==sp]; b.save(manifest,subset)
        folder=o.OUT/f'beam15-{sp}'
        scores,_=o.checked(manifest,folder/'nbest.json',folder/'gpt2.json')
        if scores['protocol']!=freeze['protocol']:
            raise ValueError('original comparison protocol')
        originals[sp],_=h.evaluate(subset,scores['records'],.3,-.02)
    package=review_package(rows,all_records)
    unchanged(policy)
    result=o.seal(dict(summary=summaries,records=details,original_same_policy=originals,
        durations=dict(curated=b.duration_summary(rows),original=b.duration_summary(old)),
        timings=timings,validation=validations,package=package,network=b.load(OUT/'network.json'),
        selection_counts=selection['counts'],retained=selection['retained'],complete=False,target=60,
        independent_audit='jiwer word/character totals and denominators checked for every corpus method/policy; S/D/I alternative alignments retained',
        candidate_diversity={sp:o.diagnostic(b.load(OUT/sp/'gpt2.json'),{(r['source'],r['id']):r for r in rows},.3,-.02)['diversity'] for sp in ('validation','test')},
        input_sha256={str(p.relative_to(b.ROOT)):b.file_hash(p) for p in [DATA/'manifest.json',OUT/'inference-freeze.json',OUT/'selection.json',OUT/'screening.json']},
        seconds=time.monotonic()-started))
    b.save(OUT/'results.json',result)
    report(result,screening)
    print(json.dumps(dict(summary=summaries,timings=timings,package=package,seconds=result['seconds']),indent=2))


def report(result,screening):
    lines=['# FLEURS general-content — feasible23 frozen benchmark; requested60 INCOMPLETE','',
        '**Not a completed native30/30 general60.** Sealed policy yields12 validation/11 test,23 distinct texts at15–30s after excluding fresh30. No silent relaxation, duplicated texts, split reassignment or post-inference replacement. Original datasets remain active and byte-identical.',
        '', '## Reference-only policy and screening', '',
        'Policy persisted before screening; selection and source eligibility persisted before acquisition and decoding. Full pinned TSV:1218 recordings;278 at15–30s;175 automatic-clear records manually read. Exclude explicit digits/spoken numeric constructs and foreign proper entities/acronyms/transliterated terms; indefinite một and generic loanwords such as protein/blog allowed. No capitalization-only NER. Exact matched spans/reasons for every pool/current60 row are in screening.json. No hypotheses were inputs to screening. Accent unknown; no accent guesses or restrictions.',
        'Requested30/30 is blocked under these conservative constraints. Broader10–15s sampling was not silently activated. These feasible counts are policy-specific, not an assertion about every imaginable eligibility policy or global maximum. A future wider-duration dataset requires its own pre-inference selection.',
        '',f"Retained11/23 ({100*11/23:.2f}%) from current60;12 replacement recordings. Not fresh/independent. Superseded-pool usage of replacements was not exhaustively audited and is explicitly unknown. All fresh30 native/text/audio/PCM excluded. No speaker/training independence claim.",
        '', '| Original ID | Decision | Reference-only reason/spans |','|---|---|---|']
    for r in screening['current60']:
        lines.append('| '+r['id']+' | '+('retained' if r['retained'] else 'excluded from supplement')+' | '+('; '.join(f"{f['reason']}: {f['span']}" for f in r['flags']) or 'ordinary content')+' |')
    lines += ['', '## Frozen results', '', 'Beam15, alpha0.3/beta−0.02, GPT2 revision `'+o.REV+'`, invariant v3 protocol exactly matched to the existing winner. No tuning, downloads of models, training or new dependencies. All selected23 decoded once and scored once. All errors remain in the dataset. A=source transcription; D=unchanged symmetric conservative numeric rules on refs/hyps; current=preserved active labels for retained rows and official source for new rows. No new label rewrites. CER includes normalized spaces.',
        '', '| Split | Policy | Method | Words | Chars | S | D | I | Errors | WER % | CER % |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for sp,policies in result['summary'].items():
        for conv,methods in policies.items():
            for method,x in methods.items():
                if method!='comparisons_vs_beam15':
                    lines.append(f"| {sp} | {conv} | {method} | {x['words']} | {x['characters']} | {x['S']} | {x['D']} | {x['I']} | {x['errors']} | {100*x['wer']:.4f} | {100*x['cer']:.4f} |")
            lines.append(f"\n{sp}/{conv} win/tie/loss versus beam15: `{json.dumps(methods['comparisons_vs_beam15'])}`.")
    lines += ['', '## Original60 comparison — SAME raw/symmetric policy', '', '| Split | Policy | Original GPT2 WER % | Curated GPT2 WER % |','|---|---|---:|---:|']
    for sp in ('validation','test'):
        for conv in ('A','D'):
            lines.append(f"| {sp} | {conv} | {100*result['original_same_policy'][sp][conv]['gpt2_frozen']['wer']:.4f} | {100*result['summary'][sp][conv]['gpt2_frozen']['wer']:.4f} |")
    lines += ['', 'This is different content/duration/denominator sampling, NOT model improvement or a generalization estimate. Original raw references retain historical label differences; current human-v3 is separate, not compared to raw as if identical. Human corrections are span-limited; whole-sentence listening remains pending. Word oracle is diagnostic whole-candidate minimum word errors, decoder-order ties; its CER is not a character oracle.',
        '', '## Audits, timings and artifacts', '', 'Independent jiwer checks word/character totals and denominators for every corpus method/policy, original and curated. Production deterministic S/D/I and independent alternative backtraces are both retained. Full real WAV byte/decoded-PCM/duration validation passed for original60 and curated23. Strict candidate IDs/source caches/protocols/fingerprints checked. Copied WAV hashes, portable HTML links, ZIP exact member set, CRC and SHA inventory verified.',
        '', '```json',json.dumps({k:result[k] for k in ('durations','timings','network','candidate_diversity','package','seconds')},ensure_ascii=False,indent=2),'```',
        '', 'Artifacts: `data/benchmark/fleurs-general60/manifest.json` and native split files (directory retains requested target name, but manifest has23). `outputs/benchmarks/long/fleurs-general60-20261104/`: policy.json, full-pool.json, screening.json, selection.json, acquired-seal.json, inference-freeze.json, split N-best/GPT2, results.json, timings.json. Review package includes every23 copied WAV plus raw/current/GPT2 text, highlights and pending decisions. Open review.html from the ZIP; no file:// or remote dependencies.',
        '', 'Reproduce stages separately with `.venv-benchmark/bin/python tools/curate_fleurs_general.py inventory|screen|acquire|benchmark` (one stage argument, existing sealed outputs refused). Offline benchmark invocation: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python tools/curate_fleurs_general.py benchmark`.',
        '', 'Scope caveats: native30/30 target NOT met; conservative lexical screening is not exhaustive NER or human audio certification; very small selected read-speech subset; overlap and difficulty bias; unknown speaker/training overlap; no accent verification. No commitments/pushes. Full suites skipped; exact isolated smoke and required real artifact checks only.']
    (b.ROOT/'docs/fleurs-general60-benchmark.md').write_text('\n'.join(lines)+'\n')


def verify():
    """Publication audit only; never repeats decoding/scoring or selection."""
    started=time.monotonic()
    policy=b.load(OUT/'policy.json'); unchanged(policy)
    result=b.load(OUT/'results.json'); o.verify(result)
    for p,sha in result['input_sha256'].items():
        if b.file_hash(b.ROOT/p)!=sha:
            raise ValueError('result input binding')
    screening=b.load(OUT/'screening.json'); o.verify(screening)
    selection=b.load(OUT/'selection.json'); o.verify(selection)
    acquired=b.load(OUT/'acquired-seal.json'); o.verify(acquired)
    inference=b.load(OUT/'inference-freeze.json'); o.verify(inference)
    if (screening['policy_sha256']!=b.file_hash(OUT/'policy.json') or
        screening['full_pool_sha256']!=b.file_hash(OUT/'full-pool.json') or
        selection['screening_sha256']!=b.file_hash(OUT/'screening.json') or
        inference['acquired_seal_sha256']!=b.file_hash(OUT/'acquired-seal.json') or
        acquired['manifest_sha256']!=b.file_hash(DATA/'manifest.json')):
        raise ValueError('seal chain')
    if not policy['created_unix']<screening['created_unix']<=selection['created_unix']<acquired['created_unix']<inference['created_unix']:
        raise ValueError('seal order')
    overlay=b.load(SOURCE/'source-reference-v2-overlay.json'); o.verify(overlay)
    pool=b.load(OUT/'full-pool.json'); expected=[]
    for sp,native in [('validation','dev'),('test','test')]:
        path=SOURCE/f'official-{native}.tsv'
        if b.file_hash(path)!=overlay['metadata_sha256'][sp]:
            raise ValueError('pinned metadata hash')
        for i,s in enumerate(b.parse_fleurs_tsv(path.read_text())):
            expected.append(dict(key=f'{sp}:{i}',split=sp,index=i,native_id=s[0],filename=s[1],raw=s[2],reference=s[3],
                duration=int(s[5])/16000,metadata_sha256=b.file_hash(path),automatic=automatic_flags(s[2]+' '+s[3])))
    if pool!=expected or len(screening['pool'])!=len(pool):
        raise ValueError('full immutable pool reconstruction')
    for row,logged in zip(pool,screening['pool'],strict=True):
        flags=manual_flags(row['raw'],row['reference'])
        if not 15<=row['duration']<30:
            flags.append(dict(reason='duration',span=str(row['duration'])))
        if logged!=dict(row,flags=flags,content_eligible=not flags):
            raise ValueError('screening reconstruction')
    rows=b.load(DATA/'manifest.json'); unique(rows)
    for sp in ('validation','test'):
        manifest=DATA/f'splits/{sp}.json'
        scores,_=o.checked(manifest,OUT/sp/'nbest.json',OUT/sp/'gpt2.json')
        o.verify_test(policy['freeze'],scores,b.file_hash(manifest))
        items=[r for r in rows if r['split']==sp]
        raw=[dict(r,reference=r['reference_raw']) for r in items]
        observed,ds=h.evaluate(raw,scores['records'],.3,-.02)
        current,cd=h.evaluate(items,scores['records'],.3,-.02)
        observed['current']=current['A']
        if observed!=result['summary'][sp]:
            raise ValueError('published metric reconstruction')
    backups={stamp:b.load(OLD.parent/'replacement-backups'/stamp/'manifest.json') for stamp in h.BACKUPS}
    usage=[]
    for r in rows:
        matches=[dict(backup=stamp,id=x['id'],same_audio=x['sha256']==r['sha256'],same_pcm=x['pcm_sha256']==r['pcm_sha256'])
                 for stamp,oldrows in backups.items() for x in oldrows
                 if x['sha256']==r['sha256'] or x['pcm_sha256']==r['pcm_sha256'] or
                    (x['split']==r['split'] and str(x['original_id'])==str(r['original_id']))]
        usage.append(dict(id=r['id'],role=r['curation_role'],known_backup_matches=matches))
    package=b.ROOT/result['package']['path']; hashes=b.load(package/'SHA256.json')
    with zipfile.ZipFile(b.ROOT/result['package']['zip']) as z:
        if z.testzip() or set(z.namelist())!=set(hashes)|{'SHA256.json'}:
            raise ValueError('published zip')
        for name,sha in hashes.items():
            if b.file_hash(package/name)!=sha or b.digest(z.read(name))!=sha:
                raise ValueError('published package hashes')
    unchanged(policy)
    evidence=o.seal(dict(passed=True,seconds=time.monotonic()-started,pool_rows=len(pool),
        seal_order=True,pinned_metadata=True,originals_unchanged=True,metrics_reconstructed=True,zip_verified=True,
        known_superseded_usage=usage,manifest_sha256=b.file_hash(DATA/'manifest.json'),
        current_code_sha256=b.file_hash(Path(__file__)),execution_code_sha256=inference['code_sha256']))
    b.save(OUT/'postpublication-verification.json',evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k!='known_superseded_usage'},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inventory','screen','acquire','benchmark','verify'])
    args=parser.parse_args()
    globals()[args.stage]()
