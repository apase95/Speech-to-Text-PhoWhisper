"""Bounded official fresh30 acquisition and once-only frozen beam15 evaluation."""
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.runners import long_audio_benchmark as b
from benchmarks.runners import gpt2_optimization as o
from benchmarks.audits import reevaluate_numeric_metrics as m

DATA=b.ROOT/'data/benchmark/fleurs-holdout30'
OUT=b.ROOT/'outputs/benchmarks/long/fleurs-holdout30-20261104'
BACKUPS=('1793746871837140826','1793746959492519032','1793747082868629642')


def disjoint(row, others):
    for old in others:
        if row['source']==old['source'] and (row['id']==old['id'] or str(row['original_id'])==str(old['original_id'])):
            return False
        if any(row[k]==old[k] for k in ('sha256','pcm_sha256')):
            return False
        if b.normalize(row['reference']) in {b.normalize(old['reference']),b.normalize(old.get('reference_raw',old['reference']))}:
            return False
    return True


def evaluate(rows, records, alpha, beta):
    all_metrics, details = {}, []
    mapping={(r['source'],r['id']):r for r in rows}
    for convention in ('A','D'):
        refs, methods=[],{k:[] for k in ('greedy','beam15_top1','gpt2_frozen','oracle')}
        comparisons={k:dict(win=0,tie=0,loss=0) for k in ('greedy','gpt2_frozen')}
        for r in records:
            raw=mapping[r['source'],r['id']]['reference']
            ref=m.normalized_pair(raw,'',convention=='D')[0]
            pick=o.choose(r['candidates'],alpha,beta)
            cs=[dict(id=c['id'],text=m.normalized_pair('',c['text'],convention=='D')[1]) for c in r['candidates']]
            oracle=min(cs,key=lambda c:b.distance(ref.split(),c['text'].split()))
            hyps=dict(greedy=r['greedy'],beam15_top1=r['candidates'][0]['text'],gpt2_frozen=pick['text'])
            hyps={k:m.normalized_pair('',v,convention=='D')[1] for k,v in hyps.items()}
            hyps['oracle']=oracle['text']
            costs={k:b.distance(ref.split(),v.split()) for k,v in hyps.items()}
            for k in comparisons:
                delta=costs[k]-costs['beam15_top1']
                comparisons[k]['win' if delta<0 else 'loss' if delta>0 else 'tie']+=1
            refs.append(ref)
            for k,v in hyps.items():
                methods[k].append(v)
            details.append(dict(id=r['id'],convention=convention,reference=ref,hypotheses=hyps,
                                errors=costs,selected_candidate_id=pick['id'],oracle_candidate_id=oracle['id'],
                                unique_candidates=len({c['text'] for c in cs}),candidates=len(cs)))
        all_metrics[convention]={k:m.measure(refs,hyps) for k,hyps in methods.items()}
        all_metrics[convention]['comparisons_vs_beam15']=comparisons
    return all_metrics,details


def main():
    resume = sys.argv[1:] == ['--resume-acquisition']
    if (DATA.exists() or OUT.exists()) and not resume:
        raise ValueError('refusing existing holdout/output; no repeated decode')
    if resume and (OUT/'nbest.json').exists():
        raise ValueError('no repeated decode')
    started=time.monotonic()
    oldpath=b.ROOT/'data/benchmark/long-audio/manifest.json'
    paths=[oldpath]+[oldpath.parent/'replacement-backups'/stamp/'manifest.json' for stamp in BACKUPS]
    blocked=[r for p in paths for r in b.load(p)]
    old=b.load(oldpath)
    winner=b.load(o.OUT/'winner.json'); freeze=b.load(o.OUT/'beam15-freeze.json')
    o.verify(winner); o.verify(freeze)
    if winner['beam']!=15 or winner['freeze_sha256']!=b.file_hash(o.OUT/'beam15-freeze.json') or (freeze['alpha'],freeze['beta'],freeze['revision'])!=(.3,-.02,o.REV):
        raise ValueError('winner mismatch')
    DATA.mkdir(parents=True,exist_ok=resume); OUT.mkdir(parents=True,exist_ok=resume)
    sealed = o.seal(dict(winner=winner,freeze=freeze,
        freeze_source_sha256=b.file_hash(o.OUT/'beam15-freeze.json'),
        policies=['A-original-standard','D-symmetric-conservative-verbalize-v1'],
        normalizer_sha256=b.file_hash(Path(m.__file__)),
        source_sha256={str(p.relative_to(b.ROOT)):b.file_hash(p) for p in paths},
        persisted_before_acquisition_and_inference=True,created_unix=time.time()))
    if resume:
        persisted=b.load(OUT/'frozen-protocol.json'); o.verify(persisted)
        if any(persisted[k]!=sealed[k] for k in ('winner','freeze','freeze_source_sha256','policies','normalizer_sha256','source_sha256')):
            raise ValueError('resume seal mismatch')
    else:
        b.save(OUT/'frozen-protocol.json',sealed)
    class Pool(list):
        def append(self,row):
            if disjoint(row,blocked+list(self)):
                super().append(row)
    # Existing validation rows satisfy the helper's validation quota without
    # acquiring or selecting validation data. They are never holdout members.
    pool=Pool([r for r in old if r['split']=='validation'])
    if resume:
        for r in b.load(DATA/'partial-manifest.json'):
            pool.append(r)
    stage=DATA/'acquisition-staging'; stage.mkdir(exist_ok=resume)
    previous=b.load(OUT/'acquisition.json') if resume else None
    status={}; budget=1024*1024*1024-previous['bytes'] if resume else 256*1024*1024
    http=b.BudgetHTTP(budget,240)
    try:
        b.fleurs_archive(http,stage,pool,old[0]['source_revision'],30,status,
                         scan_limit=2000,seed=42,min_duration=15 if resume else 20,blocked=blocked)
    except Exception as exc:
        status['blocker']=type(exc).__name__  # never arbitrary URL-bearing text
    network=dict(bytes=budget-http.remaining,requests=http.requests,seconds=time.monotonic()-http.started,
                 max_bytes=budget,max_seconds=240,status=status)
    if previous:
        network['previous_attempt']=previous
        network['total_bytes']=network['bytes']+previous['bytes']
        network['expansion_reason']='256MiB prefix yielded only2 disjoint20+; resume with explicit15–20 fallback within1GiB total'
    fresh=[dict(r,role='fresh_holdout',selection='first eligible official archive order; initial20+, resumed15+ fallback; seed42 recorded, no shuffle or ASR selection') for r in pool if r['split']=='test']
    b.save(OUT/'acquisition.json',network)
    if len(fresh)!=30:
        b.save(DATA/'partial-manifest.json',fresh)
        raise RuntimeError(f'bounded acquisition incomplete: {len(fresh)}/30; no inference')
    if not all(disjoint(r,blocked+fresh[:i]) for i,r in enumerate(fresh)):
        raise ValueError('global fresh disjointness failed')
    manifest=DATA/'manifest.json'; b.save(manifest,fresh)
    # Existing60 contains repeated native texts within splits; retain/report,
    # but enforce strict fresh-vs-all-old and fresh-vs-fresh disjointness above.
    b.save(OUT/'validation.json',dict(current=b.validate(old),fresh=b.validate(fresh),
        combined_audio=b.validate(old+fresh),
        fresh_vs_old_and_superseded='no ID/native ID/audio SHA/PCM SHA/normalized-text overlap',
        old_normalized_unique=len({b.normalize(r.get('reference_raw',r['reference'])) for r in old}),
        old_samples=len(old),speaker_independence='not established',training_overlap='unknown'))
    os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
    timings={}
    for name,fn in [('decode',lambda:b.decode(NS(manifest=manifest,output=OUT/'nbest.json',beam_size=15))),
                    ('score',lambda:b.score(NS(input=OUT/'nbest.json',output=OUT/'gpt2.json',lm='gpt2',revision=o.REV)))]:
        t=time.monotonic(); fn(); timings[name]=time.monotonic()-t
        b.save(OUT/'timings.json',timings)
        if name=='decode' and b.load(OUT/'nbest.json')['protocol']!=freeze['protocol']['acoustic']:
            raise ValueError('acoustic freeze mismatch')
    scores,_=o.checked(manifest,OUT/'nbest.json',OUT/'gpt2.json')
    o.verify_test(freeze,scores,b.file_hash(manifest))
    summary,details=evaluate(fresh,scores['records'],freeze['alpha'],freeze['beta'])
    result=o.seal(dict(summary=summary,records=details,timings=timings,network=network,
                      manifest_sha256=b.file_hash(manifest),frozen_protocol_sha256=b.file_hash(OUT/'frozen-protocol.json'),
                      nbest_sha256=b.file_hash(OUT/'nbest.json'),scores_sha256=b.file_hash(OUT/'gpt2.json'),
                      seconds=time.monotonic()-started,human_confirmation='pending'))
    b.save(OUT/'results.json',result)
    lines=['\n## Fresh30 — actual frozen evaluation','',
        f"30 additional native official FLEURS TEST clips, 15–30 seconds, no concatenation; {sum(r['duration']>=20 for r in fresh)} at least20s, {sum(r['duration']<20 for r in fresh)} explicit15–20 fallback. First eligible archive order, seed42 recorded (no randomized-order claim). Strict disjoint ID/native ID/audio/PCM/raw normalized transcript against current60 and all three known replacement backups, and within fresh30. No speaker/training independence claim. No holdout tuning or manual reference edits. All listening pending.",
        '| Convention | Method | S | D | I | WER % | CER % |','|---|---|---:|---:|---:|---:|---:|']
    for conv,methods in summary.items():
        for name,x in methods.items():
            if name!='comparisons_vs_beam15':
                lines.append(f"| {conv} | {name} | {x['S']} | {x['D']} | {x['I']} | {100*x['wer']:.4f} | {100*x['cer']:.4f} |")
        lines.append(f"\n{conv} per-sample comparisons versus beam15: `{json.dumps(methods['comparisons_vs_beam15'])}`.")
    lines += ['',f'Network: `{json.dumps(network)}`. Stage seconds: `{json.dumps(timings)}`. Total {result["seconds"]:.3f}s.',
              'Word-oracle diagnostic uses convention-specific minimum whole-candidate word errors, decoder-order ties. CER is that word-oracle candidate, not a character oracle. Candidate diversity per sample and chosen IDs in results.json. Independent jiwer word/character totals checked. No new model downloads, training, PhoBERT or configuration search.',
              f'Artifacts: `{OUT.relative_to(b.ROOT)}`. Protocol sealed before acquisition/inference, manifest/cache hashes sealed after. Generic D verbalizer applies identically to raw refs and hypotheses. A and D remain separate provisional analyses; human transcript confirmation pending.']
    with (b.ROOT/'docs/gpt2-error-audit-holdout.md').open('a') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(summary=summary,timings=timings,network=network,seconds=result['seconds']),indent=2))


if __name__=='__main__':
    main()
