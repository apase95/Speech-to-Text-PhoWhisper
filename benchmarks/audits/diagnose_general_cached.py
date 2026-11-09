"""Targeted general23 cached deletion/screening audit. No model or network calls."""
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.data import curate_fleurs_general as c

b,o,m=c.b,c.o,c.m


def outcome(errors,baseline):
    return 'win' if errors<baseline else 'loss' if errors>baseline else 'tie'


def contains(text,phrase):
    return (' '+phrase+' ') in (' '+text+' ')


def sample(reference,greedy,candidates,alpha,beta):
    ref=b.normalize(reference); greedy=b.normalize(greedy)
    pick=o.choose(candidates,alpha,beta)
    selected=b.normalize(pick['text'])
    hyps=dict(gpt2=selected,greedy=greedy,beam=b.normalize(candidates[0]['text']))
    metrics={k:m.measure([ref],[text]) for k,text in hyps.items()}
    spans=[]
    # Independent alignment describes a deletion; tied alignments are not causal.
    for edit in m.edit_spans(ref,selected):
        if edit['type']=='delete':
            phrase=edit['reference']
            spans.append(dict(phrase=phrase,greedy_contains_phrase=contains(greedy,phrase),
                candidate_indices_containing_phrase=[i for i,x in enumerate(candidates) if contains(b.normalize(x['text']),phrase)]))
    evidence=[]
    for i,x in enumerate(candidates):
        text=b.normalize(x['text']); metric=m.measure([ref],[text])
        evidence.append(dict(index=i,id=x['id'],text=text,words=len(text.split()),
            avg_logprob=x['avg_logprob'],lm=x['lm'],length_term=beta*len(text.split()),
            score=x['avg_logprob']+alpha*x['lm']+beta*len(text.split()),
            metrics=metric,selected=x['id']==pick['id']))
    return dict(reference=ref,hypotheses=hyps,selected_id=pick['id'],metrics=metrics,
        deletion_spans=spans,candidates=evidence,greedy_exact_candidate_indices=[i for i,x in enumerate(evidence) if x['text']==greedy])


def reason_counts(screening):
    result={}
    order=['digits','spoken_numeric','foreign_entity_or_term','explicit_numeric_construct']
    for sp in ('validation','test'):
        pool=[r for r in screening['pool'] if r['split']==sp]
        duration=[r for r in pool if not any(f['reason']=='duration' for f in r['flags'])]
        flags=Counter(); spans=Counter(); waterfall=Counter()
        for r in duration:
            reasons={f['reason'] for f in r['flags']}
            flags.update(reasons)
            spans.update({f['span'] for f in r['flags'] if f['reason'] in ('foreign_entity_or_term','explicit_numeric_construct')})
            waterfall[next((x for x in order if x in reasons),'content_clear')]+=1
        exclusions=[x for x in screening['exclusions'] if x['key'].startswith(sp+':')]
        selection=Counter(reason for x in exclusions for reason in set(x['reasons']))
        result[sp]=dict(pool=len(pool),under15=sum(r['duration']<15 for r in pool),
            at_least30=sum(r['duration']>=30 for r in pool),duration_eligible=len(duration),
            reason_record_counts_overlapping=dict(flags),exclusive_waterfall=dict(waterfall),
            content_clear=sum(r['content_eligible'] for r in duration),
            selection_skip_records=len(exclusions),selection_skip_reasons_overlapping=dict(selection),
            lexical_span_record_counts=dict(spans),
            content_clear_unique_native_ids=len({r['native_id'] for r in duration if r['content_eligible']}))
    current={}
    for sp in ('validation','test'):
        rows=[r for r in screening['current60'] if r['split']==sp]
        current[sp]=dict(retained=sum(r['retained'] for r in rows),excluded=sum(not r['retained'] for r in rows),
            reasons_overlapping=dict(Counter(reason for r in rows for reason in {f['reason'] for f in r['flags']})))
    return dict(pool=result,current60=current)


def main():
    started=time.monotonic()
    if sys.argv[1:]==['--summary']:
        audit=b.load(c.OUT/'cached-deletion-diagnosis.json'); o.verify(audit)
        print(json.dumps(dict(samples=[dict(id=r['id'],duration=r['duration'],
            D={k:x['D'] for k,x in r['metrics'].items()},errors={k:x['errors'] for k,x in r['metrics'].items()},
            **({k:r[k] for k in ('reference','hypotheses','deletion_spans','greedy_exact_candidate_indices')} if r['excess_D']>0 else {})) for r in audit['samples']],
            seconds=audit['seconds']),ensure_ascii=False,indent=2))
        for row in audit['samples']:
            if row['excess_D']>0:
                phrase='thuộc nhân dân liên quan đến'
                print(json.dumps(dict(id=row['id'],phrase=phrase,
                    reference_contains=contains(row['reference'],phrase),
                    greedy_contains=contains(row['hypotheses']['greedy'],phrase),
                    candidate_indices=[x['index'] for x in row['candidates'] if contains(x['text'],phrase)],
                    candidate_D=[dict(index=x['index'],D=x['metrics']['D'],words=x['words'],score=x['score']) for x in row['candidates']]),ensure_ascii=False,indent=2))
        return
    paths=[c.DATA/'manifest.json',c.DATA/'splits/validation.json',c.DATA/'splits/test.json',
        c.OUT/'policy.json',c.OUT/'screening.json',c.OUT/'selection.json',c.OUT/'results.json',
        c.OUT/'inference-freeze.json',c.OUT/'test/nbest.json',c.OUT/'test/gpt2.json']
    before={str(p.relative_to(b.ROOT)):b.file_hash(p) for p in paths}
    policy=b.load(c.OUT/'policy.json'); c.unchanged(policy)
    result=b.load(c.OUT/'results.json'); o.verify(result)
    screening=b.load(c.OUT/'screening.json'); o.verify(screening)
    for p,sha in result['input_sha256'].items():
        if b.file_hash(b.ROOT/p)!=sha:
            raise ValueError('result binding')
    freeze=policy['freeze']; o.verify(freeze)
    rows=b.load(c.DATA/'manifest.json'); c.unique(rows)
    scores,_=o.checked(c.DATA/'splits/test.json',c.OUT/'test/nbest.json',c.OUT/'test/gpt2.json')
    o.verify_test(freeze,scores,b.file_hash(c.DATA/'splits/test.json'))
    mapping={r['id']:r for r in rows}; samples=[]
    for record in scores['records']:
        row=mapping[record['id']]
        item=sample(row['reference_raw'],record['greedy'],record['candidates'],freeze['alpha'],freeze['beta'])
        samples.append(dict(item,id=row['id'],duration=row['duration'],physical_tsv_index=row['physical_tsv_index'],
            native_id=row['original_id'],excess_D=item['metrics']['gpt2']['D']-item['metrics']['greedy']['D']))
    comparisons={base:dict(Counter(outcome(r['metrics']['gpt2']['errors'],r['metrics'][base]['errors']) for r in samples)) for base in ('beam','greedy')}
    for value in comparisons.values():
        for key in ('win','tie','loss'):
            value.setdefault(key,0)
    for method,key in [('gpt2','gpt2_frozen'),('greedy','greedy'),('beam','beam15_top1')]:
        for count in ('S','D','I','errors'):
            if sum(r['metrics'][method][count] for r in samples)!=result['summary']['test']['A'][key][count]:
                raise ValueError('saved sample metric reconstruction')
    durations={sp:dict(samples=len([r for r in rows if r['split']==sp]),
        min=min(r['duration'] for r in rows if r['split']==sp),max=max(r['duration'] for r in rows if r['split']==sp),
        mean=sum(r['duration'] for r in rows if r['split']==sp)/sum(r['split']==sp for r in rows),
        words=sum(len(b.normalize(r['reference_raw']).split()) for r in rows if r['split']==sp)) for sp in ('validation','test')}
    c.unchanged(policy)
    if any(b.file_hash(b.ROOT/p)!=sha for p,sha in before.items()):
        raise ValueError('cached inputs changed')
    audit=o.seal(dict(samples=samples,comparisons=comparisons,durations=durations,screening=reason_counts(screening),
        inputs_sha256=before,seconds=time.monotonic()-started,unchanged=True))
    target=c.OUT/'cached-deletion-diagnosis.json'
    if target.exists():
        raise ValueError('existing diagnosis; refusing overwrite')
    b.save(target,audit)
    print(json.dumps(dict(comparisons=comparisons,durations=durations,screening=audit['screening'],
        samples=[{k:v for k,v in r.items() if k!='candidates'} for r in samples],
        candidate_evidence=[dict(id=r['id'],candidates=r['candidates']) for r in samples if r['excess_D']>0],
        seconds=audit['seconds']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
