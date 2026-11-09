"""Cached-only local alignment diagnostic and explicitly pending listening queue."""
import argparse
import csv
import html
import json
from pathlib import Path
import re
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.runners import long_audio_benchmark as b
from benchmarks.runners import gpt2_optimization as o
from tools import normalize_benchmark_references as n
from benchmarks.audits import reevaluate_numeric_metrics as m


def alignment(ref, hyp):
    """Diagonal/deletion/insertion tie break; all optimal mappings per ref index.

    Optimal-DAG traversal exposes position ambiguity without enumerating paths.
    Options contain (operation, hypothesis index), including deletions.
    """
    d = [[0]*(len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j
    for i, x in enumerate(ref, 1):
        for j, y in enumerate(hyp, 1):
            d[i][j] = min(d[i-1][j-1]+(x != y), d[i-1][j]+1, d[i][j-1]+1)
    def parents(i, j):
        edges = []
        if i and j and d[i][j] == d[i-1][j-1]+(ref[i-1] != hyp[j-1]):
            edges.append((i-1, j-1, 'E' if ref[i-1] == hyp[j-1] else 'S', i-1, j-1))
        if i and d[i][j] == d[i-1][j]+1:
            edges.append((i-1, j, 'D', i-1, None))
        if j and d[i][j] == d[i][j-1]+1:
            edges.append((i, j-1, 'I', None, j-1))
        return edges
    path, i, j = [], len(ref), len(hyp)
    while i or j:
        i, j, op, ri, hi = parents(i, j)[0]
        path.append((op, ri, hi))
    options = {i:set() for i in range(len(ref))}
    stack, seen = [(len(ref), len(hyp))], set()
    while stack:
        point = stack.pop()
        if point in seen:
            continue
        seen.add(point)
        for i, j, op, ri, hi in parents(*point):
            if ri is not None:
                options[ri].add((op, hi))
            stack.append((i, j))
    return list(reversed(path)), options


def local_substitutions(reference, candidates, selected_id):
    ref = reference.split()
    if len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('duplicate candidate IDs')
    aligned = {c['id']:alignment(ref, c['text'].split()) for c in candidates}
    selected = next(c for c in candidates if c['id'] == selected_id)
    rows = []
    for op, ri, hi in aligned[selected_id][0]:
        if op != 'S':
            continue
        correct = [c['id'] for c in candidates if any(x[0] == 'E' for x in aligned[c['id']][1][ri])]
        # Conservative: even differing equal-token mappings are marked ambiguous.
        ambiguous = any(len(aligned[c['id']][1][ri]) != 1 for c in candidates)
        category = ('alignment_ambiguous' if ambiguous else
                    'local_selection_opportunity' if correct else 'missing_from_aligned_set')
        rows.append(dict(reference_index=ri, reference_token=ref[ri],
                         selected_token=selected['text'].split()[hi],
                         context=' '.join(ref[max(0,ri-3):ri+4]),
                         classification=category, correct_candidate_ids=correct,
                         selected_candidate_id=selected_id))
    return rows


def highlighted(text):
    return re.sub(r'\d+(?:[.,]\d+)*', lambda x:'<mark>'+x[0]+'</mark>', html.escape(text))


def review_queue(output, active, raw, v1, fixed):
    fm = {r['id']:r for r in fixed['records']}
    queue, cards = [], []
    triples = sorted(zip(active,raw,v1), key=lambda t:(t[0]['id']!='test:266', t[0]['id']))
    for r, original, first in triples:
        if r['reference'] == original['reference']:
            continue
        row = dict(id=r['id'], source=r['source'], original_id=r['original_id'],
                   audio_path=r['audio_path'], audio_sha256=r['sha256'],
                   original=original['reference'], raw=original['reference'],
                   v1=first['reference'], v2=r['reference'],
                   untrusted_suggestions=fm[r['id']]['hypotheses'], decision='pending',
                   human_verified=False, reviewer='', notes='', override='',
                   questions='Confirm actual reading, omissions, dates/years, nghìn/ngàn and linh/lẻ. Citation 2002 must be checked, not inferred from ASR.')
        queue.append(row)
        cards.append('<section><h2>'+html.escape(r['id'])+'</h2><audio controls preload="none" src="'+
                     html.escape(Path(r['audio_path']).as_uri(), quote=True)+'"></audio>'+''.join(
                         '<p><b>'+label+'</b>: '+highlighted(row[label])+'</p>' for label in ('original','raw','v1','v2'))+
                     '<details><summary>UNTRUSTED cached suggestions — not listening evidence</summary><pre>'+html.escape(json.dumps(row['untrusted_suggestions'],ensure_ascii=False,indent=2))+'</pre></details>'+
                     '<p>Decision remains pending in review-queue.json. Record reviewer, notes and exact override there after listening.</p></section>')
    b.save(output/'review-queue.json',dict(human_verification='pending', records=queue))
    (output/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Pending human listening</title><h1>Human listening PENDING</h1><p>No audio has been certified by this tool. Listen before viewing suggestions where possible. All text is escaped. Browser file audio controls support keyboard navigation. Edit review-queue.json to record pending/accepted/override, reviewer and notes; no decisions are automatically applied.</p>'+''.join(cards),encoding='utf-8')
    return len(queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=b.ROOT/'outputs/benchmarks/long/error-audit-20261104')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('refusing existing output')
    started = time.monotonic()
    paths = [b.ROOT/'data/benchmark/long-audio/manifest.json',n.DEFAULT,n.DEFAULT_V2,
             o.OUT/'winner.json',o.OUT/'beam15-freeze.json',
             b.ROOT/'outputs/benchmarks/long/numeric-metric-v2-20261104-final/fixed-hypotheses.json']
    paths += [o.OUT/f'beam15-{s}'/name for s in ('validation','test') for name in ('nbest.json','gpt2.json')]
    paths += [Path(__file__),Path(b.__file__),Path(o.__file__),Path(m.__file__),Path(n.__file__)]
    hashes = {str(p.relative_to(b.ROOT)):b.file_hash(p) for p in paths}
    active = b.load(paths[0]); raw = m.original_rows(active)
    v1 = n.transform(raw,b.load(n.DEFAULT))
    if n.transform(raw,b.load(n.DEFAULT_V2)) != active:
        raise ValueError('active annotations mismatch')
    fixed = b.load(paths[5]); o.verify(fixed)
    m.identity_join(active,fixed['records'])
    winner, freeze = b.load(paths[3]), b.load(paths[4])
    o.verify(winner); o.verify(freeze)
    if winner['beam'] != 15 or winner['freeze_sha256'] != b.file_hash(paths[4]) or (freeze['alpha'],freeze['beta']) != (.3,-.02):
        raise ValueError('frozen winner mismatch')
    summary, substitutions = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for split in ('validation','test'):
            manifest = Path(tmp)/f'{split}.json'
            b.save(manifest,[r for r in raw if r['split']==split])
            scored, mapping = o.checked(manifest,o.OUT/f'beam15-{split}/nbest.json',o.OUT/f'beam15-{split}/gpt2.json')
            if scored['protocol'] != freeze['protocol']:
                raise ValueError('protocol mismatch')
            for convention in ('C','D'):
                refs, hyps, oracles, local = [], [], [], []
                for r in scored['records']:
                    saved = next(x for x in fixed['records'] if x['id']==r['id'])['hypotheses']['gpt2_tuned_beam15']
                    pick = o.choose(r['candidates'],freeze['alpha'],freeze['beta'])
                    if (pick['id'],pick['text']) != (saved['candidate_id'],saved['text']):
                        raise ValueError('frozen selected identity mismatch')
                    original = mapping[r['source'],r['id']]['reference']
                    ref = next(x['reference'] for x in active if x['id']==r['id']) if convention=='C' else original
                    ref = m.normalized_pair(ref,'',convention=='D')[0]
                    cs = [dict(id=c['id'],text=m.normalized_pair('',c['text'],convention=='D')[1]) for c in r['candidates']]
                    hyp = next(c['text'] for c in cs if c['id']==pick['id'])
                    oracle = min(cs,key=lambda c:b.distance(ref.split(),c['text'].split()))
                    rows = local_substitutions(ref,cs,pick['id'])
                    for row in rows:
                        row.update(id=r['id'],split=split,convention=convention)
                    local.extend(rows); refs.append(ref); hyps.append(hyp); oracles.append(oracle['text'])
                measured, oracle_metrics = m.measure(refs,hyps), m.measure(refs,oracles)
                if measured['S'] != len(local):
                    raise ValueError('alignment substitution count mismatch')
                summary.append(dict(split=split,convention=convention,selected=measured,oracle=oracle_metrics,
                                    oracle_gap_errors=measured['errors']-oracle_metrics['errors'],
                                    substitutions={k:sum(x['classification']==k for x in local) for k in
                                        ('local_selection_opportunity','missing_from_aligned_set','alignment_ambiguous')}))
                substitutions.extend(local)
    if any(b.file_hash(b.ROOT/p)!=sha for p,sha in hashes.items()):
        raise ValueError('input changed')
    output.mkdir(parents=True)
    queued = review_queue(output,active,raw,v1,fixed)
    b.save(output/'analysis.json',o.seal(dict(inputs_sha256=hashes,summary=summary,substitutions=substitutions)))
    with (output/'substitutions.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(substitutions[0])); writer.writeheader(); writer.writerows(substitutions)
    elapsed = time.monotonic()-started
    lines = ['# GPT2 error audit and fresh holdout status','','## Executed cached audit',
             f'Wall time {elapsed:.3f}s. {queued} modified references queued; zero human approvals. No inference/network/training/tuning.',
             'A original standard remains reported historically; C annotated-v2 and D symmetric-conservative are provisional, not definitive main metrics.',
             '| Split | Metric | S/D/I | WER % | Oracle WER % | Whole-candidate gap errors | Local opportunity | Missing | Ambiguous |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        x=r['selected']; z=r['substitutions']
        lines.append(f"| {r['split']} | {r['convention']} | {x['S']}/{x['D']}/{x['I']} | {100*x['wer']:.4f} | {100*r['oracle']['wer']:.4f} | {r['oracle_gap_errors']} | {z['local_selection_opportunity']} | {z['missing_from_aligned_set']} | {z['alignment_ambiguous']} |")
    lines += ['', 'All candidate alignments use an optimal Levenshtein DAG. A location is ambiguous if ANY candidate has multiple optimal mappings there. The chosen deterministic backtrace matches the existing metric. Independent jiwer validates total word/character errors and denominators (its equally optimal S/D/I may differ).',
              'Local opportunities are not additive attainable improvements: correct tokens can occur in mutually incompatible candidates. Missing means absent at that aligned location, NOT necessarily absent anywhere in the text. Neither category establishes acoustic versus LM causality. Whole-candidate word oracle uses the same convention and decoder-order tie-break; no deployed choice is changed.',
              '## Human action required',
              f'Open `{output.relative_to(b.ROOT)}/review.html` locally. All 13 modified references are prioritized, citation test:266 first. Listen to the complete audio; resolve actual citation omission/presence, year readings, nghìn/ngàn, linh/lẻ and abbreviations. Cached hypotheses are UNTRUSTED. Leave uncertainty pending. Record accepted/override, reviewer, notes, exact transcript and human_verified only after real listening.',
              'Guarded application is deliberately not automatic: review-queue.json binds audio SHA and raw/v2 text. Before any later override application, require human_verified=true, named reviewer, nonempty notes and exact expected current text/audio hash; validate the entire proposed manifest and partitions and back up originals. Existing text-only annotation transformer rejects audio_validated=true; do not misuse it as a listening approval importer.',
              '## Fresh30 status', 'NOT YET EXECUTED in this partial stage. No fresh30 result or disjointness claim is made. Continue the handoff acquisition/frozen inference stage; cached60 is not a fresh holdout.',
              '## Artifacts', f'`{output.relative_to(b.ROOT)}`: sealed analysis.json with source hashes, substitutions.csv with candidate IDs/context, pending review-queue.json, escaped local review.html. No old artifacts or references changed.']
    report='\n\n'.join(lines)+'\n'
    (output/'report.md').write_text(report)
    (b.ROOT/'docs/gpt2-error-audit-holdout.md').write_text(report)
    print(json.dumps(dict(seconds=elapsed,queued=queued,summary=summary),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
