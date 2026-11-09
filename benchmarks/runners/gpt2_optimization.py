"""Offline, separate GPT2 mean-score/word-length experiment; original artifacts untouched."""
import itertools
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.runners import long_audio_benchmark as b

ALPHAS = [0, .05, .1, .2, .3, .4, .5, .6, .8, 1.0]
BETAS = [-.02, -.01, 0, .01, .02]
REV = '65818d14816b42be09e2201933bf07106d9a2647'
ROOT = b.ROOT
CACHE = ROOT / 'outputs/benchmarks/long/fleurs60-v3-20261104'
OUT = ROOT / 'outputs/benchmarks/long/gpt2-optimization-20261104'

def seal(value):
    return dict(value, fingerprint=b.digest(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()))

def verify(value):
    if seal({k: v for k, v in value.items() if k != 'fingerprint'}) != value:
        raise ValueError('freeze integrity')

def edits(ref, hyp):
    d = [[0] * (len(hyp)+1) for _ in range(len(ref)+1)]
    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j
    for i, x in enumerate(ref, 1):
        for j, y in enumerate(hyp, 1):
            d[i][j] = min(d[i-1][j-1]+(x != y), d[i-1][j]+1, d[i][j-1]+1)
    i, j = len(ref), len(hyp)
    counts = dict(S=0, D=0, I=0)
    while i or j:
        if i and j and d[i][j] == d[i-1][j-1]+(ref[i-1] != hyp[j-1]):
            counts['S'] += ref[i-1] != hyp[j-1]
            i, j = i-1, j-1
        elif i and d[i][j] == d[i-1][j]+1:
            counts['D'] += 1
            i -= 1
        else:
            counts['I'] += 1
            j -= 1
    return counts

def choose(cs, alpha, beta):
    return max(cs, key=lambda c: c['avg_logprob'] + alpha*c['lm'] + beta*len(b.normalize(c['text']).split()))

def select_validation(records, grid):
    b.require_validation(records)
    return min(grid,key=lambda g:(g['errors'],g['chars'],g['beta'] != 0,abs(g['beta']),g['alpha'],g['beta']))

def verify_test(f, scores, manifest_hash):
    verify(f)
    if f['protocol'] != scores['protocol'] or f['manifest_sha256'] == manifest_hash:
        raise ValueError('frozen test protocol mismatch')

def checked(manifest, nbest, scored):
    n, s = b.load(nbest), b.load(scored)
    b.verify_configuration(n)
    b.verify_configuration(s)
    mapping = b.join(manifest, s)
    b.join(manifest, n)
    if s['kind'] != 'scored' or n['kind'] != 'nbest' or s['lm'] != 'gpt2' or s['nbest_sha256'] != b.file_hash(nbest) or s['protocol']['acoustic'] != n['protocol']:
        raise ValueError('source/protocol mismatch')
    if s['resolved_revision'] != REV or s['protocol']['lm']['revision'] != REV:
        raise ValueError('revision mismatch')
    for nr, sr in zip(n['records'], s['records'], strict=True):
        if (nr['source'], nr['id'], nr['split']) != (sr['source'], sr['id'], sr['split']):
            raise ValueError('record identity')
        if not nr['candidates'] or len({c['id'] for c in nr['candidates']}) != len(nr['candidates']):
            raise ValueError('candidate IDs')
        if nr['candidates'] != [{k: v for k, v in c.items() if k != 'lm'} for c in sr['candidates']]:
            raise ValueError('candidate source mismatch')
        for c in sr['candidates']:
            if not all(math.isfinite(c[k]) for k in ('lm', 'avg_logprob')):
                raise ValueError('nonfinite')
    return s, mapping

def diagnostic(s, mapping, alpha, beta):
    rows = []
    for r in s['records']:
        ref = b.normalize(mapping[r['source'], r['id']]['reference'])
        cs = r['candidates']
        texts = [b.normalize(c['text']) for c in cs]
        word = min(range(len(cs)), key=lambda i: (b.distance(ref.split(), texts[i].split()), i))
        char = min(range(len(cs)), key=lambda i: (b.distance(ref, texts[i]), i))
        hyps = dict(beam=cs[0]['text'], baseline=choose(cs,.4,0)['text'], tuned=choose(cs,alpha,beta)['text'], word_oracle=texts[word], char_oracle=texts[char])
        metrics = {}
        for name, hyp in hyps.items():
            hyp = b.normalize(hyp)
            counts = edits(ref.split(), hyp.split())
            metrics[name] = dict(**counts, errors=sum(counts.values()), chars=b.distance(ref,hyp), hypothesis=hyp)
        pairs = [b.distance(a.split(),z.split()) / max(1,len(a.split()),len(z.split())) for a,z in itertools.combinations(texts,2)]
        rows.append(dict(id=r['id'], reference=ref, words=len(ref.split()), characters=len(ref), metrics=metrics, candidates=len(cs), unique=len(set(texts)), pairwise=sum(pairs)/len(pairs) if pairs else 0))
    summary = {}
    for name in rows[0]['metrics']:
        totals = {k: sum(r['metrics'][name][k] for r in rows) for k in ('S','D','I','errors','chars')}
        summary[name] = dict(**totals, wer=totals['errors']/sum(r['words'] for r in rows), cer=totals['chars']/sum(r['characters'] for r in rows))
    return dict(summary=summary, records=rows, diversity=dict(candidates=sum(r['candidates'] for r in rows), unique=sum(r['unique'] for r in rows), mean_pairwise=sum(r['pairwise'] for r in rows)/len(rows)))

def main():
    OUT.mkdir(exist_ok=False)
    timings = {}
    def stage(name, fn):
        t = time.monotonic()
        fn()
        timings[name] = time.monotonic()-t
        b.save(OUT/'timings.json', timings)
    frozen = {}
    validation = {}
    for beam in (5,10,15):
        split = 'validation'
        manifest = ROOT/f'data/benchmark/long-audio/splits/{split}.json'
        folder = CACHE/split if beam == 5 else OUT/f'beam{beam}-{split}'
        if beam != 5:
            stage(f'decode-{beam}-{split}', lambda: b.decode(NS(manifest=manifest, output=folder/'nbest.json', beam_size=beam)))
            stage(f'score-{beam}-{split}', lambda: b.score(NS(input=folder/'nbest.json',output=folder/'gpt2.json',lm='gpt2',revision=REV)))
        s, mapping = checked(manifest,folder/'nbest.json',folder/'gpt2.json')
        b.require_validation(s['records'])
        grid = []
        # Cache candidate edit distances; reference never enters score computation.
        costs = []
        for r in s['records']:
            ref = b.normalize(mapping[r['source'],r['id']]['reference'])
            costs.append({c['id']:(b.distance(ref.split(),b.normalize(c['text']).split()),b.distance(ref,b.normalize(c['text']))) for c in r['candidates']})
        for alpha,beta in itertools.product(ALPHAS,BETAS):
            es = [cost[choose(r['candidates'],alpha,beta)['id']] for r,cost in zip(s['records'],costs)]
            grid.append(dict(alpha=alpha,beta=beta,errors=sum(e[0] for e in es),chars=sum(e[1] for e in es)))
        best = select_validation(s['records'],grid)
        f = seal(dict(**best,protocol=s['protocol'],revision=REV,grid=dict(alpha=ALPHAS,beta=BETAS),scoring='avg_logprob + alpha * mean_LM + beta * normalized_word_count',manifest_sha256=b.file_hash(manifest),scores_sha256=b.file_hash(folder/'gpt2.json'),nbest_sha256=b.file_hash(folder/'nbest.json')))
        frozen[beam] = f
        b.save(OUT/f'beam{beam}-freeze.json',f)
        b.save(OUT/f'beam{beam}-grid.json',grid)
        validation[beam] = diagnostic(s,mapping,best['alpha'],best['beta'])
        b.save(OUT/f'beam{beam}-validation.json',validation[beam])
    winner = min(frozen,key=lambda beam:(frozen[beam]['errors'],frozen[beam]['chars'],beam))
    b.save(OUT/'winner.json',seal(dict(beam=winner,freeze_sha256=b.file_hash(OUT/f'beam{winner}-freeze.json'),selection='validation errors, character errors, smaller beam')))
    # All decisions now persisted, before accessing test references or hypotheses.
    for beam in sorted({5,winner}):
        manifest = ROOT/'data/benchmark/long-audio/splits/test.json'
        folder = CACHE/'test' if beam == 5 else OUT/f'beam{beam}-test'
        if beam != 5:
            stage(f'decode-{beam}-test',lambda: b.decode(NS(manifest=manifest,output=folder/'nbest.json',beam_size=beam)))
            stage(f'score-{beam}-test',lambda: b.score(NS(input=folder/'nbest.json',output=folder/'gpt2.json',lm='gpt2',revision=REV)))
        f = b.load(OUT/f'beam{beam}-freeze.json')
        verify(f)
        s,mapping = checked(manifest,folder/'nbest.json',folder/'gpt2.json')
        verify_test(f,s,b.file_hash(manifest))
        b.save(OUT/f'beam{beam}-test.json',diagnostic(s,mapping,f['alpha'],f['beta']))
    print(json.dumps(dict(winner=winner,configs={k:(v['alpha'],v['beta']) for k,v in frozen.items()},timings=timings)))

def report():
    import csv
    lines = ['# GPT2 optimization — FLEURS60 (2026-11-04)', '',
        'Executed offline with the original v3 scorer/decoder; original artifacts untouched. '
        '30 validation clips (1267 words), 30 test clips (1353 words). NFC/lowercase/alphanumeric normalization; CER includes spaces.', '',
        '## Protocol and selection', '',
        '`score = avg_logprob + alpha * mean_LM + beta * normalized_word_count`. '
        'Alpha grid: '+str(ALPHAS)+'; beta grid: '+str(BETAS)+'. '
        'Beta is log-score units per whitespace-delimited normalized word (not LM token). '
        'No inferred token counts or sum-LM variants. All 50 configurations per beam are persisted. '
        'Tie-break: word errors, character errors, beta zero, smaller absolute beta, smaller alpha, beta. '
        'Beam selection: validation word errors, character errors, smaller beam. '
        'Beam15 won before test access. Test comparisons are exploratory on an already-seen test set.', '',
        'Freeze files bind exact protocol, grid, scoring formula, revision, validation manifest/score/N-best hashes. '
        'Candidate source identity and finite scores are checked against hashed N-best; test requires identical invariant protocol. '
        'Fingerprints detect corruption, not cryptographic authenticity against an adversary who can rewrite all artifacts.', '',
        'GPT2 revision `'+REV+'`; checkpoint `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d`. '
        'Runtime and tokenizer vocabulary fingerprints are preserved in each freeze file.', '',
        '| Split | Beam | Method | WER % | CER % | S | D | I |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    details = []
    for split,beam in [('validation',5),('validation',10),('validation',15),('test',5),('test',15)]:
        d = b.load(OUT/f'beam{beam}-{split}.json')
        for method,m in d['summary'].items():
            lines.append(f"| {split} | {beam} | {method} | {100*m['wer']:.4f} | {100*m['cer']:.4f} | {m['S']} | {m['D']} | {m['I']} |")
        v = d['diversity']
        details.append(f"- {split} beam{beam}: {v['candidates']} candidates, {v['unique']} unique normalized strings (summed within samples), mean pairwise normalized word edit distance {v['mean_pairwise']:.5f}.")
        for r in d['records']:
            for name in ('baseline','tuned','word_oracle'):
                diff = r['metrics']['beam']['errors']-r['metrics'][name]['errors']
                r[name+'_vs_beam'] = 'win' if diff > 0 else 'loss' if diff < 0 else 'tie'
        b.save(OUT/f'beam{beam}-{split}-comparisons.json',d)
        for name in ('baseline','tuned'):
            counts = {x:sum(r[name+'_vs_beam']==x for r in d['records']) for x in ('win','tie','loss')}
            details.append(f'- {split} beam{beam} {name} versus beam: {counts}.')
        with (OUT/f'beam{beam}-{split}.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['id','method','S','D','I','word_errors','character_errors'])
            for r in d['records']:
                for name,m in r['metrics'].items():
                    writer.writerow([r['id'],name,m['S'],m['D'],m['I'],m['errors'],m['chars']])
    lines += ['', '## Diversity and per-sample comparisons', '']+details
    lines += ['', '## Attribution and findings', '',
        '1. Candidate-set limitations dominate: beam5 validation 310 → baseline GPT2 292 → oracle 282 errors; test 251 → 237 → 228. '
        'The oracle residual is not correctable by whole-candidate selection in those sets; it is not evidence of exclusively acoustic errors. '
        'For tuned beam15, validation 281 → oracle 271; test 228 → oracle 218. Substitutions dominate all these totals, followed by insertions, not deletions. Counts alone do not establish cause.',
        '2. Larger beam improves word oracle: validation 282 / 276 / 271 errors for beams5/10/15. '
        'Tuned validation errors 289 / 286 / 281. However beam top1 stays at 310 and test beam15 top1 regresses from 251 to 254. '
        'Word-oracle CER also regresses across validation beams: word and character oracles are separately labeled, with ties resolved by original decoder order.',
        '3. Length/alpha tuning gives small gains: beam5 validation 292 → 289, test 237 → 235; '
        'beam15 validation 284 → 281, test 231 → 228. Beam5 test CER worsens despite WER improvement. '
        'Selected configs: beam5 (alpha .3, beta -.02), beam10 (.4, -.02), beam15 (.3, -.02). '
        'Beta hits the predefined lower boundary; expanding the grid would be a new validation experiment, not grounds to retune test.', '',
        'Illustrative validation: reference “bơi 100m và 200m” versus beam “hơn một trăm mét và hai trăm mét”; '
        'reference “khối thịnh vượng” versus “khói thịnh vượng”. Test: “lông vũ” versus “lâm vũ”. '
        'These illustrate numeric formatting and lexical differences, not verified causal diagnoses. Full normalized references/hypotheses and per-sample S/D/I are in diagnostic JSON.', '',
        'Next: manually audit transcripts and number-format conventions, then evaluate candidate-generation changes on fresh held-out speech. '
        'Remaining oracle errors outweigh selection headroom; another rescoring-only grid cannot fix missing candidates.', '',
        '## Reproduction and timing', '',
        'Run `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-benchmark/bin/python benchmarks/runners/gpt2_optimization.py` '
        '(refuses an existing output directory). Report/CSV generation: same interpreter and script with `--report`. '
        'No training, PhoBERT, credentials, packages or downloads requested. CPU GPT2, existing GPU checkpoint for acoustic decode. '
        'Each run used a 1800-second outer bound; sequential model stages only.', '',
        'Measured successful stage wall seconds: `'+json.dumps(b.load(OUT/'timings.json'))+'`.', '',
        '## Caveats', '',
        'Human transcript audit pending; training overlap unknown; small read-speech clips under30 seconds; '
        'no significance/generalization or long-context claim. Test was already seen historically. '
        'Beam10 test was intentionally skipped: only cached-tuned beam5 and the validation-selected winner were evaluated. '
        'No sum-LM/token-normalization grid because sufficient token statistics were not stored. '
        'Original smoke fixtures are stale relative to v3: the unchanged test_long_audio_smoke.py fixture declares v2 and omits prefix/vocabulary fields.', '',
        '## Verification', '',
        'All tests used `.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v`; '
        'standard unittest discovery restricts execution to that filename. '
        'New test RED: missing module ImportError (0.000s). GREEN: 3 passed (0.060s), covering alignment, empty-reference/hypothesis edges, '
        'freeze tampering and cached-source missing/forged/protocol/source-hash rejection. '
        'test_gpt2_prefix_smoke.py: 2 passed (0.000s); test_nbest_ranker_smoke.py: 2 passed (0.000s unittest timing, excluding imports). '
        'test_long_audio_smoke.py: 12 passed, 3 errors (0.037s), all at the unchanged v2 fixture versus current v3 schema; not fixed outside authorized new files. '
        'Environment checker passed 138 effective pins (0.441s). Both split manifest validations passed full audio/hash checks; '
        'validation 611.34s audio, test 624.9s audio. Individual manifest-command wall durations were not instrumented. '
        '`git diff --check` passed. Full suites skipped; only explicitly requested isolated files run. '
        'No commits. Known coverage limitation: no dedicated malformed test-freeze end-to-end unit case; '
        'actual frozen test execution checked protocol equality successfully.']
    (ROOT/'docs/gpt2-optimization-fleurs60.md').write_text('\n\n'.join(lines)+'\n')

def audit():
    """Independent persisted-result audit, no model operations."""
    from jiwer import process_words, process_characters
    audit_rows = []
    for split,beam in [('validation',5),('validation',10),('validation',15),('test',5),('test',15)]:
        data = b.load(OUT/f'beam{beam}-{split}.json')
        refs = [r['reference'] for r in data['records']]
        totals = {}
        for method, expected in data['summary'].items():
            hyps = [r['metrics'][method]['hypothesis'] for r in data['records']]
            w, c = process_words(refs,hyps), process_characters(refs,hyps)
            observed = dict(S=w.substitutions,D=w.deletions,I=w.insertions,
                            errors=w.substitutions+w.deletions+w.insertions,
                            chars=c.substitutions+c.deletions+c.insertions,wer=w.wer,cer=c.cer)
            if any(observed[k] != expected[k] for k in ('errors','chars','wer','cer')):
                raise ValueError('independent metric audit mismatch')
            if any(sum(r['metrics'][method][k] for r in data['records']) != expected[k] for k in ('S','D','I')):
                raise ValueError('persisted S/D/I aggregation mismatch')
            totals[method] = dict(jiwer=observed, persisted=expected)
        audit_rows.append(dict(split=split,beam=beam,words=sum(len(r.split()) for r in refs),characters=sum(map(len,refs)),metrics=totals))
    b.save(OUT/'independent-audit.json',audit_rows)
    lines = ['## Final independent verification (supersedes historical test status above)', '',
             'All saved word/character error totals and WER/CER independently reproduced with jiwer. Persisted S/D/I independently aggregated from sample records. '
             'Jiwer may choose a different equally optimal backtrace, so its individual S/D/I can differ; both are retained in the audit artifact. '
             'Reference totals: validation **1267 words / 5750 characters**; test **1353 words / 6085 characters**. '
             'Characters include normalized spaces. Evidence: `independent-audit.json`.', '',
             'Original GPT2 means cached beam5 alpha .4; selected means validation-selected beam15 alpha .3, beta -.02. '
             'Validation: 292 → 281 errors (−11), WER 23.0466 → 22.1784 (−0.8682 percentage points), '
             'CER 13.7391 → 13.7739 (+0.0348 points). Test: 237 → 228 (−9), '
             'WER 17.5166 → 16.8514 (−0.6652 points), CER 10.6984 → 10.6163 (−0.0822 points). '
             'Versus cached beam5 top1: validation 310 → 281 (−29; −2.2889 WER points), '
             'test 251 → 228 (−23; −1.6999 WER points). '
             'Versus same-beam15 top1: validation 310 → 281 (−29), test 254 → 228 (−26; −1.9217 WER points).', '',
             'Concrete normalized snippets, directly checked against saved hypotheses:', '']
    snippets = [ ('validation',5,'validation:151','bơi 100m và 200m','hơn một trăm mét và hai trăm mét'),
                 ('validation',5,'validation:151','khối thịnh vượng','khói thịnh vượng'),
                 ('test',5,'test:524','lông vũ','lâm vũ') ]
    for split,beam,id_,ref,hyp in snippets:
        record = next(r for r in b.load(OUT/f'beam{beam}-{split}.json')['records'] if r['id']==id_)
        if ref not in record['reference'] or hyp not in record['metrics']['beam']['hypothesis']:
            raise ValueError('snippet not evidenced')
        w = process_words(ref,hyp)
        lines.append(f'- {id_}, beam{beam} top1: reference `{ref}` → hypothesis `{hyp}`; isolated-span alignment S={w.substitutions}, D={w.deletions}, I={w.insertions}.')
    lines += ['', 'The first snippet exhibits numeric orthography differences and additional hypothesis words; '
              'it does not prove number formatting caused the error. Isolated-span alignments need not equal the full-utterance alignment attribution.', '',
              'Only stale v2 fixture fields in test_long_audio_smoke.py were migrated to strict v3. '
              'Production schema/checks unchanged. Added tests exercise validation-only selection/tie-break, frozen protocol mismatch, '
              'reuse of validation manifest rejection, persisted S/D/I and reference totals, and winner/freeze persistence before test access. '
              'The orchestration-order test isolates model/I/O boundaries; the saved-metric audit uses independent jiwer rather than the production backtrace.', '',
              'No model rerun. Final scoped commands use `.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v` '
              'for the four exact filenames; no full suite. Final results recorded below.']
    with (ROOT/'docs/gpt2-optimization-fleurs60.md').open('a') as f:
        f.write('\n\n'+'\n\n'.join(lines)+'\n')

if __name__ == '__main__':
    if '--audit' in sys.argv:
        audit()
    elif '--report' in sys.argv:
        report()
    else:
        main()
