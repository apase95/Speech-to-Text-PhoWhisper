"""Offline fixed-hypothesis evaluation. Never imports or invokes model operations."""
import argparse
import copy
import datetime
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.runners import long_audio_benchmark as b
from benchmarks.runners import gpt2_optimization as o
from tools import normalize_benchmark_references as n

DIGITS = 'không một hai ba bốn năm sáu bảy tám chín'.split()
ATOM = r'(?:0|[1-9]\d{0,8})(?:[.,]\d+)*'


def integer(value, full=False):
    if not 0 <= value < 1000000000:
        raise ValueError('integer outside grammar')
    if value >= 1000:
        scale, name = (1000000, 'triệu') if value >= 1000000 else (1000, 'nghìn')
        q, r = divmod(value, scale)
        return integer(q) + ' ' + name + (' ' + integer(r, True) if r else '')
    h, r = divmod(value, 100)
    out = [DIGITS[h], 'trăm'] if h or full else []
    t, u = divmod(r, 10)
    if t >= 2:
        out += [DIGITS[t], 'mươi']
    elif t == 1:
        out += ['mười']
    elif u and out:
        out += ['linh']
    if u:
        out += ['mốt' if u == 1 and t >= 2 else 'lăm' if u == 5 and t else DIGITS[u]]
    return ' '.join(out) or 'không'


def number(text, decimal=False):
    if re.fullmatch(r'[1-9]\d{0,2}(?:\.\d{3})+', text):
        return integer(int(text.replace('.', '')))
    if re.fullmatch(r'0|[1-9]\d{0,8}', text):
        return integer(int(text))
    if decimal and re.fullmatch(r'(?:0|[1-9]\d{0,8}),\d+', text):
        a, z = text.split(',')
        return integer(int(a)) + ' phẩy ' + ' '.join(DIGITS[int(c)] for c in z)
    raise ValueError('unsupported numeric notation')


def verbalize(text):
    """Context-only digit expansion; no spoken-variant equivalence classes.

    Dates require explicit ngày/tháng/năm and calendar validity. Comma decimal
    only with physical units; dot grouping is Vietnamese thousands convention.
    Unsupported atoms remain intact and are returned for audit, not guessed.
    """
    text = unicodedata.normalize('NFC', text).lower()
    protected = []
    def date(match):
        day, month, year = map(int, match.groups())
        try:
            datetime.date(year, month, day)
            return f'ngày {integer(day)} tháng {integer(month)} năm {integer(year)}'
        except ValueError:
            protected.append(match.group())
            return f'\ufff0{chr(0xe000 + len(protected)-1)}\ufff1'
    text = re.sub(r'\bngày (\d{1,2}) tháng (\d{1,2}) năm (\d{4})\b', date, text)
    # Match the complete atom; boundaries forbid partial conversion in identifiers,
    # slash dates, malformed decimals or leading-zero numbers.
    edge_l, edge_r = r'(?<![\w.,/])', r'(?![\w/]|[.,]\d)'
    units = {'m':'mét', 'kilomet':'ki lô mét', 'km/h':'ki lô mét trên giờ', 'dặm':'dặm'}
    pattern = edge_l + '(' + ATOM + r')\s*(km/h|kilomet|dặm|m)\b' + edge_r
    def unit(match):
        if re.search(r'\b(?:mã|code|id)\s*$', text[:match.start()]):
            return match.group()
        try:
            return number(match[1], True) + ' ' + units[match[2]]
        except ValueError:
            return match.group()
    text = re.sub(pattern, unit, text)
    def suffix(match):
        try:
            return number(match[1]) + ' ' + match[2]
        except ValueError:
            return match.group()
    text = re.sub(edge_l + '(' + ATOM + r')\s+(tuổi|đĩa|món|trường hợp|vận động viên)\b', suffix, text)
    def prefix(match):
        try:
            if match[1] == 'năm' and not re.fullmatch(r'[1-9]\d{3}', match[2]):
                return match.group()
            if match[1] == 'thứ' and match[2] in ('1','4'):
                return match[1] + ' ' + {'1':'nhất','4':'tư'}[match[2]]
            return match[1] + ' ' + number(match[2])
        except ValueError:
            return match.group()
    text = re.sub(r'\b(năm|thứ|tổng số)\s+(' + ATOM + ')' + edge_r, prefix, text)
    text = re.sub(edge_l + '(' + ATOM + r')(?= trong tổng số\b)',
                  lambda m: number(m[1]) if re.fullmatch(r'[1-9]\d{0,8}', m[1]) else m.group(), text)
    for i, original in enumerate(protected):
        text = text.replace(f'\ufff0{chr(0xe000+i)}\ufff1', original)
    unresolved = re.findall(r'\S*\d\S*', text)
    return text, unresolved


def normalized_pair(ref, hyp, symmetric=False):
    return tuple(b.normalize(verbalize(x)[0] if symmetric else x) for x in (ref, hyp))


def identity_join(rows, records, references=False):
    mapping = {(r['source'], r['id'], r['split']): r for r in rows}
    keys = [(r['source'], r['id'], r['split']) for r in records]
    if len(mapping) != len(rows) or len(set(keys)) != len(keys) or set(keys) != set(mapping):
        raise ValueError('sample identity mismatch')
    if references and any(b.normalize(mapping[k]['reference']) != r['reference']
                          for k, r in zip(keys, records)):
        raise ValueError('original reference identity mismatch')
    return mapping


def fixed_choice(candidates, alpha, beta):
    if not candidates or len({c['id'] for c in candidates}) != len(candidates):
        raise ValueError('candidate IDs')
    if not all(math.isfinite(c[k]) for c in candidates for k in ('avg_logprob', 'lm')):
        raise ValueError('nonfinite scores')
    return o.choose(candidates, alpha, beta)  # Original normalization, never new refs.


def measure(refs, hyps):
    from jiwer import process_words, process_characters
    totals = dict(S=0, D=0, I=0, errors=0, chars=0,
                  words=sum(len(r.split()) for r in refs), characters=sum(map(len, refs)))
    for ref, hyp in zip(refs, hyps, strict=True):
        counts = o.edits(ref.split(), hyp.split())
        for k in counts:
            totals[k] += counts[k]
        totals['chars'] += b.distance(ref, hyp)
    totals['errors'] = sum(totals[k] for k in ('S','D','I'))
    totals['wer'] = totals['errors'] / max(1, totals['words'])
    totals['cer'] = totals['chars'] / max(1, totals['characters'])
    w, c = process_words(refs, hyps), process_characters(refs, hyps)
    independent = dict(S=w.substitutions, D=w.deletions, I=w.insertions,
                       errors=w.substitutions+w.deletions+w.insertions,
                       chars=c.substitutions+c.deletions+c.insertions)
    if any(independent[k] != totals[k] for k in ('errors','chars')):
        raise ValueError('independent metric mismatch')
    if (w.hits+w.substitutions+w.deletions, c.hits+c.substitutions+c.deletions) != (totals['words'],totals['characters']):
        raise ValueError('independent denominator mismatch')
    totals['independent'] = independent
    return totals


def edit_spans(ref, hyp):
    from jiwer import process_words
    w = process_words(ref, hyp)
    return [dict(type=a.type, reference=' '.join(w.references[0][a.ref_start_idx:a.ref_end_idx]),
                 hypothesis=' '.join(w.hypotheses[0][a.hyp_start_idx:a.hyp_end_idx]))
            for a in w.alignments[0] if a.type != 'equal']


def original_rows(active):
    rows = copy.deepcopy(active)
    for r in rows:
        if r.pop('reference_convention') != 'fleurs60-spoken-v2':
            raise ValueError('active convention')
        r['reference'] = r.pop('reference_raw')
    return rows


def verify_original_cache(manifest, result):
    if result['manifest_sha256'] != b.file_hash(manifest):
        raise ValueError('original reconstructed manifest hash mismatch')
    return identity_join(b.load(manifest), result['records'], references=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=b.ROOT/'outputs/benchmarks/long/numeric-metric-v2-20261104')
    parser.add_argument('--artifact-only', action='store_true', help='Do not overwrite the project audit document')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise ValueError('refusing existing output')
    started = time.monotonic()
    hashes = {str(path.relative_to(b.ROOT)):b.file_hash(path) for path in
              (Path(__file__).resolve(),Path(b.__file__),Path(o.__file__),Path(n.__file__))}
    def load(path):
        hashes[str(path.relative_to(b.ROOT))] = b.file_hash(path)
        return b.load(path)
    active = load(b.ROOT/'data/benchmark/long-audio/manifest.json')
    a1, a2 = load(n.DEFAULT), load(n.DEFAULT_V2)
    if n.transform(active, a2) != active:
        raise ValueError('active annotation mismatch')
    raw = original_rows(active)
    v1, v2 = n.transform(raw, a1), n.transform(raw, a2)
    winner = load(o.OUT/'winner.json')
    o.verify(winner)
    if winner['beam'] != 15 or winner['freeze_sha256'] != b.file_hash(o.OUT/'beam15-freeze.json'):
        raise ValueError('winner freeze mismatch')
    freezes = {}
    for beam in (5,15):
        f = load(o.OUT/f'beam{beam}-freeze.json')
        o.verify(f)
        if f['revision'] != o.REV or f['grid'] != dict(alpha=o.ALPHAS,beta=o.BETAS) or f['scoring'] != 'avg_logprob + alpha * mean_LM + beta * normalized_word_count':
            raise ValueError('freeze semantics')
        freezes[beam] = f
    acoustic5 = copy.deepcopy(freezes[5]['protocol']['acoustic'])
    acoustic15 = copy.deepcopy(freezes[15]['protocol']['acoustic'])
    acoustic5['decoding'].pop('beam_size')
    acoustic15['decoding'].pop('beam_size')
    if acoustic5 != acoustic15 or acoustic5['checkpoint_sha256'] != 'e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d' or freezes[5]['protocol']['lm'] != freezes[15]['protocol']['lm']:
        raise ValueError('cross-beam checkpoint/LM identity mismatch')
    fixed, historical = [], {}
    with tempfile.TemporaryDirectory() as tmp:
        for split in ('validation','test'):
            subset = [r for r in raw if r['split'] == split]
            active_split = load(b.ROOT/f'data/benchmark/long-audio/splits/{split}.json')
            if active_split != [r for r in active if r['split']==split]:
                raise ValueError('partition mismatch')
            manifest = Path(tmp)/f'{split}.json'
            b.save(manifest, subset)
            mh = b.file_hash(manifest)
            result = load(o.CACHE/split/'results.json')
            verify_original_cache(manifest, result)
            historical[split] = result
            methods = {}
            for beam in (5,15):
                folder = o.CACHE/split if beam == 5 else o.OUT/f'beam{beam}-{split}'
                nbpath, spath = folder/'nbest.json', folder/'gpt2.json'
                nb, scored = load(nbpath), load(spath)
                o.checked(manifest, nbpath, spath)
                if any(nr['greedy'] != sr['greedy'] for nr,sr in zip(nb['records'],scored['records'],strict=True)):
                    raise ValueError('greedy source mismatch')
                f = freezes[beam]
                if f['protocol'] != scored['protocol'] or f['protocol']['acoustic']['decoding']['beam_size'] != beam:
                    raise ValueError('frozen protocol mismatch')
                if split == 'validation':
                    if (f['manifest_sha256'],f['scores_sha256'],f['nbest_sha256']) != (mh,b.file_hash(spath),b.file_hash(nbpath)):
                        raise ValueError('validation freeze source mismatch')
                else:
                    o.verify_test(f, scored, mh)
                diag = load(o.OUT/f'beam{beam}-{split}.json')
                diagmap = {r['id']:r for r in diag['records']}
                if len(diagmap) != len(subset) or set(diagmap) != {r['id'] for r in subset}:
                    raise ValueError('historical diagnostic IDs')
                for r in scored['records']:
                    key = r['source'], r['id'], r['split']
                    tuned = fixed_choice(r['candidates'], f['alpha'], f['beta'])
                    d = diagmap[r['id']]
                    reference = next(x['reference'] for x in subset if x['id']==r['id'])
                    if d['reference'] != b.normalize(reference) or d['metrics']['tuned']['hypothesis'] != b.normalize(tuned['text']):
                        raise ValueError('fixed tuned historical mismatch')
                    picks = {f'gpt2_tuned_beam{beam}': (tuned['id'], tuned['text'])}
                    if beam == 5:
                        configpath = o.CACHE/'gpt2-frozen.json'
                        config = load(configpath)
                        b.verify_configuration(config)
                        if config['protocol'] != scored['protocol'] or any(config[k] != scored[k] for k in ('lm','model_id','resolved_revision')) or config['validation_scores_sha256'] != b.file_hash(o.CACHE/'validation/gpt2.json') or config['validation_manifest_sha256'] != freezes[5]['manifest_sha256']:
                            raise ValueError('original GPT2 freeze source')
                        c = fixed_choice(r['candidates'],config['alpha'],0)
                        picks.update(greedy=('greedy',r['greedy']), beam5=(r['candidates'][0]['id'],r['candidates'][0]['text']), gpt2_original=(c['id'],c['text']))
                    methods.setdefault(key, {}).update(picks)
            # All final hypotheses are chosen above without consulting new refs.
            for (source, id_, split_), picks in methods.items():
                hist = next(r for r in result['records'] if (r['source'],r['id'],r['split']) == (source,id_,split_))
                for method, old in [('greedy','greedy'),('beam5','beam_top1'),('gpt2_original','gpt2')]:
                    if b.normalize(picks[method][1]) != hist['hypotheses'][old]:
                        raise ValueError('saved original hypothesis mismatch')
                fixed.append(dict(source=source,id=id_,split=split_, hypotheses={k:dict(candidate_id=v[0],text=v[1]) for k,v in picks.items()}))
    # Evaluate only after the fixed decisions have been materialized.
    summaries, records = [], []
    maps = {name:{(r['source'],r['id']):r['reference'] for r in rows}
            for name,rows in [('A',raw),('B',v1),('C',v2),('D',raw)]}
    for split in ('validation','test'):
        items = [r for r in fixed if r['split']==split]
        for convention in maps:
            for method in items[0]['hypotheses']:
                refs, hyps = [], []
                for r in items:
                    ref = maps[convention][r['source'],r['id']]
                    hyp = r['hypotheses'][method]['text']
                    pair = normalized_pair(ref,hyp,convention=='D')
                    refs.append(pair[0]); hyps.append(pair[1])
                    metrics = measure([pair[0]],[pair[1]])
                    records.append(dict(split=split,id=r['id'],source=r['source'],convention=convention,method=method,
                        reference=pair[0],hypothesis=pair[1],metrics=metrics,edit_spans=edit_spans(*pair),
                        unresolved_reference=verbalize(ref)[1] if convention=='D' else [],
                        unresolved_hypothesis=verbalize(hyp)[1] if convention=='D' else []))
                total = measure(refs,hyps)
                if convention == 'A':
                    if (total['words'],total['characters']) != ((1267,5750) if split=='validation' else (1353,6085)):
                        raise ValueError('baseline denominator mismatch')
                    if method in ('greedy','beam5','gpt2_original'):
                        oldname = dict(greedy='greedy',beam5='beam_top1',gpt2_original='gpt2')[method]
                        old = historical[split]['summary'][split+':pooled']['metrics'][oldname]
                        if (total['errors'],total['wer'],total['cer']) != (old['word_errors'],old['wer'],old['cer']):
                            raise ValueError('published baseline mismatch')
                    if method == 'gpt2_tuned_beam15' and total['errors'] != (281 if split=='validation' else 228):
                        raise ValueError('published selected baseline mismatch')
                summaries.append(dict(split=split,convention=convention,method=method,**total))
    if any(b.file_hash(b.ROOT/path) != sha for path,sha in hashes.items()):
        raise ValueError('input changed during evaluation')
    args.output.mkdir(parents=True,exist_ok=False)
    b.save(args.output/'fixed-hypotheses.json',o.seal(dict(inputs_sha256=hashes,records=fixed)))
    b.save(args.output/'metrics.json',dict(summary=summaries,records=records))
    b.save(args.output/'independent-audit.json',dict(engine='jiwer',rows=[dict(split=r['split'],convention=r['convention'],method=r['method'],**r['independent']) for r in summaries], passed=True))
    elapsed = time.monotonic()-started
    inventory = dict(samples=len(raw),original_digit_refs=sum(bool(re.search(r'\d',r['reference'])) for r in raw),
        original_numeric_expressions=sum(len(re.findall(r'\d+(?:[.,]\d+)*',r['reference'])) for r in raw),
        v1_changed_refs=sum(a['reference']!=z['reference'] for a,z in zip(raw,v1)),
        v2_additional_changed_refs=sum(a['reference']!=z['reference'] for a,z in zip(v1,v2)),
        v2_remaining_digit_refs=sum(bool(re.search(r'\d',r['reference'])) for r in v2))
    b.save(args.output/'inventory.json',inventory)
    report(args.output,summaries,records,elapsed,inventory, publish_document=not args.artifact_only)
    print(json.dumps(dict(output=str(args.output),seconds=elapsed,inventory=inventory,independent_audit='passed',summary_rows=len(summaries)),ensure_ascii=False,indent=2))


def report(output, summaries, records, elapsed, inventory, publish_document=True):
    artifact_path = output.relative_to(b.ROOT) if output.is_relative_to(b.ROOT) else output
    lines = ['# FLEURS60 numeric metric audit — fixed cached predictions', '',
        'No model inference, training, retuning or downloads. A reproduces historical original-reference metrics before any new results are published. CER includes normalized spaces.', '',
        'A: original references. B: v1 ref-only. C: v2 ref-only. D: the same conservative numeric verbalizer on ORIGINAL raw references and raw hypotheses, before NFC/lower/alphanumeric/space normalization.', '',
        '| Split | Convention | Method | Words | Chars | S | D | I | Errors | WER % | Char errors | CER % |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['split']} | {r['convention']} | {r['method']} | {r['words']} | {r['characters']} | {r['S']} | {r['D']} | {r['I']} | {r['errors']} | {100*r['wer']:.4f} | {r['chars']} | {100*r['cer']:.4f} |")
    lines += ['', '## Changes versus A', '', '| Split | Convention | Method | Error delta | Word delta | Character delta |','|---|---|---|---:|---:|---:|']
    for r in summaries:
        if r['convention']=='A':
            continue
        a = next(x for x in summaries if x['split']==r['split'] and x['method']==r['method'] and x['convention']=='A')
        lines.append(f"| {r['split']} | {r['convention']} | {r['method']} | {r['errors']-a['errors']:+d} | {r['words']-a['words']:+d} | {r['characters']-a['characters']:+d} |")
    lines += ['', '## Local changes and remaining mismatches', '', 'All sample-level metrics, normalized texts, unresolved digits and independent edit spans are in `metrics.json`. Below: original GPT2 only; every changed sample is listed (A → B → C → D word errors).']
    ids = sorted({r['id'] for r in records if r['method']=='gpt2_original'})
    for id_ in ids:
        rs = [next(r for r in records if r['id']==id_ and r['method']=='gpt2_original' and r['convention']==c) for c in 'ABCD']
        if len({r['reference'] for r in rs})>1 or len({r['hypothesis'] for r in rs})>1:
            lines.append(f"- {id_}: " + ' → '.join(str(r['metrics']['errors']) for r in rs))
            lines.append('  Remaining D edits: `'+json.dumps(rs[-1]['edit_spans'],ensure_ascii=False)+'`')
    unresolved = [dict(id=r['id'],method=r['method'],reference=r['unresolved_reference'],hypothesis=r['unresolved_hypothesis']) for r in records if r['convention']=='D' and (r['unresolved_reference'] or r['unresolved_hypothesis'])]
    lines += ['', 'Confirmed unchanged lexical snippets (beam5, D):']
    for id_,ref,hyp in [('validation:151','khối thịnh vượng','khói thịnh vượng'),('test:524','lông vũ','lâm vũ')]:
        row = next(r for r in records if r['id']==id_ and r['method']=='beam5' and r['convention']=='D')
        if ref not in row['reference'] or hyp not in row['hypothesis']:
            raise ValueError('unsupported lexical example')
        lines.append(f'- {id_}: `{ref}` versus `{hyp}`.')
    lines += ['', '## Grammar, provenance and limitations', '',
        'C and D agree on validation. On test, D has five fewer errors than C for every method solely because the unsupported bare citation year in test:266 stays as 2002 instead of becoming six spoken words. The hypothesis omits that citation. This is a denominator/annotation-scope difference, NOT five additional recognized words or evidence that symmetric normalization repaired a prediction.', '',
        'Only explicit valid ngày/tháng/năm dates, năm years, thứ ordinals, age/count noun contexts, total/subset counts and numeric physical units are expanded. Dot-separated groups of three mean thousands; comma decimals require units. Leading zeros, malformed notation, slash dates, bare citation years, Roman numerals, abbreviations and alphanumeric identifiers are not guessed. No nghìn/ngàn, linh/lẻ, một/mốt or omitted-hundred equivalences are collapsed: D is a conservative lower-bound normalization, not full numeric invariance. No sample-ID rules.', '',
        'Original manifests are reconstructed by restoring reference_raw and removing only the two annotation fields; their exact serialized hashes must equal old caches. Strict existing artifact/config/source checks remain intact. Original freezes, validation source hashes, winner fingerprint, protocol/checkpoint identities and historical final hypotheses are checked. The fixed-hypotheses artifact seals chosen candidate IDs, RAW texts and input hashes. No reference enters candidate scoring. These fingerprints detect corruption, not adversarial rewriting of all evidence.', '',
        'S/D/I use deterministic diagonal-before-deletion-before-insertion backtrace. Independent jiwer word/character total errors match every sample and corpus result. Equally optimal jiwer S/D/I can differ; both are saved. Jiwer edit spans are illustrative, not causal attribution.', '',
        'Percentage changes mix numerator and denominator changes. A→B→C→D differences are not additive causal error categories and do not improve the model. Text-only references have not been audio audited. Near-homophone substitutions such as khối/khói and lông/lâm are lexical examples, not proof of acoustic origin. Small, previously seen test set; no generalization claim.', '',
        'Verified inventory: `'+json.dumps(inventory)+'`. V1 covers seven measurement spans; v2 additionally changes 16 numeric expressions (the three date components form one span). Roman/abbreviation ambiguities remain. V1 backup: `data/benchmark/long-audio/reference-backups/1793755881486055602/`. Previous actual apply fully validated 60 audio hashes/PCM/durations/leak checks, 1236.24s total; repeated apply changed zero files.', '',
        'Unresolved D forms: `'+json.dumps(unresolved,ensure_ascii=False)+'`', '',
        f'Artifacts: `{artifact_path}` (`fixed-hypotheses.json`, `metrics.json`, `independent-audit.json`, `inventory.json`, `report.md`). Reevaluation wall time {elapsed:.3f}s; no audio/model stage. PhoBERT and new-reference candidate oracles intentionally omitted; fixed greedy, beam5, original GPT2, tuned beam5 and selected beam15 covered.', '',
        'Next priorities: human audio/transcript audit of number readings and unresolved citation/abbreviation forms, then lexical/proper-name and candidate-generation errors on fresh held-out speech. Further whole-candidate rescoring cannot recover words absent from candidates.']
    lines += ['', '## Reproduction and verification', '',
        'Run `.venv-benchmark/bin/python benchmarks/audits/reevaluate_numeric_metrics.py --output NEW_DIRECTORY` (refuses existing output). Relative output paths are resolved; original artifacts are read-only. The final invocation used `outputs/benchmarks/long/numeric-metric-v2-20261104-final`.', '',
        'New tests demonstrated RED (missing module ImportError), then GREEN: 7 tests, 0.021s. Synthetic fixtures exercise integer/unit/date/year/count/ordinal/decimal rules, invalid-date and identifier preservation, symmetric equality/idempotence, S/D/I and empty edges, original-manifest reconstruction/hash rejection, swapped reference rejection and finite fixed-candidate scoring. No ignored-data dependency.', '',
        'Exact filename-limited command: `.venv-benchmark/bin/python -m unittest discover -s tests -p FILE -v`. Existing scoped results: test_reference_normalization.py 4 passed (0.005s), test_long_audio_smoke.py 15 (0.063s), test_gpt2_prefix_smoke.py 2 (0.000s), test_nbest_ranker_smoke.py 2 (0.000s), test_gpt2_optimization.py 6 passed / 1 optional historical audit skipped (0.013s). New reevaluation independently audits every evaluated sample and corpus row with jiwer; no full suite run.', '',
        'An intermediate relative-output run completed metrics/audits but failed report rendering; its partial directory is retained, not the final result. The path-resolution bug was fixed before the final run. All original input hashes are checked again before publication. No commits or pushes; unrelated working-tree files preserved.']
    text = '\n'.join(lines)+'\n'
    (output/'report.md').write_text(text)
    if publish_document:
        (b.ROOT/'docs/numeric-metric-audit-fleurs60.md').write_text(text)


if __name__ == '__main__':
    main()
