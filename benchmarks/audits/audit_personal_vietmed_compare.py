"""Audit saved artifacts only; never rerun models or change saved measurements."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.metrics import normalize_vietnamese
from jiwer import process_words, process_characters
from benchmarks.runners.personal_vietmed_compare import sha, save, identity


def main():
    out = Path('outputs/benchmarks/personal-vietmed-compare-20261104T035943Z')
    report = json.loads((out/'report.json').read_text())
    audit = dict(profiles={}, personal={})
    for name in ('original-production','current-long'):
        path = out/(name+'.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        expected = sum([json.loads(Path(f'data/benchmark/vietmed100/{s}.json').read_text()) for s in ('train','dev','test')], [])
        identity(expected, rows)
        assert [r['reference'] for r in rows] == [r['reference'] for r in expected]
        assert sha(path) == report['results'][name]['predictions_sha256']
        refs = [normalize_vietnamese(r['reference']) for r in rows]
        hyps = [normalize_vietnamese(r['hypothesis']) for r in rows]
        w,c = process_words(refs,hyps),process_characters(refs,hyps)
        assert w.wer == report['results'][name]['wer'] and c.cer == report['results'][name]['cer']
        stages = {key:sum(r[key] for r in rows) for key in ('audio_io_seconds','preprocessing_seconds','decode_seconds','lm_seconds')}
        assert abs(sum(stages.values())-report['results'][name]['inference_seconds']) < 1e-8
        audit['profiles'][name] = dict(stages=stages, words=sum(len(r.split()) for r in refs),characters=sum(map(len,refs)),
            saved_predictions_verified=True)
        if name == 'original-production':
            archived = Path('outputs/benchmarks/vietmed-local-20261025T233412Z/base-gpt2-all100.jsonl')
            if archived.is_file():
                old = [json.loads(line) for line in archived.read_text().splitlines()]
                identity(old,rows)
                audit['archived_baseline_exact_hypothesis_matches'] = sum(a['hypothesis']==b['hypothesis'] for a,b in zip(old,rows))
    segments = json.loads((out/'personal-segments.json').read_text())
    assert segments[0]['start_sample'] == 0
    assert all(a['end_sample']==b['start_sample'] for a,b in zip(segments,segments[1:]))
    assert segments[-1]['end_sample'] == report['personal']['decoded_samples']
    audit['personal'] = dict(stages={key:sum(s[key] for s in segments) for key in ('preprocessing_seconds','decode_seconds','lm_seconds')},
        bounds=[(s['start_seconds'],s['end_seconds']) for s in segments], coverage_verified=True,
        warm_wall_rtf=report['personal']['processing_wall_seconds']/float(report['personal']['probe']['format']['duration']),
        cold_accounted_wall_rtf=(report['personal']['processing_wall_seconds']+sum(report['loads'].values()))/float(report['personal']['probe']['format']['duration']),
        note='cold_accounted_wall adds measured model loads to measured personal processing wall; excludes imports, hashing, config/probe overhead; not whole process cold wall')
    save(out/'independent-audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
