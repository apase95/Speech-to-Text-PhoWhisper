"""Offline staged benchmark; preserve prior artifacts and stop on first failure."""
import contextlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.runners import long_audio_benchmark as b

OUT = Path('outputs/benchmarks/long/fleurs60-v3-20261104')
REVISIONS = dict(gpt2='65818d14816b42be09e2201933bf07106d9a2647', phobert='86cd7fd4c148980922ac11a2cf5e257f2ba639e1')


def main():
    if OUT.exists():
        raise ValueError('Use a fresh directory; existing evidence preserved')
    b.save(OUT/'progress.json', [])
    progress = []
    def run(stage, function, **kwargs):
        started = time.monotonic()
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                function(SimpleNamespace(**kwargs))
            entry = dict(stage=stage, status='passed', seconds=time.monotonic()-started)
        except Exception as exc:
            entry = dict(stage=stage, status='blocked', error_type=type(exc).__name__, seconds=time.monotonic()-started)
        progress.append(entry)
        b.save(OUT/'progress.json', progress)
        print(json.dumps(entry), flush=True)
        if entry['status'] != 'passed':
            sys.exit(2)
    def scores(split, nbest):
        for lm, revision in REVISIONS.items():
            run(split+'-'+lm, b.score, input=nbest, output=OUT/split/(lm+'.json'), lm=lm, revision=revision)
        g, p = (b.load(OUT/split/(lm+'.json')) for lm in REVISIONS)
        for data in (g, p):
            b.verify_configuration(data)
        assert g['nbest_sha256'] == p['nbest_sha256']
        assert [[(c['id'], c['text'], c['avg_logprob']) for c in r['candidates']] for r in g['records']] == [[(c['id'], c['text'], c['avg_logprob']) for c in r['candidates']] for r in p['records']]
    scores('smoke', Path('outputs/benchmarks/long/smoke-20261104-staged/nbest.json'))
    for split in ('validation', 'test'):
        manifest = Path('data/benchmark/long-audio/splits')/(split+'.json')
        run(split+'-decode', b.decode, manifest=manifest, output=OUT/split/'nbest.json', beam_size=5)
        scores(split, OUT/split/'nbest.json')
        if split == 'validation':
            for lm in REVISIONS:
                run('freeze-'+lm, b.tune, manifest=manifest, input=OUT/split/(lm+'.json'), output=OUT/(lm+'-frozen.json'))
        run(split+'-evaluate', b.evaluate, manifest=manifest, gpt2=OUT/split/'gpt2.json', phobert=OUT/split/'phobert.json', gpt2_config=OUT/'gpt2-frozen.json', phobert_config=OUT/'phobert-frozen.json', output=OUT/split/'results.json')


if __name__ == '__main__':
    main()
