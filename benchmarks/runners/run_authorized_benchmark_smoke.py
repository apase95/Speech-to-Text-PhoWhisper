"""Bounded real-model smoke; safe diagnostics, no training or test decoding."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.runners import long_audio_benchmark as b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise ValueError('Output already exists; choose a fresh run directory')
    rows = b.load('data/benchmark/long-audio/manifest.json')
    validation = [r for r in rows if r['split'] == 'validation']
    assert validation == b.load('data/benchmark/long-audio/splits/validation.json')
    b.save(out / 'manifest.json', validation[:1])
    steps = []
    def run(name, function, **kwargs):
        started = time.monotonic()
        try:
            # Third-party exceptions/progress can include signed URLs: never emit them.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                function(SimpleNamespace(**kwargs))
        except Exception as exc:
            steps.append(dict(stage=name, status='blocked', error_type=type(exc).__name__, seconds=time.monotonic()-started))
            b.save(out / 'progress.json', steps)
            print(json.dumps(steps[-1]), flush=True)
            return False
        steps.append(dict(stage=name, status='passed', seconds=time.monotonic()-started))
        b.save(out / 'progress.json', steps)
        print(json.dumps(steps[-1]), flush=True)
        return True
    if not run('decode', b.decode, manifest=out/'manifest.json', output=out/'nbest.json', beam_size=5):
        return 2
    for lm, revision in [('gpt2', '65818d14816b42be09e2201933bf07106d9a2647'), ('phobert', '86cd7fd4c148980922ac11a2cf5e257f2ba639e1')]:
        if not run(lm, b.score, input=out/'nbest.json', output=out/(lm+'.json'), lm=lm, revision=revision):
            return 2
    g, p = (b.load(out/(lm+'.json')) for lm in ('gpt2', 'phobert'))
    for data in (g, p):
        b.verify_configuration(data)
    assert g['nbest_sha256'] == p['nbest_sha256']
    print('PASS: shared persisted candidates; finite LM scores; validation-only smoke')
    return 0


if __name__ == '__main__':
    sys.exit(main())
