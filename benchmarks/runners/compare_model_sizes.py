"""Offline orchestration of isolated, no-rescore base/small benchmark processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def validate_pairs(a, b):
    def index(rows):
        indexed = {r['id']: r for r in rows}
        if not rows or len(indexed) != len(rows) or any(r.get('error') for r in rows):
            raise ValueError('Empty, duplicate or failed predictions')
        return indexed
    left, right = index(a), index(b)
    if left.keys() != right.keys():
        raise ValueError('Different sample IDs')
    paired = []
    for key, row in left.items():
        other = right[key]
        for field in ('reference', 'audio_path', 'audio_sha256'):
            if not row.get(field) or row[field] != other.get(field):
                raise ValueError(f'Mismatched {field}: {key}')
        paired.append(other)
    return paired


def compare(base_dir, small_dir):
    from benchmarks.metrics import corpus_error_rates, paired_bootstrap_test
    summaries, predictions = [], []
    for directory in (base_dir, small_dir):
        summaries.append(json.loads((directory / 'summary.json').read_text()))
        predictions.append([json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()])
    a, b = summaries
    for field in ('protocol', 'manifest_sha256', 'environment', 'source'):
        if not a.get(field) or a[field] != b.get(field):
            raise ValueError(f'Incomparable {field}')
    for summary, rows in zip(summaries, predictions):
        if summary['failures'] or summary['samples'] != len(rows):
            raise ValueError('Failed or incomplete run')
        metrics = corpus_error_rates([r['reference'] for r in rows], [r['hypothesis'] for r in rows])
        if any(abs(metrics[k] - summary[k]) > 1e-10 for k in metrics):
            raise ValueError('Summary metrics disagree with predictions')
    paired = validate_pairs(*predictions)
    bootstrap = paired_bootstrap_test([r['reference'] for r in predictions[0]],
                                     [r['hypothesis'] for r in predictions[0]],
                                     [r['hypothesis'] for r in paired])
    quality = {k: {'base': a[k], 'small': b[k], 'absolute_B_minus_A': b[k] - a[k],
                   'relative_B_minus_A': (b[k] - a[k]) / a[k] if a[k] else None}
               for k in ('wer', 'cer')}
    resources = {k: {'base': a[k], 'small': b[k],
                      'small_over_base': b[k] / a[k] if a[k] and b[k] is not None else None}
                 for k in ('rtf', 'model_load_seconds', 'latency_median_seconds', 'latency_p95_seconds',
                           'peak_process_rss_bytes', 'peak_cuda_memory_bytes', 'peak_cuda_reserved_bytes')}
    return {'quality': quality, 'resources': resources, 'paired_wer_bootstrap_percentage_points': bootstrap,
            'pilot': bool(a['pilot'] or b['pilot']), 'automatic_promotion': False,
            'decision': 'Manual review required; runtime default unchanged by this comparison'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, required=True)
    parser.add_argument('--small-checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--disjoint-from', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--warmup', type=int, default=0)
    args = parser.parse_args()
    from benchmarks.run import read_manifest
    rows = read_manifest(args.manifest, None)
    other = read_manifest(args.disjoint_from, None)
    if ({r['id'] for r in rows} & {r['id'] for r in other}
            or {r['sha256'] for r in rows} & {r['sha256'] for r in other}):
        raise ValueError('Overlapping IDs or audio hashes')
    for checkpoint in (args.base_checkpoint, args.small_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    directories = []
    for label, checkpoint in [('base', args.base_checkpoint), ('small', args.small_checkpoint)]:
        root = args.output_dir / label
        command = [sys.executable, '-m', 'benchmarks.run', '--manifest', str(args.manifest),
                   '--checkpoint', str(checkpoint), '--name', label, '--output-dir', str(root),
                   '--warmup', str(args.warmup)]
        if args.limit is not None:
            command += ['--limit', str(args.limit)]
        subprocess.run(command, check=True)
        directories.append(next(root.iterdir()))
    report = compare(*directories)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2))
    (args.output_dir / 'comparison.md').write_text('# Base / small comparison\n\n'
        'B = small; A = base. No automatic promotion. Bootstrap units: percentage points.\n\n```json\n'
        + json.dumps(report, indent=2) + '\n```\n')


if __name__ == '__main__':
    main()
