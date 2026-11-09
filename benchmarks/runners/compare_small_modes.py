"""Frozen small public FAST (A) versus RESCORE (B), fresh sequential processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from benchmarks.runners.compare_model_sizes import validate_pairs
from benchmarks.runners.compare_vietmed_modes import summarize, validate, save, PROTOCOL
from benchmarks.metrics import corpus_error_rates, paired_bootstrap_test


def compare(directories, expected):
    summaries, cells = [], []
    for mode, directory in zip(('fast', 'rescore'), directories, strict=True):
        summary = json.loads((directory / 'summary.json').read_text())
        rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text().splitlines()]
        if summary['mode'] != mode or summary['failures'] or summary['samples'] != len(expected):
            raise ValueError('Failed or incomplete run')
        if summary['protocol']['decode'] != dict(public_api=True, fast=PROTOCOL['fast'],
                rescoring=PROTOCOL['rescoring'] if mode == 'rescore' else None):
            raise ValueError('Not frozen public decoding')
        validate(rows, expected)
        rates = corpus_error_rates([r['reference'] for r in rows], [r['hypothesis'] for r in rows])
        if any(abs(rates[k] - summary[k]) > 1e-10 for k in rates):
            raise ValueError('Metrics disagree')
        summaries.append(summary)
        cells.append(rows)
    a, b = summaries
    for field in ('manifest_sha256', 'checkpoint_sha256', 'environment', 'source', 'pilot'):
        if a[field] != b[field]:
            raise ValueError('Incomparable ' + field)
    protocols = [{k: v for k, v in s['protocol'].items() if k not in ('mode', 'decode', 'rescore')} for s in summaries]
    if protocols[0] != protocols[1]:
        raise ValueError('Incomparable protocol')
    paired = validate_pairs(*cells)
    gates = {key: 0 for key in ('single_candidate', 'acoustic_gate', 'actual_lm_scored', 'winner_changed')}
    for row in paired:
        for key in gates:
            gates[key] += bool(row['diagnostics'][key])
    return dict(a=summarize(cells[0]), b=summarize(paired), resources=dict(zip(('fast', 'rescore'), summaries)),
                paired_wer_bootstrap_percentage_points=paired_bootstrap_test(
                    [r['reference'] for r in cells[0]], [r['hypothesis'] for r in cells[0]],
                    [r['hypothesis'] for r in paired]), gate_counts=gates,
                automatic_promotion=False, pilot=a['pilot'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/phowhisper-small.pt'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--disjoint-from', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    from benchmarks.run import read_manifest, file_sha256, validate_public_audio
    rows = read_manifest(args.manifest, None)
    dev = read_manifest(args.disjoint_from, None)
    if args.limit is not None and not 0 < args.limit <= len(rows):
        parser.error('limit must be 1..manifest size')
    if ({r['id'] for r in rows} & {r['id'] for r in dev}
            or {r['sha256'] for r in rows} & {r['sha256'] for r in dev}):
        raise ValueError('Overlapping IDs or audio hashes')
    metadata_path = Path(str(args.checkpoint) + '.provenance.json')
    metadata = json.loads(metadata_path.read_text())
    if (metadata['resolved_revision'] != 'a86b604c346caf7148c37512eafe783a16420adb'
            or metadata['native_sha256'] != file_sha256(args.checkpoint)):
        raise ValueError('Small checkpoint provenance mismatch')
    expected = rows[:args.limit] if args.limit else rows
    validate_public_audio([r['audio_path'] for r in expected] + [dev[0]['audio_path']])
    paths = [args.manifest, args.disjoint_from, args.checkpoint, metadata_path,
             Path('benchmarks/run.py'), Path('benchmarks/metrics.py'),
             Path('benchmarks/runners/compare_small_modes.py'),
             Path('benchmarks/runners/compare_vietmed_modes.py'),
             Path('src/phowhisper/pipeline.py'), Path('src/phowhisper/rescorer.py')]
    paths += [Path(r['audio_path']) for r in expected + [dev[0]]]
    before = {str(p): file_sha256(p) for p in paths}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='blocked', valid_complete=False, pilot=args.limit is not None,
                  automatic_promotion=False, before=before, checkpoint_metadata=metadata,
                  policy='One excluded dev[0] warmup, same path both modes; frozen defaults; no tuning')
    try:
        directories = []
        for mode in ('fast', 'rescore'):
            root = args.output_dir / mode
            command = [sys.executable, '-m', 'benchmarks.run', '--mode', mode,
                       '--checkpoint', str(args.checkpoint), '--manifest', str(args.manifest),
                       '--output-dir', str(root), '--name', mode, '--warmup', '1',
                       '--warmup-audio', dev[0]['audio_path']]
            if args.limit:
                command += ['--limit', str(args.limit)]
            subprocess.run(command, check=True)
            children = list(root.iterdir())
            if len(children) != 1:
                raise ValueError('Ambiguous run directory')
            directories.append(children[0])
        comparison = compare(directories, expected)
        save(args.output_dir / 'comparison.json', comparison)
        report.update(status='completed', valid_complete=args.limit is None,
                      samples=len(expected), failures=0)
    except Exception as exc:
        report['blocker'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        try:
            report['after'] = {str(p): file_sha256(p) for p in paths}
            if report['after'] != before:
                report.update(status='blocked', valid_complete=False, blocker='Input/source mutation')
                raise ValueError(report['blocker'])
        finally:
            save(args.output_dir / 'report.json', report)


if __name__ == '__main__':
    main()
