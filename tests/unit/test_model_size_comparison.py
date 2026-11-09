import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock
from benchmarks.runners.compare_model_sizes import validate_pairs, compare


class ComparisonSmoke(unittest.TestCase):
    def test_pairs_by_id(self):
        rows = [dict(id='a', reference='xin', audio_path='a.wav', audio_sha256='abc', error=None)]
        self.assertEqual(validate_pairs(rows, rows), rows)

    def test_rejects_changed_audio(self):
        row = dict(id='a', reference='xin', audio_path='a.wav', audio_sha256='abc', error=None)
        with self.assertRaises(ValueError):
            validate_pairs([row], [dict(row, audio_sha256='def')])

    def test_run_mocked_happy_and_preflight_hash_failure(self):
        from benchmarks import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / 'a.wav'
            audio.write_bytes(b'audio')
            checkpoint = root / 'model.pt'
            checkpoint.write_bytes(b'weights')
            manifest = root / 'manifest.csv'
            manifest.write_text(f'id,audio_path,reference,split,sha256\na,{audio},xin,test,{run.file_sha256(audio)}\n')
            model = Mock()
            model.transcribe.return_value = {'text': 'xin'}
            with patch('sys.argv', ['benchmark', '--manifest', str(manifest), '--output-dir', str(root / 'out')]), \
                    patch.object(run, 'get_model', return_value=model), \
                    patch.object(run, 'model_path', return_value=checkpoint), \
                    patch.object(run.torch.cuda, 'is_available', return_value=False), \
                    patch.object(run.sf, 'info', return_value=Mock(duration=2)), \
                    patch('builtins.print'):
                run.main()
            options = model.transcribe.call_args.kwargs
            self.assertEqual(options['temperature'], 0)
            self.assertIsNone(options['beam_size'])
            self.assertIsNone(options['best_of'])
            summary = json.loads(next((root / 'out').iterdir()).joinpath('summary.json').read_text())
            self.assertEqual(summary['name'], 'phowhisper-small')
            self.assertEqual(summary['word_edits']['hits'], 1)
            self.assertEqual(summary['wer'], 0)
            self.assertEqual(summary['rtf'], summary['total_inference_seconds'] / 2)
            self.assertIsNone(summary['peak_cuda_reserved_bytes'])
            audio.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                run.read_manifest(manifest, None)

    def test_bootstrap_report_and_failure_rejection(self):
        from benchmarks.metrics import corpus_error_rates
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / 'base', Path(tmp) / 'small']
            for directory, hypothesis in zip(dirs, ['sai', 'xin']):
                directory.mkdir()
                row = dict(id='a', reference='xin', hypothesis=hypothesis,
                           audio_path='a.wav', audio_sha256='abc', error=None)
                summary = dict(protocol={'temperature': 0}, manifest_sha256='hash',
                               environment={'device': 'cpu'}, source={'revision': 'test'},
                               samples=1, failures=0, pilot=False)
                summary.update(corpus_error_rates(['xin'], [hypothesis]))
                summary.update({k: 1 for k in ('rtf', 'model_load_seconds',
                    'latency_median_seconds', 'latency_p95_seconds', 'peak_process_rss_bytes',
                    'peak_cuda_memory_bytes', 'peak_cuda_reserved_bytes')})
                (directory / 'summary.json').write_text(json.dumps(summary))
                (directory / 'predictions.jsonl').write_text(json.dumps(row) + '\n')
            report = compare(*dirs)
            self.assertFalse(report['automatic_promotion'])
            self.assertEqual(report['paired_wer_bootstrap_percentage_points']['ci_upper'], -100)
            summary['failures'] = 1
            (dirs[1] / 'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                compare(*dirs)


if __name__ == '__main__':
    unittest.main()
