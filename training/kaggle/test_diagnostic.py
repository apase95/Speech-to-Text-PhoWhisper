import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kaggle_training import diagnostic as d


class DiagnosticTests(unittest.TestCase):
    def test_cpu_files_and_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(d.main(['--output', tmp, '--heartbeat-test-seconds', '0']), 0)
            result = json.loads((Path(tmp) / 'diagnostic_result.json').read_text())
            self.assertEqual(result['status'], 'success')
            events = [json.loads(x) for x in (Path(tmp) / 'diagnostic_events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['status'], 'success')
            self.assertIn('+00:00', events[0]['utc'])
            with patch('builtins.print') as printer:
                d.Reporter(Path(tmp)).emit('test')
                self.assertTrue(printer.call_args.kwargs['flush'])

    def test_metadata_counts_and_unsafe_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'wavs').mkdir()
            (root / 'wavs/audio1.wav').touch()
            for split in ('train', 'test'):
                (root / f'metadata_{split}.csv').write_text('audio_file|text|speaker_name\naudio1.wav|private transcript|private speaker\n')
            summary, sample = d.metadata(root)
            self.assertEqual(summary['train']['rows'], 1)
            self.assertEqual(summary['test']['speakers'], 1)
            self.assertEqual(sample, root / 'wavs/audio1.wav')
            self.assertNotIn('private', json.dumps(summary))
            for name in ('../audio1.wav', '/audio1.wav', 'wavs/../../audio1.wav'):
                with self.assertRaises(ValueError):
                    d.safe_audio(root, name)
            (root / 'wavs/audio2.wav').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):
                d.safe_audio(root, 'audio2.wav')

    def test_failure_and_subprocess_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(d, 'run_child', side_effect=RuntimeError('bounded failure')):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(d.main(['--stage', 'gpu', '--output', tmp]), 1)
            result = json.loads((Path(tmp) / 'diagnostic_result.json').read_text())
            self.assertEqual(result['status'], 'failure')
            self.assertIn('RuntimeError', result['traceback'])
            with patch.object(d.subprocess, 'run', side_effect=d.subprocess.TimeoutExpired('python', 120)):
                with self.assertRaises(d.subprocess.TimeoutExpired):
                    d.run_child('pass')


if __name__ == '__main__':
    unittest.main()
