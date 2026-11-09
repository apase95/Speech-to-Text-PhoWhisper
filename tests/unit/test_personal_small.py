import unittest
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from benchmarks.runners import compare_personal_small as runner


class Smoke(unittest.TestCase):
    def test_clean_clone_checkpoint_and_forwarding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root/'models/phowhisper-small.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'small fixture, not weights')
            sidecar = Path(str(checkpoint)+'.provenance.json')
            metadata = dict(origin='huggingface', model_id='vinai/PhoWhisper-small',
                resolved_revision='a86b604c346caf7148c37512eafe783a16420adb',
                native_sha256=runner.sha(checkpoint))
            runner.save(sidecar, metadata)
            audio = root/'input.aac'
            audio.write_bytes(b'mocked audio')
            lm = root/'lm'
            lm.mkdir()
            (lm/'config.json').write_text('{}')
            shared = SimpleNamespace(SOURCE=(), LM='pinned-lm', LM_REVISION='revision', LM_PATH=lm)
            output = root/'output'
            commands = []

            def external(command, **kwargs):
                if command[0] == 'ffprobe':
                    return SimpleNamespace(stdout='{}')
                commands.append(command)
                self.assertEqual(command[command.index('--checkpoint')+1], str(checkpoint.resolve()))
                mode = command[command.index('--worker')+1]
                name = 'small-fast' if mode == 'fast' else 'small-rescoring'
                rows = json.loads((output/'manifest.json').read_text())['segments']
                runner.save(output/(name+'.segments.json'), [dict(row, hypothesis='xin chào') for row in rows])
                runner.save(output/(name+'.summary.json'), dict(status='completed'))

            with patch.dict(sys.modules, {'benchmarks.runners.compare_vietmed_modes': shared,
                    'whisper': SimpleNamespace(load_audio=lambda _: np.zeros(3, dtype=np.float32))}), \
                    patch.object(runner.subprocess, 'run', side_effect=external), \
                    patch.dict(os.environ), patch.object(sys, 'argv', ['runner', '--audio', str(audio),
                        '--output-dir', str(output), '--checkpoint', str(checkpoint)]):
                runner.main()
            self.assertEqual(len(commands), 2)
            report = json.loads((output/'report.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['before'], report['after'])
            self.assertIn(str(checkpoint.resolve()), report['before'])
            self.assertIn(str(sidecar.resolve()), report['before'])

    def test_checkpoint_provenance_rejects_wrong_identity_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)/'small.pt'
            checkpoint.write_bytes(b'fixture')
            metadata = dict(origin='huggingface', model_id='vinai/PhoWhisper-small',
                resolved_revision='a86b604c346caf7148c37512eafe783a16420adb', native_sha256=runner.sha(checkpoint))
            sidecar = Path(str(checkpoint)+'.provenance.json')
            runner.save(sidecar, metadata)
            self.assertEqual(runner.validate_checkpoint(checkpoint), sidecar)
            for key, value in (('origin', 'HF'), ('origin', 'local'), ('model_id', 'vinai/PhoWhisper-base'),
                               ('resolved_revision', 'wrong'), ('native_sha256', 'wrong')):
                with self.subTest(key=key):
                    runner.save(sidecar, dict(metadata, **{key: value}))
                    with self.assertRaisesRegex(ValueError, 'Checkpoint provenance mismatch'):
                        runner.validate_checkpoint(checkpoint)

    def test_default_checkpoint_forwarded_to_worker(self):
        with patch.object(sys, 'argv', ['runner', '--audio', 'input.aac',
                '--output-dir', 'output', '--worker', 'fast']), \
                patch.dict(os.environ), patch.object(runner, 'worker') as worker:
            runner.main()
        self.assertEqual(worker.call_args.args[3], Path('models/phowhisper-small.pt').resolve())

    def test_worker_sets_selected_pipeline_path_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            audio = output/'input.aac'
            checkpoint = output/'chosen-small.pt'
            runner.save(output/'manifest.json', dict(audio_path=str(audio.resolve()), segments=[]))
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {'benchmarks.runners.compare_vietmed_modes': SimpleNamespace(),
                    'torch': fake_torch, 'phowhisper': SimpleNamespace(pipeline=SimpleNamespace())}), \
                    patch.dict(os.environ), patch.object(runner, 'validate_checkpoint') as validate:
                with self.assertRaisesRegex(RuntimeError, 'CUDA required'):
                    runner.worker('fast', output, audio, checkpoint)
                validate.assert_called_once_with(checkpoint)
                self.assertEqual(os.environ['PHOWHISPER_MODEL_PATH'], str(checkpoint.resolve()))

    def test_coverage_and_float_roundtrip(self):
        import tempfile
        from pathlib import Path
        import soundfile as sf
        audio = np.linspace(-.9, .9, 448003, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            rows = runner.write_segments(audio, Path(directory))
            restored = np.concatenate([sf.read(r['audio_path'], dtype='float32')[0] for r in rows])
        np.testing.assert_array_equal(audio, restored)
        self.assertEqual([(r['start_sample'], r['end_sample']) for r in rows], [(0, 448000), (448000, 448003)])
        self.assertEqual(runner.compare(rows, rows)['wer'], None)

    def test_invalid_audio_and_identity(self):
        for audio in (np.array([], dtype=np.float32), np.array([np.nan], dtype=np.float32), np.zeros((2, 2)), np.zeros(2, dtype=np.float64)):
            with self.assertRaises(ValueError):
                runner.validate_audio(audio)
        row = dict(start_sample=0, end_sample=1, sha256='a', frames=1, audio_path='a')
        with self.assertRaises(ValueError):
            runner.compare([row], [dict(row, sha256='b')])
        with self.assertRaises(ValueError):
            runner.compare([row], [])

    def test_join_all_chunks_without_reference(self):
        rows = [dict(hypothesis=t) for t in (' xin  chào ', '', ' cuối\n cùng ')]
        self.assertEqual(runner.join_transcript(rows), 'xin chào cuối cùng')


if __name__ == '__main__':
    unittest.main()
