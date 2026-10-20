import tempfile
import unittest
from pathlib import Path
from kaggle_training.build_pilot_source import build

from kaggle_training.pilot import metadata, select, epoch_steps


class PilotSmoke(unittest.TestCase):
    def test_progress_counts_and_failure(self):
        import json
        from kaggle_training.train import Progress
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'progress.json'
            progress = Progress(path)
            with progress.stage('audit'):
                progress.update('audit', processed=1, total=1200)
                self.assertEqual(json.loads(path.read_text())['processed'], 1)
            self.assertEqual(json.loads(path.read_text())['status'], 'done')
            self.assertEqual(json.loads(path.read_text())['last_progress']['processed'], 1)
            with self.assertRaises(ValueError), progress.stage('load'):
                raise ValueError('network unavailable')
            self.assertEqual(json.loads(path.read_text())['status'], 'failed')

    def test_optimizer_progress_warmup(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from kaggle_training.train import StepProgress
        progress = Mock()
        callback = StepProgress(progress, clock=iter(range(20)).__next__)
        state = SimpleNamespace(global_step=0, max_steps=125)
        callback.on_train_begin(None, state, None)
        for step in range(1, 8):
            state.global_step = step
            callback.on_step_end(None, state, None)
        values = progress.update.call_args.kwargs
        self.assertEqual(values['processed'], 7)
        self.assertEqual(values['total'], 125)
        self.assertEqual(values['eta_seconds'], 118)
        self.assertEqual(values['eta_scope'], 'training optimizer steps only; excludes evaluation/save/reporting')

    def test_layout_and_source_provenance(self):
        from kaggle_training.pilot import validate_layout, source_provenance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'dataset layout'):
                validate_layout(root)
            (root / 'wavs').mkdir()
            for split in ('train', 'test'):
                (root / f'metadata_{split}.csv').write_text('audio_file|text|speaker_name\n')
            validate_layout(root)
        provenance = source_provenance()
        self.assertIsNone(provenance['git_revision'])
        self.assertEqual(len(provenance['executed_source_sha256']), 64)
        self.assertIn('kaggle_training/train.py', provenance['executed_source_files'])

    def test_entry_dependency_policy(self):
        source = build(Path(__file__).resolve().parents[1])
        self.assertIn('/kaggle/input/datasets/tuandaodev/viet-bud500-formatted', source)
        self.assertIn("'jiwer==3.0.5'", source)
        self.assertIn('validate_layout(Path(args.root))', source)

    def test_balanced_disjoint_selection(self):
        rows = [dict(id=f'audio{n}.wav', speaker=f's{n % 4}', text='xin chào',
                     audio_path=f'/unused/audio{n}.wav') for n in range(1251, 1271)]
        train, validation, stats = select(rows, set(), 8, 4, 42)
        self.assertEqual((len(train), len(validation)), (8, 4))
        self.assertFalse({r['speaker'] for r in train} & {r['speaker'] for r in validation})
        self.assertEqual(select(rows, set(), 8, 4, 42), (train, validation, stats))
        self.assertEqual(epoch_steps(1000, 2, 4, 1), 125)
        full = [dict(id=f'audio{n}.wav', speaker=f's{n % 40}', text='xin')
                for n in range(1251, 5251)]
        train, validation, _ = select(full, set())
        self.assertEqual((len(train), len(validation)), (1000, 200))
        self.assertFalse({r['speaker'] for r in train} & {r['speaker'] for r in validation})

    def test_exclusions_empty_speakers_and_embedded_source(self):
        rows = [dict(id=f'audio{n}.wav', speaker=f's{n % 4}', text='xin')
                for n in range(1251, 1271)]
        rows += [dict(id='audio1.wav', speaker='s', text='x'),
                 dict(id='audio3000.wav', speaker='', text='x')]
        train, validation, _ = select(rows, {'audio1251.wav'}, 8, 4, 42)
        self.assertNotIn('audio1251.wav', {r['id'] for r in train + validation})
        self.assertTrue(all(r['speaker'] for r in train + validation))
        source = build(Path(__file__).resolve().parents[1])
        scope = {'__name__': 'artifact_validation'}
        exec(compile(source, 'artifact', 'exec'), scope)
        self.assertEqual(len(scope['PAYLOAD']['kaggle_training/exclusions/viet_bud500_1000.csv'].splitlines()), 1001)
        self.assertEqual(len(scope['PAYLOAD']['kaggle_training/exclusions/viet_bud500_dev_250.csv'].splitlines()), 251)
        for name, content in scope['PAYLOAD'].items():
            if name.endswith('.py'):
                compile(content, name, 'exec')

    def test_conflicting_metadata_and_insufficient_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata_train.csv').write_text(
                'audio_file|text|speaker_name\naudio1251.wav|a| S \naudio1251.wav|b|s\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                metadata(root, 'train')
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select([dict(id='audio1251.wav', speaker='s')], set(), 8, 4, 42)


if __name__ == '__main__':
    unittest.main()
