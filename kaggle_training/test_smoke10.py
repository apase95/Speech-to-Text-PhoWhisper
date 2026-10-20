import unittest
from pathlib import Path
from kaggle_training.smoke10 import select, metadata
from kaggle_training.build_smoke10_notebook import build
import tempfile


class Smoke10(unittest.TestCase):
    def test_selected_audio_audit_and_duplicate_bytes(self):
        from kaggle_training.smoke10 import audit
        from unittest.mock import Mock, patch
        from types import SimpleNamespace
        import sys
        np = Mock()
        np.isfinite.return_value.all.return_value = True
        sf = Mock()
        sf.info.return_value = SimpleNamespace(frames=16000, samplerate=16000, duration=1)
        sf.read.return_value = (object(), 16000)
        processor = SimpleNamespace(tokenizer=lambda text: SimpleNamespace(input_ids=[1, 2]))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, numpy=np, soundfile=sf):
            root = Path(directory)
            paths = [root / 'a.wav', root / 'b.wav']
            for path in paths: path.write_bytes(b'audio')
            rows = [dict(audio_path=str(p), text='xin') for p in paths]
            audit(rows[:1], root, processor)
            self.assertEqual(len(rows[0]['sha256']), 64)
            with self.assertRaisesRegex(ValueError, 'Duplicate selected'):
                audit(rows, root, processor)
            np.isfinite.return_value.all.return_value = False
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                audit(rows[:1], root, processor)

    def test_happy_and_candidate(self):
        rows = [dict(id=f'audio{n}.wav', speaker=f's{n % 4}', text='xin')
                for n in range(1251, 1271)]
        train, val = select(rows, {'audio1251.wav'})
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse({r['speaker'] for r in train} & {r['speaker'] for r in val})
        self.assertNotIn('audio1251.wav', {r['id'] for r in train + val})
        self.assertEqual(select(rows, {'audio1251.wav'}), (train, val))
        notebook = build(Path(__file__).resolve().parents[1])
        self.assertEqual(len(notebook['cells']), 7)
        for cell in notebook['cells']:
            compile(''.join(cell['source']), '<cell>', 'exec')
        source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
        for text in ('fp32', 'PYTHONUNBUFFERED', '--require-finite-training', 'heartbeat', 'resolved_revision'):
            self.assertIn(text, source)

    def test_conflict_and_insufficient(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata_train.csv').write_text('audio_file|text|speaker_name\naudio1251.wav|a| S \naudio1251.wav|b|s\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                metadata(root, 'train')
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select([dict(id='audio1251.wav', speaker='s')], set())


if __name__ == '__main__':
    unittest.main()
