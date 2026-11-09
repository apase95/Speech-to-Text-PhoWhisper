import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import json
from kaggle_training.train import normalize, read_metadata
from kaggle_training.train import prepare, Collator


class FakeTensor:
    def __init__(self, rows):
        self.rows = rows

    def ne(self, value):
        return FakeTensor([[x != value for x in row] for row in self.rows])

    def masked_fill(self, mask, value):
        return FakeTensor([[value if flag else x for x, flag in zip(row, flags)]
                           for row, flags in zip(self.rows, mask.rows)])


class FakeTokenizer:
    def pad(self, features, return_tensors):
        assert return_tensors == 'pt'
        rows = [f['input_ids'] for f in features]
        width = max(map(len, rows))
        return SimpleNamespace(
            input_ids=FakeTensor([r + [0] * (width - len(r)) for r in rows]),
            attention_mask=FakeTensor([[1] * len(r) + [0] * (width - len(r)) for r in rows]))


class FakeExtractor:
    def pad(self, features, return_tensors):
        assert return_tensors == 'pt'
        return {'input_features': [f['input_features'] for f in features]}


class Smoke(unittest.TestCase):
    def test_collator_mixed_prefix_and_padding(self):
        processor = SimpleNamespace(feature_extractor=FakeExtractor(), tokenizer=FakeTokenizer())
        batch = Collator(processor, 99)([
            {'input_features': 'a', 'labels': [99, 4, 5]},
            {'input_features': 'b', 'labels': [6]}])
        self.assertEqual(batch['input_features'], ['a', 'b'])
        self.assertEqual(batch['labels'].rows, [[4, 5], [6, -100]])

    def test_collator_empty_labels_rejected(self):
        processor = SimpleNamespace(feature_extractor=FakeExtractor(), tokenizer=FakeTokenizer())
        for labels in ([], [99]):
            with self.subTest(labels=labels), self.assertRaisesRegex(ValueError, 'Empty decoder labels'):
                Collator(processor, 99)([{'input_features': 'a', 'labels': labels}])

    def test_happy(self):
        self.assertEqual(normalize('  XIN, chào! '), 'xin chào')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'wavs').mkdir()
            (root / 'wavs/audio1251.wav').touch()
            (root / 'metadata_train.csv').write_text('audio_file|text|speaker_name\naudio1251.wav|Xin chào|speaker1\n')
            self.assertEqual(len(read_metadata(root, 'train')), 1)

    def test_duplicate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'wavs').mkdir()
            (root / 'wavs/audio1251.wav').touch()
            (root / 'metadata_train.csv').write_text('audio_file|text|speaker_name\naudio1251.wav|a|s\naudio1251.wav|b|s\n')
            with self.assertRaises(ValueError):
                read_metadata(root, 'train')


    def test_exclusion_and_speaker_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'wavs').mkdir()
            for number in (1, 1251, 1252, 2000):
                (root / f'wavs/audio{number}.wav').write_bytes(str(number).encode())
            (root / 'metadata_train.csv').write_text('audio_file|text|speaker_name\naudio1.wav|x|a\naudio1251.wav|y|a\naudio1252.wav|z|b\n')
            (root / 'metadata_test.csv').write_text('audio_file|text|speaker_name\naudio2000.wav|t|c\n')
            prepare(SimpleNamespace(root=root, output=root, exclude_manifest=[], seed=42,
                validation_fraction=.5, train_cap=0, validation_cap=0))
            train = json.loads((root / 'train.json').read_text())
            validation = json.loads((root / 'validation.json').read_text())
            self.assertEqual(len(train) + len(validation), 2)
            self.assertFalse({r['speaker'] for r in train} & {r['speaker'] for r in validation})

    def test_unsafe_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata_train.csv').write_text('audio_file|text|speaker_name\n../audio1251.wav|a|s\n')
            with self.assertRaises(ValueError):
                read_metadata(root, 'train')


if __name__ == '__main__':
    unittest.main()
