"""Isolated relocation contract; no model imports."""
import importlib
import json
from pathlib import Path
import unittest


class RelocationSmoke(unittest.TestCase):
    def test_canonical_training(self):
        module = importlib.import_module('training.kaggle.train')
        self.assertEqual(module.normalize(' XIN! '), 'xin')

    def test_legacy_local_is_same_module(self):
        old = importlib.import_module('tools.vietmed100_train')
        new = importlib.import_module('training.local.vietmed100_train')
        self.assertIs(old, new)
        with self.assertRaises(ValueError):
            new.teacher_tokens([], [1], 2)

    def test_launch_paths_and_diagnostic_sync(self):
        root = Path(__file__).resolve().parent / 'kaggle'
        for launch in ('launch-diagnostic', 'launch-diagnostic-gpu',
                       'launch-diagnostic-data', 'launch-pilot1000'):
            metadata = json.loads((root / launch / 'kernel-metadata.json').read_text())
            self.assertTrue((root / launch / metadata['code_file']).is_file())
        for stage in ('gpu', 'data'):
            self.assertEqual((root / f'diagnostic_{stage}_source.py').read_text(),
                             f"import os\nos.environ['KAGGLE_DIAGNOSTIC_STAGE'] = {stage!r}\n"
                             + (root / 'diagnostic.py').read_text())


if __name__ == '__main__':
    unittest.main()
