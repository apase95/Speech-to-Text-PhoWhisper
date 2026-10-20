import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

from kaggle_training import build_phowhisper3_notebook as builder


class SetupSmoke(unittest.TestCase):
    def test_installed_dependency_set(self):
        for absent, expected in (
            ({'jiwer', 'rapidfuzz'}, ['jiwer==3.0.5', 'rapidfuzz>=3,<4']),
            ({'rapidfuzz'}, ['rapidfuzz>=3,<4']),
            (set(), []),
        ):
            with self.subTest(absent=absent), patch.object(
                builder.importlib.util, 'find_spec',
                side_effect=lambda module: None if module in absent else object(),
            ), patch.object(builder.importlib, 'invalidate_caches'), patch(
                'importlib.metadata.version', return_value='test-version',
            ):
                run = Mock()
                report = {}
                namespace = {
                    'missing_dependencies': builder.missing_dependencies,
                    'sys': builder.sys, 'importlib': builder.importlib,
                    'run': run, 'REPORT': report, 'save_report': Mock(),
                    'stage': lambda function: function(),
                    '__import__': Mock(return_value=object()),
                }
                exec(builder.DEPENDENCIES, namespace)
                self.assertEqual(report.get('dependencies_installed', []), expected)
                if expected:
                    run.assert_called_once_with(
                        [builder.sys.executable, '-u', '-m', 'pip', 'install', '--no-deps', *expected],
                        timeout=180,
                    )
                else:
                    run.assert_not_called()
                self.assertIn('jiwer', report['versions'])
                self.assertIn('rapidfuzz', report['versions'])

    def test_artifacts_and_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            builder.build(Path(directory))
            meta = json.loads((Path(directory) / 'kernel-metadata.json').read_text())
            self.assertEqual(meta['id'], 'hngthiduy/phowhisper3')
            self.assertTrue(meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'])
            for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources'):
                self.assertEqual(meta[key], [])
            notebook = json.loads((Path(directory) / 'phowhisper3.ipynb').read_text())
            for cell in notebook['cells']:
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), '<cell>', 'exec')
            source = '\n'.join(''.join(c['source']) for c in notebook['cells'])
            self.assertNotIn('pip install -r', source)
            self.assertNotIn('numpy==1.26', source)

    def test_missing_torch_and_stream(self):
        with patch.object(builder.importlib.util, 'find_spec', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'image'):
                builder.missing_dependencies()
        with tempfile.TemporaryDirectory() as directory:
            namespace = {'__name__': 'smoke'}
            # Exercise the actual generated streaming helper, without Kaggle paths.
            source = builder.BOOTSTRAP.replace("Path('/kaggle/working/outputs/benchmarks')", repr(Path(directory)))
            namespace['PosixPath'] = Path
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(source, namespace)
                namespace['run']([builder.sys.executable, '-u', '-c', "print('stream-ok', flush=True)"], timeout=2)
            self.assertIn('stream-ok', output.getvalue())
            self.assertIn('stream-ok', namespace['LOG'].read_text())
            with self.assertRaises(TimeoutError):
                namespace['run']([builder.sys.executable, '-u', '-c', 'import time; time.sleep(5)'], timeout=0.05)


if __name__ == '__main__':
    unittest.main()
