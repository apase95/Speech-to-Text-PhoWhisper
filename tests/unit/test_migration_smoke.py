"""Isolated structural contract; no model, GPU or optional runtime imports."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


class MigrationSmoke(unittest.TestCase):
    def test_canonical_core_is_independent(self):
        for name in ('pipeline', 'rescorer', 'audio_utils'):
            tree = ast.parse((ROOT / 'src/phowhisper' / (name + '.py')).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse((node.module or '').startswith(('benchmarks', 'tools', 'src.')))

    def test_old_and_new_help_without_runtime_dependencies(self):
        for module in ('src.inference', 'phowhisper.cli'):
            result = subprocess.run([sys.executable, '-S', '-m', module, '--help'],
                cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(ROOT / 'src')},
                capture_output=True, text=True, timeout=2)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--audio', result.stdout)


if __name__ == '__main__':
    unittest.main()
