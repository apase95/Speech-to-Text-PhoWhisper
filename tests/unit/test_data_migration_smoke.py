"""Isolated canonical data-script and legacy identity contracts."""
import importlib
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
NAMES = ('correct_fleurs_sources', 'curate_fleurs_general',
         'download_viet_bud500_subset', 'human_reference_review',
         'normalize_benchmark_references')


class DataMigrationSmoke(unittest.TestCase):
    def test_canonical_imports_and_legacy_identity(self):
        for name in NAMES:
            with self.subTest(name=name):
                canonical = importlib.import_module('scripts.data.' + name)
                self.assertIs(importlib.import_module('tools.' + name), canonical)
                self.assertEqual(Path(canonical.__file__).parent, ROOT / 'scripts/data')

    def test_direct_scripts_outside_repository(self):
        for prefix in ('scripts/data', 'tools'):
            result = subprocess.run(
                [sys.executable, str(ROOT / prefix / 'curate_fleurs_general.py'), '--help'],
                cwd='/tmp', capture_output=True, text=True, timeout=2)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('inventory', result.stdout)


if __name__ == '__main__':
    unittest.main()
