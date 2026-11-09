"""Structural relocation contracts, without models or ignored output dependencies."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    'unit': ('app', 'benchmark', 'conversion', 'reference_normalization'),
    'integration': ('dynamic_gating', 'gpt2_prefix_smoke'),
    'regressions': (
        'fleurs_general', 'fleurs_holdout', 'fleurs_source_correction',
        'general_cached_diagnosis', 'gpt2_error_audit', 'gpt2_optimization',
        'holdout_deletion_audit', 'human_reference_review', 'long_audio_smoke',
        'matched_rescoring', 'nbest_ranker_smoke', 'numeric_metrics',
        'personal_vietmed_compare', 'review_package', 'section23_smoke',
        'vietmed100', 'vietmed100_train', 'vietmed_local'),
}
REPORTS = (
    'benchmark-results-fleurs60', 'fleurs-general60-benchmark',
    'gpt2-error-audit-holdout', 'gpt2-optimization-fleurs60',
    'human-review-benchmark-fleurs60', 'long-audio-benchmark',
    'numeric-metric-audit-fleurs60', 'personal-audio-vietmed100-comparison',
    'reference-normalization-fleurs60',
)


class RelocationSmoke(unittest.TestCase):
    def test_grouped_tests_remain_discoverable_without_root_duplicates(self):
        for group, names in GROUPS.items():
            self.assertTrue((ROOT / 'tests' / group / '__init__.py').is_file())
            for name in names:
                filename = f'test_{name}.py'
                with self.subTest(group=group, filename=filename):
                    self.assertTrue((ROOT / 'tests' / group / filename).is_file())
                    self.assertFalse((ROOT / 'tests' / filename).exists())

    def test_reports_and_relative_links_survive_relocation(self):
        for name in REPORTS:
            report = ROOT / 'reports' / f'{name}.md'
            with self.subTest(report=name):
                self.assertTrue(report.is_file())
                self.assertFalse((ROOT / 'docs' / report.name).exists())
                for target in re.findall(r'\]\(([^)]+)\)', report.read_text()):
                    if ':' not in target and not target.startswith('#'):
                        self.assertTrue((report.parent / target.split('#')[0]).exists(),
                                        f'{report.name}: broken relative link {target}')


if __name__ == '__main__':
    unittest.main()
