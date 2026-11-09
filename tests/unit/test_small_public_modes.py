import unittest
from unittest.mock import Mock, patch
from benchmarks import run
from benchmarks.runners import compare_small_modes as comparison
from benchmarks.runners import compare_vietmed_modes as helpers
import json
import tempfile
from pathlib import Path


class PublicSmoke(unittest.TestCase):
    def test_fast_public_defaults_no_lm(self):
        from phowhisper import pipeline
        with patch.object(pipeline, 'transcribe', return_value='xin') as call:
            with run.PublicMode('fast') as mode:
                self.assertEqual(mode.transcribe('a.wav'), 'xin')
            call.assert_called_once_with('a.wav', rescore=False)

    def test_long_audio_rejected_before_inference(self):
        with patch.object(run.sf, 'info', return_value=Mock(duration=30.01)):
            with self.assertRaises(ValueError):
                run.validate_public_audio(['long.wav'])

    def test_rescore_pinned_and_warning_fails_closed(self):
        from phowhisper import pipeline, rescorer
        with patch.object(helpers, 'pinned_loader', return_value=(Mock(), Mock())) as loader, \
                patch.object(run.torch.cuda, 'synchronize'), \
                patch.object(pipeline, 'transcribe', side_effect=lambda *a, **k: rescorer.logger.warning('fallback')):
            with run.PublicMode('rescore') as mode:
                loader.assert_called_once_with()
                with self.assertRaises(RuntimeError):
                    mode.transcribe('a.wav')

    def test_comparison_happy_failed_and_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / m for m in ('fast', 'rescore')]
            expected = [dict(id='a', reference='xin', split='test', audio_path='a.wav', sha256='hash')]
            row = dict(expected[0], audio_sha256='hash', hypothesis='xin', audio_seconds=2,
                       inference_seconds=.1, diagnostics=dict(single_candidate=True, acoustic_gate=False,
                       actual_lm_scored=False, winner_changed=False))
            for mode, directory in zip(('fast', 'rescore'), dirs):
                directory.mkdir()
                summary = dict(mode=mode, failures=0, samples=1, wer=0, cer=0,
                               manifest_sha256='m', checkpoint_sha256='c', environment={'gpu':'mock'},
                               source={'hash':'s'}, pilot=True, protocol=dict(seed=42, mode=mode))
                summary['protocol']['decode'] = dict(public_api=True, fast=helpers.PROTOCOL['fast'],
                    rescoring=helpers.PROTOCOL['rescoring'] if mode == 'rescore' else None)
                (directory / 'summary.json').write_text(json.dumps(summary))
                (directory / 'predictions.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(comparison.compare(dirs, expected)['gate_counts']['single_candidate'], 1)
            for bad in ([dict(row, error='LM failure')], []):
                (dirs[1] / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.assertRaises(ValueError):
                    comparison.compare(dirs, expected)


if __name__ == '__main__':
    unittest.main()
