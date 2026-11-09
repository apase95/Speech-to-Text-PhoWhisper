import unittest
from unittest.mock import Mock, patch
import logging
import sys
from benchmarks.runners import compare_vietmed_modes as runner


class Smoke(unittest.TestCase):
    def test_pinned_offline_loader(self):
        transformers = Mock()
        with patch.dict(sys.modules, transformers=transformers):
            runner.pinned_loader()
        for loader in (transformers.AutoTokenizer, transformers.AutoModelForCausalLM):
            self.assertEqual(loader.from_pretrained.call_args.kwargs,
                             dict(revision=runner.LM_REVISION, local_files_only=True))
        transformers.AutoModelForCausalLM.from_pretrained.return_value.float.return_value.to.assert_called_once_with('cuda')

    def test_gate_diagnostics_and_warning(self):
        d = runner.Diagnostics()
        d.candidates = [dict(text='xin', avg_logprob=0), dict(text='sai')]
        self.assertTrue(d.finish('xin')['acoustic_gate'])
        d.candidates[0]['avg_logprob'] = -1
        with self.assertRaises(ValueError):
            d.finish('xin')
        d.lm_calls = 1
        d.candidates[0]['lm_logprob_per_token'] = -2
        self.assertTrue(d.finish('sai')['winner_changed'])
        with self.assertRaises(RuntimeError):
            runner.FailWarnings().emit(logging.LogRecord('test', logging.WARNING, '', 0, 'fallback', (), None))

    def test_pairing_splits_and_saved_metrics(self):
        rows = [dict(id=s, split=s, reference='xin', hypothesis='xin',
                     audio_path=s, audio_sha256=s, sha256=s, audio_seconds=2,
                     inference_seconds=.1) for s in ('train', 'dev', 'test')]
        import json
        saved = json.loads(json.dumps(rows))
        comparison = runner.compare({c: saved for c in runner.CELLS}, rows)
        self.assertEqual(set(comparison['base-fast->small-fast']), {'all100', 'train', 'dev', 'test'})
        with self.assertRaises(ValueError):
            runner.validate([dict(rows[0], audio_sha256='wrong')], rows[:1])

    def test_four_cells_and_roundtrip(self):
        self.assertEqual(len(runner.CELLS), 4)
        row = dict(id='a', split='test', reference='xin', hypothesis='xin',
                   audio_path='a', audio_sha256='hash', audio_seconds=2,
                   inference_seconds=.5)
        self.assertEqual(runner.summarize([row])['wer'], 0)
        self.assertEqual(runner.summarize([row])['word_edits']['hits'], 1)
        runner.validate([row], [dict(row, sha256='hash')])

    def test_failure_and_partial_rejected(self):
        with self.assertRaises(ValueError):
            runner.validate([], [dict(id='a')])
        with self.assertRaises(ValueError):
            runner.compare({}, [])


if __name__ == '__main__':
    unittest.main()
