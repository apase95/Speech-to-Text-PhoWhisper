import unittest
import json
from tools.gpt2_optimization import seal, verify
from tools.personal_vietmed_compare import chunks, decision, rtf, identity


class Smoke(unittest.TestCase):
    def test_profile_reproducibility(self):
        profile = seal({'alpha': .3, 'beta': -.02, 'beam': 15, 'prefix': 50256})
        verify(json.loads(json.dumps(profile)))
        profile['alpha'] = .4
        with self.assertRaises(ValueError):
            verify(profile)

    def test_happy(self):
        self.assertEqual(chunks(900000), [(0, 448000), (448000, 896000), (896000, 900000)])
        self.assertEqual(rtf(2, 8), .25)
        self.assertEqual(decision({'wer': .2, 'cer': .1}, {'wer': .1, 'cer': .09}), 'keep-production; long opt-in (no automatic promotion)')
        identity([{'id': 'a', 'split': 'test'}], [{'id': 'a', 'split': 'test'}])

    def test_edges(self):
        for n in (1, 448000, 448001):
            spans = chunks(n)
            self.assertEqual(sum(b-a for a,b in spans), n)
            self.assertEqual(spans[0][0], 0)
            self.assertEqual(spans[-1][1], n)
            self.assertTrue(all(a[1] == b[0] for a,b in zip(spans, spans[1:])))
        with self.assertRaises(ValueError):
            chunks(0)
        with self.assertRaises(ValueError):
            rtf(1, 0)
        with self.assertRaises(ValueError):
            identity([{'id': 'a', 'split': 'test'}], [{'id': 'b', 'split': 'test'}])
        self.assertIn('regression', decision({'wer': .2, 'cer': .1}, {'wer': .19, 'cer': .11}))
