"""No decoding: exercise the installed Whisper ranker on synthetic candidates."""
import unittest

try:
    from whisper.decoding import MaximumLikelihoodRanker
except ImportError:
    MaximumLikelihoodRanker = None
from tools.nbest_ranking import rank_candidates


@unittest.skipIf(MaximumLikelihoodRanker is None, 'requires benchmark environment Whisper')
class RankerSmoke(unittest.TestCase):
    def test_default_length_not_eos_average(self):
        candidates = [dict(tokens=[1], sum_logprob=-1.0),
                      dict(tokens=[1, 2, 3], sum_logprob=-2.4)]
        ranked = rank_candidates(candidates, MaximumLikelihoodRanker(None))
        self.assertIs(ranked[0], candidates[1])
        self.assertEqual(len(candidates), 2)

    def test_ties_preserve_decoder_order(self):
        candidates = [dict(tokens=[1], sum_logprob=-1.0),
                      dict(tokens=[2], sum_logprob=-1.0)]
        self.assertEqual(rank_candidates(candidates, MaximumLikelihoodRanker(None)), candidates)
        self.assertEqual(rank_candidates([], MaximumLikelihoodRanker(None)), [])
