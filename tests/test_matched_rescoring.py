"""Unit tests for matched rescoring analysis, switch metrics, and semantic tracking."""

import unittest
from tools.analyze_matched_rescoring import (
    classify_switch,
    select_candidate_generic,
    track_critical_words,
)


class TestMatchedRescoring(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            {
                "text": "tôi đi học",
                "normalized_text": "tôi đi học",
                "avg_logprob": -0.10,
                "gpt2_score": -4.0,
                "phobert_score": -3.5,
                "rank": 1,
            },
            {
                "text": "tôi không đi học",
                "normalized_text": "tôi không đi học",
                "avg_logprob": -0.25,
                "gpt2_score": -2.0,
                "phobert_score": -1.5,
                "rank": 2,
            },
            {
                "text": "tôi đi",
                "normalized_text": "tôi đi",
                "avg_logprob": -0.40,
                "gpt2_score": -1.0,
                "phobert_score": -1.0,
                "rank": 3,
            },
        ]

    def test_candidate_selection_ungated(self) -> None:
        # With high alpha and no acoustic gate (gamma = -inf), rank 3 (high LM score) or rank 2 could win
        # Rank 1 score: -0.10 + 0.2*(-4.0) = -0.90
        # Rank 2 score: -0.25 + 0.2*(-2.0) = -0.65 (diff avg_lp: -0.15)
        # Rank 3 score: -0.40 + 0.2*(-1.0) = -0.60 (diff avg_lp: -0.30)
        cand, rank = select_candidate_generic(
            self.candidates,
            score_key="gpt2_score",
            alpha=0.20,
            beta=0.0,
            gamma=-float("inf"),
        )
        self.assertEqual(rank, 3)
        self.assertEqual(cand["normalized_text"], "tôi đi")

    def test_candidate_selection_gated(self) -> None:
        # With gamma = -0.20, rank 3 has diff -0.30 < -0.20 (rejected by acoustic gate!)
        # Rank 2 has diff -0.15 >= -0.20 (allowed by acoustic gate)
        cand, rank = select_candidate_generic(
            self.candidates,
            score_key="gpt2_score",
            alpha=0.20,
            beta=0.0,
            gamma=-0.20,
        )
        self.assertEqual(rank, 2)
        self.assertEqual(cand["normalized_text"], "tôi không đi học")

    def test_classify_switch(self) -> None:
        ref = "tôi không đi học"
        top1 = "tôi đi học"       # errors = 1 (deletion of 'không')
        better = "tôi không đi học" # errors = 0
        worse = "anh đi"          # errors = 3
        tie = "tôi đã đi học"     # errors = 1 (sub 'đã' for 'không')

        # Win
        res_win = classify_switch(ref, top1, better)
        self.assertTrue(res_win["is_switch"])
        self.assertEqual(res_win["outcome"], "WIN")
        self.assertEqual(res_win["delta_errors"], -1)

        # Loss
        res_loss = classify_switch(ref, top1, worse)
        self.assertTrue(res_loss["is_switch"])
        self.assertEqual(res_loss["outcome"], "LOSS")
        self.assertGreater(res_loss["delta_errors"], 0)

        # Tie
        res_tie = classify_switch(ref, top1, tie)
        self.assertTrue(res_tie["is_switch"])
        self.assertEqual(res_tie["outcome"], "TIE")
        self.assertEqual(res_tie["delta_errors"], 0)

        # No switch
        res_none = classify_switch(ref, top1, top1)
        self.assertFalse(res_none["is_switch"])
        self.assertEqual(res_none["outcome"], "NO_SWITCH")

    def test_track_critical_words(self) -> None:
        ref = "tôi không có hai chiếc xe"
        top1 = "tôi không có 2 chiếc xe"  # contains negation 'không', contains number '2'
        cand_short = "tôi có xe"          # dropped 'không' and dropped number

        res = track_critical_words(ref, top1, cand_short)
        self.assertEqual(res["negation"]["ref_count"], 1)
        self.assertEqual(res["negation"]["top1_count"], 1)
        self.assertEqual(res["negation"]["rescored_count"], 0)
        self.assertEqual(res["negation"]["false_deletions"], 1)
        self.assertEqual(res["negation"]["recovered"], 0)

        self.assertEqual(res["numbers"]["ref_count"], 1)
        self.assertEqual(res["numbers"]["top1_count"], 1)
        self.assertEqual(res["numbers"]["rescored_count"], 0)
        self.assertEqual(res["numbers"]["false_deletions"], 1)


if __name__ == "__main__":
    unittest.main()
