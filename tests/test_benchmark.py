import unittest

from benchmarks.metrics import corpus_error_rates, normalize_vietnamese
from tools.download_viet_bud500_subset import numeric_audio_id


class BenchmarkTests(unittest.TestCase):
    def test_normalizer_preserves_vietnamese_diacritics(self):
        self.assertEqual(normalize_vietnamese(" Xin, CHÀO! "), "xin chào")

    def test_corpus_metrics_are_zero_for_equivalent_text(self):
        rates = corpus_error_rates(["Xin chào!"], ["xin chào"])
        self.assertEqual(rates, {"wer": 0.0, "cer": 0.0})

    def test_numeric_audio_id_is_not_lexicographic(self):
        self.assertEqual(numeric_audio_id("wavs/audio1000.wav"), 1000)


if __name__ == "__main__":
    unittest.main()
