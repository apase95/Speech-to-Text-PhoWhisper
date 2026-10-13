import unittest

import numpy as np

from benchmarks.run import add_silence_padding
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

    def test_add_silence_padding_uses_whisper_sample_rate(self):
        audio = np.ones(160, dtype=np.float32)
        padded = add_silence_padding(audio, 200)

        self.assertEqual(len(padded), 160 + 2 * 3200)
        np.testing.assert_array_equal(padded[3200:-3200], audio)
        self.assertFalse(padded[:3200].any())
        self.assertFalse(padded[-3200:].any())

    def test_add_silence_padding_rejects_negative_duration(self):
        with self.assertRaises(ValueError):
            add_silence_padding(np.array([], dtype=np.float32), -1)


if __name__ == "__main__":
    unittest.main()
