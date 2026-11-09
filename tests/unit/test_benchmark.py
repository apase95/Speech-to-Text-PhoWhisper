import unittest

import numpy as np

from benchmarks.run import add_silence_padding
from benchmarks.metrics import (
    corpus_error_rates,
    harmonize_vietnamese_tone_placement,
    normalize_vietnamese,
    normalize_vietnamese_harmonized,
)
from src.audio_utils import apply_time_stretch
from tools.download_viet_bud500_subset import numeric_audio_id


class BenchmarkTests(unittest.TestCase):
    def test_normalizer_preserves_vietnamese_diacritics(self):
        self.assertEqual(normalize_vietnamese(" Xin, CHÀO! "), "xin chào")

    def test_harmonize_tone_placement(self):
        self.assertEqual(harmonize_vietnamese_tone_placement("hóa học"), "hoá học")
        self.assertEqual(harmonize_vietnamese_tone_placement("khỏe khoắn"), "khoẻ khoắn")
        self.assertEqual(harmonize_vietnamese_tone_placement("thủy triều"), "thuỷ triều")
        self.assertEqual(normalize_vietnamese_harmonized(" Hóa Học! "), "hoá học")

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

    def test_apply_time_stretch_rejects_non_positive_rate(self):
        with self.assertRaises(ValueError):
            apply_time_stretch(np.zeros(160, dtype=np.float32), 0.0)
        with self.assertRaises(ValueError):
            apply_time_stretch(np.zeros(160, dtype=np.float32), -0.5)

    def test_apply_time_stretch_identity_on_rate_one(self):
        audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        result = apply_time_stretch(audio, 1.0)
        np.testing.assert_array_equal(audio, result)

    def test_apply_time_stretch_changes_audio_length(self):
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        stretched_slow = apply_time_stretch(audio, 0.8)
        stretched_fast = apply_time_stretch(audio, 1.2)
        self.assertGreater(len(stretched_slow), len(audio))
        self.assertLess(len(stretched_fast), len(audio))


if __name__ == "__main__":
    unittest.main()
