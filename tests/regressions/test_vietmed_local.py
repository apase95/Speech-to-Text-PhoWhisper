import unittest
from tools.vietmed_local import summarize


class SummaryTests(unittest.TestCase):
    def test_happy(self):
        result = summarize([dict(reference='Xin chào!', hypothesis='xin chào', audio_seconds=2, inference_seconds=0.5)])
        self.assertEqual(result['wer'], 0)
        self.assertEqual(result['rtf'], 0.25)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])
