import unittest
from tools.section23_benchmark import summarize


class Smoke(unittest.TestCase):
    def test_happy(self):
        result = summarize([dict(reference='Xin chào!', hypothesis='xin chào', audio_seconds=2, inference_seconds=.5)])
        self.assertEqual(result['wer'], 0)
        self.assertEqual(result['cer'], 0)
        self.assertEqual(result['exact_matches'], 1)
        self.assertEqual(result['rtf'], .25)

    def test_zero_audio(self):
        with self.assertRaises(ValueError):
            summarize([dict(reference='a', hypothesis='', audio_seconds=0, inference_seconds=1)])


if __name__ == '__main__':
    unittest.main()
