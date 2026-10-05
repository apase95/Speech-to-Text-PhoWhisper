import unittest
from unittest.mock import Mock, patch

from src import inference


class InferenceTests(unittest.TestCase):
    def tearDown(self):
        inference.get_model.cache_clear()

    @patch("src.inference.whisper.load_model")
    @patch("src.inference.Path.is_file", return_value=True)
    def test_model_is_loaded_once(self, _is_file, load_model):
        first = inference.get_model()
        second = inference.get_model()
        self.assertIs(first, second)
        load_model.assert_called_once()

    @patch("src.inference.torch.cuda.is_available", return_value=False)
    @patch("src.inference.get_model")
    def test_transcribe_uses_vietnamese_and_cpu_precision(self, get_model, _cuda):
        get_model.return_value = Mock()
        get_model.return_value.transcribe.return_value = {"text": " xin chào "}

        self.assertEqual(inference.transcribe("sample.wav"), "xin chào")
        get_model.return_value.transcribe.assert_called_once_with(
            "sample.wav", language="vi", task="transcribe", fp16=False
        )


if __name__ == "__main__":
    unittest.main()
