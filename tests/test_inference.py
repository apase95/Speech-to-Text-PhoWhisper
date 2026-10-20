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
            "sample.wav",
            language="vi",
            task="transcribe",
            fp16=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )

    @patch("src.audio_utils.apply_time_stretch")
    @patch("src.inference.whisper.load_audio")
    @patch("src.inference.torch.cuda.is_available", return_value=False)
    @patch("src.inference.get_model")
    def test_transcribe_with_time_stretch_rate(
        self, get_model, _cuda, load_audio, apply_time_stretch
    ):
        get_model.return_value = Mock()
        get_model.return_value.transcribe.return_value = {"text": " xin chào "}
        dummy_audio = Mock()
        stretched_audio = Mock()
        load_audio.return_value = dummy_audio
        apply_time_stretch.return_value = stretched_audio

        self.assertEqual(
            inference.transcribe("sample.wav", time_stretch_rate=0.9), "xin chào"
        )
        load_audio.assert_called_once_with("sample.wav")
        apply_time_stretch.assert_called_once_with(dummy_audio, 0.9)
        get_model.return_value.transcribe.assert_called_once_with(
            stretched_audio,
            language="vi",
            task="transcribe",
            fp16=False,
            without_timestamps=True,
            condition_on_previous_text=False,
        )

    @patch("src.rescorer.rescore_audio", return_value="xin chào rescored")
    @patch("src.inference.get_model")
    def test_transcribe_with_rescore_enabled(self, get_model, rescore_audio):
        get_model.return_value = Mock()
        result = inference.transcribe("sample.wav", rescore=True, beam_size=5)

        self.assertEqual(result, "xin chào rescored")
        rescore_audio.assert_called_once_with(
            audio_input="sample.wav",
            whisper_model=get_model.return_value,
            beam_size=5,
            confidence_threshold=-0.02,
        )
        get_model.return_value.transcribe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
