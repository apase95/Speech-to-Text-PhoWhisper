import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf

from src.app import build_app, clear_outputs, render_spectrogram, run_transcription, theme


class AppTests(unittest.TestCase):
    def test_transcribe_button_uses_blue_primary_theme(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            app = build_app()
        self.addCleanup(app.close)

        buttons = [
            component
            for component in app.get_config_file()["components"]
            if component["type"] == "button"
        ]
        self.assertEqual(theme.primary_500, "#3b82f6")
        self.assertEqual(buttons[0]["props"]["variant"], "primary")

    def test_audio_change_clears_previous_outputs(self):
        self.assertEqual(clear_outputs(), (None, ""))

    def test_missing_audio_is_rejected(self):
        with self.assertRaises(gr.Error) as caught:
            run_transcription(None)
        self.assertEqual(caught.exception.message, "Please choose a WAV file")

    def test_non_wav_is_rejected(self):
        with self.assertRaises(gr.Error) as caught:
            render_spectrogram("recording.mp3")
        self.assertEqual(caught.exception.message, "Only WAV files are supported")

    def test_corrupt_wav_is_reported_as_gradio_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.wav"
            path.write_text("not audio", encoding="utf-8")
            with self.assertLogs("src.app", level="ERROR"):
                with self.assertRaises(gr.Error) as caught:
                    render_spectrogram(str(path))

        self.assertEqual(caught.exception.message, "Could not read WAV file")

    @patch("src.app.transcribe")
    def test_corrupt_wav_is_rejected_before_model_call(self, transcribe):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.wav"
            path.write_text("not audio", encoding="utf-8")

            with self.assertLogs("src.app", level="ERROR"):
                with self.assertRaises(gr.Error) as caught:
                    run_transcription(str(path))

        self.assertEqual(caught.exception.message, "Could not read WAV file")
        transcribe.assert_not_called()

    @patch("src.app.transcribe", return_value="xin chào")
    def test_transcription_callback_returns_text(self, transcribe):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(160, dtype=np.float32), 16_000)

            self.assertEqual(run_transcription(str(path)), "xin chào")

        transcribe.assert_called_once_with(str(path))

    @patch(
        "src.app.create_logmel_spectrogram",
        side_effect=RuntimeError("private decoder details"),
    )
    def test_spectrogram_failure_logs_details_and_shows_short_error(self, _):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(160, dtype=np.float32), 16_000)

            with self.assertLogs("src.app", level="ERROR") as logs:
                with self.assertRaises(gr.Error) as caught:
                    render_spectrogram(str(path))

        self.assertEqual(caught.exception.message, "Could not create spectrogram")
        self.assertIn("private decoder details", logs.output[0])

    @patch("src.app.transcribe", side_effect=RuntimeError("private model details"))
    def test_transcription_failure_logs_details_and_shows_short_error(self, _):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(160, dtype=np.float32), 16_000)

            with self.assertLogs("src.app", level="ERROR") as logs:
                with self.assertRaises(gr.Error) as caught:
                    run_transcription(str(path))

        self.assertEqual(caught.exception.message, "Transcription failed")
        self.assertIn("private model details", logs.output[0])

    def test_build_app_returns_blocks(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            app = build_app()
        self.addCleanup(app.close)
        self.assertIsInstance(app, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
