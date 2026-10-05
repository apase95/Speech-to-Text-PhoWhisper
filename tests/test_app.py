import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import numpy as np
import soundfile as sf

from src.app import build_app, clear_outputs, run_transcription


class AppTests(unittest.TestCase):
    def test_interface_accepts_upload_and_microphone(self):
        app = build_app()
        self.addCleanup(app.close)
        audio = next(
            item for item in app.get_config_file()["components"] if item["type"] == "audio"
        )
        self.assertEqual(audio["props"]["sources"], ["upload", "microphone"])

    def test_audio_change_clears_outputs(self):
        self.assertEqual(clear_outputs(), (None, ""))

    def test_missing_audio_is_rejected(self):
        with self.assertRaises(gr.Error):
            run_transcription(None)

    @patch("src.app.transcribe", return_value="xin chào")
    def test_transcription_callback(self, transcribe):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            sf.write(path, np.zeros(160, dtype=np.float32), 16_000)
            self.assertEqual(run_transcription(str(path)), "xin chào")
        transcribe.assert_called_once_with(str(path))


if __name__ == "__main__":
    unittest.main()
