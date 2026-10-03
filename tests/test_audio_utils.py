import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure

from src.audio_utils import create_logmel_spectrogram


class SpectrogramTests(unittest.TestCase):
    def test_normalizes_stereo_8khz_wav_and_returns_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            time = np.linspace(0, 1, 8_000, endpoint=False)
            signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)
            sf.write(path, np.column_stack([signal, signal]), 8_000)

            figure = create_logmel_spectrogram(str(path))

            self.assertIsInstance(figure, Figure)
            self.assertEqual(figure.axes[0].get_xlabel(), "Time")
            plt.close(figure)

    def test_silent_wav_has_finite_spectrogram(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silent.wav"
            sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000)

            figure = create_logmel_spectrogram(str(path))

            image_data = figure.axes[0].collections[0].get_array()
            self.assertTrue(np.isfinite(image_data).all())
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
