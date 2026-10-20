import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import whisper

from benchmarks.metrics import paired_bootstrap_test
from tools.generate_dev_nbest import NBestDecodingTask, extract_standard_mel


class PipelineParityTests(unittest.TestCase):
    def test_extract_standard_mel_shape_and_padding(self):
        # Create a mock model with dims
        model = SimpleNamespace(
            dims=SimpleNamespace(n_mels=80),
            device=torch.device("cpu"),
        )
        # Create dummy 1.5s audio array (24,000 samples at 16kHz)
        audio = np.sin(np.linspace(0, 100, 24000)).astype(np.float32)

        with patch("whisper.load_audio", return_value=audio):
            mel = extract_standard_mel("dummy.wav", model)

        # Expected output shape: (80, N_FRAMES)
        self.assertEqual(mel.shape, (80, whisper.audio.N_FRAMES))
        self.assertEqual(mel.device.type, "cpu")

    def test_paired_bootstrap_test_significance(self):
        refs = [
            "tôi đi học",
            "trời hôm nay đẹp quá",
            "học viện công nghệ",
            "mô hình nhận dạng tiếng nói",
        ] * 10
        # System A has some errors
        hyps_a = [
            "tôi đi chơi",
            "trời hôm nay xấu quá",
            "học viện khoa học",
            "mô hình nhận dạng tiếng việt",
        ] * 10
        # System B is exact match
        hyps_b = list(refs)

        result = paired_bootstrap_test(refs, hyps_a, hyps_b, n_bootstraps=500, seed=42)

        self.assertGreater(result["wer_a"], result["wer_b"])
        self.assertLess(result["mean_diff"], 0.0)
        self.assertLess(result["ci_upper"], 0.0)
        self.assertLess(result["p_value"], 0.01)
        self.assertTrue(result["statistically_significant"])

    def test_end_to_end_parity_with_real_sample(self):
        model_path = Path("models/phowhisper-base.pt")
        sample_audio = Path("data/benchmark/viet-bud500/audio/audio1.wav")

        if not model_path.is_file() or not sample_audio.is_file():
            self.skipTest("Real model or audio1.wav not available for end-to-end parity test")

        from src.inference import get_model

        model = get_model()

        # Branch A: whisper.transcribe
        res_transcribe = model.transcribe(
            str(sample_audio),
            beam_size=5,
            without_timestamps=True,
            condition_on_previous_text=False,
            language="vi",
            task="transcribe",
        )
        text_a = res_transcribe["text"].strip()

        # Branch B: extract_standard_mel + NBestDecodingTask
        mel = extract_standard_mel(sample_audio, model).unsqueeze(0)
        options = whisper.DecodingOptions(
            task="transcribe",
            language="vi",
            beam_size=5,
            without_timestamps=True,
            fp16=torch.cuda.is_available(),
        )
        task = NBestDecodingTask(model, options)
        nbest = task.run_nbest(mel)[0]
        text_b = nbest[0]["text"].strip()

        # Regression assertion: exact equality
        self.assertEqual(
            text_a,
            text_b,
            f"Regression mismatch detected between transcribe ({text_a}) and NBestDecodingTask ({text_b})",
        )


if __name__ == "__main__":
    unittest.main()
