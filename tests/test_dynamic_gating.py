"""Unit tests for Dynamic Acoustic Confidence Gating."""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import whisper

from src.rescorer import rescore_audio, DEFAULT_CONFIDENCE_THRESHOLD


class TestDynamicAcousticGating(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_whisper = MagicMock()
        self.mock_whisper.dims.n_mels = 80
        self.mock_whisper.device = torch.device("cpu")

    @patch("src.rescorer.extract_standard_mel")
    @patch("src.rescorer.NBestDecodingTask")
    @patch("src.rescorer.get_lm_model")
    def test_skips_lm_when_confidence_exceeds_threshold(
        self,
        mock_get_lm: MagicMock,
        mock_task_cls: MagicMock,
        mock_extract_mel: MagicMock,
    ) -> None:
        # Mock mel extraction
        mock_extract_mel.return_value = torch.zeros((80, 3000), dtype=torch.float32)

        # Mock NBestDecodingTask to return a highly confident top1
        # e.g. avg_logprob = -0.01 >= threshold (-0.02)
        mock_task = MagicMock()
        mock_task.run_nbest.return_value = [
            [
                {
                    "text": "tôi đi học",
                    "tokens": [1, 2, 3],
                    "sum_logprob": -0.03,
                    "avg_logprob": -0.01,
                    "rank": 1,
                },
                {
                    "text": "tôi không đi học",
                    "tokens": [1, 4, 2, 3],
                    "sum_logprob": -0.80,
                    "avg_logprob": -0.20,
                    "rank": 2,
                },
            ]
        ]
        mock_task_cls.return_value = mock_task

        # Call rescore_audio with confidence_threshold = -0.02
        result = rescore_audio(
            audio_input="dummy.wav",
            whisper_model=self.mock_whisper,
            lm_model_name="NlpHUST/gpt2-vietnamese",
            confidence_threshold=-0.02,
        )

        self.assertEqual(result, "tôi đi học")
        # LM model should NOT be loaded or called at all!
        mock_get_lm.assert_not_called()

    @patch("src.rescorer.extract_standard_mel")
    @patch("src.rescorer.NBestDecodingTask")
    @patch("src.rescorer.get_lm_model")
    def test_invokes_lm_when_confidence_below_threshold(
        self,
        mock_get_lm: MagicMock,
        mock_task_cls: MagicMock,
        mock_extract_mel: MagicMock,
    ) -> None:
        # Mock mel extraction
        mock_extract_mel.return_value = torch.zeros((80, 3000), dtype=torch.float32)

        # Mock NBestDecodingTask with lower confidence
        # e.g. avg_logprob = -0.15 < threshold (-0.02)
        mock_task = MagicMock()
        mock_task.run_nbest.return_value = [
            [
                {
                    "text": "tôi đi học",
                    "tokens": [1, 2, 3],
                    "sum_logprob": -0.45,
                    "avg_logprob": -0.15,
                    "rank": 1,
                },
                {
                    "text": "tôi không đi học",
                    "tokens": [1, 4, 2, 3],
                    "sum_logprob": -0.48,
                    "avg_logprob": -0.16,
                    "rank": 2,
                },
            ]
        ]
        mock_task_cls.return_value = mock_task

        # Mock LM model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        # Mock loss
        mock_output = MagicMock()
        mock_output.loss.item.return_value = 2.0
        mock_model.return_value = mock_output
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        mock_get_lm.return_value = (mock_model, mock_tokenizer)

        result = rescore_audio(
            audio_input="dummy.wav",
            whisper_model=self.mock_whisper,
            lm_model_name="NlpHUST/gpt2-vietnamese",
            confidence_threshold=-0.02,
        )

        # LM model SHOULD be called because confidence was below threshold
        mock_get_lm.assert_called_once()
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
