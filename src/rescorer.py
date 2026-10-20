"""N-best decoding and Language Model rescoring for PhoWhisper."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisper

from benchmarks.metrics import normalize_vietnamese

logger = logging.getLogger(__name__)

# Default locked parameters from 5-fold cross-validation on Dev 250
DEFAULT_LM_MODEL = "NlpHUST/gpt2-vietnamese"
DEFAULT_ALPHA = 0.200
DEFAULT_BETA = -0.020
DEFAULT_GAMMA = -0.20


def extract_standard_mel(audio_input: str | Path | np.ndarray, model: whisper.Whisper) -> torch.Tensor:
    """Extract standard mel spectrogram matching whisper.transcribe exactly."""
    if isinstance(audio_input, (str, Path)):
        audio = whisper.load_audio(str(audio_input))
    else:
        audio = audio_input

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mel_full = whisper.log_mel_spectrogram(audio, model.dims.n_mels, padding=whisper.audio.N_SAMPLES)
    content_frames = mel_full.shape[-1] - whisper.audio.N_FRAMES
    segment_size = min(whisper.audio.N_FRAMES, max(0, content_frames))
    mel_segment = mel_full[:, 0:segment_size]
    return whisper.pad_or_trim(mel_segment, whisper.audio.N_FRAMES).to(model.device).to(dtype)


class NBestDecodingTask(whisper.decoding.DecodingTask):
    """Whisper decoding task that collects and exports all N-best candidates from beam search."""

    def run_nbest(self, mel: torch.Tensor) -> list[list[dict[str, Any]]]:
        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]

        audio_features = self._get_audio_features(mel)
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        tokens, sum_logprobs, _ = self._main_loop(audio_features, tokens)

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)

        results: list[list[dict[str, Any]]] = []
        for a_idx in range(n_audio):
            cands: list[dict[str, Any]] = []
            for t_seq, lp in zip(tokens[a_idx], sum_logprobs[a_idx]):
                eot_matches = (t_seq == tokenizer.eot).nonzero()
                eot_idx = (
                    eot_matches[0, 0].item()
                    if len(eot_matches) > 0
                    else len(t_seq)
                )
                cand_tokens = t_seq[self.sample_begin : eot_idx].tolist()
                text = tokenizer.decode(cand_tokens).strip()
                sum_lp = float(lp.item() if hasattr(lp, "item") else lp)
                avg_lp = float(sum_lp / (len(cand_tokens) + 1))
                cands.append(
                    {
                        "text": text,
                        "tokens": cand_tokens,
                        "sum_logprob": sum_lp,
                        "avg_logprob": avg_lp,
                    }
                )
            cands.sort(key=lambda x: x["sum_logprob"], reverse=True)
            for r_idx, c in enumerate(cands, start=1):
                c["rank"] = r_idx
            results.append(cands)
        return results


@lru_cache(maxsize=1)
def get_lm_model(model_name: str = DEFAULT_LM_MODEL) -> tuple[Any, Any]:
    """Lazy-load and cache the language model and tokenizer for rescoring."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformers is required for LM rescoring. "
            "Install it via `pip install transformers`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def rescore_audio(
    audio_input: str | Path | np.ndarray,
    whisper_model: whisper.Whisper,
    lm_model_name: str = DEFAULT_LM_MODEL,
    beam_size: int = 5,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> str:
    """Perform beam search ASR decoding and conservative Language Model rescoring."""
    mel = extract_standard_mel(audio_input, whisper_model)
    use_fp16 = torch.cuda.is_available()

    options = whisper.DecodingOptions(
        language="vi",
        task="transcribe",
        beam_size=beam_size,
        without_timestamps=True,
        fp16=use_fp16,
    )
    task = NBestDecodingTask(whisper_model, options)

    with torch.no_grad():
        cands = task.run_nbest(mel.unsqueeze(0))[0]

    if not cands:
        return ""

    top1 = cands[0]
    if len(cands) <= 1:
        return top1["text"].strip()

    try:
        lm_model, lm_tokenizer = get_lm_model(lm_model_name)
        device = next(lm_model.parameters()).device

        for c in cands:
            c["normalized_text"] = normalize_vietnamese(c["text"])
            if not c["normalized_text"]:
                c["lm_logprob_per_token"] = -20.0
                continue
            inp = lm_tokenizer.encode(c["normalized_text"], return_tensors="pt").to(device)
            with torch.no_grad():
                loss = lm_model(inp, labels=inp).loss
                c["lm_logprob_per_token"] = -float(loss.item())

        top1_ac = top1["avg_logprob"]
        best_cand = top1
        best_score = top1_ac + alpha * top1.get("lm_logprob_per_token", 0.0) + beta * len(top1["normalized_text"].split())

        for c in cands[1:]:
            score = c["avg_logprob"] + alpha * c.get("lm_logprob_per_token", 0.0) + beta * len(c["normalized_text"].split())
            if score > best_score and (c["avg_logprob"] - top1_ac) >= gamma:
                best_score = score
                best_cand = c

        return best_cand["text"].strip()

    except Exception as exc:
        logger.warning("Rescoring failed, falling back to top-1 ASR output: %s", exc)
        return top1["text"].strip()
