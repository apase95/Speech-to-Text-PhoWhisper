"""Tạo giả thuyết beam search rồi tái xếp hạng bằng GPT-2 hoặc PhoBERT.

Gate âm học có thể trả top-1 trước khi tải LM. Điểm kết hợp chỉ chọn trong
các giả thuyết sẵn có, không sinh câu sửa mới và không bảo đảm tăng độ đúng.
Nhánh này lấy một cửa sổ Mel đầu tiên, không lặp phân đoạn như transcribe
cho toàn bộ tệp dài. Phụ thuộc API giải mã nội bộ của OpenAI Whisper.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisper

from phowhisper.text.normalization import normalize_vietnamese

logger = logging.getLogger(__name__)

# Default locked parameters from 5-fold cross-validation on Dev 250
DEFAULT_LM_MODEL = "NlpHUST/gpt2-vietnamese"
DEFAULT_ALPHA_GPT2 = 0.200
DEFAULT_BETA_GPT2 = -0.020
DEFAULT_GAMMA_GPT2 = -0.20

DEFAULT_ALPHA_PHOBERT = 0.130
DEFAULT_BETA_PHOBERT = -0.008
DEFAULT_GAMMA_PHOBERT = -0.20

# Default confidence threshold for dynamic gating (skip LM when acoustic avg_lp >= threshold)
DEFAULT_CONFIDENCE_THRESHOLD = -0.020


def extract_standard_mel(audio_input: str | Path | np.ndarray, model: whisper.Whisper) -> torch.Tensor:
    """Đọc đường dẫn hoặc waveform, lấy cửa sổ Mel đầu và pad/trim đủ N_FRAMES.

    Số dải Mel theo model; tensor được đưa lên model.device, với float16
    nếu CUDA khả dụng, ngược lại float32. Không giải mã các cửa sổ tiếp theo.
    """
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
    """Mở rộng DecodingTask để thu các ứng viên beam thay vì chỉ một kết quả."""

    def run_nbest(self, mel: torch.Tensor) -> list[list[dict[str, Any]]]:
        """Giải mã batch Mel và trả danh sách ứng viên riêng cho từng âm thanh.

        Mỗi ứng viên có text, tokens, tổng logprob và trung bình chia cho
        số token cộng một. Rank bắt đầu từ 1 theo tổng logprob giảm dần;
        decoder được reset trước lượt chạy.
        """
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


@lru_cache(maxsize=2)
def get_lm_model(model_name: str = DEFAULT_LM_MODEL) -> tuple[Any, Any]:
    """Nạp lười và cache tối đa hai cặp model/tokenizer theo tên Hugging Face.

    Tên chứa phobert dùng masked LM, còn lại dùng causal LM; đặt eval và
    chọn CUDA/CPU. from_pretrained có thể tải và ghi cache trên đĩa. Thiếu
    transformers hoặc lỗi tải được truyền ra để rescore_audio fallback.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformers is required for LM rescoring. "
            "Install it via `pip install transformers`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "phobert" in model_name.lower():
        model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()
    return model, tokenizer


def rescore_audio(
    audio_input: str | Path | np.ndarray,
    whisper_model: whisper.Whisper,
    lm_model_name: str = DEFAULT_LM_MODEL,
    beam_size: int = 5,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float | None = None,
    confidence_threshold: float | None = DEFAULT_CONFIDENCE_THRESHOLD,
) -> str:
    """Trả văn bản ứng viên được chọn từ một cửa sổ âm thanh.

    Beam mặc định 5; không có ứng viên trả chuỗi rỗng, chỉ một ứng viên
    hoặc top-1 có avg_logprob >= -0.020 thì trả ngay, không nạp LM.
    confidence_threshold=None tắt gate. Bộ alpha/beta/gamma mặc định là
    0.200/-0.020/-0.20 cho GPT-2 và 0.130/-0.008/-0.20 cho PhoBERT.

    Điểm là avg_logprob + alpha * điểm LM + beta * số từ chuẩn hóa.
    Chỉ thay top-1 khi điểm tăng và chênh lệch âm học >= gamma. Chuẩn hóa
    phục vụ chấm điểm, kết quả vẫn là text gốc của ứng viên, bỏ khoảng trắng
    hai đầu. Lỗi trong bước LM/chọn điểm được log warning rồi trả top-1;
    lỗi trích Mel hoặc beam search trước đó không được fallback ở đây.
    """
    is_phobert = "phobert" in lm_model_name.lower()
    if alpha is None:
        alpha = DEFAULT_ALPHA_PHOBERT if is_phobert else DEFAULT_ALPHA_GPT2
    if beta is None:
        beta = DEFAULT_BETA_PHOBERT if is_phobert else DEFAULT_BETA_GPT2
    if gamma is None:
        gamma = DEFAULT_GAMMA_PHOBERT if is_phobert else DEFAULT_GAMMA_GPT2

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

    # Dynamic Acoustic Confidence Gating: skip LM rescoring if Whisper Top-1 is highly confident
    if confidence_threshold is not None and top1["avg_logprob"] >= confidence_threshold:
        logger.debug(
            "Acoustic confidence %.4f >= threshold %.4f, skipping LM rescoring.",
            top1["avg_logprob"],
            confidence_threshold,
        )
        return top1["text"].strip()

    try:
        lm_model, lm_tokenizer = get_lm_model(lm_model_name)
        device = next(lm_model.parameters()).device

        # PhoBERT dùng pyvi và trung bình logprob khi che từng token theo lô 16.
        if is_phobert:
            import torch.nn.functional as F
            from pyvi import ViTokenizer

            max_chunk_size = 16
            max_seq_len = getattr(lm_tokenizer, "model_max_length", 256)
            if max_seq_len > 512:
                max_seq_len = 256

            for c in cands:
                c["normalized_text"] = normalize_vietnamese(c["text"])
                if not c["normalized_text"]:
                    c["lm_logprob_per_token"] = -20.0
                    continue
                segmented = ViTokenizer.tokenize(c["normalized_text"])
                input_ids = lm_tokenizer.encode(
                    segmented,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                ).to(device)[0]

                L = len(input_ids)
                if L <= 2:
                    c["lm_logprob_per_token"] = 0.0
                    continue

                positions = torch.arange(1, L - 1, device=device)
                target_ids = input_ids[positions]
                T = len(positions)
                token_lps = []

                for start in range(0, T, max_chunk_size):
                    end = min(start + max_chunk_size, T)
                    pos = positions[start:end]
                    tgt = target_ids[start:end]
                    k = len(pos)

                    chunk_batch = input_ids.unsqueeze(0).repeat(k, 1)
                    chunk_batch[torch.arange(k, device=device), pos] = lm_tokenizer.mask_token_id

                    with torch.no_grad():
                        sub_logits = lm_model(chunk_batch).logits
                    m_logits = sub_logits[torch.arange(k, device=device), pos]
                    token_lps.append(F.log_softmax(m_logits, dim=-1)[torch.arange(k, device=device), tgt])

                c["lm_logprob_per_token"] = float(torch.cat(token_lps).mean().item())
        else:
            # GPT-2 chấm văn bản chuẩn hóa bằng âm của causal language-model loss.
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
