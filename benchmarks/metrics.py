"""Chuẩn hóa văn bản và đo sai số nhận dạng bằng jiwer.

WER/CER trả tỷ lệ, không phải phần trăm. Chuẩn hóa mặc định giữ dấu tiếng
Việt và chữ số; hài hòa vị trí dấu thanh là lựa chọn riêng, không tự bật.
"""

from jiwer import cer, wer
from phowhisper.text.normalization import (
    TONE_MARKS,
    harmonize_vietnamese_tone_placement,
    normalize_vietnamese,
    normalize_vietnamese_harmonized,
)


def corpus_error_rates(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Trả wer/cer toàn corpus sau chuẩn hóa mặc định qua jiwer.

    Hai danh sách phải cùng độ dài và không rỗng, nếu không gây ValueError.
    Không lấy trung bình các tỷ lệ riêng từng câu.
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    if not references:
        raise ValueError("At least one prediction is required")

    normalized_references = [normalize_vietnamese(text) for text in references]
    normalized_hypotheses = [normalize_vietnamese(text) for text in hypotheses]
    return {
        "wer": wer(normalized_references, normalized_hypotheses),
        "cer": cer(normalized_references, normalized_hypotheses),
    }


def paired_bootstrap_test(
    references: list[str],
    hypotheses_a: list[str],
    hypotheses_b: list[str],
    n_bootstraps: int = 2000,
    seed: int = 42,
    harmonized: bool = False,
) -> dict[str, float | bool]:
    """So sánh WER hai hệ thống bằng lấy mẫu lại các cặp câu có hoàn lại.

    Mặc định 2.000 lượt, seed=42, không hài hòa dấu. Danh sách lệch độ dài
    hoặc rỗng gây ValueError. Mẫu tham chiếu rỗng được tính mẫu số ít nhất
    một từ. wer_a/wer_b là tỷ lệ; mean_diff và khoảng percentile 95% là
    chênh lệch B - A theo điểm phần trăm, âm nghĩa là B ít lỗi hơn.
    p_value là phần lượt có B - A >= 0; cờ ý nghĩa dùng p_value < 0.05,
    tức kiểm định một phía theo hướng B tốt hơn A, không phải hai phía.
    """
    import jiwer
    import numpy as np

    if not (len(references) == len(hypotheses_a) == len(hypotheses_b)):
        raise ValueError("References and all hypotheses must have the same length")
    if not references:
        raise ValueError("At least one prediction is required")

    norm_fn = normalize_vietnamese_harmonized if harmonized else normalize_vietnamese
    norm_refs = [norm_fn(r) for r in references]
    norm_a = [norm_fn(h) for h in hypotheses_a]
    norm_b = [norm_fn(h) for h in hypotheses_b]

    n = len(norm_refs)
    ref_lens = np.array([max(1, len(r.split())) for r in norm_refs])
    errs_a = np.array([
        jiwer.process_words(r, h).substitutions
        + jiwer.process_words(r, h).deletions
        + jiwer.process_words(r, h).insertions
        for r, h in zip(norm_refs, norm_a)
    ])
    errs_b = np.array([
        jiwer.process_words(r, h).substitutions
        + jiwer.process_words(r, h).deletions
        + jiwer.process_words(r, h).insertions
        for r, h in zip(norm_refs, norm_b)
    ])

    total_words = float(ref_lens.sum())
    base_wer_a = float(errs_a.sum()) / total_words
    base_wer_b = float(errs_b.sum()) / total_words

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstraps, n))

    sampled_errs_a = errs_a[indices].sum(axis=1)
    sampled_errs_b = errs_b[indices].sum(axis=1)
    sampled_lens = ref_lens[indices].sum(axis=1)

    wer_diffs = ((sampled_errs_b / sampled_lens) - (sampled_errs_a / sampled_lens)) * 100.0

    p_value = float(np.mean(wer_diffs >= 0.0))
    ci_lower = float(np.percentile(wer_diffs, 2.5))
    ci_upper = float(np.percentile(wer_diffs, 97.5))
    mean_diff = float(np.mean(wer_diffs))

    return {
        "wer_a": base_wer_a,
        "wer_b": base_wer_b,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "statistically_significant": bool(p_value < 0.05),
    }
