import re
import unicodedata

from jiwer import cer, wer


TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def harmonize_vietnamese_tone_placement(text: str) -> str:
    """Normalize Vietnamese tone mark placement to modern canonical standard.

    In modern standard, tone marks in 'oa', 'oe', 'uy' diphthongs are placed on
    the second vowel (e.g., 'hoá', 'khoẻ', 'thuỷ', 'toà') rather than the first.
    """
    nfd = unicodedata.normalize("NFD", text)
    chars = list(nfd)
    result = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if (
            i + 2 < len(chars)
            and c in ("o", "O")
            and chars[i + 1] in TONE_MARKS
            and chars[i + 2] in ("a", "e", "A", "E")
        ):
            result.extend([c, chars[i + 2], chars[i + 1]])
            i += 3
        elif (
            i + 2 < len(chars)
            and c in ("u", "U")
            and chars[i + 1] in TONE_MARKS
            and chars[i + 2] in ("y", "Y")
        ):
            result.extend([c, chars[i + 2], chars[i + 1]])
            i += 3
        else:
            result.append(c)
            i += 1
    return unicodedata.normalize("NFC", "".join(result))


def normalize_vietnamese_harmonized(text: str) -> str:
    base = normalize_vietnamese(text)
    return harmonize_vietnamese_tone_placement(base)


def normalize_vietnamese(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()


def corpus_error_rates(references: list[str], hypotheses: list[str]) -> dict[str, float]:
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
    """Perform paired bootstrap significance test between System A and System B on WER."""
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
