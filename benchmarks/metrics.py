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
