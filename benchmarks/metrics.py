import re
import unicodedata

from jiwer import cer, wer


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
