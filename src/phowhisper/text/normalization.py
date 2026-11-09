"""Chuẩn hóa văn bản tiếng Việt; không phụ thuộc mô hình hoặc benchmark."""
import re
import unicodedata


TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}


def harmonize_vietnamese_tone_placement(text: str) -> str:
    """Chuyển dấu thanh từ nguyên âm đầu sang nguyên âm sau trong oa/oe/uy.

    Xử lý chuỗi NFD rồi trả NFC, ví dụ hòa thành hoà. Đây là phép đổi mẫu
    ký tự cụ thể, không phải bộ kiểm tra chính tả tiếng Việt tổng quát.
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
    """Chuẩn hóa cơ bản rồi hài hòa vị trí dấu thanh oa/oe/uy."""
    base = normalize_vietnamese(text)
    return harmonize_vietnamese_tone_placement(base)


def normalize_vietnamese(text: str) -> str:
    """Đưa về NFC/casefold, thay dấu câu Unicode bằng cách và gộp khoảng trắng.

    Không chuyển số thành chữ, không bỏ dấu thanh và không hài hòa dấu.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(char).startswith("P") else char for char in text)
    return re.sub(r"\s+", " ", text).strip()
