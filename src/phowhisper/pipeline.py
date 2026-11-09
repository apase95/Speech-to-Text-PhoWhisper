"""API và CLI suy luận PhoWhisper từ checkpoint native trên máy cục bộ.

Luồng xử lý chọn checkpoint, tùy chọn chỉnh tốc độ rồi nhận dạng tiếng Việt.
LM chỉ được import khi yêu cầu rescoring; nhánh mặc định dùng transcribe
của Whisper, không hậu xử lý hài hòa dấu thanh cho transcript đầu ra.
"""

import argparse
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import whisper

DEFAULT_MODEL_PATH = Path("models/phowhisper-small.pt")
def model_path() -> Path:
    """Lấy PHOWHISPER_MODEL_PATH hoặc models/phowhisper-small.pt theo cwd."""
    return Path(os.environ.get("PHOWHISPER_MODEL_PATH", DEFAULT_MODEL_PATH))
@lru_cache(maxsize=1)
def get_model():
    """Nạp và cache một mô hình trên CUDA nếu có, nếu không dùng CPU.

    Checkpoint thiếu gây FileNotFoundError kèm gợi ý chuyển đổi. Cache
    không tự nạp lại khi biến môi trường hoặc nội dung tệp thay đổi.
    """
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint {path}. "
            "Hãy chuyển đổi bằng scripts/convert/convert_phowhisper.py hoặc "
            "kích hoạt bản small đã xác minh bằng scripts/models/activate_small.py."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(str(path), device=device)


def transcribe(
    audio_path: str,
    time_stretch_rate: float = 1.0,
    rescore: bool = False,
    beam_size: int = 5,
    confidence_threshold: float | None = -0.020,
) -> str:
    """Nhận đường dẫn âm thanh và trả transcript đã bỏ khoảng trắng hai đầu.

    Mặc định rate=1 giữ nguyên tốc độ và rescore=False. Khi bật rescoring,
    beam_size=5 và ngưỡng -0.020 được chuyển sang rescorer; None tắt gate.
    Chỉnh tốc độ cần ffmpeg và tải waveform trước khi suy luận. FP16 chỉ
    bật khi CUDA khả dụng; lỗi tải âm thanh/mô hình và giải mã truyền ra ngoài.
    """
    use_fp16 = torch.cuda.is_available()
    audio_input: str | np.ndarray = audio_path
    if not math.isclose(time_stretch_rate, 1.0, rel_tol=1e-5):
        from phowhisper.audio_utils import apply_time_stretch

        audio = whisper.load_audio(audio_path)
        audio_input = apply_time_stretch(audio, time_stretch_rate)

    # Tách nhánh LM để chế độ mặc định không phải nạp phụ thuộc rescoring.
    if rescore:
        from phowhisper.rescorer import rescore_audio

        return rescore_audio(
            audio_input=audio_input,
            whisper_model=get_model(),
            beam_size=beam_size,
            confidence_threshold=confidence_threshold,
        )

    result = get_model().transcribe(
        audio_input,
        language="vi",
        task="transcribe",
        fp16=use_fp16,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    return result["text"].strip()


def main() -> None:
    """Đọc CLI, kiểm tra đường dẫn tồn tại và in transcript ra stdout.

    Giá trị ngưỡng nhỏ hơn -100 được đổi thành None để tắt gate; quy ước
    này chỉ áp dụng tại CLI, không áp dụng trực tiếp trong API transcribe.
    """
    parser = argparse.ArgumentParser(description="PhoWhisper + GPT-2: Vietnamese Speech-to-Text")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument(
        "--time-stretch-rate",
        type=float,
        default=1.0,
        help="Speech rate adjustment (e.g. 0.9 for fast speech)",
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Enable accurate mode with N-best Language Model rescoring (Beam 5 + GPT-2)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size when rescoring is enabled (default: 5)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=-0.020,
        help="Acoustic confidence threshold to skip LM rescoring (default: -0.020; set to -999 to disable)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")

    conf_th = None if args.confidence_threshold < -100.0 else args.confidence_threshold
    print("Transcript:")
    print(
        transcribe(
            str(audio_path),
            time_stretch_rate=args.time_stretch_rate,
            rescore=args.rescore,
            beam_size=args.beam_size,
            confidence_threshold=conf_th,
        )
    )


if __name__ == "__main__":
    main()
