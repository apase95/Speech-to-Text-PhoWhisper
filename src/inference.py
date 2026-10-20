import argparse
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import whisper

DEFAULT_MODEL_PATH = Path("models/phowhisper-base.pt")
def model_path() -> Path:
    return Path(os.environ.get("PHOWHISPER_MODEL_PATH", DEFAULT_MODEL_PATH))
@lru_cache(maxsize=1)
def get_model():
    path = model_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint {path}. "
            "Hãy chạy tools/convert_phowhisper.py trước."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(str(path), device=device)


def transcribe(
    audio_path: str,
    time_stretch_rate: float = 1.0,
    rescore: bool = False,
    beam_size: int = 5,
) -> str:
    use_fp16 = torch.cuda.is_available()
    audio_input: str | np.ndarray = audio_path
    if not math.isclose(time_stretch_rate, 1.0, rel_tol=1e-5):
        from src.audio_utils import apply_time_stretch

        audio = whisper.load_audio(audio_path)
        audio_input = apply_time_stretch(audio, time_stretch_rate)

    if rescore:
        from src.rescorer import rescore_audio

        return rescore_audio(
            audio_input=audio_input,
            whisper_model=get_model(),
            beam_size=beam_size,
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
    parser = argparse.ArgumentParser(description="PhoWhisper Vietnamese Speech-to-Text")
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
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")

    print("Transcript:")
    print(
        transcribe(
            str(audio_path),
            time_stretch_rate=args.time_stretch_rate,
            rescore=args.rescore,
            beam_size=args.beam_size,
        )
    )


if __name__ == "__main__":
    main()
