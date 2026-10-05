import argparse
import os
from functools import lru_cache
from pathlib import Path

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


def transcribe(audio_path: str) -> str:
    use_fp16 = torch.cuda.is_available()
    result = get_model().transcribe(
        audio_path,
        language="vi",
        task="transcribe",
        fp16=use_fp16,
    )
    return result["text"].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")

    print("Transcript:")
    print(transcribe(str(audio_path)))


if __name__ == "__main__":
    main()
