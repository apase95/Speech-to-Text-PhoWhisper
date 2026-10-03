import argparse
import logging
from pathlib import Path

import torch
from transformers import pipeline
from transformers.utils import logging as transformers_logging

MODEL_NAME = "vinai/PhoWhisper-base"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()


def transcribe(audio_path: str) -> str:
    device = 0 if torch.cuda.is_available() else -1
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=device,
        dtype=dtype,
    )

    result = pipe(audio_path, generate_kwargs={"language": "vi", "task": "transcribe"})

    return result["text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {audio_path}")

    text = transcribe(str(audio_path))
    print("Transcript:")
    print(text)


if __name__ == "__main__":
    main()
