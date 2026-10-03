import argparse
import logging
from functools import lru_cache
from pathlib import Path

import torch
from transformers import pipeline
from transformers.utils import logging as transformers_logging

MODEL_NAME = "vinai/PhoWhisper-base"

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
transformers_logging.set_verbosity_error()


@lru_cache(maxsize=1)
def get_pipeline():
    use_cuda = torch.cuda.is_available()
    return pipeline(
        task="automatic-speech-recognition",
        model=MODEL_NAME,
        device=0 if use_cuda else -1,
        dtype=torch.float16 if use_cuda else torch.float32,
    )


def transcribe(audio_path: str) -> str:
    result = get_pipeline()(
        audio_path,
        generate_kwargs={"language": "vi", "task": "transcribe"},
    )
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
