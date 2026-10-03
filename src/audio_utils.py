from pathlib import Path

import librosa
import soundfile as sf


def convert_to_16khz_mono(input_path: str, output_path: str) -> None:
    audio, _ = librosa.load(input_path, sr=16_000, mono=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sf.write(output, audio, 16_000, subtype="PCM_16")
