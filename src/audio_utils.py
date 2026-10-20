import math
import subprocess
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure


def _build_atempo_filter(rate: float) -> str:
    filters = []
    current = rate
    while current < 0.5:
        filters.append("atempo=0.5")
        current /= 0.5
    while current > 2.0:
        filters.append("atempo=2.0")
        current /= 2.0
    filters.append(f"atempo={current}")
    return ",".join(filters)


def apply_time_stretch(
    audio: np.ndarray, rate: float, sr: int = 16_000
) -> np.ndarray:
    """Time-stretch an audio waveform without pitch alteration.

    rate > 1.0 speeds up the audio (shortens duration).
    rate < 1.0 slows down the audio (lengthens duration).
    Uses ffmpeg's atempo filter (WSOLA) for speech quality and phase preservation.
    """
    if rate <= 0:
        raise ValueError("Time-stretch rate must be positive")
    if math.isclose(rate, 1.0, rel_tol=1e-5):
        return audio
    if audio.size == 0:
        return audio

    pcm_bytes = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-filter:a",
        _build_atempo_filter(rate),
        "-f",
        "s16le",
        "-ar",
        str(sr),
        "-ac",
        "1",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm_bytes, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, np.int16).flatten().astype(np.float32) / 32768.0


def convert_to_16khz_mono(input_path: str, output_path: str) -> None:
    audio, _ = librosa.load(input_path, sr=16_000, mono=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, audio, 16_000, subtype="PCM_16")


def create_logmel_spectrogram(audio_path: str) -> Figure:
    audio, sample_rate = librosa.load(audio_path, sr=16_000, mono=True)
    if audio.size == 0:
        raise ValueError("Audio file is empty")

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=80, n_fft=400, hop_length=160
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    figure, axis = plt.subplots(figsize=(10, 4))
    image = librosa.display.specshow(
        log_mel,
        sr=sample_rate,
        hop_length=160,
        x_axis="time",
        y_axis="mel",
        ax=axis,
    )
    axis.set(title="Log-Mel Spectrogram", xlabel="Time", ylabel="Mel frequency")
    figure.colorbar(image, ax=axis, format="%+2.0f dB")
    figure.tight_layout()
    return figure
