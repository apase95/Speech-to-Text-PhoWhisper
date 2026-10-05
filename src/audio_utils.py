from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure


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
