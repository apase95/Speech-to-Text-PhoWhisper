import logging
from pathlib import Path

import gradio as gr
import soundfile as sf

from src.audio_utils import create_logmel_spectrogram
from src.inference import transcribe

logger = logging.getLogger(__name__)
theme = gr.themes.Default(primary_hue="blue")


def _require_wav(audio_path: str | None) -> str:
    if not audio_path:
        raise gr.Error("Please choose a WAV file")
    if Path(audio_path).suffix.lower() != ".wav":
        raise gr.Error("Only WAV files are supported")
    try:
        sf.info(audio_path)
    except Exception:
        logger.exception("Could not read WAV file: %s", audio_path)
        raise gr.Error("Could not read WAV file") from None
    return audio_path


def render_spectrogram(audio_path: str | None):
    path = _require_wav(audio_path)
    try:
        return create_logmel_spectrogram(path)
    except Exception:
        logger.exception("Could not create spectrogram: %s", path)
        raise gr.Error("Could not create spectrogram") from None


def run_transcription(audio_path: str | None) -> str:
    path = _require_wav(audio_path)
    try:
        return transcribe(path)
    except Exception:
        logger.exception("Transcription failed: %s", path)
        raise gr.Error("Transcription failed") from None


def clear_outputs():
    return None, ""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PhoWhisper Vietnamese STT") as app:
        gr.Markdown(
            "Choose a WAV file to view its log-Mel spectrogram and transcript"
        )
        audio = gr.Audio(label="WAV audio", sources=["upload"], type="filepath")
        spectrogram = gr.Plot(label="Log-Mel Spectrogram")
        transcribe_button = gr.Button("Transcribe", variant="primary")
        transcript = gr.Textbox(label="Transcript", interactive=False, lines=5)

        audio.change(
            clear_outputs,
            outputs=[spectrogram, transcript],
            queue=False,
        ).then(render_spectrogram, inputs=audio, outputs=spectrogram)
        transcribe_button.click(run_transcription, inputs=audio, outputs=transcript)
    return app


if __name__ == "__main__":
    build_app().launch(theme=theme)
