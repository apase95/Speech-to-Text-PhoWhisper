"""Giao diện Gradio cho WAV, phổ log-Mel và nhận dạng tiếng Việt.

Callback kiểm tra tệp trước khi xử lý, ghi chi tiết lỗi vào logger và chỉ
hiển thị thông báo ngắn cho người dùng. Việc kiểm tra này không thay thế
giới hạn tài nguyên hay cơ chế bảo mật khi triển khai dịch vụ.
"""

import logging
from pathlib import Path

import gradio as gr
import soundfile as sf

from phowhisper.audio_utils import create_logmel_spectrogram
from phowhisper.pipeline import transcribe

logger = logging.getLogger(__name__)
theme = gr.themes.Default(primary_hue="blue")


def _require_wav(audio_path: str | None) -> str:
    """Trả lại đường dẫn có đuôi WAV đọc được bởi soundfile, hoặc báo gr.Error."""
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
    """Tạo Figure từ WAV; chuyển lỗi vẽ phổ thành thông báo giao diện."""
    path = _require_wav(audio_path)
    try:
        return create_logmel_spectrogram(path)
    except Exception:
        logger.exception("Could not create spectrogram: %s", path)
        raise gr.Error("Could not create spectrogram") from None


def run_transcription(audio_path: str | None, rescore: bool = False) -> str:
    """Trả transcript, mặc định không dùng LM; ghi log và báo lỗi nếu suy luận hỏng."""
    path = _require_wav(audio_path)
    try:
        return transcribe(path, rescore=rescore)
    except Exception:
        logger.exception("Transcription failed: %s", path)
        raise gr.Error("Transcription failed") from None


def clear_outputs():
    """Xóa biểu đồ và transcript cũ khi người dùng đổi âm thanh."""
    return None, ""


def build_app() -> gr.Blocks:
    """Dựng Blocks và nối sự kiện; chỉ launch khi chạy mô-đun trực tiếp."""
    with gr.Blocks(title="PhoWhisper + GPT-2 Vietnamese STT") as app:
        gr.Markdown("Upload or drop a WAV file, or record audio with your microphone")
        audio = gr.Audio(
            label="Upload WAV or record audio",
            sources=["upload", "microphone"],
            type="filepath",
        )
        spectrogram = gr.Plot(label="Log-Mel Spectrogram")
        rescore_checkbox = gr.Checkbox(
            label="Bật Rescoring LM (PhoWhisper + GPT-2 Accurate Mode)",
            value=False,
            info="Tăng độ chính xác ngữ nghĩa bằng cách tái xếp hạng giả thuyết N-best qua mô hình ngôn ngữ tiếng Việt.",
        )
        transcribe_button = gr.Button("Transcribe", variant="primary")
        transcript = gr.Textbox(label="Transcript", interactive=False, lines=5)

        # Đổi nguồn âm thanh sẽ xóa kết quả cũ trước khi tạo phổ mới.
        audio.change(
            clear_outputs, outputs=[spectrogram, transcript], queue=False
        ).then(render_spectrogram, inputs=audio, outputs=spectrogram)
        transcribe_button.click(
            run_transcription,
            inputs=[audio, rescore_checkbox],
            outputs=transcript,
        )
    return app


if __name__ == "__main__":
    build_app().launch(theme=theme)
