# Local PhoWhisper Web MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local single-page Gradio app that previews one WAV file, displays its log-Mel spectrogram, and transcribes it with PhoWhisper.

**Architecture:** Keep the existing Python-only project. Add one reusable spectrogram function, cache the existing Hugging Face pipeline once per process, and connect both operations to a small Gradio Blocks interface.

**Tech Stack:** Python, Gradio 6.29.1, Transformers, PyTorch, Librosa, Matplotlib, SoundFile, stdlib `unittest`

**Spec:** `docs/superpowers/specs/2026-10-04-local-stt-web-mvp-design.md`

## Global Constraints

- Run locally and process one WAV file at a time.
- Gradio is the only new direct dependency.
- Keep the existing `python src/inference.py --audio ...` CLI working.
- Do not add accounts, persistence, history, deployment configuration, or a separate API/frontend.
- Do not commit or push changes; the user will handle Git operations.

## Review Focus

- No selected file: callbacks should show a concise user-facing error and perform no processing.
- Non-WAV or corrupt input: reject it with a concise Gradio error rather than a traceback in the page.
- Stereo or non-16-kHz WAV: Librosa should normalize it to mono 16 kHz before plotting.
- Silent WAV: spectrogram generation should return a valid figure without NaN/Inf image data.
- Repeated transcription: the Hugging Face pipeline should be constructed only once per process.

---

### Task 1: Log-Mel Spectrogram

**Files:**
- Modify: `src/audio_utils.py`
- Create: `tests/test_audio_utils.py`

**Interfaces:**
- Consumes: filesystem path to a readable WAV file.
- Produces: `create_logmel_spectrogram(audio_path: str) -> matplotlib.figure.Figure`.

- [ ] **Step 1: Write the failing spectrogram tests**

```python
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from matplotlib.figure import Figure

from src.audio_utils import create_logmel_spectrogram


class SpectrogramTests(unittest.TestCase):
    def test_normalizes_stereo_8khz_wav_and_returns_figure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            time = np.linspace(0, 1, 8_000, endpoint=False)
            signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)
            sf.write(path, np.column_stack([signal, signal]), 8_000)

            figure = create_logmel_spectrogram(str(path))

            self.assertIsInstance(figure, Figure)
            self.assertEqual(figure.axes[0].get_xlabel(), "Time")
            plt.close(figure)

    def test_silent_wav_has_finite_spectrogram(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "silent.wav"
            sf.write(path, np.zeros(16_000, dtype=np.float32), 16_000)

            figure = create_logmel_spectrogram(str(path))

            image_data = figure.axes[0].collections[0].get_array()
            self.assertTrue(np.isfinite(image_data).all())
            plt.close(figure)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm the missing interface fails**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_audio_utils.py' -v`

Expected: import failure because `create_logmel_spectrogram` does not exist.

- [ ] **Step 3: Add the minimal spectrogram implementation**

Add imports for `librosa.display`, `matplotlib.pyplot`, `numpy`, and `Figure`, then add:

```python
def create_logmel_spectrogram(audio_path: str) -> Figure:
    audio, sample_rate = librosa.load(audio_path, sr=16_000, mono=True)
    if audio.size == 0:
        raise ValueError("Audio file is empty")

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=80,
        n_fft=400,
        hop_length=160,
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
```

- [ ] **Step 4: Run the spectrogram tests**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_audio_utils.py' -v`

Expected: two tests pass.

### Task 2: Cache PhoWhisper Pipeline

**Files:**
- Modify: `src/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Produces: `get_pipeline()` returning the cached Transformers ASR pipeline.
- Preserves: `transcribe(audio_path: str) -> str` and the existing CLI.

- [ ] **Step 1: Write the failing cache test**

```python
import unittest
from unittest.mock import patch

from src import inference


class InferenceTests(unittest.TestCase):
    def tearDown(self):
        inference.get_pipeline.cache_clear()

    @patch("src.inference.pipeline")
    def test_get_pipeline_builds_model_once(self, pipeline):
        inference.get_pipeline.cache_clear()

        first = inference.get_pipeline()
        second = inference.get_pipeline()

        self.assertIs(first, second)
        pipeline.assert_called_once()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm the missing interface fails**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_inference.py' -v`

Expected: failure because `get_pipeline` does not exist.

- [ ] **Step 3: Move pipeline construction into a one-entry stdlib cache**

Import `lru_cache`, then implement:

```python
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
```

- [ ] **Step 4: Run the inference cache test and CLI help smoke test**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_inference.py' -v && .venv/bin/python src/inference.py --help`

Expected: test passes and CLI usage is printed without loading the model.

### Task 3: Local Gradio Interface

**Files:**
- Create: `src/app.py`
- Create: `tests/test_app.py`
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: `create_logmel_spectrogram(str) -> Figure` and `transcribe(str) -> str`.
- Produces: `render_spectrogram(audio_path)`, `run_transcription(audio_path)`, `build_app() -> gr.Blocks`, and `python -m src.app` launch command.

- [ ] **Step 1: Add and install the single UI dependency**

Append `gradio==6.29.1` to `requirements.txt`, preserving the existing entries, then run:

`.venv/bin/python -m pip install gradio==6.29.1`

Expected: Gradio installs successfully.

- [ ] **Step 2: Write failing callback and app-construction tests**

```python
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from src.app import build_app, render_spectrogram, run_transcription


class AppTests(unittest.TestCase):
    def test_missing_audio_is_rejected(self):
        with self.assertRaises(gr.Error):
            run_transcription(None)

    def test_non_wav_is_rejected(self):
        with self.assertRaises(gr.Error):
            render_spectrogram("recording.mp3")

    def test_corrupt_wav_is_reported_as_gradio_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.wav"
            path.write_text("not audio", encoding="utf-8")
            with self.assertRaises(gr.Error):
                render_spectrogram(str(path))

    @patch("src.app.transcribe", return_value="xin chào")
    def test_transcription_callback_returns_text(self, transcribe):
        self.assertEqual(run_transcription("sample.wav"), "xin chào")
        transcribe.assert_called_once_with("sample.wav")

    def test_build_app_returns_blocks(self):
        self.assertIsInstance(build_app(), gr.Blocks)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests and confirm the app module is missing**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_app.py' -v`

Expected: import failure because `src.app` does not exist.

- [ ] **Step 4: Implement callbacks and the Gradio page**

Create `src/app.py` with:

```python
from pathlib import Path

import gradio as gr

from src.audio_utils import create_logmel_spectrogram
from src.inference import transcribe


def _require_wav(audio_path: str | None) -> str:
    if not audio_path:
        raise gr.Error("Please choose a WAV file.")
    if Path(audio_path).suffix.lower() != ".wav":
        raise gr.Error("Only WAV files are supported.")
    return audio_path


def render_spectrogram(audio_path: str | None):
    path = _require_wav(audio_path)
    try:
        return create_logmel_spectrogram(path)
    except Exception as error:
        raise gr.Error(f"Could not read WAV file: {error}") from error


def run_transcription(audio_path: str | None) -> str:
    path = _require_wav(audio_path)
    try:
        return transcribe(path)
    except Exception as error:
        raise gr.Error(f"Transcription failed: {error}") from error


def build_app() -> gr.Blocks:
    with gr.Blocks(title="PhoWhisper Vietnamese STT") as app:
        gr.Markdown("# PhoWhisper Vietnamese STT\nChoose a WAV file to view its log-Mel spectrogram and transcript.")
        audio = gr.Audio(label="WAV audio", sources=["upload"], type="filepath")
        spectrogram = gr.Plot(label="Log-Mel Spectrogram")
        transcribe_button = gr.Button("Transcribe", variant="primary")
        transcript = gr.Textbox(label="Transcript", interactive=False, lines=5)

        audio.change(render_spectrogram, inputs=audio, outputs=spectrogram)
        transcribe_button.click(run_transcription, inputs=audio, outputs=transcript)
    return app


if __name__ == "__main__":
    build_app().launch()
```

- [ ] **Step 5: Run the app tests**

Run: `.venv/bin/python -m unittest discover -s tests -p 'test_app.py' -v`

Expected: five tests pass without loading PhoWhisper.

- [ ] **Step 6: Document the local launch command**

Add this section to `README.md`:

````markdown
## Local web demo

Start the Gradio interface:

```bash
python -m src.app
```

Open the local URL shown in the terminal, choose a WAV file, inspect its log-Mel spectrogram, and click **Transcribe**.
````

- [ ] **Step 7: Run all automated checks and import smoke test**

Run: `.venv/bin/python -m unittest discover -s tests -v && .venv/bin/python -c "from src.app import build_app; print(type(build_app()).__name__)"`

Expected: all tests pass and the final line prints `Blocks`.

- [ ] **Step 8: Perform the manual local smoke test**

Run: `.venv/bin/python -m src.app`

Open the printed local URL, select a valid Vietnamese WAV, confirm playback and spectrogram display, click **Transcribe**, and confirm Vietnamese text appears. Stop the server with Ctrl+C. Do not commit or push; leave the verified working tree for the user.
