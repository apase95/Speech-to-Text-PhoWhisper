# Local PhoWhisper Web MVP Design

## Goal

Provide a local, single-page demo where a user selects one WAV file, previews it, sees its log-Mel spectrogram, runs PhoWhisper, and reads the Vietnamese transcript.

## Scope

The MVP supports one file at a time and runs on the user's computer. It has no accounts, database, transcript history, public deployment, or separate frontend/backend services.

## Technology

- Gradio for the local web interface, upload, audio preview, plot, button, loading state, and text output.
- Existing Transformers/PyTorch PhoWhisper pipeline for transcription.
- Existing Librosa and Matplotlib packages for audio loading and spectrogram rendering.

Gradio is the only new direct dependency. A separate API and JavaScript frontend are intentionally excluded.

## Components

### `src/inference.py`

Keep the CLI behavior, but cache model pipeline construction so the model loads once per process. `transcribe(audio_path)` reuses that pipeline and continues forcing Vietnamese transcription.

### `src/audio_utils.py`

Add a function that loads audio as mono at 16 kHz and returns a Matplotlib log-Mel spectrogram. Use 80 Mel bins and convert power to decibels. Existing audio conversion remains unchanged.

### `src/app.py`

Define a Gradio page containing:

1. Title and one-sentence instruction.
2. WAV audio input with browser playback.
3. Spectrogram plot updated after file selection.
4. Transcribe button.
5. Read-only transcript output.

The module starts the local Gradio server when run directly.

## Data Flow

1. The user selects a WAV file.
2. Gradio supplies its temporary local path.
3. The spectrogram callback loads the file at 16 kHz mono and returns a plot.
4. On button click, the transcription callback passes the same path to the cached PhoWhisper pipeline.
5. The returned text is displayed without persistence.

## Validation and Errors

- The upload control accepts WAV files only.
- Missing input produces a short user-facing message rather than invoking processing.
- Librosa and model failures surface as concise Gradio errors.
- Files remain under Gradio's temporary-file lifecycle; the app creates no permanent upload copies.

A duration limit is omitted for the local MVP. Add one before public hosting or when long files cause unacceptable memory or latency.

## Verification

- Add a small unit test for spectrogram generation using a temporary synthetic WAV and assert that a figure is returned.
- Preserve the existing CLI inference entry point.
- Run the project's tests and import the Gradio app without launching it.
- Manually smoke-test one valid Vietnamese WAV through upload, spectrogram, and transcription when model weights and suitable hardware are available.

## Success Criteria

- The app launches locally with one command.
- Selecting a valid WAV displays playback and a log-Mel spectrogram.
- Clicking Transcribe displays PhoWhisper's Vietnamese text.
- Repeated transcriptions do not reconstruct the model pipeline.
