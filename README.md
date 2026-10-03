# PhoWhisper Vietnamese STT

Project tối thiểu để chạy inference PhoWhisper base trong VS Code/Jupyter và chuẩn bị fine-tuning trên Kaggle.

## VS Code

1. Mở folder này trong VS Code.
2. Chọn interpreter: `${workspaceFolder}/.venv/bin/python`.
3. Với notebook, chọn kernel Python từ `.venv`.

## Cài dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Test môi trường

```bash
python src/check_environment.py
```

Lần đầu sẽ tải config/tokenizer của `vinai/PhoWhisper-base` từ Hugging Face.

## Test inference bằng file audio

Đặt file test tại `data/raw/test/sample.wav`, hoặc truyền path khác:

```bash
python src/inference.py --audio data/raw/test/sample.wav
```

Trong VS Code có thể dùng `Run and Debug` -> `Run PhoWhisper Inference`.

## Test bằng Jupyter

Mở `notebooks/01_smoke_test.ipynb`, chọn kernel `.venv`, chạy cell đầu để kiểm tra model config. Cell thứ hai chỉ chạy khi đã có audio test.

## Local web demo

Start the Gradio interface:

```bash
python -m src.app
```

Open the local URL shown in the terminal, choose a WAV file, inspect its log-Mel spectrogram, and click **Transcribe**.

## Fine-tuning trên Kaggle

Trên Kaggle nên bật GPU T4/P100, upload notebook/dataset, rồi cài các package chính:

```python
!pip install -q transformers datasets accelerate evaluate jiwer librosa soundfile
```

Model name dùng thống nhất:

```python
MODEL_NAME = "vinai/PhoWhisper-base"
```

Dataset nên có tối thiểu 2 cột: audio path/array và transcript tiếng Việt. Fine-tuning cần notebook riêng sau khi chốt format dataset.
# Speech-to-Text-with-Phowisper
