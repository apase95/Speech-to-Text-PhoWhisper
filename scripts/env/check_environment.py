"""In chẩn đoán PyTorch, Whisper, CUDA và vị trí checkpoint ra stdout.

Mã chạy ngay cả khi import mô-đun. Chỉ kiểm tra checkpoint có là tệp hay
không, không tải trọng số hay xác nhận tương thích; lỗi import/thăm dò GPU
được truyền ra ngoài thay vì chuyển thành kết luận môi trường hợp lệ.
"""

import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

import torch
import whisper

from phowhisper.pipeline import model_path

print("PyTorch:", torch.__version__)
print("OpenAI Whisper:", getattr(whisper, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
print("Checkpoint:", model_path())
print("Checkpoint exists:", model_path().is_file())
