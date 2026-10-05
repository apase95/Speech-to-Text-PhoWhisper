import torch
import whisper

from src.inference import model_path

print("PyTorch:", torch.__version__)
print("OpenAI Whisper:", getattr(whisper, "__version__", "unknown"))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
print("Checkpoint:", model_path())
print("Checkpoint exists:", model_path().is_file())
