import torch
import transformers
from transformers import AutoProcessor

MODEL_NAME = "vinai/PhoWhisper-base"

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")

processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model config OK:", MODEL_NAME)
print("Tokenizer vocab size:", len(processor.tokenizer))
