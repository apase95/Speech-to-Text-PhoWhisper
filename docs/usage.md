# Sử dụng PhoWhisper-small

Xem [README](../README.md) cho cài đặt, conversion pinned, CLI/API/web và số đo.
Default là `models/phowhisper-small.pt` theo cwd; `PHOWHISPER_MODEL_PATH` override.
Restart worker khi đổi checkpoint vì model được cache. Package không chứa weights.

```bash
phowhisper --audio audio.wav
phowhisper --audio short.wav --rescore --beam-size 5 --confidence-threshold -0.020
# Tắt gate, chấm LM nếu có nhiều candidate:
phowhisper --audio short.wav --rescore --confidence-threshold -999
```

Fast giữ temperature fallback, không phải benchmark greedy temperature=0 cũ.
Rescore chọn N-best, không bảo đảm sửa đúng. `--time-stretch-rate 0.9` dùng
FFmpeg chỉnh tốc độ, không bảo đảm tăng chất lượng.

```python
from phowhisper.pipeline import transcribe
text = transcribe("audio.wav")
text_short = transcribe("short.wav", rescore=True, confidence_threshold=None)
```

API dùng `confidence_threshold=None` để tắt gate; -999 chỉ là quy ước CLI.
Mặc định `rescore=False`, rate=1.0, beam_size=5. CUDA ASR dùng FP16;
public API có đường CPU khi không CUDA nhưng chưa đo CPU throughput.

## Audio dài và web

**Public rescore chỉ lấy 30 giây đầu**, không tự chia bản ghi. Web
`PYTHONPATH=src python -m app.web.app` dùng WAV và có cùng giới hạn checkbox.
Không khuyến nghị đưa cả hội thảo/video vào rescore trực tiếp.

```bash
PYTHONPATH=.:src python -m benchmarks.runners.compare_personal_small \
  --audio recording.m4a --output-dir outputs/benchmarks/personal-small-new
```

Runner chia cùng mảnh 28 giây cho hai mode, độc lập không chéo ngữ cảnh;
cần CUDA, checkpoint/LM local chuẩn bị sẵn theo [benchmark guide](../benchmarks/README.md).
Không reference thì không WER/CER. Shared decode, load và warmup tách khỏi timing inference.

Core Python >=3.10; benchmark >=3.11. Tooling đầy đủ:
`pip install -e '.[web,benchmark,rescore,conversion]'` vì một số audio imports
benchmark cần thư viện plot từ extra web. Cần FFmpeg trên PATH.
Training dùng môi trường riêng, không cần train để inference.

Output là candidate gốc đã trim, không ép harmonize dấu thanh. Lỗi LM warning
rồi top-1; lỗi audio/beam có thể truyền ra caller. Xem [kiến trúc](architecture.md),
[so sánh](comparison.md), [rollback](model-migration-small.md).
