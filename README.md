# PhoWhisper PyTorch

Web demo nhận dạng tiếng Việt có giao diện tương đương `phowhisper-vietnamese`, nhưng runtime dùng trực tiếp `openai-whisper`/PyTorch thay cho `transformers.pipeline`.

## Kiến trúc

- `src/app.py`: giao diện Gradio upload/ghi âm, spectrogram và transcript.
- `src/inference.py`: nạp `torch.nn.Module` từ checkpoint `.pt` và suy luận tiếng Việt.
- `tools/convert_phowhisper.py`: chuyển trọng số Safetensors của `vinai/PhoWhisper-base` sang định dạng OpenAI Whisper.
- Không có dependency `transformers`, `accelerate`, `datasets` hoặc `huggingface_hub` trong runtime.

## Cài đặt

Yêu cầu Python 3.10+ và `ffmpeg`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Nếu cần CUDA, nên cài bản `torch` phù hợp driver từ trang PyTorch trước khi cài requirements.

## Chuẩn bị checkpoint

PhoWhisper chỉ phát hành trọng số gốc theo định dạng Transformers `pytorch_model.bin`. Lệnh sau tải trực tiếp hai file công khai và chuyển một lần sang `.pt` (không cần dùng Transformers API):

```bash
python tools/convert_phowhisper.py --download
```

Có thể chuyển từ file đã tải sẵn:

```bash
python tools/convert_phowhisper.py \
  --source /path/to/pytorch_model.bin \
  --config /path/to/config.json \
  --output models/phowhisper-base.pt
```

Đặt checkpoint ở vị trí khác bằng biến `PHOWHISPER_MODEL_PATH`.

## Chạy

```bash
python -m src.check_environment
python -m src.inference --audio data/raw/test/sample.wav
python -m src.app
```

## Có cần train lại không?

**Không cần** nếu mục tiêu là chạy đúng PhoWhisper-base bằng PyTorch. Chuyển đổi chỉ đổi tên tensor và vỏ checkpoint; kiến trúc, tokenizer multilingual Whisper và toàn bộ trọng số fine-tune tiếng Việt được giữ nguyên. Cần kiểm chứng transcript/WER giữa bản Transformers và bản PyTorch trên một tập âm thanh trước khi triển khai.

Chỉ cần fine-tune/train tiếp khi muốn thích nghi domain (tổng đài, y tế, giọng vùng miền, tiếng ồn), thêm dữ liệu riêng, hoặc khi phép chuyển đổi không tương thích với một phiên bản checkpoint khác. Train từ đầu gần như không thực tế; nên fine-tune từ checkpoint PhoWhisper.

## Vấn đề cần lưu ý

1. **Định dạng checkpoint:** tên tensor của Transformers và OpenAI Whisper khác nhau. Script hiện nhắm tới `PhoWhisper-base`; checkpoint shard hoặc biến thể kiến trúc cần mở rộng và kiểm thử ánh xạ.
2. **Nguồn model:** runtime không dùng Hugging Face, nhưng trọng số chính thức vẫn được lưu trên máy chủ Hugging Face. Có thể tải thủ công và chuyển offline nếu môi trường không được truy cập mạng.
3. **Tokenizer:** runtime dùng tokenizer multilingual gốc của OpenAI Whisper. Phải giữ đúng vocabulary và special token của checkpoint; không thay tokenizer tiếng Việt tùy ý.
4. **Sai khác decoder:** hai implementation có thể khác mặc định beam search, temperature và xử lý timestamp, nên transcript không nhất thiết giống từng ký tự dù trọng số giống nhau.
5. **Tài nguyên:** PhoWhisper-base cần hàng trăm MB RAM/VRAM; CPU chạy được nhưng chậm. CUDA dùng FP16, CPU dùng FP32.
6. **Audio dài:** Whisper xử lý theo cửa sổ 30 giây. File dài cần quan tâm chunking, timestamp, bộ nhớ và nguy cơ lặp/mất ngữ cảnh.
7. **FFmpeg:** `openai-whisper` cần binary `ffmpeg` để đọc audio khi inference.
8. **Giấy phép và bảo mật:** cần kiểm tra giấy phép model/dataset trước khi thương mại hóa; audio nhạy cảm nên được xử lý local và không ghi log nội dung.

## Test

```bash
python -m unittest discover -s tests -v
```

Unit test không tải model. Sau khi chuyển checkpoint, nên chạy smoke test bằng WAV thật và so sánh WER với project cũ.
