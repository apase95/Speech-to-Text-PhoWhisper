# Hướng Dẫn Sử Dụng PhoWhisper PyTorch

Tài liệu hướng dẫn chi tiết cách cài đặt, cấu hình và sử dụng hệ thống nhận dạng tiếng nói PhoWhisper PyTorch qua dòng lệnh (CLI), giao diện web (Gradio), và lập trình (Python API).

---

## 1. Yêu cầu hệ thống & Cài đặt

### Yêu cầu tiên quyết
- **Python**: 3.10 trở lên.
- **FFmpeg**: Cần thiết để đọc định dạng âm thanh và thực hiện tính năng Time-Stretching WSOLA (`atempo`).
  ```bash
  # Ubuntu/Debian:
  sudo apt-get update && sudo apt-get install -y ffmpeg

  # Fedora/RHEL:
  sudo dnf install -y ffmpeg

  # macOS (Homebrew):
  brew install ffmpeg
  ```

### Cài đặt môi trường Python
```bash
# Tạo và kích hoạt môi trường ảo
python3 -m venv .venv
source .venv/bin/activate

# Cài đặt các gói phụ thuộc cơ bản
pip install -r requirements.txt

# (Tùy chọn) Cài đặt transformers nếu muốn dùng tính năng Accurate LM Rescoring:
pip install transformers
```

### Chuẩn bị Checkpoint mô hình
Hệ thống sử dụng checkpoint native PyTorch của PhoWhisper-base. Tải và chuyển đổi một lần duy nhất:
```bash
python tools/convert_phowhisper.py --download
```
File trọng số sẽ được lưu tại `models/phowhisper-base.pt`.

---

## 2. Các Chế Độ Nhận Dạng (Fast vs Accurate)

Hệ thống hỗ trợ 2 chế độ suy luận phù hợp cho các tình huống sử dụng khác nhau:

| Tiêu chí | Chế độ Nhanh (Fast Mode) | Chế độ Chính xác (Accurate Mode) |
|---|---|---|
| **Cơ chế** | Greedy decoding (`without_timestamps=True`) | Beam 5 search + GPT-2 Conservative Rescoring |
| **Tốc độ (RTF)** | **~0,20x** (nhanh gấp 5 lần thời gian phát thực) | **~0,25x - 0,35x** (nhanh gấp 3-4 lần thời gian thực) |
| **VRAM tiêu thụ** | **~560 MB** | **~850 MB** |
| **Độ chính xác (WER)** | 10,49% (Viet-BUD500) | **8,32%** (Viet-BUD500) |
| **Số câu đúng hoàn hảo**| ~45,1% | **51,4%** |
| **Tình huống phù hợp** | Chatbot thời gian thực, voice agent, thiết bị nhúng | Bóc băng hội thảo, phụ đề video, hồ sơ y tế, văn bản pháp lý |

---

## 3. Sử dụng qua dòng lệnh (CLI)

Entry point chính là file `src/inference.py`:

### 3.1. Chế độ Mặc định (Fast Mode)
Nhận dạng âm thanh với tốc độ tối đa và tiêu thụ tài nguyên tối thiểu:
```bash
python -m src.inference --audio path/to/audio.wav
```

### 3.2. Chế độ Chính xác cao (Accurate Mode với `--rescore`)
Kích hoạt Beam 5 cùng mô hình ngôn ngữ tiếng Việt để tái xếp hạng giả thuyết, sửa lỗi ngữ nghĩa:
```bash
python -m src.inference --audio path/to/audio.wav --rescore
```

### 3.3. Xử lý giọng nói nhanh / lướt âm (`--time-stretch-rate`)
Khi gặp đoạn nói quá nhanh hoặc nuốt âm, sử dụng cờ co giãn thời gian không làm đổi cao độ giọng (khuyến nghị `0.9` cho giảm 10% tốc độ):
```bash
# Chạy Fast mode với giãn tốc độ 0.9x:
python -m src.inference --audio path/to/audio.wav --time-stretch-rate 0.9

# Chạy Accurate mode kết hợp giãn tốc độ 0.9x:
python -m src.inference --audio path/to/audio.wav --time-stretch-rate 0.9 --rescore
```

### 3.4. Tùy chỉnh kích thước Beam (`--beam-size`)
Khi bật `--rescore`, mặc định hệ thống dùng `beam_size=5`. Bạn có thể tùy chỉnh kích thước chùm tia:
```bash
python -m src.inference --audio path/to/audio.wav --rescore --beam-size 3
```

---

## 4. Sử dụng qua Python API

Hàm `transcribe()` trong `src.inference` cho phép tích hợp trực tiếp vào ứng dụng Python của bạn:

```python
from src.inference import transcribe

audio_file = "data/benchmark/viet-bud500/audio/audio1.wav"

# 1. Nhận dạng nhanh (Mặc định):
text_fast = transcribe(audio_file)
print("Fast:", text_fast)

# 2. Nhận dạng chính xác cao (Accurate Rescoring):
text_accurate = transcribe(audio_file, rescore=True)
print("Accurate:", text_accurate)

# 3. Nhận dạng âm thanh nói nhanh:
text_stretched = transcribe(audio_file, time_stretch_rate=0.9, rescore=True)
print("Stretched + Rescored:", text_stretched)
```

### Chi tiết tham số hàm `transcribe()`:
- `audio_path` (*str*): Đường dẫn file âm thanh (định dạng WAV, MP3,...).
- `time_stretch_rate` (*float*, mặc định `1.0`): Hệ số co giãn tốc độ âm thanh (ví dụ: `0.9` là nói chậm lại 10%, `1.1` là nói nhanh hơn 10%).
- `rescore` (*bool*, mặc định `False`): `True` để bật Beam 5 và GPT-2 Conservative Rescoring; `False` để chạy Greedy siêu tốc.
- `beam_size` (*int*, mặc định `5`): Kích thước chùm tia khi bật rescore.

---

## 5. Sử dụng qua Giao diện Web (Gradio Web UI)

Khởi động web app phục vụ người dùng cuối:

```bash
python src/app.py
```
Mặc định ứng dụng sẽ mở tại `http://localhost:7860`.

### Các tính năng trên giao diện:
1. **Tải lên hoặc Ghi âm trực tiếp**: Kéo thả file WAV hoặc nhấn biểu tượng microphone để nói trực tiếp từ trình duyệt.
2. **Biểu đồ Log-Mel Spectrogram**: Tự động trích xuất và hiển thị trực quan phổ âm thanh để kiểm tra chất lượng thu âm.
3. **Checkbox `Bật Rescoring LM`**:
   - Bỏ chọn: Suy luận tức thì (Fast mode).
   - Chọn vào: Kích hoạt chế độ Accurate (tự động tải GPT-2 và rescore cho độ chính xác cao nhất).
4. **Hộp thoại Transcript**: Hiển thị kết quả văn bản tiếng Việt chuẩn hóa.

---

## 6. Chạy Benchmark & Tái lập kết quả

Để kiểm chứng hiệu năng và độ chính xác trên tập dữ liệu chuẩn Viet-BUD500 (1.000 mẫu):

```bash
# 1. Chạy Benchmark Beam 5 gốc (Nhánh A):
python -m benchmarks.run \
  --manifest benchmarks/manifests/viet_bud500_1000.csv \
  --name test1000-beam5 \
  --beam-size 5 \
  --without-timestamps \
  --no-condition-on-previous-text

# 2. Chạy 5-Fold Cross-Validation và đánh giá Conservative Rescoring (Nhánh C):
python tools/rescore_eval.py

# 3. Chạy toàn bộ bộ test kiểm thử tự động (Regression + Unit Tests):
python -m unittest discover tests
```

---

## 7. Tài liệu liên quan
- [Nhật ký Nâng cấp & Tối ưu hóa](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/docs/upgrade.md): Bảng tổng hợp 11 cải tiến kỹ thuật so với bản gốc.
- [Báo cáo Kiểm chứng Rescoring Đối chứng](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/outputs/viet-bud500-controlled-rescoring-benchmark-check.md): Phân tích thực nghiệm 3 nhánh A / B / C trên 1.000 mẫu test.
- [Báo cáo Conservative Rescoring & Thống kê](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/outputs/viet-bud500-conservative-rescoring-report.md): Chi tiết cơ chế Acoustic Margin $\gamma$ và Paired Bootstrap Significance test.
