# PhoWhisper Benchmark

Benchmark dùng một manifest cố định để so sánh các thay đổi model, decoding và runtime theo cùng dữ liệu.

## Viet-BUD500 subset

Tập con gồm đúng 1.000 ID liên tiếp `audio1.wav` đến `audio1000.wav` từ Kaggle dataset `tuandaodev/viet-bud500-formatted`. Thứ tự theo ID số, không theo thứ tự từ điển của Kaggle API. Transcript được ghép từ `metadata_train.csv` và `metadata_test.csv`.

Tải hoặc kiểm tra lại dữ liệu và tái tạo manifest:

```bash
source .venv/bin/activate
python tools/download_viet_bud500_subset.py
```

Downloader hỗ trợ resume và kiểm tra mỗi WAV bằng libsndfile. Manifest lưu SHA-256 của từng file; không chỉnh sửa test set hoặc dùng các mẫu này để fine-tune.

## Chạy benchmark

Smoke test 10 file:

```bash
python -m benchmarks.run \
  --manifest benchmarks/manifests/viet_bud500_1000.csv \
  --name baseline-smoke \
  --limit 10
```

Baseline đầy đủ:

```bash
python -m benchmarks.run \
  --manifest benchmarks/manifests/viet_bud500_1000.csv \
  --name baseline
```

Mỗi run tạo thư mục riêng trong `outputs/benchmarks/` gồm:

- `summary.json`: WER, CER, RTF, latency tổng, model load time, peak RAM/VRAM, hash model/manifest và môi trường.
- `predictions.jsonl`: reference, hypothesis, thời lượng, latency, RTF và lỗi của từng file.

WER/CER được tính trên toàn corpus sau khi chuẩn hóa NFC, lowercase, bỏ dấu câu và chuẩn hóa khoảng trắng. Dấu thanh tiếng Việt được giữ nguyên.

Khi so sánh, chỉ thay một yếu tố mỗi run và luôn dùng cùng manifest/checkpoint hash. Lần chạy đầu có thể chịu chi phí warm-up; nên chạy cùng một quy trình cho tất cả cấu hình.
