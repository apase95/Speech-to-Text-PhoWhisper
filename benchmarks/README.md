# PhoWhisper Benchmark

Benchmark dùng một manifest cố định để so sánh các thay đổi model, decoding và runtime theo cùng dữ liệu.

Cài core và benchmark tooling bằng `pip install -e '.[benchmark]'`; GPT-2 runners
cần thêm extra `rescore`. Overlay `requirements-benchmark.txt` giữ chính sách
môi trường bridge riêng, không phải standalone lock. Lệnh từ checkout cần core
đã cài hoặc `PYTHONPATH=.:src`. Implementations nằm trong `benchmarks/runners`
và `benchmarks/audits`; normalization của core không phụ thuộc benchmarks.

## Báo cáo lịch sử

- [FLEURS60](../reports/benchmark-results-fleurs60.md)
- [FLEURS general60](../reports/fleurs-general60-benchmark.md)
- [GPT-2 error audit](../reports/gpt2-error-audit-holdout.md)
- [GPT-2 optimization](../reports/gpt2-optimization-fleurs60.md)
- [Human reference review](../reports/human-review-benchmark-fleurs60.md)
- [Long audio](../reports/long-audio-benchmark.md)
- [Numeric metrics](../reports/numeric-metric-audit-fleurs60.md)
- [Personal audio/VietMed](../reports/personal-audio-vietmed100-comparison.md)
- [Reference normalization](../reports/reference-normalization-fleurs60.md)
- [VietMed JSON](../outputs/benchmarks/vietmed_benchmark_results.json)

Migration không chạy lại hoặc xác nhận các kết quả lịch sử này.

## Viet-BUD500 subset

Tập con gồm đúng 1.000 ID liên tiếp `audio1.wav` đến `audio1000.wav` từ Kaggle dataset `tuandaodev/viet-bud500-formatted`. Thứ tự theo ID số, không theo thứ tự từ điển của Kaggle API. Transcript được ghép từ `metadata_train.csv` và `metadata_test.csv`.

Tải hoặc kiểm tra lại dữ liệu và tái tạo manifest:

```bash
source .venv/bin/activate
python scripts/data/download_viet_bud500_subset.py
```

Downloader hỗ trợ resume và kiểm tra mỗi WAV bằng libsndfile. Manifest lưu SHA-256 của từng file; không chỉnh sửa test set hoặc dùng các mẫu này để fine-tune.

Dev set cố định gồm 250 ID tiếp theo, `audio1001.wav` đến `audio1250.wav`, không trùng với benchmark 1.000 mẫu:

```bash
python scripts/data/download_viet_bud500_subset.py \
  --count 250 \
  --start-id 1001 \
  --manifest benchmarks/manifests/viet_bud500_dev_250.csv
```

Chọn cấu hình decoding trên dev set này trước khi chạy lại test set. Không dùng kết quả test để chọn beam size hoặc siêu tham số khác.

## Chạy benchmark

Cấu hình mặc định được khuyến nghị cho utterance ngắn là greedy, `without_timestamps=True`, `condition_on_previous_text=False` và không silence padding. Không truyền `--beam-size` để giữ greedy decoding.

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

Thử silence padding ở mỗi biên sau khi audio được resample về 16 kHz:

```bash
python -m benchmarks.run \
  --manifest benchmarks/manifests/viet_bud500_dev_250.csv \
  --name dev250-greedy-padding200 \
  --without-timestamps \
  --no-condition-on-previous-text \
  --silence-padding-ms 200
```

Thử time-stretching (co dãn tốc độ phát âm không đổi cao độ) trước khi inference:

```bash
python -m benchmarks.run \
  --manifest benchmarks/manifests/viet_bud500_dev_250.csv \
  --name dev250-greedy-stretch09 \
  --without-timestamps \
  --no-condition-on-previous-text \
  --time-stretch-rate 0.9
```

RTF vẫn được tính theo thời lượng file gốc để so sánh nhất quán giữa các mức padding và stretch rate.

Mỗi run tạo thư mục riêng trong `outputs/benchmarks/` gồm:

- `summary.json`: WER, CER, RTF, latency tổng, model load time, peak RAM/VRAM, hash model/manifest và môi trường.
- `predictions.jsonl`: reference, hypothesis, thời lượng, latency, RTF và lỗi của từng file.

WER/CER được tính trên toàn corpus sau khi chuẩn hóa NFC, lowercase, bỏ dấu câu và chuẩn hóa khoảng trắng. Dấu thanh tiếng Việt được giữ nguyên.

Khi so sánh, chỉ thay một yếu tố mỗi run và luôn dùng cùng manifest/checkpoint hash. Lần chạy đầu có thể chịu chi phí warm-up; nên chạy cùng một quy trình cho tất cả cấu hình.
