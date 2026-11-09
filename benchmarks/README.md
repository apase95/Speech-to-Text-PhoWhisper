# PhoWhisper Benchmark

## Current small public-mode evidence — 2026-11-10

Official small is now default by user decision, not a reversal of the historical
base/small gate. [Migration/archive](../docs/model-migration-small.md).
[Holdout500 public modes report](../reports/phowhisper-small-public-modes-vietbud500-20261110.md)
and [curated JSON](../configs/evaluations/phowhisper-small-public-modes-vietbud500.json):
fast/rescore WER 9.4854%/7.8500%, CER 6.4873%/5.8158%, RTF .18649/.44159.
Public fast retains Whisper temperature fallback; not the old temperature=0 protocol.
Full 500 rows each, no failures. This local cohort was previously evaluated;
upstream exposure is unknown. No zero-leakage assertion.

Python >=3.11 for current benchmark imports. Full tooling:
`pip install -e '.[web,benchmark,rescore,conversion]'` (some audio imports need
plot libraries in web). FFmpeg, local audio and cached pinned LM are prerequisites.
No installation or inference was run by this documentation task.

Reproduction command (GPU inference, not a documentation validation command):

```bash
PYTHONPATH=.:src python -m benchmarks.runners.compare_small_modes \
  --manifest benchmarks/manifests/viet_bud500_holdout_500.csv \
  --disjoint-from benchmarks/manifests/viet_bud500_dev_250.csv \
  --output-dir outputs/benchmarks/small-public-modes-holdout500-rerun
```

Use a fresh output directory. LM revision
`65818d14816b42be09e2201933bf07106d9a2647`, offline CUDA FP32; ASR CUDA FP16.
Prepare its local cache once with network access before running the offline runners:

```bash
hf download NlpHUST/gpt2-vietnamese --revision 65818d14816b42be09e2201933bf07106d9a2647
```

One excluded warmup on dev audio1001; no holdout tuning. Public rescore only
handles its first 30 seconds, so do not apply directly to long recordings.
For equal independent 28-second pieces:

```bash
PYTHONPATH=.:src python -m benchmarks.runners.compare_personal_small \
  --audio recording.m4a --output-dir outputs/benchmarks/personal-small-new
```

This runner defaults to `models/phowhisper-small.pt` and its converter provenance
sidecar; `--checkpoint` selects another verified official-small artifact. It does
not require base weights or a dated acquisition directory. The pinned local LM
cache must still be prepared before use.
No cross-piece context. Without reference it reports timing, not WER/CER.

## Historical no-rescore base/small decision — 2026-11-09

For frozen official base/small no-rescore comparisons, use the
[evaluation protocol](../docs/phowhisper-small-evaluation.md). The runner now pins
temperature=0 explicitly (older runs may have used Whisper fallback temperatures).
Historical results are not directly comparable without matching the protocol.
Actual 2026-11-09 dev250/holdout500 results and checkpoint provenance are in the
[base/small report](../reports/phowhisper-base-small-vietbud500-20261109.md).
Base was retained at that date because small failed the CER and two-sided 95% CI
gates despite lower WER. Later user promotion does not change that evidence.

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
- VietMed JSON local-only (gitignored): `outputs/benchmarks/vietmed_benchmark_results.json`

Migration không chạy lại hoặc xác nhận các kết quả lịch sử này.

## Viet-BUD500 subset — legacy data preparation

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

## Legacy benchmark examples — not the current public-mode comparison

Các ví dụ cũ bên dưới phục vụ data/decoding experiments, không thay thế lệnh
`compare_small_modes` hiện tại ở đầu trang. Không suy diễn public fast là greedy
temperature=0 chỉ từ việc không truyền `--beam-size`; cần kiểm tra protocol đã lưu.
Giữ `without_timestamps=True`, `condition_on_previous_text=False` và không silence
padding khi tái lập protocol utterance ngắn tương ứng.

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

- [VietMed100 official base/small fast/rescoring: kết quả thực và audit 2026-11-09](../reports/vietmed100-base-small-fast-rescoring-20261109.md).
