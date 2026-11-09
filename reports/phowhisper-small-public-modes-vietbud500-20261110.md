# PhoWhisper-small: public fast và GPT-2 rescore — 2026-11-10

Đây là **kết quả đã chạy**, không phải protocol chờ inference. Hai mode dùng
official small revision `a86b604c346caf7148c37512eafe783a16420adb`, không local
fine-tune. Small mặc định theo quyết định người dùng; phép đo này không đảo
[gate base/small lịch sử](phowhisper-base-small-vietbud500-20261109.md).
[JSON curated](../configs/evaluations/phowhisper-small-public-modes-vietbud500.json)
chỉ giữ metrics/protocol/provenance, không chứa transcript cá nhân hay secrets.

## Dữ liệu, protocol và provenance

- Manifest `benchmarks/manifests/viet_bud500_holdout_500.csv`, SHA256
  `46f6777ff565a168cae35e31d0c6e0a1f2df9f55e0b4b4e705860e3f12621a0b`.
  500 mẫu, 1.300,656099773244 giây; kiểm tra disjoint ID/hash với dev250.
- Cohort local đã được đánh giá trước đây, **không phải holdout hoàn toàn mới**.
  Không bảo đảm loại trừ khỏi upstream training, không tuyên bố zero leakage.
- Một warmup dev audio1001 mỗi mode, loại khỏi scoring/timing. Không tuning.
- Fast public giữ temperature fallback Whisper; khác temperature=0 greedy cũ.
  Rescore beam 5, alpha .2, beta -.02, gamma -.2, gate -.020 kế thừa base.
- GPT-2 `NlpHUST/gpt2-vietnamese` revision
  `65818d14816b42be09e2201933bf07106d9a2647`, offline local-only CUDA FP32;
  ASR CUDA FP16. Fast không nạp LM. Fresh sequential processes, batch 1, seed 42.
- Word/character corpus ratios sau normalization giữ dấu/chữ số, không harmonize
  tone. Exact match cũng sau normalization, không phải đúng nguyên văn từng byte.
- Native SHA256 `15632cc15395f5ee8d9fec382133f3d6429c63d1c2209518984fb126058e85c5`;
  model, sidecar, manifest, audio và source snapshot trước/sau đều khớp trong report.
- HEAD `a051afe91f82440cab09f25365400875bc3f3478`, **dirty=true**; không chỉ dùng HEAD
  để tái lập. JSON giữ diff/status hash; raw reports giữ source/input hashes chi tiết.
- Linux, Python 3.14.8, torch 2.14.1+cu130, Whisper 20250625, CUDA 13.0,
  RTX 3050 Laptop, capacity 3.951.296.512 bytes (class 4GB).

Raw artifacts **chỉ cục bộ, bị gitignore**, không phải link tải GitHub:
`outputs/benchmarks/small-public-modes-holdout500-20261110/{report.json,comparison.json}`;
`fast/fast-20261109T175900Z/{summary.json,predictions.jsonl}` và
`rescore/rescore-20261109T180312Z/{summary.json,predictions.jsonl}` bên trong thư mục đó.
Tên run dùng UTC 2026-11-09; ngày báo cáo/controller hoàn tất là 2026-11-10.

## Kết quả đầy đủ, không lỗi

| Chỉ số | Fast | Rescore |
|---|---:|---:|
| Mẫu / lỗi | 500 / 0 | 500 / 0 |
| WER | 9,4853903% | 7,8499782% |
| CER | 6,4872581% | 5,8157978% |
| Exact match | 239/500 (47,8%) | 260/500 (52,0%) |
| Word S / D / I / hits | 271 / 60 / 104 / 4255 | 195 / 80 / 85 / 4311 |
| Character S / D / I / hits | 422 / 300 / 505 / 18192 | 293 / 378 / 429 / 18243 |
| Inference seconds | 242,563499855 | 574,351590986 |
| RTF | 0,186493186 | 0,441586051 |
| Median latency s | 0,436333831 | 1,157514428 |
| p95 interpolated s (summary) | 0,689674054 | 1,526489304 |
| p95 nearest-rank s (comparison) | 0,689599074 | 1,526204572 |

Timing gồm audio I/O và API call, không gồm model/LM load và warmup.
Hai cách tính p95 được ghi rõ thay vì coi các khác biệt nhỏ là lỗi dữ liệu.
WER giảm 1,6354121 điểm phần trăm (~17,24% tương đối), nhưng đây là
**so sánh workflow** (fallback fast so với beam+LM), không cô lập gain của LM.
Deletion tăng 60 → 80; không bảo đảm giữ mọi từ hoặc từ phủ định.

Bootstrap cặp theo utterance, 2.000 draws seed 42, B−A tính theo điểm phần trăm:
mean -1,6370071, percentile 95% CI [-2,6665928; -0,6954226]. Có 0/2.000
resample không âm (empirical one-sided p=0), **không phải xác suất bằng zero**.
Bootstrap mô tả cohort/protocol này, không chứng minh chất lượng ngoài domain.

Rescore: single candidate 0, acoustic gate 253/500 (50,6%), thực chấm LM 247,
winner đổi so với beam top-1 104. Đây không phải 104 câu chắc chắn sửa đúng,
và gate rate không phải phần trăm tiết kiệm tổng runtime hay tỷ lệ phổ quát.

## Bộ nhớ và khởi động

| Chỉ số bytes / seconds | Fast | Rescore |
|---|---:|---:|
| CUDA measured peak allocated | 1.118.336.512 | 1.635.197.440 |
| CUDA measured peak reserved | 2.128.609.280 | 2.254.438.400 |
| CUDA load peak allocated | 1.936.234.496 | 1.936.234.496 |
| CUDA load peak reserved | 2.128.609.280 | 2.139.095.040 |
| Process peak RSS | 1.751.040.000 | 2.325.856.256 |
| ASR load s | 3,166660 | 4,459488 |
| LM load s | 0 | 5,750056 |
| Warmup s | 0,700820 | 1,452611 |

Measured CUDA peaks reset sau warmup, vẫn gồm resident weights. PyTorch reserved
không phải tổng VRAM nvidia-smi; RSS là high-water toàn process, không riêng model.
Quan sát batch 1 fit GPU 4GB, không cam kết nhiều worker, training hay CPU fit/tốc độ.
Một lần chạy thứ tự cố định chịu ảnh hưởng nhiệt/load/cache; chưa so HF-small cùng
protocol, không tuyên bố giảm 70% VRAM.

## Tái lập và kiểm tra không inference

Lệnh GPU tái lập (không chạy trong tác vụ docs), dùng output mới:

```bash
PYTHONPATH=.:src python -m benchmarks.runners.compare_small_modes \
  --manifest benchmarks/manifests/viet_bud500_holdout_500.csv \
  --disjoint-from benchmarks/manifests/viet_bud500_dev_250.csv \
  --output-dir outputs/benchmarks/small-public-modes-holdout500-rerun
```

Cần đúng local audio, native checkpoint và cache LM pinned; xem
[benchmark guide](../benchmarks/README.md). Public rescore chỉ cửa sổ đầu 30 giây,
không dùng trực tiếp cho toàn bản ghi dài. Bản audit docs tính lại WER/CER,
S/D/I, exact match, timing và bootstrap từ predictions đã lưu, kiểm tra pairing
ID/reference/audio hash, report completed/500/zero failures và before=after.
Không tải model, không inference, không chỉnh source hay báo cáo/config lịch sử.

VietMed test15 cho thấy rescoring có thể làm xấu WER; xem
[báo cáo VietMed](vietmed100-base-small-fast-rescoring-20261109.md).
Không suy diễn chất lượng ASR từ unit tests hoặc nhận xét clip không reference.
