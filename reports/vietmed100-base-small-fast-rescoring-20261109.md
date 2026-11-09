# VietMed100: base/small × fast/rescoring — kết quả thực 2026-11-09

## Kết luận và phạm vi

Đã chạy inference thực tế và sau đó audit độc lập các transcript đã lưu: **4/4 cấu hình hoàn tất, mỗi cấu hình 100 câu, không lỗi**, `status=completed`, `valid4cellcomplete=true`, `pilot=false`. Không huấn luyện hoặc thay trọng số. Small-rescore có WER/CER thấp nhất trên all100; **small-fast tốt nhất trên test15**. Rescoring không có lợi đồng đều.

Giữ nguyên default base. Small-fast là ứng viên opt-in cân bằng chất lượng/tốc độ; small-rescore chỉ nên dùng khi chấp nhận latency và đã xác nhận trên dữ liệu đích. Không tự promote từ 100 câu chọn sẵn hoặc test15 nhỏ; kết quả VietBud500 trước đó vẫn có CER regression và chưa qua các gate CI. Rollback: đặt `PHOWHISPER_MODEL_PATH=models/phowhisper-base.pt` rồi restart process để xóa cache model. Không thay checkpoint baseline.

## Dữ liệu và protocol

100 utterance, **625 giây = 10 phút 25 giây**, không phải 100 giờ hoặc toàn corpus. Split frozen train70/dev15/test15 có nguồn từ thử nghiệm fine-tune cũ; lần này chỉ dùng checkpoint official, không dùng local fine-tuned base. Preflight lưu kết quả kiểm tra manifest/hash WAV 16 kHz mono và nguồn/speaker không giao nhau giữa split. Clip tối đa 8 giây nên giới hạn cửa sổ đầu của rescorer không cắt mất nội dung ở đây. Không khẳng định dữ liệu chưa từng xuất hiện trong pretraining upstream hoặc đã kiểm tra near-duplicate với dữ liệu upstream; quy mô subset nhỏ hạn chế khả năng tổng quát hóa.

API public giữ nguyên: fast có temperature fallback mặc định Whisper; rescore beam5, alpha/beta/gamma `0.2/-0.02/-0.2`, acoustic gate `-0.020`. So sánh **toàn workflow**, không phải ablation thuần LM; không tune small hoặc dựa reference để chọn cấu hình. Scoring LM public không được sửa dù có hướng cải tiến prefix ở nơi khác.

Một GPU RTX 3050 Laptop, CUDA 13.0, torch 2.14.1+cu130; ASR FP16, LM GPT-2 FP32 CUDA offline/local-only; seed42, 4 threads, batch1, rate1. Bốn fresh process tuần tự, mỗi process chỉ giữ một ASR model; rescore preload LM kể cả câu bị gate, fast không nạp LM. Mỗi cell bỏ một warmup dev[0]. Timing đồng bộ CUDA gồm audio I/O và API inference, không gồm load/warmup. Một lượt thứ tự cố định không chứng minh tốc độ có ý nghĩa thống kê; RTF<1 không phải cam kết streaming hoặc concurrent throughput.

## All100: số mô tả gộp

WER/CER dưới đây là %, corpus-weighted, không trung bình WER từng câu. Chuẩn hóa NFC, casefold, dấu câu thành khoảng trắng, giữ dấu tiếng Việt; không harmonize thanh điệu.

| Cell | WER % | CER % | Word S/D/I | Tổng giây | RTF | Median / p95 giây |
|---|---:|---:|---|---:|---:|---|
| base-fast | 19.6203 | 13.6689 | 289/97/110 | 37.4338 | 0.05989 | 0.37335 / 0.45473 |
| base-rescore | 18.1171 | 13.1891 | 243/115/100 | 75.4868 | 0.12078 | 0.70187 / 1.12339 |
| small-fast | 15.8623 | 11.6275 | 205/89/107 | 74.2621 | 0.11882 | 0.71346 / 1.02669 |
| small-rescore | 15.0712 | 11.1571 | 191/92/98 | 187.6739 | 0.30028 | 1.79726 / 2.64844 |

## Test15 riêng biệt — N nhỏ

| Cell | WER % | CER % |
|---|---:|---:|
| base-fast | 22.3958 | 17.0504 |
| base-rescore | 20.0521 | 15.4947 |
| small-fast | **17.4479** | **12.4456** |
| small-rescore | 18.4896 | 13.2545 |

Small-rescore giảm all100 WER 0.7911 điểm phần trăm so với small-fast nhưng tăng test15 WER 1.0417 điểm; chi phí tổng thời gian khoảng 2.53× small-fast trong lượt này. Không suy ra lợi ích rescoring phổ quát.

## Bộ nhớ

Đơn vị byte, không phải incremental VRAM. Measured peak reset sau warmup nhưng vẫn bao gồm model/LM resident; load peak tách riêng tránh đánh giá thấp VRAM triển khai. RSS là high-water mark toàn process.

| Cell | Measured CUDA allocated | Measured reserved | Load peak allocated | Process RSS peak |
|---|---:|---:|---:|---:|
| base-fast | 377452544 | 601882624 | 584074240 | 1496977408 |
| base-rescore | 902985216 | 996147200 | 806732800 | 2027728896 |
| small-fast | 1119700992 | 2130706432 | 1936234496 | 1715658752 |
| small-rescore | 1642756096 | 2264924160 | 1936234496 | 2296254464 |

## Diagnostics rescoring thực tế

| Cell | Single candidate | Acoustic gate | Thực sự chấm LM | Winner khác beam top1 |
|---|---:|---:|---:|---:|
| base-rescore | 0 | 6 | 94 | 55 |
| small-rescore | 0 | 33 | 67 | 35 |

Gate là hành vi hợp lệ, không phải lỗi/fallback. Winner changed so với beam top1, không so với fast và không đồng nghĩa cải thiện. Handler runner biến warning rescorer thành lỗi; không silent fallback được chấp nhận. Khi nạp LM, transformers có thông báo chọn loss mặc định `ForCausalLMLoss`; đây không phải rescorer fallback.

## PairwiseCI và thắng/hòa/thua

Paired bootstrap theo utterance, 2.000 draws, seed42. CI percentile hai phía 95% và chênh lệch WER `B-A` theo **điểm phần trăm**; âm có lợi B. W/T/L là B ít/bằng/nhiều **word errors S+D+I thực từ jiwer**, không so tỷ lệ làm tròn. p là tỷ phần bootstrap `B-A >= 0`, một phía; `p=0` nghĩa 0/2000 draws, không xác suất bằng không. Không đồng nhất gate p<0.05 một phía với CI hai phía. Các cặp/split dùng chung mẫu, mang tính khám phá, không hiệu chỉnh đa so sánh, không phải các xác nhận độc lập.

| A → B | All100 Δ quan sát | All100 CI95 | p | B W/T/L | Test15 CI95 | p | B W/T/L |
|---|---:|---|---:|---|---|---:|---|
| base-fast → small-fast | -3.7579 | [-5.2799, -2.2798] | 0 | 57/26/17 | [-10.5913, -0.5361] | .0155 | 8/5/2 |
| base-rescore → small-rescore | -3.0459 | [-4.5126, -1.6243] | 0 | 52/26/22 | [-7.4361, 3.1580] | .3365 | 5/5/5 |
| base-fast → base-rescore | -1.5032 | [-2.5743, -0.5751] | .0010 | 34/49/17 | [-4.5921, -0.2481] | .0205 | 7/6/2 |
| small-fast → small-rescore | -0.7911 | [-1.4607, -0.1980] | .0045 | 20/72/8 | [-0.2639, 2.4216] | .9685 | 1/10/4 |

## Provenance và bằng chứng audit

- [Raw report](../outputs/benchmarks/vietmed-base-small-modes-20261109-full/report.json), [comparison](../outputs/benchmarks/vietmed-base-small-modes-20261109-full/comparison.json), [manifest](../outputs/benchmarks/vietmed-base-small-modes-20261109-full/manifest.json); JSONL và resource JSON từng cell nằm cùng thư mục. Không sao chép transcript y tế vào báo cáo.
- [Config curated](../configs/evaluations/vietmed100-base-small-modes.json) lưu SHA-256 raw report/comparison/manifest và cả bốn JSONL, CI, counters và quyết định. Raw report chứa đầy đủ source/model/data/LM hashes trước/sau, environment, load/warmup và train/dev/test summaries.
- HEAD thực `a051afe91f82440cab09f25365400875bc3f3478` cùng dirty snapshot được lưu trong raw report; không giả định clean checkout.
- Dataset revision `dc1a2ebef3ffa772a955f2c9b4412ad8722c0b74`; LM `NlpHUST/gpt2-vietnamese` revision `65818d14816b42be09e2201933bf07106d9a2647`.
- Official `vinai/PhoWhisper-base` revision `7ebdb9e88f5cc5271fb88f4d642c82ff9388650e`, native SHA `0a8a244bbe70938bce477bfb3e595b3743dc2937c43f5df25c59aa3783601c28`.
- Official `vinai/PhoWhisper-small` revision `a86b604c346caf7148c37512eafe783a16420adb`, native SHA `15632cc15395f5ee8d9fec382133f3d6429c63d1c2209518984fb126058e85c5`. Không phải local trained checkpoint.
- `before == after` toàn bộ snapshot đã được audit. Kiểm tra SHA-256 trực tiếp sau benchmark cũng xác nhận baseline `models/phowhisper-base.pt` giữ nguyên SHA `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d`, cùng hai file nguồn/config cũ; đường dẫn model mặc định vẫn là base.

Audit offline recompute `summarize` tất cả cell/split và `compare` toàn bộ so với raw JSON bằng exact equality; xác nhận 400 rows, thứ tự ID/reference/path/audio hash và duration khớp manifest, resource JSON khớp report. Negative controls thiếu row/đảo thứ tự đều bị validator từ chối. Audit hoàn tất trong 0.988 giây, không import torch hoặc chạy lại inference.

Kiểm thử cuối: 128 unittest, không failure, một historical-artifact test opt-in được skip; loại riêng một test parity dùng model thật vì benchmark bốn cấu hình đã được chạy tách biệt. Kaggle suite 22 test đạt; `git diff --check` đạt. Có thể chạy kiểm thử logic runner bằng `PYTHONPATH=.:src .venv-benchmark/bin/python -m unittest tests.unit.test_vietmed_mode_comparison`. Không commit/push hoặc thay model mặc định trong tác vụ này.
