# Đánh giá PhoWhisper base / small trên Viet-BUD500 — 2026-11-09

## Kết luận: giữ base vì gate không đạt

**Không promote small. Default vẫn là `models/phowhisper-base.pt`.** Controller đã chạy pilot 2 mẫu, dev250 và holdout500 thực tế, không rescoring. Trên holdout500, small giảm WER từ 10.8155255% xuống 9.4853903% (-1.3301352 điểm phần trăm; -12.2984% tương đối), nhưng CER tăng và khoảng tin cậy WER hai phía 95% vẫn chứa 0. Đây là hai điều kiện bắt buộc bị thất bại, không phải quyết định chọn base vì chưa chạy đánh giá.

Không đổi model tự động, không training mới. Small là candidate có sẵn để chọn thủ công qua biến môi trường; chưa thay thế default. Raw outputs và quyết định `Manual review required; base default unchanged` trong comparison được giữ nguyên; kết luận gate dưới đây là đánh giá sau khi đọc kết quả.

## Dữ liệu, giao thức và môi trường

- Manifest dev: `benchmarks/manifests/viet_bud500_dev_250.csv`, 250 mẫu, 632.0968707482992 giây; SHA-256 `5faf06364d345a63a3b662440187fe40d5936df384aeabb453bb7a7233c442d0`.
- Manifest holdout: `benchmarks/manifests/viet_bud500_holdout_500.csv`, 500 mẫu, 1300.656099773244 giây; SHA-256 `46f6777ff565a168cae35e31d0c6e0a1f2df9f55e0b4b4e705860e3f12621a0b`.
- Tổng 750 utterance khác nhau; comparator kiểm tra không trùng ID hoặc hash audio giữa hai manifest. Tên dev/holdout là partition của repository từ test split của dataset gốc, không chứng minh dữ liệu chưa từng xuất hiện trong pretraining. Đây không phải đánh giá toàn bộ 500 giờ.
- Cùng Vietnamese transcription, greedy `temperature=0`, `beam_size=None`, `best_of=None`, seed 42, FP16 CUDA, không timestamps, không previous-text conditioning, không padding/stretch, không LM rescoring. Chuẩn hóa NFC, lowercase, dấu câu/khoảng trắng; giữ dấu tiếng Việt và chữ số. WER/CER là corpus metrics, không trung bình tỷ lệ từng câu.
- Mỗi model một process mới, chạy tuần tự base rồi small, warmup 0; mẫu đầu cold. Giao thức đánh giá pin greedy riêng; không thay decoding defaults của pipeline thông thường.
- GPU NVIDIA GeForce RTX 3050 Laptop GPU (4 GB; PyTorch báo capacity 3951296512 byte). Python 3.14.8, torch 2.14.1+cu130, CUDA runtime 13.0, Whisper 20250625, Linux x86_64.
- HEAD được ghi nhận: `a051afe91f82440cab09f25365400875bc3f3478`, working tree dirty. HEAD không đại diện cho cây sạch; từng summary lưu source hashes, tracked diff hash `4e966378e77babe6942552bb731033c5355fac7cec3f80c91ebb8f4fe9278a3e` và status hash `99496d4a07dd49ecaab217d5ef9317f07ce8a8c42e644d80a985db2127522c80`.

## Kết quả đo

| Tập / chỉ số | Base | Small |
|---|---:|---:|
| Dev250 WER (%) | 11.1062527 | 9.9256668 |
| Dev250 CER (%) | 6.4176749 | 6.6596528 |
| Dev250 RTF | 0.10082757 | 0.16629037 |
| Holdout500 WER (%) | 10.8155255 | 9.4853903 |
| Holdout500 CER (%) | 6.2704875 | 6.4872581 |
| Holdout500 RTF | 0.12567409 | 0.20175968 |
| Holdout500 latency median (s) | 0.301863 | 0.529083 |
| Holdout500 latency p95 (s) | 0.489684 | 0.707670 |
| Holdout500 model load (s) | 1.301087 | 3.494976 |
| Holdout500 peak CUDA allocated (byte) | 584074240 | 1936234496 |
| Holdout500 peak CUDA reserved (byte) | 601882624 | 2128609280 |
| Holdout500 peak process RSS (byte) | 1532706816 | 1749868544 |
| Failures (dev / holdout) | 0 / 0 | 0 / 0 |

Holdout small/base: RTF 1.6054x, median 1.7527x, p95 1.4452x, load 2.6862x, RSS 1.1417x, CUDA allocated 3.3150x, reserved 3.5366x. CER tăng 0.216771 điểm phần trăm (+3.4570% tương đối). Word substitutions/deletions/insertions là 397/41/58 với base và 271/60/104 với small: substitutions giảm nhưng deletions và insertions tăng. Không có bằng chứng ở đây để kết luận nguyên nhân.

### Bootstrap và gate đã công bố

Paired bootstrap 2.000 resamples theo utterance, seed 42, B = small, A = base, đơn vị chênh lệch là điểm phần trăm:

- Dev250: CI hai phía 95% [-2.80294221, +0.39702678], p một phía 0.0725.
- Holdout500: CI hai phía 95% **[-2.75725059, +0.04330839]**, p một phía **0.0295**. Hàm hiện tại trả `statistically_significant=true` vì dùng `p_value < 0.05` một phía. Điều này không có nghĩa CI hai phía 95% loại trừ 0; CI thực tế chứa 0. Bootstrap mean difference -1.36491517 không phải corpus point estimate -1.33013519.

| Điều kiện promotion | Quan sát holdout | Kết luận |
|---|---|---|
| Full frozen holdout500, không failures | 500 mẫu/model, 0 failures | Đạt |
| Small WER < base WER | 9.4853903% < 10.8155255% | Đạt |
| Small CER <= base CER | 6.4872581% > 6.2704875% | **Không đạt** |
| Upper bound CI WER 95% < 0 | +0.04330839 | **Không đạt** |
| RTF < 1 | 0.20175968 | Đạt trong lượt đo |
| CUDA allocated và reserved fit | 1936234496 / 2128609280 < capacity 3951296512 byte | Chạy thành công trong điều kiện đo; không đảm bảo tải đồng thời |

Gate tổng thể **FAIL**; không promote dù WER point estimate tốt hơn. Không dùng holdout để chỉnh tham số sau kết quả này.

## Provenance và bảo toàn rollback

Thư mục checkpoint đo thực tế (đường dẫn tương đối từ root repository):

```text
models/official/acquisition-20261109/vinai/PhoWhisper-base/7ebdb9e88f5cc5271fb88f4d642c82ff9388650e/native.pt
models/official/acquisition-20261109/vinai/PhoWhisper-small/a86b604c346caf7148c37512eafe783a16420adb/native.pt
```

Mỗi file có sidecar `native.pt.provenance.json`, origin `huggingface`, model ID lần lượt `vinai/PhoWhisper-base` và `vinai/PhoWhisper-small`; requested/resolved revision đúng SHA trong đường dẫn.

| SHA-256 | Base official | Small official |
|---|---|---|
| Source weights | `910b9e8875497078efa83dc506ea7ee378e4e759a206ceecbf23e088aec9b9f8` | `9ec6f07c5bd321fc8f477cf778c43c3fd94b0e33a9f20283fc61cc6d22cbfded` |
| Config | `c75f5d458f497b7ea8748ab0c857db7ce2fd274b64cc70df30b17cb84723caf0` | `c9d7e721876727d437cfb145229936926b439d1ab483db572e5f9b5fb6b6d0bf` |
| Native | `0a8a244bbe70938bce477bfb3e595b3743dc2937c43f5df25c59aa3783601c28` | `15632cc15395f5ee8d9fec382133f3d6429c63d1c2209518984fb126058e85c5` |

Base dimensions: state 512, 6 encoder/decoder layers, 8 heads. Small: state 768, 12 encoder/decoder layers, 12 heads. Cả hai 80 mels, audio context 1500, text context 448, vocabulary 51865. Conversion timestamps UTC lần lượt 16:21:56 và 16:29:21 ngày 2026-11-09.

Hash trước và sau đánh giá xác nhận các file cũ được bảo toàn: `models/phowhisper-base.pt` SHA-256 `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d`; `models/pytorch_model.bin` SHA-256 `910b9e8875497078efa83dc506ea7ee378e4e759a206ceecbf23e088aec9b9f8`; `models/config.json` SHA-256 `c75f5d458f497b7ea8748ab0c857db7ce2fd274b64cc70df30b17cb84723caf0`. Source/config base tải mới khớp hash cũ. Native mới có hash file khác sau reserialization; kiểm tra CPU bằng `torch.equal` xác nhận dimensions, tập key, dtype và toàn bộ **245 state tensors giống hệt** native base cũ.

Lần tải base đầu tiên bị ngắt bởi timeout của công cụ sau 120 giây, để lại dữ liệu tạm cô lập tại official root ban đầu. Lượt tải lại thành công dùng `acquisition-20261109`, với thời gian chờ dài hơn; lượt tải small cũng thành công. Không xóa dữ liệu tạm/không rõ nguồn. Weights mới nằm dưới `models/official/` đã được ignore; không đưa weights vào commit.

## Artifacts gốc và tái lập

- [Dev comparison](../outputs/benchmarks/official-base-small-dev250-20261109/comparison.json), summaries: `base/base-20261109T163007Z/summary.json`, `small/small-20261109T163114Z/summary.json` dưới cùng thư mục dev.
- [Holdout comparison](../outputs/benchmarks/official-base-small-holdout500-20261109/comparison.json), summaries: `base/base-20261109T163325Z/summary.json`, `small/small-20261109T163613Z/summary.json` dưới cùng thư mục holdout. Mỗi run giữ `predictions.jsonl` bên cạnh summary.
- [Artifact quyết định máy đọc](../configs/evaluations/phowhisper-base-small-vietbud500.json); [giao thức và lệnh rerun](../docs/phowhisper-small-evaluation.md).

Lệnh rerun từ repository root, dùng đúng checkpoint đã đo; thư mục output mới phải chưa tồn tại (không chạy các lệnh này trong tác vụ tài liệu):

```sh
BASE=models/official/acquisition-20261109/vinai/PhoWhisper-base/7ebdb9e88f5cc5271fb88f4d642c82ff9388650e/native.pt
SMALL=models/official/acquisition-20261109/vinai/PhoWhisper-small/a86b604c346caf7148c37512eafe783a16420adb/native.pt
PYTHONPATH=.:src .venv-benchmark/bin/python -m benchmarks.runners.compare_model_sizes --base-checkpoint "$BASE" --small-checkpoint "$SMALL" --manifest benchmarks/manifests/viet_bud500_dev_250.csv --disjoint-from benchmarks/manifests/viet_bud500_holdout_500.csv --output-dir outputs/benchmarks/official-base-small-dev250-rerun --warmup 0
PYTHONPATH=.:src .venv-benchmark/bin/python -m benchmarks.runners.compare_model_sizes --base-checkpoint "$BASE" --small-checkpoint "$SMALL" --manifest benchmarks/manifests/viet_bud500_holdout_500.csv --disjoint-from benchmarks/manifests/viet_bud500_dev_250.csv --output-dir outputs/benchmarks/official-base-small-holdout500-rerun --warmup 0
```

Để thử candidate có chủ đích, đặt `PHOWHISPER_MODEL_PATH` tới đường dẫn small ở trên rồi restart process. Rollback: `export PHOWHISPER_MODEL_PATH=models/phowhisper-base.pt` rồi restart workers để xóa model cache (deployment nên dùng đường dẫn tuyệt đối). Default vẫn dùng base cũ nếu không override.

## Giới hạn diễn giải

Các câu ngắn khoảng 2.5 giây; kết quả không chứng minh chất lượng long-audio, VRAM cho audio dài hoặc concurrency. Chỉ một lượt/model, thứ tự cố định base → small, không warmup; nhiệt độ GPU/cache và tải cạnh tranh có thể ảnh hưởng timing. Không tuyên bố khác biệt latency có ý nghĩa thống kê hoặc steady-state. RTF gồm đọc waveform/tiền xử lý/decoding, không gồm model load hay `sf.info`. RSS là Linux process high-water; CUDA allocated/reserved là allocator peaks, không phải tổng VRAM `nvidia-smi` và không chứng minh toàn bộ capacity luôn sẵn có. Không so trực tiếp với benchmark lịch sử khác fallback temperatures/decoding/rescoring.
