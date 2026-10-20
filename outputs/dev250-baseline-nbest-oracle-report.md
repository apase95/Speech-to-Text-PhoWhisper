# PhoWhisper PyTorch - Baseline Lock, Normalization Audit & Dev250 Oracle WER Report

## Thông tin thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-21 |
| Nguồn dữ liệu | Viet-BUD500, 250 mẫu dev cố định (`audio1001` đến `audio1250`) |
| Manifest | `benchmarks/manifests/viet_bud500_dev_250.csv` |
| Manifest SHA-256 | `5faf06364d345a63a3b662440187fe40d5936df384aeabb453bb7a7233c442d0` |
| Model | PhoWhisper-base, checkpoint PyTorch đã chuyển đổi |
| Checkpoint SHA-256 | `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d` |
| Runtime | OpenAI Whisper `20250625`, PyTorch `2.14.1+cu130` |
| Thiết bị | NVIDIA GeForce RTX 3050 Laptop GPU, FP16 |
| Cấu hình N-best | `beam_size=5`, `without_timestamps=True`, `language=vi`, `task=transcribe`, `fp16=True` |

---

## 1. Khóa Baseline Hiện Tại & Xác Minh Chi Tiết Con Số 164

Thực hiện audit lại toàn bộ dữ liệu dự đoán của baseline greedy ([dev250-stretch10-20261020T165633Z](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/outputs/benchmarks/dev250-stretch10-20261020T165633Z/summary.json)):

* **Tổng số từ tham chiếu (Ground truth words)**: 2.287 từ
* **Số từ nhận dạng đúng (Hits)**: 2.059 từ
* **Tổng số lỗi từ**: 254 lỗi
  * **Substitution (Thay thế từ)**: **200 từ**
  * **Deletion (Mất từ)**: 28 từ
  * **Insertion (Chèn từ thừa)**: 26 từ
* **Corpus WER**: **11,11%** (254 / 2.287)
* **Corpus CER**: **6,42%**
* **Số câu đúng hoàn toàn (Exact Match)**: 98/250 (39,20%)

### Nguồn gốc và ý nghĩa chính xác của con số 164:
* Con số **164** là **số lượng cụm/vị trí thay thế (Substitution Chunks)** ở cấp độ căn chỉnh câu (sentence-level alignment) do JiWER trích xuất.
* Trong 164 chunks này:
  * **135 chunks là thay thế 1-1** (ví dụ: *con* $\to$ *còn*, *dò* $\to$ *do*).
  * **29 chunks là thay thế đa từ (Multi-word Substitution Chunks)** (ví dụ: *cái quyền* $\to$ *kê quyển* [2 từ], *kiên nhẫn* $\to$ *chi nhật* [2 từ], *khai lễ* $\to$ *khay lệch* [2 từ], *khóa học* $\to$ *khoá hóc* [2 từ]).
* Tổng số từ bị thay thế thực tế trong 164 chunks này là **200 từ**.

---

## 2. Audit Normalization: Bất Đồng Vị Trí Đặt Dấu Thanh

Trong tiếng Việt, tồn tại 2 trường phái đặt dấu thanh chính tả:
1. **Kiểu cũ (truyền thống)**: Dấu thanh đặt ở nguyên âm đầu trong các âm mở `oa`, `oe`, `uy` (ví dụ: *hóa, khỏe, thủy, tòa, hòa*).
2. **Kiểu mới (chuẩn giáo dục hiện đại)**: Dấu thanh đặt ở nguyên âm thứ hai (ví dụ: *hoá, khoẻ, thuỷ, toà, hoà*).

### Kết quả đếm bất đồng trên tập Dev 250:
* Quét toàn bộ 250 câu cho thấy có **3 trường hợp** trong dev set mà sai khác duy nhất giữa ground truth và dự đoán chỉ là vị trí đặt dấu thanh:
  * `[audio1048]`: *khóa* vs *khoá* (xuất hiện 2 lần trong câu).
  * `[audio1230]`: *hóa* vs *hoá* (1 lần).
* Đã triển khai hàm chuẩn hóa vị trí dấu thanh [`normalize_vietnamese_harmonized()`](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/benchmarks/metrics.py#L38-L41) vào [benchmarks/metrics.py](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/benchmarks/metrics.py).

### Bảng so sánh song song Metric Cũ và Metric Mới trên Baseline:

| Metric | Metric Cũ (`normalize_vietnamese`) | Metric Mới (`normalize_vietnamese_harmonized`) | Chênh lệch |
|---|---:|---:|---:|
| **Corpus WER** | **11,11%** (0.111063) | **10,98%** (0.109751) | **-0,13%** |
| **Corpus CER** | **6,42%** (0.064177) | **6,32%** (0.063230) | **-0,10%** |
| **Substitutions** | 200 từ | 197 từ | -3 từ |
| **Deletions** | 28 từ | 28 từ | 0 |
| **Insertions** | 26 từ | 26 từ | 0 |
| **Câu đúng hoàn toàn** | 98/250 (39,20%) | 98/250 (39,20%) | 0 |

---

## 3. Sinh N-best trên Dev 250 (Beam Size = 5)

Đã xây dựng module [tools/generate_dev_nbest.py](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/tools/generate_dev_nbest.py) can thiệp trực tiếp vào `whisper.decoding.BeamSearchDecoder.finalize()` để trích xuất đầy đủ $N$ giả thuyết từ chùm tia.

* **Thư mục lưu trữ**: `outputs/benchmarks/dev250-nbest-beam5-20261020T172658Z/`
* **File dữ liệu**: `nbest_predictions.jsonl` (250 dòng JSON).
* **Cấu trúc mỗi mẫu**:
  ```json
  {
    "id": "audio1018",
    "audio_path": "data/benchmark/viet-bud500/audio/audio1018.wav",
    "reference": "quá lớn của kẻ đã bị hắn hớt tay trên",
    "top1_candidate": { "rank": 1, "text": "quá lớn của kẻ đã bị hắn hớt cay trên.", "avg_logprob": -0.047 },
    "oracle_candidate_std": { "rank": 2, "text": "quá lớn của kẻ đã bị hắn hớt tay trên.", "avg_logprob": -0.055 },
    "all_candidates": [
      { "rank": 1, "text": "...hớt cay trên.", "avg_logprob": -0.047 },
      { "rank": 2, "text": "...hớt tay trên.", "avg_logprob": -0.055 },
      { "rank": 3, "text": "...hớt cây trên.", "avg_logprob": -0.130 },
      { "rank": 4, "text": "...hết tay trên.", "avg_logprob": -0.159 },
      { "rank": 5, "text": "...vớt tay trên.", "avg_logprob": -0.172 }
    ]
  }
  ```

---

## 4. Đo Oracle WER & Phân Tích Dư Địa (Headroom)

Oracle WER là mức lỗi nhỏ nhất về mặt lý thuyết nếu có một bộ Rescorer hoàn hảo chọn ra candidate tốt nhất trong 5-best candidates cho từng audio.

### Bảng Đối Chiếu Toàn Diện: Baseline vs Beam Top-1 vs Oracle

| Chỉ số | Greedy Baseline | Beam Top-1 (Rank 1) | **Oracle (5-best)** | **Cải thiện Oracle so với Baseline** |
|---|---:|---:|---:|---:|
| **WER (Metric Cũ)** | 11,11% | 11,67% | **6,47%** | **-4,64%** (giảm 41,8% lỗi) |
| **WER (Metric Mới)** | 10,98% | 11,54% | **6,43%** | **-4,55%** (giảm 41,4% lỗi) |
| **CER (Metric Cũ)** | 6,42% | 7,70% | **4,52%** | **-1,90%** |
| **CER (Metric Mới)** | 6,32% | 7,61% | **4,47%** | **-1,85%** |
| **Câu đúng hoàn toàn** | 98/250 (39,20%) | 87/250 (34,80%) | **143/250 (57,20%)** | **+45 câu** (+18,0 điểm %) |
| Substitutions | 200 | 171 | **90** | **-110 lỗi từ** |
| Deletions | 28 | 26 | **18** | **-10 lỗi từ** |
| Insertions | 26 | 70 | **40** | +14 lỗi từ |
| Tổng số từ lỗi | 254 | 267 | **148** | **-106 lỗi từ** |

### Phân phối vị trí (Rank) của Oracle Candidates:

```
  Rank 1 (Top-1 đã là tốt nhất):  150 clips (60,0%)
  Rank 2:                         51 clips (20,4%)
  Rank 3:                         22 clips (8,8%)
  Rank 4:                         21 clips (8,4%)
  Rank 5:                          6 clips (2,4%)
```
* **Có đến 100/250 clips (40,0%)** sở hữu candidate tốt hơn nằm ở các Rank 2, 3, 4, 5.
* **Có đúng 56 câu** đạt độ chính xác 100% (Exact Match) nếu chọn đúng candidate ở Rank 2, 3, 4, 5 trong khi Rank 1 bị sai!

### Các trường hợp ngữ nghĩa tiêu biểu được cứu bởi Beam Candidates:

1. **Thành ngữ / Cụm từ cố định**:
   * `[audio1018]`:
     * Reference: *quá lớn của kẻ đã bị hắn **hớt tay trên***
     * Top-1 (Rank 1, logprob: -0.047): *quá lớn của kẻ đã bị hắn **hớt cay trên*** (vô nghĩa)
     * Oracle (Rank 2, logprob: -0.055): *quá lớn của kẻ đã bị hắn **hớt tay trên*** (khớp 100%)
     * *Nhận xét*: Điểm âm học chỉ chênh lệch 0,008 logprob, một mô hình ngữ nghĩa (LM) sẽ ngay lập tức thưởng điểm cho cụm *"hớt tay trên"*.
2. **Từ vựng chính thống vs Từ nghe nhầm**:
   * `[audio1025]`:
     * Reference: ***biện pháp** bảo mật cần thiết tuy nhiên*
     * Top-1 (Rank 1, logprob: -0.107): ***tiện pháp** bảo mật cần thiết tuy nhiên*
     * Oracle (Rank 3, logprob: -0.142): ***biện pháp** bảo mật cần thiết tuy nhiên* (khớp 100%)
3. **Từ láy tượng thanh**:
   * `[audio1001]`:
     * Reference: *trái tim của hoàng phượng đập **thình***
     * Top-1 (Rank 1, logprob: -0.034): *trái tim của hoàng phượng đập **thỉnh***
     * Oracle (Rank 2, logprob: -0.129): *trái tim của hoàng phượng đập **thình*** (khớp 100%)

---

## 5. Kết Luận & Quyết Định Kỹ Thuật (GO / NO-GO)

### **Quyết định: GO (KHẲNG ĐỊNH TIẾP TỤC RESCORING)**

1. **Cơ sở thực nghiệm không thể chối cãi**:
   * Oracle WER đạt **6,47%**, mở ra **dư địa cải thiện khổng lồ 4,64% WER** so với baseline greedy hiện tại.
   * Số câu đúng hoàn toàn có tiềm năng tăng vọt từ **39,20% lên 57,20%**.
   * Số lỗi thay thế từ giảm hơn một nửa (từ 200 từ xuống còn 90 từ).
2. **Hiện tượng âm học đã được xác nhận**:
   * Các từ đúng không hề bị biến mất trong quá trình encoder xử lý; chúng **thực sự đã được sinh ra và nằm sẵn ở Rank 2 và Rank 3 của Whisper beam search**, chỉ vì điểm logprob âm học của Whisper-base bị lệch nhẹ vài phần trăm so với Rank 1.
   * Lớp ngữ nghĩa (Semantic Rescorer - ví dụ PhoBERT hoặc KenLM) hoàn toàn có khả năng bù đắp khoảng cách logprob nhỏ này để kéo các candidate đúng lên Rank 1.
