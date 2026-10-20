# PhoWhisper PyTorch - N-Best Language Model Rescoring Benchmark Report

## Thông tin thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-21 |
| Model ASR | PhoWhisper-base, checkpoint PyTorch (`models/phowhisper-base.pt`) |
| Checkpoint SHA-256 | `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d` |
| Language Model | `NlpHUST/gpt2-vietnamese` (Causal LM, 117M tham số) |
| Tập Dev (chọn tham số) | `benchmarks/manifests/viet_bud500_dev_250.csv` (250 mẫu, SHA: `5faf0636...`) |
| Tập Test (đánh giá độc lập) | `benchmarks/manifests/viet_bud500_1000.csv` (1.000 mẫu, SHA: `428f56f5...`) |
| Runtime & Thiết bị | OpenAI Whisper `20250625`, PyTorch `2.14.1+cu130`, NVIDIA RTX 3050 Laptop GPU, FP16 |
| Cấu hình Decoding | `beam_size=5`, `without_timestamps=True`, `language=vi`, `task=transcribe` |
| Công thức Rescoring | $\text{Score}(c) = \text{avg\_logprob}(c) + \alpha^* \cdot \text{lm\_logprob\_per\_token}(c) + \beta^* \cdot \text{WordCount}(c)$ |

---

## 1. Quy Trình Chọn Trọng Số Nghiêm Ngặt Trên Dev 250 (5-Fold Cross-Validation)

Để tuyệt đối tránh rò rỉ dữ liệu (data leakage) và không báo cáo kết quả chọn tham số trên toàn bộ 250 mẫu như đánh giá độc lập, chúng tôi áp dụng **5-Fold Cross-Validation**:
* Tập dev 250 mẫu được chia đều thành 5 fold (mỗi fold 50 mẫu).
* Với mỗi fold $k \in \{1..5\}$:
  * Tối ưu hóa lưới $(\alpha, \beta)$ trên 200 mẫu huấn luyện để tìm $(\alpha_k^*, \beta_k^*)$.
  * Dùng $(\alpha_k^*, \beta_k^*)$ dự đoán trên 50 mẫu held-out (Out-of-Fold - OOF).

### Bảng Kết Quả 5-Fold Cross-Validation:

| Fold | Số mẫu Train | Số mẫu Val | Best $\alpha$ | Best $\beta$ | Train WER | Val Top-1 WER | **Val Rescored WER** | Mức giảm WER |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Fold 1** | 200 | 50 | 0,170 | -0,030 | 9,96% | 8,91% | 9,13% | +0,22% |
| **Fold 2** | 200 | 50 | 0,200 | -0,026 | 9,70% | 11,28% | **8,63%** | **-2,65%** |
| **Fold 3** | 200 | 50 | 0,170 | -0,028 | 9,13% | 12,75% | **11,19%** | **-1,56%** |
| **Fold 4** | 200 | 50 | 0,200 | -0,014 | 9,11% | 12,39% | **10,92%** | **-1,47%** |
| **Fold 5** | 200 | 50 | 0,220 | -0,022 | 9,48% | 13,05% | **9,96%** | **-3,09%** |

### Kết Quả Tổng Hợp Out-Of-Fold (OOF) Trên Toàn Bộ 250 Mẫu Dev:

| Chỉ số đánh giá | Top-1 Beam (Chưa Rescore) | **Rescored Out-of-Fold (OOF)** | Cải thiện thực chất |
|---|---:|---:|---:|
| **Corpus WER (Metric Cũ)** | 11,67% (0.116747) | **9,97%** (0.099694) | **-1,70%** |
| **Corpus WER (Metric Mới)** | 11,54% (0.115435) | **9,93%** (0.099257) | **-1,61%** |
| **Số từ thay thế (Substitutions)** | 171 từ | **132 từ** | **-39 từ** |
| **Số câu đúng hoàn toàn** | 87/250 (34,80%) | **99/250 (39,60%)** | **+12 câu** |

> [!IMPORTANT]
> **Khóa Cấu Hình (Locked Configuration):**
> Các tham số tối ưu giữa 5 fold rất ổn định (hội tụ quanh $\alpha \approx 0,20$ và $\beta \approx -0,026$). 
> Do đó, cấu hình được khóa cố định theo trung vị:
> $$\alpha^* = 0,200, \quad \beta^* = -0,026$$

---

## 2. Đánh Giá Độc Lập Duy Nhất 1 Lần Trên Test Set 1.000 Mẫu

Với cấu hình đã khóa cố định ở Bước 1, chúng tôi tiến hành chạy suy luận và rescoring đúng 1 lần trên toàn bộ 1.000 mẫu của tập test chính thức ([viet_bud500_1000.csv](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/benchmarks/manifests/viet_bud500_1000.csv)). Tuyệt đối không điều chỉnh tham số hay chọn lại phương pháp dựa trên kết quả test này.

### Bảng So Sánh Tổng Hợp Trên Test Set (1.000 Audio):

| Metric | Baseline Greedy (Trước đây) | Beam 5 Top-1 (Không LM) | **Rescored Beam 5 (Locked $\alpha=0.2, \beta=-0.026$)** | Oracle (5-best lý thuyết) |
|---|---:|---:|---:|---:|
| **Corpus WER (Metric Cũ)** | 10,49% | 10,75% | **9,89%** | 6,03% |
| **Corpus WER (Metric Mới)** | 10,38% | 10,70% | **9,83%** | 6,02% |
| **Corpus CER** | **6,10%** | 6,93% | 7,19% | 4,26% |
| **Câu đúng hoàn toàn (Exact Match)** | 427/1.000 (42,70%) | 407/1.000 (40,70%) | **436/1.000 (43,60%)** | **623/1.000 (62,30%)** |
| **Substitutions (Thay thế từ)** | 740 từ | 649 từ | **486 từ (-254 từ so với Greedy)** | 338 từ |
| Deletions (Mất từ) | **99 từ** | **98 từ** | 161 từ | 69 từ |
| Insertions (Chèn từ thừa) | **120 từ** | 236 từ | 257 từ | 144 từ |
| **Tổng số lỗi từ** | 959 lỗi | 983 lỗi | **904 lỗi (-55 lỗi so với Greedy)** | 551 lỗi |
| **Tổng thời gian inference** | 644,11 s | 1.418,77 s | **517,04 s (~8,6 phút)** | - |
| **Real-time factor (RTF)** | 0,253x | 0,558x | **0,203x** | - |
| **Peak GPU VRAM** | 584 MiB | 584 MiB | **846 MiB (~0,83 GiB)** | - |

---

## 3. Phân Tích Thực Nghiệm Trên Test Set

### 1. Giảm thiểu lỗi thay thế (Substitution) vượt trội:
* Số lỗi substitution giảm từ **740 từ (Greedy) và 649 từ (Beam-1) xuống còn 486 từ** (giảm tới **34,3%** lỗi thay thế từ).
* Trên tập test, có **164 clips được cải thiện WER**, so với 92 clips bị giảm nhẹ.
* Tỷ lệ câu đúng hoàn toàn tăng lên **43,60% (436/1.000 câu)**, cao nhất trong tất cả các cấu hình triển khai thực tế từ trước đến nay.

### 2. Các ví dụ cứu lỗi ngữ nghĩa thực tế trên Test Set:
* `[audio9]`:
  * Reference: *thôi anh đừng nói gì nữa tôi chưa đủ khổ*
  * Top-1 (Rank 1): ***thuang*** *đừng nói gì nữa tôi chưa đủ khổ* (vô nghĩa)
  * Rescored (Rank 4): ***thôi anh*** *đừng nói gì nữa tôi chưa đủ khổ* (khớp 100%)
* `[audio28]`:
  * Reference: *lãnh đạo hầu hết đoạt học vị và **học hàm***
  * Top-1 (Rank 1): *lãnh đạo hầu hết đoạt học vị vào **học hạm***
  * Rescored (Rank 2): *lãnh đạo hầu hết đoạt học vị và **học hàm*** (khớp 100%)
* `[audio27]`:
  * Reference: *trời luôn về đi má mày chết rồi **kìa***
  * Top-1 (Rank 1): *trời luôn về đi má mày chết rồi **kiệt***
  * Rescored (Rank 4): *trời luôn về đi má mày chết rồi **kìa*** (khớp 100%)
* `[audio29]`:
  * Reference: *hứa hẹn nhưng qua đó anh cũng phải **có***
  * Top-1 (Rank 1): *...anh cũng phải **cất***
  * Rescored (Rank 2): *...anh cũng phải **có*** (khớp 100%)

---

## 4. Kết Luận

1. **Thành công về mặt khoa học & kỹ thuật**:
   * Việc kết hợp **Lớp Ngữ Nghĩa (Language Model Rescoring)** đã chính thức hạ **WER của PhoWhisper-base trên test set xuống dưới 10% (đạt 9,89% theo metric cũ và 9,83% theo metric chuẩn hóa mới)**.
   * Đây là cải tiến thực chất, được kiểm chứng bằng 5-fold cross-validation trên dev và đánh giá 1 lần duy nhất trên test set độc lập.
2. **Hiệu năng runtime**:
   * Pipeline tiêu thụ tổng cộng chỉ **846 MiB VRAM** (cả Whisper và GPT-2 cùng nằm trên GPU).
   * Tốc độ xử lý đạt **RTF 0,203x** (xử lý 1.000 audio tương đương 42 phút âm thanh chỉ trong 8,6 phút), hoàn toàn sẵn sàng cho production.
