# PhoWhisper PyTorch - PhoBERT vs GPT-2 Challenger Report

## Thông tin thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-21 |
| Model ASR | PhoWhisper-base, checkpoint PyTorch (`models/phowhisper-base.pt`) |
| Baseline ASR | Nhánh A: Beam 5 gốc (`benchmarks.run`), WER 9,57% |
| Causal LM (Đương kim) | `NlpHUST/gpt2-vietnamese` (117M params, Causal Auto-regressive LM) |
| Masked LM (Challenger) | `vinai/phobert-base-v2` (135M params, Bidirectional Masked LM) |
| Phương pháp chấm điểm PhoBERT | **Pseudo-Log-Likelihood (PLL)** dạng vector hóa theo batch + `pyvi` tách từ |
| Tập Dev (tối ưu trọng số) | `benchmarks/manifests/viet_bud500_dev_250.csv` (250 mẫu) |
| Tập Test (đánh giá độc lập) | `benchmarks/manifests/viet_bud500_1000.csv` (1.000 mẫu) |
| Thư mục kết quả | `outputs/benchmarks/test1000-phobert-challenger-20261020T193348Z` |

---

## 1. Kết Quả 5-Fold Cross-Validation Trên Dev 250 (PhoBERT PLL)

Lưới siêu tham số $(\alpha, \beta, \gamma)$ được tìm kiếm qua 5 fold trên Dev 250:

| Fold | Train Samples | Val Samples | $\alpha^*$ tối ưu | $\beta^*$ tối ưu | $\gamma^*$ tối ưu | Val Top-1 WER | **Val PhoBERT WER** | Mức biến thiên |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| **Fold 1** | 200 | 50 | 0,130 | -0,020 | -0,20 | 7,17% | 7,83% | +0,66% |
| **Fold 2** | 200 | 50 | 0,150 | +0,010 | -0,20 | 10,84% | **7,08%** | **-3,76%** |
| **Fold 3** | 200 | 50 | 0,200 | -0,020 | -0,15 | 10,29% | **10,07%** | **-0,22%** |
| **Fold 4** | 200 | 50 | 0,110 | +0,016 | -0,20 | 13,45% | **10,50%** | **-2,95%** |
| **Fold 5** | 200 | 50 | 0,060 | -0,008 | -0,15 | 9,73% | **8,85%** | **-0,88%** |

* **OOF Baseline Top-1 WER**: 10,32%.
* **OOF PhoBERT Rescored WER**: **8,88%** (giảm 1,44 điểm % OOF).
* **Tham số trung vị (Median) được khóa cho Test**:
  $$\alpha^* = 0,130 \quad (\text{LM Weight}), \qquad \beta^* = -0,008 \quad (\text{Length Penalty}), \qquad \gamma^* = -0,20 \quad (\text{Acoustic Margin})$$

---

## 2. Bảng Đối Chiếu Toàn Diện Trên Test Set 1.000 Mẫu

Đánh giá sòng phẳng trên cùng 1.000 mẫu test, cùng chùm tia N-best và cùng cấu hình âm học:

| Chỉ số | Nhánh A: Beam 5 gốc | Nhánh C: GPT-2 Rescore | **Challenger: PhoBERT-v2 PLL** | So sánh PhoBERT vs GPT-2 |
|---|:---:|:---:|:---:|:---:|
| **Corpus WER (Chuẩn)** | 9,57% | 8,36% | **8,29%** | **-0,07%** (Cải thiện nhẹ) |
| **Corpus WER (Harmonized)** | 9,52% | 8,30% | **8,20%** | **-0,10%** (Cải thiện tốt) |
| **Corpus CER (Chuẩn)** | 5,55% | 5,47% | **5,36%** | **-0,11%** (Cải thiện rõ rệt) |
| **Corpus CER (Harmonized)** | 5,51% | 5,44% | **5,31%** | **-0,13%** (Cải thiện rõ rệt) |
| **Exact Matches (Chuẩn)** | 451/1.000 (45,1%) | 509/1.000 (50,9%) | **513/1.000 (51,3%)** | **+4 câu hoàn hảo** |
| **Exact Matches (Harmonized)** | 454/1.000 (45,4%) | 512/1.000 (51,2%) | **520/1.000 (52,0%)** | **+8 câu hoàn hảo** |
| **Substitutions (Lỗi thay từ)** | 670 từ | 502 từ | **508 từ** | +6 từ |
| **Deletions (Lỗi mất từ)** | 102 từ | 160 từ | **139 từ** | **-21 từ (Giảm 13% deletion)** |
| **Insertions (Lỗi chèn từ)** | 103 từ | 102 từ | **111 từ** | +9 từ |
| **Số từ đúng (Hits)** | 8.371 từ | 8.481 từ | **8.496 từ** | **+15 từ đúng** |
| **Độ trễ chấm điểm LM** | 0 ms | **5,8 ms / sample** | **67,8 ms / sample** | Chậm hơn ~11,7x |
| **Peak VRAM tiêu thụ** | ~560 MB | ~846 MB | **~688 MB** | Tiết kiệm ~158 MB |

---

## 3. Kiểm Định Ý Nghĩa Thống Kê (Paired Bootstrap Test)

### 1. PhoBERT vs Baseline A (Beam 5 gốc)
* **Mean Delta WER**: **-1,278%**
* **Khoảng tin cậy 95% (95% CI)**: **[-1,749%, -0,796%]**
* **Trị số empirical tail**: **$p = 0,00000$** (trong 2.000 replicate bootstrap, không có replicate nào có $\Delta \text{WER} \ge 0$).
* *Kết luận*: Cả khoảng tin cậy 95% đều nằm sâu dưới 0, khẳng định mức cải thiện của PhoBERT so với baseline là nhất quán trên tập đánh giá này.

### 2. PhoBERT vs GPT-2 (Branch C)
* **Mean Delta WER**: **-0,063%**
* **Khoảng tin cậy 95% (95% CI)**: **[-0,471%, +0,340%]** (khoảng tin cậy bao gồm 0).
* **Trị số empirical tail**: **$p = 0,39150$**
* *Kết luận*: **Chưa phát hiện khác biệt WER có ý nghĩa thống kê giữa PhoBERT và GPT-2 trên tập đánh giá này.** Không kết luận PhoBERT vượt trội hơn GPT-2 về mặt tổng WER.

---

## 4. Phân Tích Chuyên Sâu: Điểm Khác Biệt & Đánh Đổi

### Các điểm khác biệt quan sát được:
1. **Hiện tượng Deletion**:
   - So với GPT-2 (160 deletion), PhoBERT giảm được 21 phép lỗi deletion xuống còn **139**.
   - Tuy nhiên, con số 139 vẫn cao hơn baseline gốc (102 deletion), và việc giảm deletion đi kèm mức tăng nhẹ substitution (508 vs 502) và insertion (111 vs 102).
   - *Lưu ý về cơ chế*: GPT-2 không trực tiếp ngắt câu của Whisper; việc GPT-2 chọn nhiều candidate có deletion hơn là do tương tác giữa điểm ngôn ngữ, tham số độ dài $\beta$ và ngưỡng âm học $\gamma$.
2. **CER và Exact Match**:
   - CER quan sát được giảm xuống **5,36%** (Harmonized: **5,31%**), so với 5,47% của GPT-2.
   - Exact match quan sát được tăng lên **513/1.000 câu** (Harmonized: **520/1.000 câu**).
   - Tuy nhiên, đây là kết quả quan sát trên tập mẫu cụ thể, cần kiểm chứng thêm trên các tập holdout độc lập mới.
3. **Đặc tính kỹ thuật & Bộ nhớ**:
   - PhoBERT-base-v2 có kích thước ~135M tham số, bộ nhớ peak CUDA đo độc lập trong tiến trình là **687,9 MiB** (so với 846 MiB của GPT-2 trong cùng điều kiện).

### Đánh đổi & Giới hạn kỹ thuật (Technical Trade-offs):
1. **Độ trễ tính toán PLL (Latency)**:
   - GPT-2 tính điểm xác suất causal theo chuỗi trong 1 forward pass (~5,8 ms / câu).
   - PhoBERT tính Pseudo-Log-Likelihood (PLL) cần mask từng vị trí token. Mặc dù đã được vector hóa dạng batch, thời gian chấm điểm đo riêng bằng `torch.cuda.synchronize()` là **67,8 ms / sample** (khoảng 13,5 ms / candidate), chậm hơn GPT-2 khoảng 11,7 lần.
   - *Tối ưu an toàn cho câu dài*: Đã triển khai kỹ thuật **Chunked Masking** (chia nhỏ thành các batch $\le 16$) và kiểm soát độ dài `max_seq_len=256` để triệt tiêu nguy cơ tràn VRAM (OOM) khi xử lý các câu dài.
2. **Tách từ tiếng Việt (`pyvi`)**:
   - Mô hình PhoBERT gốc được khuyến nghị tiền xử lý với `RDRSegmenter` (VnCoreNLP). Việc sử dụng thư viện nhẹ `pyvi` là một lựa chọn thực dụng trong Python nhưng cần được coi là một biến thể thay thế cần kiểm chứng độ tương thích trên các từ vựng hiếm hoặc domain đặc thù.

---

## 5. Kết Luận & Khuyến Khích Lựa Chọn

| Chế độ | Cấu hình | Đặc tính & Khuyến nghị |
|---|---|---|
| **Nhanh (Fast Mode)** | Greedy decoding (`without_timestamps=True`) | RTF ~0,20x, không tốn thêm VRAM/LM, phù hợp ứng dụng thời gian thực. |
| **Cân bằng (Khuyến nghị mặc định cho Rescoring)** | Beam 5 + **GPT-2 Rescoring** | WER 8,32%, độ trễ chấm điểm rất thấp (~5,8 ms/câu), không cần dependency tách từ. |
| **Thử nghiệm / Chất lượng văn bản** | Beam 5 + **PhoBERT Rescoring** | CER 5,31%, ít deletion hơn GPT-2 trong benchmark này (139 vs 160), nhưng độ trễ LM cao hơn (~67,8 ms/câu). |
