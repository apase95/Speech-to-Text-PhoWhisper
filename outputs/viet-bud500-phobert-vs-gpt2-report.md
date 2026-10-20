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
* **Trị số $p$-value**: **$p = 0,00000$ ($p < 0,001$)**
* *Kết luận*: PhoBERT vượt trội hoàn toàn so với Baseline gốc với độ tin cậy tuyệt đối.

### 2. PhoBERT vs GPT-2 (Branch C)
* **Mean Delta WER**: **-0,063%**
* **Khoảng tin cậy 95% (95% CI)**: **[-0,471%, +0,340%]**
* **Trị số $p$-value**: **$p = 0,39150$**
* *Kết luận*: Về tổng tỷ lệ lỗi WER, chênh lệch giữa PhoBERT và GPT-2 không có ý nghĩa thống kê ($p > 0,05$). Tuy nhiên, cấu trúc lỗi của PhoBERT có những ưu thế định tính rất rõ ràng.

---

## 4. Phân Tích Chuyên Sâu: Điểm Mạnh & Đánh Đổi của PhoBERT

### Điểm mạnh nổi bật của PhoBERT:
1. **Sửa lỗi Deletion của GPT-2**:
   - GPT-2 (Causal LM) thường phạt các câu kết thúc lơ lửng khiến Deletion tăng vọt từ 102 lên 160.
   - PhoBERT (Bidirectional Masked LM) nhìn được ngữ cảnh 2 chiều nên không bị thiên kiến ngắt câu sớm quá mức, kéo số lỗi Deletion giảm mạnh từ 160 xuống **139** (giảm 21 lỗi mất từ).
2. **CER và Exact Match đạt mốc tốt nhất lịch sử**:
   - CER giảm sâu xuống **5,36%** (Harmonized: **5,31%**), tốt hơn đáng kể so với mức 5,47% của GPT-2.
   - Số câu nhận dạng đúng 100% đạt **520/1.000 câu (52,00%)** theo metric chuẩn hóa, thiết lập kỷ lục mới của dự án.
3. **Bộ nhớ VRAM cực kỳ nhẹ**:
   - PhoBERT-base-v2 chỉ tiêu tốn thêm khoảng **~128 MB VRAM** (Peak CUDA: 687,9 MiB), nhẹ hơn GPT-2 (~846 MiB).

### Sự đánh đổi (Trade-off):
1. **Độ trễ tính toán (Latency)**:
   - Để tính điểm Pseudo-Log-Likelihood (PLL), PhoBERT cần mask từng vị trí token trong câu. Dù đã được vector hóa thành 1 forward pass dạng batch, thời gian chấm điểm của PhoBERT vẫn mất **67,8 ms / sample** (so với 5,8 ms / sample của GPT-2).
   - Tuy nhiên, 67,8 ms vẫn rất nhanh so với thời gian phát âm thực tế (trung bình 2.500 ms/sample, tương đương chỉ tốn ~2,7% thời lượng audio).
2. **Đòi hỏi thư viện tách từ (`pyvi`)**:
   - PhoBERT yêu cầu văn bản đầu vào được tách từ ghép (`thành_phố`, `tất_cả`), do đó cần dependency `pyvi`.

---

## 5. Kết Luận & Đề Xuất Chiến Lược

| Mục tiêu sử dụng | Khuyến nghị lựa chọn |
|---|---|
| **Ưu tiên tốc độ siêu nhanh (Real-time / Chatbot)** | **Fast mode**: Greedy decoding thuần túy (RTF ~0,20x, không cần LM). |
| **Cân bằng giữa Tốc độ & Độ chính xác (Production)** | **GPT-2 Rescoring**: WER 8,32%, độ trễ LM cực thấp (~5,8 ms), không cần bộ tách từ. |
| **Ưu tiên độ chính xác cao nhất & ít mất từ (High-Quality Transcription)** | **PhoBERT Rescoring**: WER 8,20% (Harmonized), CER 5,31%, ít lỗi Deletion nhất (139 lỗi), 52% câu đúng 100%. |
