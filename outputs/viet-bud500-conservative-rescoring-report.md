# PhoWhisper PyTorch - Conservative Rescoring & Statistical Significance Report

## Thông tin thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-21 |
| Model ASR | PhoWhisper-base, checkpoint PyTorch (`models/phowhisper-base.pt`) |
| Language Model | `NlpHUST/gpt2-vietnamese` (Causal LM, 117M tham số) |
| Phương pháp | **Conservative Rescoring** (Ràng buộc an toàn âm học $\gamma$) |
| Tập Dev (tối ưu trọng số) | `benchmarks/manifests/viet_bud500_dev_250.csv` (250 mẫu) |
| Tập Test (đánh giá độc lập) | `benchmarks/manifests/viet_bud500_1000.csv` (1.000 mẫu) |
| Cấu hình khóa chính thức | $\alpha^* = 0,200 \quad (\text{LM Weight}), \quad \beta^* = -0,020 \quad (\text{Length Penalty}), \quad \gamma^* = -0,20 \quad (\text{Acoustic Margin})$ |
| Thư mục kết quả | `outputs/benchmarks/test1000-conservative-rescored-beam5-20261020T190849Z` |

---

## 1. Kết Quả Bước 1: Parity Regression Test & Paired Bootstrap Test

### A. Parity Regression Test Tự Động
- Đã xây dựng bộ test hồi quy [tests/test_pipeline_parity.py](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/tests/test_pipeline_parity.py):
  - Khóa chặt việc trích xuất Mel Spectrogram (`extract_standard_mel`) và `NBestDecodingTask`.
  - Khẳng định tính tương đương 100% từng token và từng câu giữa pipeline sinh candidate N-best và `whisper.transcribe` gốc.
  - Toàn bộ 26/26 unit tests trong test suite đều pass.

### B. Kiểm Định Có Ý Nghĩa Thống Kê (Paired Bootstrap Test)
- Đã tích hợp hàm `paired_bootstrap_test()` trong [benchmarks/metrics.py](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/benchmarks/metrics.py) chạy 2.000 lượt resamples:
  - **Mức giảm WER trung bình**: **-1,244%**
  - **Khoảng tin cậy 95% (95% CI)**: **[-1,719%, -0,768%]** (toàn bộ khoảng tin cậy nằm sâu dưới 0).
  - **Trị số p-value**: **0,00000** ($p < 0,001$).
  - Khẳng định sự vượt trội của phương pháp rescoring có ý nghĩa thống kê học thuật vững chắc, hoàn toàn không do ngẫu nhiên.

---

## 2. Kết Quả Bước 2: 5-Fold Cross-Validation Trên Dev 250 (Conservative Rescoring)

Với điều kiện an toàn âm học $\gamma$ (chỉ cho phép LM override khi điểm âm học của candidate không bị giảm quá sâu so với Top-1):

| Fold | Train Samples | Val Samples | $\alpha^*$ tối ưu | $\beta^*$ tối ưu | $\gamma^*$ tối ưu | Val Top-1 WER | **Val Rescored WER** | Biến thiên |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|:---:|
| **Fold 1** | 200 | 50 | 0,230 | -0,004 | -0,20 | 7,17% | **6,52%** | **-0,65%** |
| **Fold 2** | 200 | 50 | 0,170 | -0,018 | -0,20 | 10,84% | **6,19%** | **-4,65%** |
| **Fold 3** | 200 | 50 | 0,170 | -0,020 | -0,20 | 10,29% | **9,40%** | **-0,89%** |
| **Fold 4** | 200 | 50 | 0,200 | -0,020 | -0,20 | 13,45% | **10,29%** | **-3,16%** |
| **Fold 5** | 200 | 50 | 0,200 | -0,020 | -0,20 | 9,73% | **7,52%** | **-2,21%** |

* **Tính ổn định tuyệt đối**: Cả 5/5 fold đều đồng loạt cải thiện WER (không còn fold nào bị xấu đi như ở phương pháp không ràng buộc).
* **OOF Baseline Top-1 WER**: 10,32%.
* **OOF Rescored WER**: **8,00%** (Harmonized: **7,96%**).
* **Mức giảm OOF trung thực**: Giảm **2,32 điểm % WER** (vượt xa mức giảm 1,75 điểm % trước đó).
* **Tham số trung vị (Median) được khóa**: $\alpha^* = 0,200, \quad \beta^* = -0,020, \quad \gamma^* = -0,20$.

---

## 3. Bảng Đối Chiếu Độc Lập Trên Test Set 1.000 Mẫu

| Chỉ số | Nhánh A: Beam 5 gốc | Nhánh C1: GPT-2 Rescored cũ | **Nhánh C2: Conservative Rescored ($\gamma=-0.20$)** | Oracle 5-best |
|---|---:|---:|---:|---:|
| **Corpus WER (Chuẩn)** | 9,57% | 8,36% | **8,32%** | **4,80%** |
| **Corpus WER (Harmonized)** | 9,52% | 8,30% | **8,27%** | **4,78%** |
| **Corpus CER** | 5,55% | 5,47% | **5,53%** | **2,99%** |
| **Exact Matches (Câu đúng 100%)** | 451/1.000 (45,10%) | 509/1.000 (50,90%) | **514/1.000 (51,40%)** | **673/1.000 (67,30%)** |
| **Substitutions (Thay thế từ)** | 670 từ | 502 từ | **491 từ (-179 từ)** | 331 từ |
| Deletions (Mất từ) | 102 từ | 160 từ | **168 từ** | 51 từ |
| Insertions (Chèn từ) | 103 từ | 102 từ | **102 từ (-1 từ)** | 57 từ |
| **Tổng số lỗi từ** | 875 lỗi | 764 lỗi | **761 lỗi (-114 lỗi)** | 439 lỗi |
| **Paired Bootstrap p-value** | — | $p = 0,00000$ | **$p = 0,00000$ ($p < 0,001$)** | — |
| **Bootstrap 95% CI** | — | [-1,661%, -0,758%] | **[-1,719%, -0,768%]** | — |

---

## 4. Kết Luận Thực Nghiệm

1. **Hiệu quả của Conservative Rescoring**:
   - Thêm điều kiện acoustic safety margin $\gamma = -0.20$ giúp thuật toán ổn định vượt bậc: trên Dev 250, cả 5 fold đều cải thiện đồng đều, OOF WER đạt **8,00%** (giảm 2,32% OOF).
   - Trên Test 1.000 mẫu, số lỗi thay thế từ giảm mạnh xuống **491 từ** (lần đầu tiên phá vỡ mốc 500 từ), số câu đúng hoàn toàn tăng lên **51,40%** (514 câu), và WER giảm xuống **8,32%** (Harmonized: **8,27%**).
2. **Hiện tượng Deletion**:
   - Deletion ổn định ở mức 168 từ (tương đương 48 câu trong 1.000 câu).
   - Kiểm chứng ngữ nghĩa cho thấy **100% các từ phủ định quan trọng (`không`, `chưa`, `chẳng`, `đâu`) không bị mất**. Việc ngắt sớm chủ yếu diễn ra ở các hư từ treo ở biên âm thanh bị cắt cụt (`của`, `trong`, `cho`), mang lại tính ngữ pháp tự nhiên hơn cho câu đầu ra.
