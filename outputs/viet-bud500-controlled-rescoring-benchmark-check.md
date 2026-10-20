# PhoWhisper PyTorch - Controlled 3-Branch Benchmark Check (A / B / C)

## Thông tin thực nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-21 |
| Model ASR | PhoWhisper-base, checkpoint PyTorch (`models/phowhisper-base.pt`) |
| Checkpoint SHA-256 | `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d` |
| Language Model | `NlpHUST/gpt2-vietnamese` (Causal LM, 117M tham số) |
| Tập Dev (tối ưu trọng số) | `benchmarks/manifests/viet_bud500_dev_250.csv` (250 mẫu, SHA: `5faf0636...`) |
| Tập Test (đánh giá độc lập) | `benchmarks/manifests/viet_bud500_1000.csv` (1.000 mẫu, SHA: `428f56f5...`) |
| Runtime & Thiết bị | OpenAI Whisper `20250625`, PyTorch `2.14.1+cu130`, NVIDIA RTX 3050 Laptop GPU, FP16 |
| Decoding | `beam_size=5`, `without_timestamps=True`, `language=vi`, `task=transcribe` |
| Phương pháp chọn trọng số | 5-Fold Cross-Validation trên Dev 250 (đánh giá Out-of-Fold không rò rỉ dữ liệu) |
| Tham số khóa chính thức | $\alpha^* = 0,170 \quad (\text{LM Weight}), \quad \beta^* = -0,012 \quad (\text{Length Penalty})$ |

---

## 1. Trả Lời Câu Hỏi Cốt Lõi: Vì Sao Top-1 Của N-best Cũ Bị Kém Hơn Beam 5 Gốc?

* **Nguyên nhân**: Ở pipeline N-best cũ, hàm tiền xử lý đã gọi `pad_or_trim(audio)` trên waveform thô (miền thời gian) trước khi tính STFT. Với các clip ngắn 2–3 giây, việc đệm số 0 ở miền thời gian làm lệch chuẩn hóa năng lượng cực đại và rò rỉ phổ ở biên FFT, làm suy giảm đặc trưng âm học đưa vào decoder.
* **Cách khắc phục**: Đồng nhất 100% hàm tiền xử lý Mel theo đúng `whisper.transcribe()`: tính STFT với padding phản chiếu, cắt đúng số frame thực của clip rồi mới đệm số 0 ở miền Mel.
* **Kết quả đối chứng**: 
  * **Nhánh B (Top-1 N-best chuẩn) trùng khớp 100% từng từ, từng lỗi với Nhánh A (Beam 5 gốc)**:
    * Corpus WER: đúng **9,57016%**
    * Substitutions: đúng **670**
    * Deletions: đúng **102**
    * Insertions: đúng **103**
    * Hits: đúng **8.371**
    * Exact Matches: đúng **451/1.000 câu**.

---

## 2. Kết Quả 5-Fold Cross-Validation Trên Dev 250

Tập Dev 250 mẫu được chia thành 5 fold độc lập (mỗi fold 50 câu). Trên từng fold, lưới $(\alpha, \beta)$ được tối ưu trên 200 câu huấn luyện và đánh giá trên 50 câu Out-of-Fold (OOF).

| Fold | Train (200 mẫu) | Val (50 mẫu) | $\alpha^*$ tối ưu | $\beta^*$ tối ưu | Val Top-1 WER | **Val Rescored WER** | Thay đổi WER |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| **Fold 1** | 200 | 50 | 0,200 | -0,012 | 7,17% | 7,83% | +0,66% |
| **Fold 2** | 200 | 50 | 0,110 | 0,000 | 10,84% | **7,08%** | **-3,76%** |
| **Fold 3** | 200 | 50 | 0,170 | -0,012 | 10,29% | **9,17%** | **-1,12%** |
| **Fold 4** | 200 | 50 | 0,130 | +0,006 | 13,45% | **10,92%** | **-2,53%** |
| **Fold 5** | 200 | 50 | 0,170 | -0,012 | 9,73% | **7,74%** | **-1,99%** |

* **Dev Top-1 Chuẩn (OOF)**: WER **10,32%** (khớp hoàn toàn với baseline lịch sử dev250 beam 5).
* **Dev Rescored (OOF)**: WER **8,57%** (Harmonized: **8,53%**).
* **Mức giảm OOF trung thực**: Giảm **1,75 điểm % WER** trên tập dev.
* **Tham số khóa cố định (Median)**: $\alpha^* = 0,170, \beta^* = -0,012$.

---

## 3. Bảng Đối Chiếu 3 Nhánh Trên Test Set 1.000 Mẫu

Đánh giá duy nhất 1 lần trên toàn bộ 1.000 mẫu của tập test chính thức ([viet_bud500_1000.csv](file:///home/hodangthaiduy/Workspace/AI/phowhisper-pytorch/benchmarks/manifests/viet_bud500_1000.csv)) với cấu hình đã khóa:

| Metric | Nhánh A: Beam 5 gốc (`benchmarks.run`) | Nhánh B: Top-1 N-best chuẩn | **Nhánh C: GPT-2 Rescored (Locked $\alpha=0.17, \beta=-0.012$)** | Oracle 5-best (Lý thuyết) |
|---|---:|---:|---:|---:|
| **Corpus WER (Metric Cũ)** | 9,57% (0.095702) | 9,57% (0.095702) | **8,36%** (0.083561) | **4,80%** (0.048015) |
| **Corpus WER (Metric Mới)** | 9,52% (0.095155) | 9,52% (0.095155) | **8,30%** (0.083014) | **4,78%** (0.047796) |
| **Corpus CER** | 5,55% | 5,55% | **5,47%** | **2,99%** |
| **Câu đúng hoàn toàn (Exact Match)** | 451/1.000 (45,10%) | 451/1.000 (45,10%) | **509/1.000 (50,90%)** | **673/1.000 (67,30%)** |
| **Substitutions (Thay thế từ)** | 670 từ | 670 từ | **502 từ (-168 từ)** | 331 từ |
| Deletions (Mất từ) | 102 từ | 102 từ | 160 từ (+58 từ) | 51 từ |
| Insertions (Chèn từ thừa) | 103 từ | 103 từ | **102 từ (-1 từ)** | 57 từ |
| **Tổng số lỗi từ** | 875 lỗi | 875 lỗi | **764 lỗi (-111 lỗi)** | 439 lỗi |
| **Độ khớp Nhánh A vs Nhánh B** | 100% (chuẩn gốc) | **100% (1.000/1.000)** | - | - |

---

## 4. Phân Tích Thực Nghiệm & Kết Luận Trực Diện

### 1. Nhánh C có thực sự vượt Nhánh A không?
* **CÓ, VƯỢT TRỘI HOÀN TOÀN:**
  * WER giảm từ **9,57% xuống 8,36%** (giảm tới **1,21 điểm % WER tuyệt đối**, tương đương giảm **12,7% lượng lỗi tổng thể**).
  * CER giảm từ **5,55% xuống 5,47%** (cải thiện cả cấp độ ký tự).
  * Số câu đúng hoàn toàn tăng thêm **58 câu** (vượt mốc 50% câu đúng hoàn toàn, đạt **50,90%**).
  * Số lỗi thay thế từ giảm mạnh từ **670 từ xuống còn 502 từ** (giảm 168 lỗi substitution).

### 2. Trần Oracle thực tế:
* Trên chùm tia chuẩn, **Oracle 5-best đạt WER 4,80%** (CER 2,99%, 67,3% câu đúng hoàn toàn).
* Điều này khẳng định chùm tia chuẩn của Whisper có tiềm năng âm học rất lớn, và GPT-2 Rescoring đã hấp thụ được một phần đáng kể tiềm năng này để đưa WER thực tế xuống mức **8,36%**.

---

Thư mục run tương ứng: `outputs/benchmarks/test1000-rescored-beam5-20261020T183131Z/`.
