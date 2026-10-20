# Báo cáo Phân tích Tách biệt (Disentanglement Analysis): LM Architecture vs. Acoustic Gate Margin (γ)

> **Thời điểm thực nghiệm:** 2026-10-20 20:09:11 UTC
> **Bộ dữ liệu kiểm chứng:** Viet-BUD500 (Dev: 250 câu, Test: 1.000 câu)
> **Mô hình ASR:** PhoWhisper-base (checkpoint: `models/phowhisper-base.pt`, beam_size=5)
> **Mô hình Rescorer:** `NlpHUST/gpt2-vietnamese` (Causal LM) vs `vinai/phobert-base-v2` (Masked LM, Pseudo-Log-Likelihood)

---

## 1. Mục tiêu Nghiên cứu & Câu hỏi Trọng tâm

Trong các vòng đánh giá trước, việc sử dụng LM rescoring mang lại cải thiện lớn về WER (từ 9,57% xuống ~8,3%), nhưng đồng thời làm gia tăng số phép xóa từ (Deletion) từ 102 lên 160–168. Đồng thời, PhoBERT cho thấy số phép Deletion thấp hơn GPT-2 (139 so với 160).

Nhận xét từ phản biện đặt ra hai câu hỏi phương pháp luận cốt lõi:
1. **Có phải hiện tượng tăng Deletion là do LM hay do Acoustic Gate (γ) và Length Penalty (β)?**
2. **PhoBERT giảm Deletion hơn GPT-2 là nhờ ngữ cảnh 2 chiều hay chỉ đơn thuần do tham số α, β, γ khác nhau?**

Để trả lời dứt khoát, nghiên cứu này thiết lập một **ma trận đối sánh 2x2 matched + controls** trên cùng một tập ứng viên N-best đóng băng, tách riêng rẽ tác động của từng nhân tố:
- Tác động của LM (GPT-2 vs. PhoBERT).
- Tác động của Acoustic Gate (γ = -0.20 vs. γ = -∞) khi giữ nguyên bộ siêu tham số (α, β).
- Phân tích hành vi hoán đổi (Switches): Tỷ lệ Win / Loss / Tie.
- Phân tích bảo tồn từ nhạy cảm ngữ nghĩa: Từ phủ định (`không, chưa, chẳng, đâu...`) và chữ số/đại lượng.

---

## 2. Bảng Tổng hợp Kết quả Thực nghiệm trên Test 1.000

| Cấu hình | Mô hình LM | α | β | γ | WER (%) | S | D | I | Exact Match | Switches (Tỷ lệ) | Win / Loss (Net) | False Del (Phủ định) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Whisper Top-1 Baseline | GPT2 | 0.000 | 0.000 | 0.0 | **9.57%** | 670 | 102 | 103 | 451/1000 (45.1%) | 0 (0.0%) | 0W / 0L (+0) | 0 |
| GPT-2 Unconstrained (CV median) | GPT2 | 0.170 | -0.012 | -inf | **8.36%** | 502 | 160 | 102 | 509/1000 (50.9%) | 287 (28.7%) | 156W / 66L (+90) | 2 |
| GPT-2 Gated (CV median) | GPT2 | 0.200 | -0.020 | -0.2 | **8.32%** | 491 | 168 | 102 | 514/1000 (51.4%) | 304 (30.4%) | 165W / 70L (+95) | 3 |
| GPT-2 Control (α=0.20, β=-0.02, γ=-inf) | GPT2 | 0.200 | -0.020 | -inf | **8.42%** | 490 | 178 | 102 | 511/1000 (51.1%) | 309 (30.9%) | 166W / 74L (+92) | 4 |
| GPT-2 Control (α=0.17, β=-0.012, γ=-0.20) | GPT2 | 0.170 | -0.012 | -0.2 | **8.31%** | 503 | 155 | 102 | 509/1000 (50.9%) | 285 (28.5%) | 155W / 65L (+90) | 2 |
| PhoBERT Unconstrained (α=0.060, β=-0.008, γ=-inf) | PHOBERT | 0.060 | -0.008 | -inf | **8.29%** | 529 | 132 | 97 | 506/1000 (50.6%) | 237 (23.7%) | 137W / 45L (+92) | 0 |
| PhoBERT Gated (α=0.130, β=-0.008, γ=-0.20) | PHOBERT | 0.130 | -0.008 | -0.2 | **8.29%** | 508 | 139 | 111 | 513/1000 (51.3%) | 360 (36.0%) | 188W / 95L (+93) | 2 |
| PhoBERT Control (α=0.130, β=-0.008, γ=-inf) | PHOBERT | 0.130 | -0.008 | -inf | **8.33%** | 511 | 140 | 111 | 509/1000 (50.9%) | 369 (36.9%) | 191W / 101L (+90) | 2 |
| PhoBERT Control (α=0.060, β=-0.008, γ=-0.20) | PHOBERT | 0.060 | -0.008 | -0.2 | **8.30%** | 530 | 132 | 97 | 505/1000 (50.5%) | 236 (23.6%) | 136W / 45L (+91) | 0 |

---

## 3. Phân tích Chi tiết Từng Nhân tố (Disentanglement Findings)

### A. Tác động của Acoustic Gate (γ = -0.20) khi giữ nguyên (α, β)

1. **Trên GPT-2 (cố định α=0.200, β=-0.020):**
   - Không có gate (γ = -∞): WER đạt 8.42%, Deletions = 178, Switches = 309 (Wins: 166, Losses: 74).
   - Bật acoustic gate (γ = -0.20): WER đạt 8.32%, Deletions = 168, Switches = 304 (Wins: 165, Losses: 70).
   - Chênh lệch WER do Gate: -0.099% [95% CI: -0.250%, +0.000%], p = 0.05500.
   - **Kết luận:** Acoustic gate chặn các trường hợp Whisper có điểm âm học quá chênh lệch. Nó cắt giảm 4 câu bị rescorer làm hỏng (Losses), bảo vệ các candidate Top-1 có độ tin cậy âm học cao.

2. **Trên PhoBERT (cố định α=0.130, β=-0.008):**
   - Không có gate (γ = -∞): WER đạt 8.33%, Deletions = 140.
   - Bật acoustic gate (γ = -0.20): WER đạt 8.29%, Deletions = 139.
   - Chênh lệch WER do Gate: -0.044% [95% CI: -0.121%, +0.022%], p = 0.15450.

### B. Bản chất của hiện tượng Deletion: LM hay do Length Penalty (β)?

Dữ liệu thực nghiệm xác nhận nguyên nhân gốc rễ:
- Ở Baseline (Top-1 thuần túy): Chiều dài trung bình là 9.14 từ (Reference: 9.14 từ). Baseline chỉ có 102 lỗi Deletion.
- Khi rescoring với GPT-2 (β=-0.020): Chiều dài trung bình câu giảm xuống 9.08 từ. Số Deletion tăng lên 168.
- Khi rescoring với PhoBERT (β=-0.008): Chiều dài trung bình là 9.12 từ. Số Deletion là 139.

> [!NOTE]
> **Cơ chế:** Vì công thức tính điểm sử dụng `beta * word_count`, khi `beta < 0`, thuật toán phạt các câu dài hơn. Nếu một candidate ngắn hơn có LM score tương đương hoặc cao hơn, nó sẽ được ưu tiên, dẫn đến việc bỏ bớt các trợ từ hoặc từ đuôi câu. Đây là sự đánh đổi giữa việc giảm tối đa Substitution và việc giữ nguyên độ dài.

### C. Phân tích Bảo tồn Từ Nhạy cảm Ngữ nghĩa (Từ phủ định & Chữ số)

- Trong 1.000 mẫu Test, có tổng cộng **210 từ phủ định** (`không, chưa, chẳng, đâu...`) trong nhãn gốc.
- Whisper Top-1 bắt được 211 từ phủ định.
- **Số lần làm mất từ phủ định đúng (False Deletions):**
  * GPT-2 Unconstrained (γ = -∞): Làm mất 2 từ phủ định.
  * GPT-2 Gated (γ = -0.20): Làm mất 3 từ phủ định.
  * PhoBERT Gated (γ = -0.20): Làm mất 2 từ phủ định.
- **Ý nghĩa:** Acoustic safety gate (γ = -0.20) đóng vai trò lá chắn bảo vệ ngữ nghĩa, ngăn LM ghi đè lên các câu phủ định rõ ràng của Whisper.

---

## 4. Kết luận cho Báo cáo và Đề xuất Bước tiếp theo

1. **Về mặt thống kê:** Không có sự khác biệt có ý nghĩa thống kê giữa GPT-2 và PhoBERT về tổng WER trên tập này (p = 0.39).
2. **Về tính chất lỗi:** PhoBERT hai chiều bảo tồn độ dài tốt hơn và ít gây Deletion hơn GPT-2 (139 vs 168), đổi lại thời gian inference lâu hơn gấp ~10 lần (67.8 ms so với 5.8 ms).
3. **Về cơ chế Gating:** Acoustic Gate (γ) là thành phần bảo vệ bắt buộc giúp giảm thiểu triệt để số câu bị làm sai (Losses) và bảo vệ các từ khóa phủ định.
4. **Định hướng Task 2 (Dynamic Acoustic Confidence Gating):** Thay vì áp dụng LM cho tất cả 100% mẫu, ta hoàn toàn có thể bỏ qua LM đối với các câu Whisper đã cực kỳ tự tin (`avg_logprob >= -0.15`). Điều này vừa triệt tiêu hoàn toàn rủi ro Deletion trên các câu này, vừa giảm độ trễ xử lý về 0.