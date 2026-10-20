# Báo cáo Triển khai Kỹ thuật: Dynamic Acoustic Confidence Gating cho N-best Rescoring

> **Thời điểm thực nghiệm:** 2026-10-20 20:32:43 UTC
> **Bộ dữ liệu kiểm chứng:** Viet-BUD500 (Dev: 250 câu, Test: 1.000 câu)
> **Mô hình ASR:** PhoWhisper-base (`models/phowhisper-base.pt`, beam_size=5)
> **Cơ chế:** Ngưỡng tin cậy âm học động $\text{avg\_logprob}(c_1) \ge \theta_{\text{confident}}$ để bỏ qua LM rescoring.

---

## 1. Đặt vấn đề và Mục tiêu Kỹ thuật

Trong pipeline ASR kết hợp Rescoring, việc chạy mô hình ngôn ngữ (GPT-2 hoặc PhoBERT) cho toàn bộ 100% câu thoại mang lại hai hạn chế:
1. **Độ trễ tính toán không cần thiết:** Với những câu thoại ngắn, rõ ràng mà Whisper đã giải mã với độ tự tin cực cao (Top-1 có xác suất âm học áp đảo), việc tiếp tục nạp và forward qua LM gây lãng phí tài nguyên GPU.
2. **Nguy cơ lỗi Deletion:** Khi điểm LM phạt độ dài ($\beta < 0$), có một tỷ lệ nhỏ các câu Top-1 vốn đã hoàn hảo bị LM hoán đổi sang một ứng viên ngắn hơn, gây tăng số phép Deletion.

**Giải pháp Dynamic Acoustic Confidence Gating:**
- Khi $\text{avg\_logprob}(c_1) \ge \theta_{\text{confident}}$: Lập tức xuất kết quả Top-1 của Whisper, **bỏ qua hoàn toàn bước chấm điểm LM**.
- Khi $\text{avg\_logprob}(c_1) < \theta_{\text{confident}}$: Thực hiện chấm điểm rescoring bình thường với bộ tham số an toàn $(\alpha, \beta, \gamma=-0.20)$.

---

## 2. Kết quả Khảo sát Ngưỡng $\theta$ trên Test 1.000 mẫu

### A. Mô hình GPT-2 Rescorer (Khóa $\alpha=0.200, \beta=-0.020, \gamma=-0.20$)

| Ngưỡng $\theta$ | Trạng thái Gating | WER (%) | S | D | I | Exact Match | Tỷ lệ Bỏ qua (%) | Thời gian LM (ms) | Net Cải thiện (Wins - Losses) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `None (Always)` | Toàn phần | **8.32%** | 491 | 168 | 102 | 514/1000 (51.4%) | 0.0% (0/1000) | 5.8 ms | 165W / 70L (+95) |
| `-0.010` | Động | **8.33%** | 492 | 168 | 102 | 513/1000 (51.3%) | 17.5% (175/1000) | 4.8 ms | 164W / 70L (+94) |
| `-0.015` | Động | **8.31%** | 492 | 166 | 102 | 515/1000 (51.5%) | 23.7% (237/1000) | 4.4 ms | 162W / 66L (+96) |
| `-0.020` **(Khóa Mặc định)** | Động | **8.33%** | 495 | 165 | 102 | 514/1000 (51.4%) | 29.4% (294/1000) | 4.1 ms | 159W / 65L (+94) |
| `-0.025` | Động | **8.35%** | 501 | 160 | 102 | 511/1000 (51.1%) | 33.8% (338/1000) | 3.8 ms | 152W / 62L (+90) |
| `-0.030` | Động | **8.36%** | 503 | 160 | 101 | 509/1000 (50.9%) | 38.4% (384/1000) | 3.6 ms | 147W / 59L (+88) |
| `-0.040` | Động | **8.49%** | 521 | 155 | 100 | 503/1000 (50.3%) | 47.3% (473/1000) | 3.1 ms | 130W / 53L (+77) |
| `-0.050` | Động | **8.72%** | 548 | 148 | 101 | 494/1000 (49.4%) | 55.8% (558/1000) | 2.6 ms | 107W / 49L (+58) |
| `-0.080` | Động | **9.08%** | 593 | 140 | 97 | 474/1000 (47.4%) | 75.4% (754/1000) | 1.4 ms | 62W / 31L (+31) |
| `-0.100` | Động | **9.18%** | 615 | 126 | 98 | 463/1000 (46.3%) | 83.9% (839/1000) | 0.9 ms | 44W / 21L (+23) |

### B. Mô hình PhoBERT Rescorer (Khóa $\alpha=0.130, \beta=-0.008, \gamma=-0.20$)

| Ngưỡng $\theta$ | Trạng thái Gating | WER (%) | S | D | I | Exact Match | Tỷ lệ Bỏ qua (%) | Thời gian LM (ms) | Net Cải thiện (Wins - Losses) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `None (Always)` | Toàn phần | **8.29%** | 508 | 139 | 111 | 513/1000 (51.3%) | 0.0% (0/1000) | 67.8 ms | 188W / 95L (+93) |
| `-0.010` | Động | **8.27%** | 507 | 139 | 110 | 515/1000 (51.5%) | 17.5% (175/1000) | 55.9 ms | 186W / 91L (+95) |
| `-0.015` | Động | **8.27%** | 509 | 137 | 110 | 514/1000 (51.4%) | 23.7% (237/1000) | 51.7 ms | 183W / 89L (+94) |
| `-0.020` **(Khóa Mặc định)** | Động | **8.28%** | 510 | 137 | 110 | 512/1000 (51.2%) | 29.4% (294/1000) | 47.9 ms | 180W / 87L (+93) |
| `-0.025` | Động | **8.29%** | 516 | 134 | 108 | 508/1000 (50.8%) | 33.8% (338/1000) | 44.9 ms | 171W / 81L (+90) |
| `-0.030` | Động | **8.32%** | 520 | 134 | 107 | 505/1000 (50.5%) | 38.4% (384/1000) | 41.8 ms | 165W / 78L (+87) |
| `-0.040` | Động | **8.42%** | 534 | 130 | 106 | 497/1000 (49.7%) | 47.3% (473/1000) | 35.7 ms | 147W / 67L (+80) |
| `-0.050` | Động | **8.63%** | 558 | 125 | 106 | 486/1000 (48.6%) | 55.8% (558/1000) | 30.0 ms | 121W / 58L (+63) |
| `-0.080` | Động | **8.94%** | 597 | 117 | 103 | 469/1000 (46.9%) | 75.4% (754/1000) | 16.7 ms | 75W / 37L (+38) |
| `-0.100` | Động | **9.15%** | 623 | 111 | 103 | 461/1000 (46.1%) | 83.9% (839/1000) | 10.9 ms | 47W / 26L (+21) |

---

## 3. Phân tích Chi tiết & Lợi ích Thực tiễn

### 1. Tiết kiệm ~30% Chi phí Tính toán và Độ trễ
- Tại ngưỡng $\theta^* = -0.020$, có đúng **29,4% số mẫu** (294/1.000 mẫu) có độ tin cậy âm học vượt ngưỡng.
- Đối với các mẫu này, pipeline không cần tính toán bất kỳ forward pass LM nào.
- Thời gian chấm điểm trung bình của PhoBERT giảm từ **67,8 ms xuống 47,9 ms/mẫu** (tiết kiệm ~20 ms/mẫu, tương đương giảm ~30% tổng chi phí LM).

### 2. Giảm thiểu Số phép Deletion và Số câu bị làm hỏng (Losses)
- **Trên GPT-2:**
  * Deletions giảm từ 168 xuống **165**.
  * Số câu bị làm hỏng (Losses) giảm từ 70 xuống **65 câu** (-5 câu được bảo toàn).
  * Exact match giữ nguyên đỉnh cao ở mức **514/1.000 câu**.
- **Trên PhoBERT:**
  * Deletions giảm từ 139 xuống **137**.
  * Số câu bị làm hỏng (Losses) giảm từ 95 xuống **87 câu** (-8 câu được bảo toàn).
  * Ở ngưỡng thận trọng $\theta = -0.010$, PhoBERT đạt kỷ lục **WER 8,27%** và **515/1.000 câu exact match**.

### 3. Kiểm định Thống kê (Paired Bootstrap, N=2.000)
- **GPT-2 (Gating $\theta=-0.020$ vs. Không Gating):** $\Delta \text{WER} = +0.009\%$ (95% CI: $[-0.065\%, +0.077\%]$, $p = 0.66100$). Không có suy giảm chất lượng nào về mặt thống kê, trong khi tiết kiệm 29,4% tài nguyên.
- **PhoBERT (Gating $\theta=-0.020$ vs. Không Gating):** $\Delta \text{WER} = -0.009\%$ (95% CI: $[-0.109\%, +0.087\%]$, $p = 0.48650$).

---

## 4. Tích hợp và Hướng dẫn Sử dụng

Cơ chế Dynamic Gating đã được tích hợp mặc định vào:
1. **Python API (`src/rescorer.py`):** Tham số `confidence_threshold=-0.020` được đặt mặc định trong `rescore_audio()`.
2. **CLI (`src/inference.py`):** Thêm cờ `--confidence-threshold -0.020` (hoặc truyền `-999` để tắt hoàn toàn gating).
3. **Gradio App (`src/app.py`):** Tự động kích hoạt khi người dùng chọn chế độ Accurate Rescoring.