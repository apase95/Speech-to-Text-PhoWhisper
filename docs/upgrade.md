# PhoWhisper PyTorch - Nhật Ký Nâng Cấp & Tối Ưu Hóa

Tổng hợp các cải tiến kỹ thuật và tối ưu decoding so với mô hình PhoWhisper gốc trên Hugging Face / GitHub.

---

### 1. Chuyển đổi sang Native PyTorch Whisper Runtime
Chuyển đổi trọng số sang định dạng native OpenAI Whisper (`.pt`), chạy FP16 trực tiếp trên GPU.
Giảm tiêu thụ VRAM xuống chỉ ~560 MB và cải thiện tốc độ suy luận vượt trội (RTF ~0.2x).

### 2. Tắt dự đoán mốc thời gian (`without_timestamps=True`)
Tắt sinh token timestamp cho các đoạn câu thoại ngắn (utterance-level ASR).
Giảm lỗi mất từ (deletion) từ 1.318 xuống 99, đưa WER trên Viet-BUD500 từ 41,78% xuống 10,49%.

### 3. Tắt phụ thuộc ngữ cảnh trước (`condition_on_previous_text=False`)
Không mớm lại transcript của câu trước làm prompt cho câu tiếp theo.
Loại bỏ hoàn toàn hiện tượng ảo giác (hallucination) và lặp từ giữa các file âm thanh độc lập.

### 4. Tối ưu Greedy Decoding cho độ trễ thấp
Sử dụng greedy decoding khi cần ưu tiên thời gian thực và tài nguyên giới hạn.
Đạt tốc độ xử lý cực nhanh (RTF ~0,25x, nhanh gấp 4 lần thời gian thực) với độ chính xác ổn định.

### 5. Beam Search Decoding (`beam_size=5`) cho độ chính xác cao
Mở rộng không gian tìm kiếm candidate âm học đa dạng trên từng frame.
Giảm WER từ 10,49% xuống 9,57%, nâng tỷ lệ câu nhận dạng đúng 100% từ 42,7% lên 45,1%.

### 6. Loại bỏ khoảng lặng nhân tạo (Zero Silence Padding)
Thực nghiệm cho thấy đệm silence (200–500 ms) làm tăng hơn gấp đôi lỗi chèn từ lặp (insertions).
Giữ nguyên biên audio gốc 0 ms giúp mô hình đạt độ nhạy nhận diện tối ưu và không bị hallucinate.

### 7. Tích hợp Time-Stretching WSOLA (`--time-stretch-rate`)
Bổ sung bộ xử lý co giãn tốc độ nói bằng thuật toán WSOLA (`atempo`) không làm đổi cao độ giọng.
Hỗ trợ giải quyết hiện tượng nói lướt, nói quá nhanh; cứu được 42 câu khó trên tập dev ở mức 0.9x.

### 8. Chuẩn hóa vị trí dấu thanh tiếng Việt (Tone Harmonization)
Đồng nhất quy tắc đặt dấu thanh kiểu mới và kiểu cũ (ví dụ: `hóa`/`hoá`, `khỏe`/`khoẻ`).
Loại bỏ các sai số giả định do định dạng chính tả của bộ gõ, giúp đánh giá ASR chuẩn xác hơn.

### 9. Đồng nhất Mel Spectrogram chuẩn (`extract_standard_mel`)
Sửa lỗi đệm số 0 trên waveform làm rò rỉ phổ biên FFT; chuyển sang reflect padding và slice frame thực.
Đảm bảo chất lượng âm học nhất quán 100% giữa pipeline sinh candidate N-best và inference gốc.

### 10. Tái xếp hạng N-best với biên an toàn âm học (Conservative LM Rescoring)
Sử dụng GPT-2 tiếng Việt chấm điểm chùm tia 5-best với acoustic margin $\gamma^*=-0.20$ khóa qua 5-Fold CV.
Ổn định cả 5 fold (OOF WER 8,00%), hạ WER test xuống kỷ lục **8,32%** và số câu đúng đạt **51,40%**.

### 11. Chế độ suy luận kép linh hoạt (Dual Fast vs Accurate Inference)
Cung cấp lựa chọn `--rescore` trong CLI và web UI: mặc định Fast (Greedy) và Accurate (Beam 5 + LM).
Đảm bảo luồng mặc định không tốn thêm bộ nhớ, tự động lazy load và có fallback an toàn khi thiếu tài nguyên.

### 12. Thử nghiệm PhoBERT Challenger (Bidirectional Masked LM)
Sử dụng `vinai/phobert-base-v2` chấm điểm Pseudo-Log-Likelihood (PLL) dạng chunked batch kết hợp tách từ `pyvi`.
Giảm deletion từ 160 xuống 139 trong benchmark này, CER đạt **5,31%**; chênh lệch tổng WER chưa có ý nghĩa thống kê so với GPT-2.
