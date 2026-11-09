# So sánh workflow small hiện tại

Mặc định từ 2026-11-10: official small, không local fine-tune; runtime OpenAI
Whisper Python/PyTorch. Chưa có đối chứng HF-small khớp protocol để kết luận
giảm VRAM hoặc tăng tốc bao nhiêu phần trăm.

| Viet-BUD500 holdout 500 | Public fast | Public rescore |
|---|---:|---:|
| WER / CER | 9,4854% / 6,4873% | 7,8500% / 5,8158% |
| Exact match | 47,8% | 52,0% |
| RTF | 0,18649 | 0,44159 |
| S / D / I | 271 / 60 / 104 | 195 / 80 / 85 |

[Báo cáo đầy đủ](../reports/phowhisper-small-public-modes-vietbud500-20261110.md).
Fast giữ fallback; rescore beam 5 + GPT-2: so sánh workflow hoàn chỉnh,
không cô lập tác dụng LM. Deletion tăng dù WER giảm. Không bảo đảm giữ nguyên
100% từ phủ định, không tuyên bố zero leakage hay cohort hoàn toàn mới.

VietMed small all100 WER fast/rescore 15,8623%/15,0712%; test15
17,4479%/18,4896%, rescoring làm xấu kết quả. Xem
[báo cáo bốn cell](../reports/vietmed100-base-small-fast-rescoring-20261109.md).
100 đoạn ~625 giây, không phải 100 giờ hoặc bằng chứng chưa thấy trong pretraining.
Audio cá nhân chỉ có timing, không reference; xem [README](../README.md).

## Bằng chứng base giữ nguyên

[Đối chứng official base/small 2026-11-09](../reports/phowhisper-base-small-vietbud500-20261109.md)
dùng temperature=0, không phải public fast fallback. Small giảm WER nhưng CER
tăng và CI hai phía chứa zero: gate cũ không đạt. Promotion sau đó do người dùng
quyết định, không đảo kết luận lịch sử. Báo cáo trong [chỉ mục](../benchmarks/README.md)
không được relabel thành small.
