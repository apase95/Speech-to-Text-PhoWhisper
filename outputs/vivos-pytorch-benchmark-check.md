# PhoWhisper PyTorch - VIVOS Benchmark Check

## Thông tin chạy

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-05 |
| Nguồn dữ liệu | Kaggle `kynthesis/vivos-vietnamese-speech-corpus-for-asr` |
| License do Kaggle công bố | CC BY-NC 4.0 |
| Manifest | `benchmarks/manifests/vivos_check_10.csv` |
| Model | PhoWhisper-base, checkpoint PyTorch đã chuyển đổi |
| Checkpoint SHA-256 | `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d` |
| Runtime | OpenAI Whisper `20250625`, PyTorch `2.14.1+cu130` |
| Thiết bị | NVIDIA GeForce RTX 3050 Laptop GPU, FP16 |
| Decoding | `language=vi`, `task=transcribe`, `without_timestamps=True` |

## Tổng hợp benchmark

| Metric | Kết quả |
|---|---:|
| Số file | 10 |
| File lỗi khi inference | 0 |
| Tổng thời lượng audio | 37,281 s |
| Tổng số từ tham chiếu | 93 |
| Tổng lỗi từ | 10 substitution, 0 deletion, 0 insertion |
| Corpus WER (micro) | **10,75%** |
| Corpus CER | **5,00%** |
| Câu đúng hoàn toàn | **4/10 (40%)** |
| Tổng thời gian inference | 6,831 s |
| Real-time factor | **0,183x** |
| Thời gian load model | 1,108 s |
| Peak CUDA memory | 584.074.240 bytes (~557 MiB) |
| Peak process RSS | 1.532.956.672 bytes (~1,43 GiB) |

## Kết quả từng file

| Mẫu | Thời lượng | Ground truth | Dự đoán PyTorch | S/D/I | WER |
|---|---:|---|---|---:|---:|
| R002 | 4,062 s | tiếng cọc cạch khựng lại của những khớp sắt | tiếng cọc cật khựng lại của những khớp sắt | 1/0/0 | 11,11% |
| R003 | 3,688 s | cũng lên tiếng ủng hộ các kiến nghị này | cũng lên tiếng ủng hộ các ký nghị này | 1/0/0 | 11,11% |
| R012 | 4,188 s | những cơn gió mạnh và mưa đóng băng gây trơn trợt | những cơn gió mạnh và mưa đóng băng ghi trơn trợt | 1/0/0 | 9,09% |
| R027 | 4,719 s | di tích có bảy phỗng ông và hai phỗng bà được làm bằng gỗ mít | di tích có bảy phỏng ông và hai phỏng bài được làm bằng gỗ mít | 3/0/0 | 20,00% |
| R028 | 4,562 s | các chú cũng thoải mái phóng uế giữa chốn đông người mà không sợ | các chú cũng thoải mái phóng ới giữa chốn đông người mà không sợ | 1/0/0 | 7,14% |
| R034 | 1,938 s | người ta nói | người ta nói | 0/0/0 | 0,00% |
| R043 | 3,375 s | anh phú chưa bú được sữa mẹ lần nào | anh phú chưa bú được sữa mẹ lần nào | 0/0/0 | 0,00% |
| R044 | 5,219 s | ngọn lửa bạo động ở trung đông và phá vỡ lộ trình hòa bình | ngọn lửa bạo động ở trung đông và phá vỡ loại trình quà bình | 2/0/0 | 14,29% |
| R055 | 2,594 s | và rất sáng tạo | và rứt sáng tạo | 1/0/0 | 25,00% |
| R057 | 2,938 s | nhưng giấc ngủ chập chờn | nhưng giấc ngủ chập chờn | 0/0/0 | 0,00% |

## Đối chiếu với hai hình

| Mẫu | WER trong hình | WER PyTorch | Khớp |
|---|---:|---:|:---:|
| R002 | 33,33% | 11,11% | Không |
| R003 | 11,11% | 11,11% | Có |
| R012 | 9,09% | 9,09% | Có |
| R027 | 20,00% | 20,00% | Có |
| R028 | 7,14% | 7,14% | Có |
| R034 | 0,00% | 0,00% | Có |
| R043 | 0,00% | 0,00% | Có |
| R044 | 14,29% | 14,29% | Có |
| R055 | 0,00% | 25,00% | Không |
| R057 | 0,00% | 0,00% | Có |

Kết quả WER câu khớp **8/10 (80%)**. Bốn trong năm trích đoạn lỗi ở Image 2 khớp; R002 không khớp vì decoder PyTorch dự đoán `cọc cật khựng`, trong khi Hugging Face pipeline cũ dự đoán `cộc cực khuẩn`.

## Kết luận

Kết quả **không hoàn toàn giống hình**. Runtime PyTorch với `without_timestamps=True` tốt hơn báo cáo cũ theo corpus WER (10,75% so với 11,83%), nhưng thay đổi lỗi ở R002 và phát sinh lỗi mới ở R055. Đây là khác biệt decoding giữa OpenAI Whisper và Transformers pipeline dù audio và trọng số model tương ứng giống nhau.

Nếu không bật `without_timestamps`, decoder long-form của OpenAI Whisper đạt WER 33,33% trên 10 clip do sinh thêm hoặc bỏ nội dung ở biên các utterance ngắn. Vì vậy, mọi lần benchmark VIVOS sau phải giữ `without_timestamps=True` để so sánh công bằng.

Kết quả thô của run nằm tại `outputs/benchmarks/vivos-check-no-timestamps-20261005T230141Z/`.
