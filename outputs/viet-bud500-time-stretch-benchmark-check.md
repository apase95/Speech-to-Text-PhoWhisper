# PhoWhisper PyTorch - Viet-BUD500 Time-Stretching Benchmark Check

## Thông tin chạy

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-21 |
| Nguồn dữ liệu | Viet-BUD500, 250 mẫu thuộc dev split (`audio1001` đến `audio1250`) |
| Manifest | `benchmarks/manifests/viet_bud500_dev_250.csv` |
| Manifest SHA-256 | `5faf06364d345a63a3b662440187fe40d5936df384aeabb453bb7a7233c442d0` |
| Model | PhoWhisper-base, checkpoint PyTorch đã chuyển đổi |
| Checkpoint SHA-256 | `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d` |
| Runtime | OpenAI Whisper `20250625`, PyTorch `2.14.1+cu130` |
| Thiết bị | NVIDIA GeForce RTX 3050 Laptop GPU, FP16 |
| Thuật toán Time-Stretch | FFmpeg `atempo` (WSOLA - Waveform Similarity Overlap-Add) |
| Cấu hình Decoding | `language=vi`, `task=transcribe`, `condition_on_previous_text=False`, `without_timestamps=True`, greedy |

## Tổng hợp benchmark so sánh 3 mức

Các lần chạy sử dụng cùng 250 file dev, cùng manifest, checkpoint và cùng phần cứng GPU để bảo đảm tính nhất quán tuyệt đối. RTF được tính dựa trên thời lượng audio gốc (632,10 s).

| Metric | Mức 1.0x (Baseline) | Mức 0.9x (Chậm 10%) | Mức 0.8x (Chậm 20%) |
|---|---:|---:|---:|
| Số file | 250 | 250 | 250 |
| File lỗi khi inference | 0 | 0 | 0 |
| Tổng thời lượng audio gốc | 632,10 s (~10 phút 32 giây) | 632,10 s | 632,10 s |
| Tổng số từ tham chiếu | 2.287 | 2.287 | 2.287 |
| **Corpus WER (micro)** | **11,11%** | 11,76% (+0,65%) | 12,81% (+1,70%) |
| **Corpus CER** | **6,42%** | 6,83% (+0,41%) | 7,63% (+1,21%) |
| **Câu đúng hoàn toàn (Exact Match)** | 98/250 (39,20%) | **102/250 (40,80%)** | 90/250 (36,00%) |
| Substitution (Thay thế) | **200** | 205 (+5) | 225 (+25) |
| Deletion (Mất từ) | 28 | **27 (-1)** | **27 (-1)** |
| Insertion (Chèn từ thừa) | **26** | 37 (+11) | 41 (+15) |
| Tổng thời gian inference | **60,096 s** | 95,072 s | 95,386 s |
| Real-time factor (RTF) | **0,095x** | 0,150x | 0,151x |
| Peak CUDA memory | 584.074.240 bytes (~557 MiB) | 584.074.240 bytes (~557 MiB) | 584.074.240 bytes (~557 MiB) |
| Peak process RSS | 1.531.953.152 bytes (~1,43 GiB) | 1.549.430.784 bytes (~1,44 GiB) | 1.546.997.760 bytes (~1,44 GiB) |

## Phân tích hiện tượng: Vì sao mức 0.9x tăng câu đúng nhưng WER tổng thể lại tăng?

Khi so sánh từng câu giữa **Mức 1.0x** và **Mức 0.9x**:
* **42 câu được cải thiện rõ rệt**: Mô hình sửa được các lỗi nhận dạng sai do nói nhanh và lướt âm ở mức 1.0x.
* **49 câu bị suy giảm chất lượng**: Mô hình sinh thêm từ lặp hoặc biến đổi các nguyên âm đã chuẩn thành nguyên âm khác.
* **159 câu giữ nguyên kết quả**.

### 1. Các trường hợp cải thiện tiêu biểu ở mức 0.9x (Bắt đúng âm lướt / nói nhanh)
Ở các câu phát âm nhanh, việc kéo dài 10% giúp các formants của nguyên âm và âm cuối không bị nén quá ngắn, giúp mô hình nhận diện chính xác:
* `[audio1026]`: 
  * Reference: *dạ dạ đâu có gì đâu chị bọn em chỉ trêu*
  * 1.0x: *dạ đâu có gì đâu chị bọn em **chị** trêu* (nhầm thanh hỏi thành thanh nặng)
  * 0.9x: *dạ đâu có gì đâu chị bọn em **chỉ** trêu* (sửa thành công)
* `[audio1041]`:
  * Reference: *các nhà phân tích thị trường chứng khoán*
  * 1.0x: *các nhà phân tích thị trường chứng **khoảng***
  * 0.9x: *các nhà phân tích thị trường chứng **khoán*** (sửa thành công)
* `[audio1048]`:
  * Reference: *trải qua một khóa học đó là khóa học*
  * 1.0x: *trải qua một **khoá hóc** đó là **khoá hóc*** (nói nhanh làm hụt nguyên âm mở)
  * 0.9x: *trải qua một **khóa học** đó là **khóa học*** (sửa thành công)
* `[audio1052]`:
  * Reference: *trách nhiệm là thế bác trả lời thế nào*
  * 1.0x: *trách nhiệm **đạt** thế bác trả lời thế nào*
  * 0.9x: *trách nhiệm **là** thế bác trả lời thế nào* (sửa thành công)

### 2. Các trường hợp suy giảm tiêu biểu ở mức 0.9x và 0.8x (Tác dụng phụ)
Với các câu người nói vốn đã phát âm với tốc độ bình thường hoặc chậm:
* **Kéo dãn nguyên âm quá mức**: Khiến decoder hiểu nhầm độ dài nguyên âm hoặc phụ âm đầu (ví dụ: `nào hãy` $\to$ `lào hãy`, `thành binh` $\to$ `thành minh`, `chính trị` $\to$ `chim trị`).
* **Tăng lỗi Insertion (chèn từ thừa)**: Lỗi insertion tăng từ 26 lên 37 ở mức 0.9x và lên 41 ở mức 0.8x. Do âm thanh bị kéo dãn thời gian, các khoảng nghỉ nhỏ giữa các từ bị rộng ra, khiến autoregressive decoder của Whisper dễ sinh thêm các trợ từ phụ (`á`, `thì`, `và`) hoặc lặp từ.

## Kết luận & Khuyến nghị

1. **Hiệu ứng hai mặt của Time-Stretching**:
   * Việc làm chậm audio bằng WSOLA (`atempo=0.9`) **thực sự phát huy hiệu quả với các trường hợp nói nhanh/lướt âm** (giúp tăng tỷ lệ câu đúng hoàn toàn từ **39,20% lên 40,80%** và sửa được nhiều lỗi nguyên âm lướt).
   * Tuy nhiên, việc áp dụng tĩnh (static / global time-stretch) lên toàn bộ corpus làm các câu tốc độ chuẩn bị kéo dài nhân tạo, dẫn đến tăng lỗi Insertion (+42% lỗi chèn từ thừa). Do đó, WER trung bình tăng nhẹ từ 11,11% lên 11,76%.
   * Mức 0.8x (chậm 20%) bị kéo dãn quá đà, làm méo mó phân phối thời lượng âm vị của Whisper, khiến cả WER (12,81%) và CER (7,63%) đều xấu đi rõ rệt.
2. **Khuyến nghị kiến trúc (Dynamic / Adaptive Time-Stretch)**:
   * **Không áp dụng Time-Stretch cố định cho toàn bộ audio đầu vào**.
   * Thay vào đó, giải pháp tối ưu trong thực tế là **Adaptive / Two-pass Decoding**:
     - *Pass 1*: Giải mã với tốc độ gốc 1.0x.
     - *Pass 2 (Chỉ kích hoạt khi cần)*: Nếu ước lượng tốc độ nói (Speaking Rate = số token dự đoán / số giây audio) vượt qua ngưỡng nói nhanh (ví dụ > 4-5 từ/giây) hoặc có độ tự tin trung bình (token logprob) thấp, lúc đó mới kích hoạt Pass 2 với `time-stretch-rate=0.9` để sửa lỗi.

---

Thư mục run tương ứng:
- Baseline 1.0x: `outputs/benchmarks/dev250-stretch10-20261020T165633Z/`
- Chậm 0.9x: `outputs/benchmarks/dev250-stretch09-20261020T165739Z/`
- Chậm 0.8x: `outputs/benchmarks/dev250-stretch08-20261020T165918Z/`
