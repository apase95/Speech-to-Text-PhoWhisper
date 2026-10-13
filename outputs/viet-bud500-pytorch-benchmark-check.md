# PhoWhisper PyTorch - Viet-BUD500 Benchmark Check

## Thông tin chạy

| Thuộc tính | Giá trị |
|---|---|
| Ngày đánh giá | 2026-10-06 |
| Nguồn dữ liệu | Viet-BUD500, 1.000 mẫu thuộc test split |
| Manifest | `benchmarks/manifests/viet_bud500_1000.csv` |
| Manifest SHA-256 | `428f56f5299a6636d2918749d55f62214140bda34f56224ec133c3f4b43da94c` |
| Model | PhoWhisper-base, checkpoint PyTorch đã chuyển đổi |
| Checkpoint SHA-256 | `e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d` |
| Runtime | OpenAI Whisper `20250625`, PyTorch `2.14.1+cu130` |
| Thiết bị | NVIDIA GeForce RTX 3050 Laptop GPU, FP16 |
| Decoding tốt nhất theo dev WER | `language=vi`, `task=transcribe`, `condition_on_previous_text=False`, `without_timestamps=True`, `beam_size=5` |

## Tổng hợp benchmark

| Metric | Kết quả |
|---|---:|
| Số file | 1.000 |
| File lỗi khi inference | 0 |
| Tổng thời lượng audio | 2.544,346 s (~42 phút 24 giây) |
| Tổng số từ tham chiếu | 9.143 |
| Tổng lỗi từ | 670 substitution, 102 deletion, 103 insertion |
| Corpus WER (micro) | **9,57%** |
| Corpus CER | **5,55%** |
| Câu đúng hoàn toàn | **451/1.000 (45,10%)** |
| Tổng thời gian inference | 1.418,774 s (~23 phút 39 giây) |
| Real-time factor | **0,558x** |
| Thời gian load model | 1,516 s |
| Peak CUDA memory | 584.074.240 bytes (~557 MiB) |
| Peak process RSS | 1.628.672.000 bytes (~1,52 GiB) |

## Chọn cấu hình trên dev set

Dev set gồm `audio1001` đến `audio1250`, không trùng ID với 1.000 mẫu test. Tất cả 250 file và transcript được cố định trong `benchmarks/manifests/viet_bud500_dev_250.csv`, manifest SHA-256 `5faf06364d345a63a3b662440187fe40d5936df384aeabb453bb7a7233c442d0`.

| Cấu hình | WER | CER | RTF |
|---|---:|---:|---:|
| Greedy, context | 11,11% | 6,42% | **0,245x** |
| Greedy, không context | 11,11% | 6,42% | 0,266x |
| Beam 3, không context | 10,36% | 6,03% | 0,426x |
| Beam 5, không context | **10,32%** | **6,01%** | 0,578x |

Beam 5 được chọn trước khi chạy test vì có WER và CER thấp nhất trên dev set. Chênh lệch độ chính xác giữa beam 3 và beam 5 rất nhỏ, trong khi beam 5 chậm hơn khoảng 35,7%; do đó beam 3 vẫn là lựa chọn cân bằng hơn khi tốc độ quan trọng.

## Đánh giá silence padding

Silence được thêm đối xứng vào đầu và cuối waveform sau khi resample về 16 kHz. Các mức 200, 300 và 500 ms được chọn trên dev set với greedy, `without_timestamps=True` và `condition_on_previous_text=False`.

| Padding mỗi biên | Dev WER | Dev CER | Dev RTF |
|---:|---:|---:|---:|
| 0 ms | **11,11%** | **6,42%** | 0,266x |
| 200 ms | 12,11% | 8,03% | **0,238x** |
| 300 ms | 14,25% | 9,93% | 0,244x |
| 500 ms | 14,25% | 9,77% | 0,249x |

Không padding cho WER dev thấp nhất. Mức dương tốt nhất là 200 ms vẫn được chạy trên 1.000 mẫu để kiểm chứng đầy đủ:

| Metric test | Greedy, 0 ms | Greedy, 200 ms | Thay đổi |
|---|---:|---:|---:|
| Corpus WER | **10,49%** | 11,99% | tăng 1,50 điểm % |
| Corpus CER | **6,10%** | 7,62% | tăng 1,52 điểm % |
| Câu đúng hoàn toàn | **427/1.000** | 365/1.000 | giảm 62 câu |
| Substitution | **740** | 765 | tăng 25 lỗi |
| Deletion | 99 | **52** | giảm 47 lỗi |
| Insertion | **120** | 279 | tăng 159 lỗi |
| Real-time factor | 0,253x | **0,247x** | gần như không đổi |

Padding giúp giữ lại nhiều token ở biên nên deletion giảm, nhưng khoảng im lặng nhân tạo làm decoder sinh thêm hoặc lặp nội dung: insertion tăng hơn gấp đôi và tổng WER xấu đi. Kết quả dev và test nhất quán, vì vậy silence padding không được đưa vào cấu hình triển khai.

## So sánh trên test set

Ba lần chạy sử dụng cùng 1.000 file, manifest và checkpoint. Cấu hình beam 5 chỉ được chọn từ kết quả dev set trước khi đánh giá trên test.

| Metric | Baseline timestamp | Greedy không timestamp | Beam 5 không timestamp |
|---|---:|---:|---:|
| Corpus WER | 41,78% | 10,49% | **9,57%** |
| Corpus CER | 34,14% | 6,10% | **5,55%** |
| Câu đúng hoàn toàn | 99/1.000 (9,90%) | 427/1.000 (42,70%) | **451/1.000 (45,10%)** |
| Substitution | 1.961 | 740 | **670** |
| Deletion | 1.318 | **99** | 102 |
| Insertion | 541 | 120 | **103** |
| Tổng thời gian inference | 916,012 s | **644,107 s** | 1.418,774 s |
| Real-time factor | 0,360x | **0,253x** | 0,558x |

## Vì sao kết quả cải thiện

Viet-BUD500 trong benchmark này gồm các utterance ngắn, trung bình khoảng 2,54 giây mỗi file, và nhiều câu bị cắt sát biên hoặc là một phần của câu dài. Ở chế độ mặc định, Whisper phải dự đoán đồng thời nội dung và token timestamp, sau đó dùng các timestamp đó để phân đoạn kết quả. Với clip ngắn hoặc bị cắt giữa câu, quyết định phân đoạn có thể khiến decoder bỏ nội dung ở đầu/cuối hoặc sinh thêm nội dung không thuộc đoạn tham chiếu.

Khi bật `without_timestamps=True`, decoder chỉ tập trung sinh token văn bản và không phải quyết định mốc thời gian cho từng segment. Điều này phù hợp hơn với bài toán một file tương ứng một utterance đã được cắt sẵn. Số liệu hỗ trợ nhận định này rõ nhất ở lỗi deletion: giảm từ 1.318 xuống 99; đồng thời substitution giảm từ 1.961 xuống 740 và insertion giảm từ 541 xuống 120.

Thời gian inference của greedy giảm vì decoder không cần sinh và xử lý chuỗi timestamp/segment. Beam search tiếp tục giảm lỗi bằng cách giữ nhiều giả thuyết token rồi chọn chuỗi có xác suất tổng thể tốt hơn, nhưng làm lượng tính toán tăng mạnh. Vì manifest, checkpoint và phần cứng không đổi, khác biệt giữa các lần chạy chủ yếu đến từ chế độ decoding. Tuy nhiên, benchmark này chứng minh tác động trên các tập đang dùng, không khẳng định cùng kết quả cho audio dài cần phân đoạn hoặc timestamp.

## Các bước tối ưu mô hình

- Chạy lại 1.000 mẫu với `without_timestamps=True`.
- Kết quả: WER giảm từ 41,78% xuống 10,49%, CER giảm từ 34,14% xuống 6,10% và RTF giảm từ 0,360x xuống 0,253x.
- Nhận định: decoder timestamp là nguyên nhân chính làm mất hoặc sinh sai nội dung ở biên các utterance ngắn; số lỗi deletion giảm mạnh từ 1.318 xuống 99.
- Tạo dev set cố định 250 mẫu không trùng test và so sánh greedy, beam 3, beam 5 cùng tùy chọn context.
- Chọn beam 5 theo dev WER rồi đánh giá lại test: WER giảm tiếp từ 10,49% xuống 9,57%, đổi lại RTF tăng từ 0,253x lên 0,558x.
- Thử silence padding 200, 300 và 500 ms trên dev; cả ba mức đều làm WER tăng. Mức 200 ms được kiểm chứng trên test và làm WER tăng từ 10,49% lên 11,99%, nên bị loại.
- Trạng thái: cấu hình triển khai an toàn là greedy, `without_timestamps=True`, không silence padding; dùng beam 5 chỉ khi ưu tiên độ chính xác cao nhất hơn tốc độ.

## Kết luận

Bật `without_timestamps=True` là tối ưu quan trọng nhất: greedy không padding đạt WER 10,49% và nhanh hơn baseline timestamp 29,68%. Beam 5 đạt WER thấp nhất 9,57% nhưng chậm hơn greedy 120,27%. Silence padding không phù hợp với tập này dù giảm deletion, vì insertion và tổng WER tăng đáng kể. Cấu hình được khuyến nghị cho triển khai là greedy, `without_timestamps=True`, không padding.

Kết quả thô của test padding 200 ms nằm tại `outputs/benchmarks/viet-bud500-greedy-padding200-20261006T014552Z/`. Kết quả beam 5 nằm tại `outputs/benchmarks/viet-bud500-beam5-20261006T010052Z/`; các run dev bắt đầu bằng `outputs/benchmarks/dev250-`; run greedy không padding nằm tại `outputs/benchmarks/viet-bud500-without-timestamps-20261006T000801Z/` và baseline tại `outputs/benchmarks/baseline-20261005T223314Z/`.
