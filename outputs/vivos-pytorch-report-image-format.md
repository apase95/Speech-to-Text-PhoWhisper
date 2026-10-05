# Kiểm chứng pipeline PhoWhisper trên VIVOS

## Kết quả WER trên 10 mẫu

Để kiểm chứng tính khả thi của pipeline và mô hình đã cài đặt, TV2 đã chạy thử nghiệm PhoWhisper Base trên 10 mẫu test cố định của tập VIVOS tải từ Kaggle. Kết quả tính toán chỉ số lỗi từ (WER) được ghi nhận chi tiết tại bảng sau.

| Mã định danh mẫu | Thời lượng audio | Số từ tham chiếu | Số từ lỗi | WER câu (%) |
|---|---:|---:|---:|---:|
| VIVOSDEV01_R002 | 4,062 s | 9 từ | 1 lỗi | 11,11% |
| VIVOSDEV01_R003 | 3,688 s | 9 từ | 1 lỗi | 11,11% |
| VIVOSDEV01_R012 | 4,188 s | 11 từ | 1 lỗi | 9,09% |
| VIVOSDEV01_R027 | 4,719 s | 15 từ | 3 lỗi | 20,00% |
| VIVOSDEV01_R028 | 4,562 s | 14 từ | 1 lỗi | 7,14% |
| VIVOSDEV01_R034 | 1,938 s | 3 từ | 0 lỗi | 0,00% (Đúng 100%) |
| VIVOSDEV01_R043 | 3,375 s | 9 từ | 0 lỗi | 0,00% (Đúng 100%) |
| VIVOSDEV01_R044 | 5,219 s | 14 từ | 2 lỗi | 14,29% |
| VIVOSDEV01_R055 | 2,594 s | 4 từ | 1 lỗi | 25,00% |
| VIVOSDEV01_R057 | 2,938 s | 5 từ | 0 lỗi | 0,00% (Đúng 100%) |

**Kết quả toàn tập:** 10 lỗi trên 93 từ tham chiếu, tương ứng corpus WER **10,75%**. Có 4/10 câu được nhận dạng đúng hoàn toàn.

## Phân tích các lỗi tiêu biểu

| Mã mẫu | Transcript chuẩn (Ground Truth) | PhoWhisper dự đoán | Phân tích nguyên nhân lỗi |
|---|---|---|---|
| R002 | cọc cạch khựng | cọc cật khựng | Âm cuối và phụ âm đầu trong từ `cạch` bị nhận nhầm thành `cật`; đây là cụm từ ít gặp và các âm có thời lượng ngắn. |
| R003 | kiến nghị | ký nghị | Âm lướt `/iə/` trong `kiến` bị decoder rút gọn thành âm `/i/` trong `ký`. |
| R012 | gây trơn trợt | ghi trơn trợt | Formant giữa nguyên âm đôi `/âj/` và `/i/` tương đối gần khi phát âm nhanh. |
| R027 | phỗng ông / phỗng bà | phỏng ông / phỏng bài | Thanh ngã bị nhận thành thanh hỏi ở hai vị trí; từ `bà` bị thêm âm cuối thành `bài`. |
| R028 | phóng uế | phóng ới | Nguyên âm đầu và thanh của từ Hán-Việt ít gặp `uế` bị nhận thành `ới`. |
| R044 | lộ trình hòa bình | loại trình quà bình | Âm tiết `lộ` bị nhận thành `loại`; phụ âm đầu `/h/` trong `hòa` bị nhận thành `/kw/` trong `quà`. |
| R055 | rất sáng tạo | rứt sáng tạo | Nguyên âm `/â/` trong `rất` bị nhận thành `/ɯ/` trong `rứt`, trong khi âm đầu, âm cuối và thanh vẫn được giữ. |

Các lỗi tập trung ở từ hiếm, nguyên âm gần nhau và thanh điệu tiếng Việt. Không có lỗi deletion hoặc insertion; toàn bộ 10 lỗi đều là substitution.

> Ghi chú: Số liệu thời lượng và số từ trong bảng này lấy trực tiếp từ WAV và transcript chính thức của VIVOS. Vì vậy, một số giá trị khác với hình tham chiếu ban đầu. Ví dụ R027 có 15 từ thay vì 10 từ và R034 có 3 từ thay vì 8 từ.
