# Fine-tune PhoWhisper-base trên Kaggle (thử nghiệm, chưa chạy GPU)

Giữ nguyên runtime OpenAI Whisper của repo. Pipeline này dùng HF đầy đủ,
không LoRA, không `device_map=auto`. API được ghim Transformers 4.46.3:
`eval_strategy`, `processing_class`, accelerate 1.1.1. Dùng Python 3.10–3.12;
không cài lại torch CUDA của Kaggle. Nếu torchvision của image không tương
thích torch, sửa image/môi trường trước, không bỏ qua lỗi import.

PhoWhisper được mô tả fine-tune trên 844 giờ; không có bằng chứng chỉ dùng
VIVOS hoặc chưa thấy Viet-BUD500. Gain trên lời nói nhanh/đồng âm là giả thuyết,
không bảo đảm WER tốt hơn. Cần dữ liệu khớp domain và kiểm tra nhãn.

## Các cell (repo đặt ở /kaggle/working/phowhisper-pytorch)

Chưa có git URL được xác nhận. Trên máy local, nén riêng thư mục
`kaggle_training/` thành ZIP (giữ nguyên tên thư mục), upload làm Kaggle Dataset
riêng rồi Add Input vào notebook. Tạo `/kaggle/working/phowhisper-pytorch` và
giải nén ZIP vào đó, sao cho có
`/kaggle/working/phowhisper-pytorch/kaggle_training/train.py`. Không cần upload
runtime, weights `.pt` hoặc toàn repo. Nếu upload bốn file không nén, copy chúng
vào thư mục `kaggle_training/` tại vị trí trên. Dataset code và dataset audio là
hai input riêng; thay đường dẫn mount bằng đường dẫn thực tế.

Bật Internet, GPU T4 x2, gắn dataset tuandaodev/viet-bud500-formatted.
Kiểm tra mount thực tế; ví dụ dưới giả định metadata ở root và `wavs/`.
Các lệnh `!` chạy từ notebook; không đặt token trong source. Model public
không cần HF token; upload tùy chọn chỉ dùng Kaggle Secrets.

```bash
%cd /kaggle/working/phowhisper-pytorch
!pip install -r kaggle_training/requirements.txt
```

**Restart kernel/session sau khi pip install**, rồi chạy lại cell `%cd` và
kiểm tra GPU trước mọi smoke/model command:

```bash
%cd /kaggle/working/phowhisper-pytorch
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() == 2"
!python kaggle_training/train.py inspect --root /kaggle/input/viet-bud500-formatted --output /kaggle/working/inspect
!python kaggle_training/train.py prepare --root /kaggle/input/viet-bud500-formatted --output /kaggle/working/manifests --train-cap 32 --validation-cap 8
!python kaggle_training/train.py evaluate --manifest /kaggle/working/manifests/validation.json --output /kaggle/working/baseline
!CUDA_VISIBLE_DEVICES=0 python kaggle_training/train.py smoke --manifest /kaggle/working/manifests/train.json --validation /kaggle/working/manifests/validation.json --output /kaggle/working/smoke --max-steps 2 --save-steps 1 --batch 1 --accumulation 1
!CUDA_VISIBLE_DEVICES=0 python kaggle_training/train.py smoke --manifest /kaggle/working/manifests/train.json --validation /kaggle/working/manifests/validation.json --output /kaggle/working/smoke --max-steps 3 --save-steps 1 --batch 1 --accumulation 1 --resume /kaggle/working/smoke/checkpoint-2
```

Xem metrics/errors, kiểm tra audio/nhãn và việc bước tiếp theo thật sự là 3.
Sau đó prepare một lần cho pilot lớn hơn (train-cap 2000, validation-cap 100),
**đóng băng manifest** và chạy lại baseline trên đúng validation mới. Không
chọn model bằng official test. Thêm `--exclude-manifest PATH` cho từng manifest
benchmark đã có (CSV chứa audio_path/id và sha256). Không tự tìm manifest.
Prepare luôn loại ID 1–1250, kiểm tra trùng ID giữa train/test, trùng ID/nhãn
trong metadata, và trùng byte audio trong train. Test chỉ ghi manifest, không
đọc audio để train. Validation chia speaker trước khi áp cap; speaker thiếu
thì dừng thay vì âm thầm random split. Xem prepare.json, train.json,
validation.json để kiểm tra số mẫu, exclusions và speaker. Hash được stream,
nhưng quét audio train có thể tốn thời gian. Near-duplicate, speaker sai,
trùng byte với official test và exposure trong pretraining chưa được loại trừ.
Để mở cap: `--train-cap 0 --validation-cap 0`; chưa nên làm ở pilot đầu.

```bash
!python kaggle_training/train.py prepare --root /kaggle/input/viet-bud500-formatted --output /kaggle/working/pilot-manifests --train-cap 2000 --validation-cap 100
!python kaggle_training/train.py evaluate --manifest /kaggle/working/pilot-manifests/validation.json --output /kaggle/working/pilot-baseline
```

Đọc `resolved_revision` trong `pilot-baseline/run.json`, gán vào biến notebook
`PILOT_REVISION = "<SHA thực tế>"` (không dùng chuỗi placeholder). Sau đó:

```bash
!torchrun --standalone --nproc_per_node=2 kaggle_training/train.py train --revision {PILOT_REVISION} --manifest /kaggle/working/pilot-manifests/train.json --validation /kaggle/working/pilot-manifests/validation.json --output /kaggle/working/pilot --max-steps 200 --save-steps 50 --batch 2 --accumulation 4
!python kaggle_training/train.py evaluate --model /kaggle/working/pilot --manifest /kaggle/working/pilot-manifests/validation.json --output /kaggle/working/post-validation
```

DDP thật: hai T4 16GB không thành một GPU 32GB. Batch hiệu dụng = batch mỗi
GPU × 2 × accumulation (ví dụ 16). Mỗi rank có model riêng. FP16 trên CUDA,
gradient checkpointing non-reentrant (`use_reentrant=False`), use_cache=false.
Không dùng reentrant mặc định với DDP. Nếu OOM giảm batch xuống 1. Greedy,
tiếng Việt, transcribe, không timestamp cho cả baseline và post-training.
Audio tải khi cần, downmix mono/resample 16k; không cache mel toàn corpus.
Audio rỗng/nonfinite/>30s hoặc transcript quá dài gây lỗi, không cắt âm thầm.
Đây không phải automatic corpus audio audit: sửa/quarantine dữ liệu có log
và tạo lại manifest nếu lỗi; không bỏ qua lỗi trong trainer.

## Ngân sách và checkpoint

Đo pilot sau warm-up (bỏ 5–10 optimizer steps đầu), đo thời gian mỗi optimizer
step và toàn bộ validation/save riêng. Ước lượng:
`max_steps <= (budget_seconds - preparation - validation_count*validation_seconds
- checkpoint_overhead - safety_margin) / warmed_step_seconds`.
Trainer log mỗi optimizer step; đo wall clock các mốc step, không suy ra từ
batch đọc audio. Dành ít nhất 20% dự phòng; không hứa 5–10 giờ hay WER mục tiêu.
Kiểm tra DDP 2 rank trước tăng steps. Run metadata lưu revision HF commit,
seed, hash manifest, world size; dùng `--revision SHA` từ baseline/run.json
cho các lần tải HF tiếp theo để tránh main thay đổi.

Checkpoint mỗi save-steps, giữ 2 bản: có model/optimizer/scheduler/RNG/trainer
state. Resume bằng `--resume /kaggle/input/checkpoints/checkpoint-N` và cùng
model/revision, manifests, batch, accumulation, world size, hyperparameters;
`max-steps` là tổng số steps, không phải số steps cộng thêm. Thử resume tiny
ở trên trước khi chạy dài. `--model /path/model` không có `--resume` là warm
start, **không** tiếp tục optimizer. Sau khi dừng, Kaggle Save Version kèm
outputs hoặc tạo checkpoint dataset riêng, kiểm tra chứa mọi file checkpoint,
processor/tokenizer và manifests. `/kaggle/working` không được bảo đảm tồn tại
sau session; save_total_limit không thay thế backup. Tránh ghi đè run metadata
khi cần lưu lịch sử resume: sao lưu thư mục output trước.

## Đánh giá cuối và giới hạn

Chỉ sau chọn model trên validation, chạy evaluate với manifests/test.json
trong output riêng. Đây có thể là toàn bộ test và tốn thời gian; không chạy
ở pilot. metrics.json: WER/CER và S/D/I (WER là tỷ lệ, không phần trăm), runtime
của Trainer; errors.csv chứa reference/hypothesis và S/D/I từng clip.
Normalization NFC, lowercase, bỏ Unicode punctuation, gom whitespace, giữ dấu
tiếng Việt. Bảo vệ transcript nhạy cảm trong output. So sánh cùng manifest,
decode/normalization. Chưa đo long-audio, punctuation hay speed riêng inference.
Checkpoint HF chưa dùng trực tiếp cho runtime `.pt`; conversion và so sánh
HF/OpenAI trên audio cố định là công việc riêng bắt buộc trước triển khai.

Local checks không tải model:
```bash
python -m unittest kaggle_training.test_smoke -v
python -m py_compile kaggle_training/train.py kaggle_training/test_smoke.py
python -m unittest discover -s tests -v
```
Không coi unit tests là bằng chứng chất lượng ASR hoặc tương thích Kaggle.
Các lệnh smoke huấn luyện, GPU/DDP và resume ở trên **chưa được thực thi**.
Test collator bằng fake objects chỉ kiểm tra strip prefix và mask padding;
không xác nhận tensor padding thực của Transformers 4.46.3. Schema prepare chỉ
chấp nhận pipe CSV với ba cột bắt buộc và `audio<ID>.wav`/`wavs/audio<ID>.wav`
tại root/wavs; không tự hỗ trợ layout, tên file hay format CSV khác.
