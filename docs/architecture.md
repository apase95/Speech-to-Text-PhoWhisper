# Kiến trúc và trạng thái chuyển đổi

Các structural moves đã hoàn tất. Không xóa datasets, checkpoints, output histories
hoặc notebook tạm. Việc dọn dữ liệu trùng lặp còn hoãn lại, không coi migration là
bằng chứng tăng tốc hoặc kiểm chứng lại các kết quả ASR lịch sử.

## Cây thư mục chuẩn

```text
src/phowhisper/
  __init__.py
  pipeline.py
  rescorer.py
  audio_utils.py
  cli.py
  text/normalization.py
app/web/app.py
scripts/
  convert/convert_phowhisper.py
  data/
  env/
benchmarks/
  run.py
  metrics.py
  runners/
  audits/
  manifests/
  annotations/
training/
  metadata.py
  kaggle/
  local/vietmed100_train.py
tests/
  unit/
  integration/
  regressions/
reports/                   # chín báo cáo chuyển từ docs, giữ tên file
docs/                      # guides và assets
outputs/benchmarks/
  vietmed_benchmark_results.json
```

JSON kết quả VietMed được di chuyển giữ nguyên hash. Các báo cáo đầu ra lịch sử
khác vẫn ở `outputs/`; xem [benchmark guide](../benchmarks/README.md).

## Entry points và dependency policy

```sh
pip install -e .
phowhisper --help
# Chạy từ checkout khi chưa cài package:
PYTHONPATH=src python -m phowhisper.cli --help
PYTHONPATH=src python -m app.web.app
python scripts/convert/convert_phowhisper.py --help
```

Package cài core `phowhisper` và app, nhưng dependencies mặc định chỉ gồm torch,
openai-whisper và numpy. Core chứa normalization riêng, không cần module benchmarks.
CLI help không import model dependencies. Web dùng extra `web`, conversion dùng
`conversion`, benchmark tooling dùng `benchmark`, GPT-2 rescoring dùng `rescore`.
Extras lấy versions từ requirements hiện có; không phải dependency lock hay cam kết
tương thích mọi Python/CUDA. `requirements.txt` giữ môi trường all-tools cũ;
`requirements-benchmark.txt` là overlay đặc thù, không phải standalone environment.
PhoBERT còn cần `pyvi` cài riêng (không thêm dependency chưa có trong requirements).
Training dùng requirements riêng tại `training/kaggle`, không kéo vào core.
Checkpoint mặc định tương đối với cwd: `models/phowhisper-base.pt`; có thể đặt
`PHOWHISPER_MODEL_PATH` tới checkpoint tuyệt đối. Package không kèm weights/data.

## Tương thích tên cũ

`src.inference`, `src.rescorer`, `src.audio_utils`, `src.app`, các stubs `tools/`
và namespace `kaggle_training` được giữ có chủ ý; implementation ở cây chuẩn trên.
Ví dụ cũ `python -m src.inference --audio audio.wav` vẫn là compatibility entry
point. Alias giữ module identity cho monkeypatch; chưa ấn định ngày bỏ.
`src.check_environment` giữ chẩn đoán lúc import, không dùng như help-only CLI.
Không chỉnh archived outputs/config snapshots hoặc xóa `temp_session_kaggle`.

## Metadata epoch

`training/metadata.py` được tự động tích hợp vào HF trainer ở `training/kaggle`
và local trainer ở `training/local/vietmed100_train.py`. Không đổi save policy,
không tự evaluate/save/delete checkpoint. Chỉ process zero ghi `<output>/epochs.json`
qua file tạm cùng thư mục và `os.replace`.

Ví dụ schema minh họa (không phải kết quả huấn luyện thực tế):

```json
{
  "schema_version": 1,
  "invocations": [
    {
      "id": "example-invocation",
      "started_at": "2026-01-01T00:00:00+00:00",
      "provenance": {"resume": "/runs/previous/checkpoint-100"}
    }
  ],
  "epochs": [
    {
      "invocation": "example-invocation",
      "epoch": 1,
      "global_step": 125,
      "timestamp": "2026-01-01T01:00:00+00:00",
      "metrics": {},
      "checkpoint_steps": []
    }
  ],
  "checkpoints": [
    {
      "invocation": "example-invocation",
      "global_step": 100,
      "path": "/runs/current/checkpoint-100",
      "timestamp": "2026-01-01T00:50:00+00:00",
      "exists": true
    }
  ],
  "final_models": [
    {
      "invocation": "example-invocation",
      "path": "/runs/current/final",
      "global_step": 125,
      "timestamp": "2026-01-01T01:01:00+00:00"
    }
  ]
}
```

Provenance do caller cấp, gồm thông tin resume; mỗi invocation có ID riêng,
giữ lịch sử cũ và không suy diễn epoch từ tên đường dẫn. Chỉ epoch nguyên dương
hoàn tất được ghi; epoch phân số bị bỏ qua. Metrics chỉ là log/evaluation quan sát
tại đúng global step, kể cả đến muộn; không gán WER step khác thành WER mỗi epoch.
NaN/Infinity thành null. `checkpoint_steps` có thể rỗng: liên kết yêu cầu cùng
invocation và exact step, bất kể save trước hay sau epoch-end. Paths checkpoint/final
là tuyệt đối. `exists` là snapshot thư mục ở lần flush, không xác nhận weights đầy
đủ hay còn tồn tại sau đó. `final_models` tách riêng khỏi checkpoint list.

## Kiểm tra

Tests đã phân nhóm, không giữ intermediate test counts trong docs. Controller
báo final validation riêng. Standalone metadata smoke không chạy trainer/model:

```sh
PYTHONPATH=.:src python tests/unit/test_epoch_metadata_smoke.py
git diff --check
```

Synthetic tests không kiểm chứng lại historical hash-bound artifacts, GPU/DDP,
chất lượng ASR hoặc builders trên Kaggle thật.
