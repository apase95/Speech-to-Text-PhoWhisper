# Small mặc định và archive base — 2026-11-10

Người dùng chọn official small sau nhận xét định tính và các số đo đã lưu.
Không phải tuyên bố gate base/small 2026-11-09 đã đạt: CER và CI hai phía không
đạt tiêu chí cũ; báo cáo/config lịch sử giữ nguyên.

Controller đã kích hoạt `models/phowhisper-small.pt` cùng sidecar provenance,
thay `models/config.json` bằng config official small. Native SHA256:
`15632cc15395f5ee8d9fec382133f3d6429c63d1c2209518984fb126058e85c5`;
config SHA256: `c9d7e721876727d437cfb145229936926b439d1ab483db572e5f9b5fb6b6d0bf`.
`_name_or_path` trong config upstream là metadata gốc, không có nghĩa repo vừa train.

Base local giữ ở `models/archive/phowhisper-base/` gồm `phowhisper-base.pt`,
`pytorch_model.bin`, `config.json`, `manifest.json`. Controller chỉ bỏ weights
base root sau khi kiểm tra hash archive. Official base cache giữ để tái lập lịch sử.
Không xóa archive hay sửa nhãn base. Weights/provenance/archive bị gitignore,
không phải weights phát hành trên GitHub.

Rollback nâng cao trên máy có archive đã xác minh:

```bash
PHOWHISPER_MODEL_PATH=models/archive/phowhisper-base/phowhisper-base.pt phowhisper --audio audio.wav
```

Dùng đường dẫn tuyệt đối cho deployment, restart worker để bỏ cache.
Không ghép config small với weights base. Clone mới dùng [conversion pinned](../README.md),
không cần activation/archive của máy hiện tại.
