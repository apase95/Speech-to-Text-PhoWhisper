"""Đường dẫn base lịch sử, không bao giờ dùng small đang hoạt động."""
from pathlib import Path


def legacy_base_file(name='phowhisper-base.pt', root=Path('models')):
    if name not in ('phowhisper-base.pt', 'pytorch_model.bin', 'config.json'):
        raise ValueError('Unknown legacy base file')
    archived = root/'archive/phowhisper-base'/name
    if archived.is_file():
        return archived
    # Khi đã lưu trữ, không trộn config small với checkpoint base cũ.
    if (root/'archive/phowhisper-base').exists():
        raise FileNotFoundError(archived)
    return root/name
