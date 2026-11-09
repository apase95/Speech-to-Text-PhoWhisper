"""Xác minh small và lưu trữ base cục bộ; mặc định chỉ kiểm tra, không ghi."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

MODEL_ID = 'vinai/PhoWhisper-small'
REVISION = 'a86b604c346caf7148c37512eafe783a16420adb'
SMALL_HASHES = dict(native_sha256='15632cc15395f5ee8d9fec382133f3d6429c63d1c2209518984fb126058e85c5',
    source_sha256='9ec6f07c5bd321fc8f477cf778c43c3fd94b0e33a9f20283fc61cc6d22cbfded',
    config_sha256='c9d7e721876727d437cfb145229936926b439d1ab483db572e5f9b5fb6b6d0bf')
BASE_HASHES = {'phowhisper-base.pt': 'e46e8147bdc39c239206f89e61a6b6759ca07141d120c1337fc1410bf25b8d3d',
    'pytorch_model.bin': '910b9e8875497078efa83dc506ea7ee378e4e759a206ceecbf23e088aec9b9f8',
    'config.json': 'c75f5d458f497b7ea8748ab0c857db7ce2fd274b64cc70df30b17cb84723caf0'}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sidecar(path):
    return path.with_name(path.name+'.provenance.json')


def verify(path, expected):
    if path.is_symlink() or not path.is_file() or digest(path) != expected:
        raise ValueError(f'Unverified file: {path}')


def publish(source, destination, expected, replace=False):
    """Sao chép vào tệp tạm cùng ổ đĩa rồi công bố nguyên tử."""
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        shutil.copyfile(source, temp)
        verify(temp, expected)
        if replace:
            os.replace(temp, destination)
        else:
            try:
                os.link(temp, destination)
            except FileExistsError:
                verify(destination, expected)
    finally:
        temp.unlink(missing_ok=True)
    verify(destination, expected)


def activate(source, models=Path('models'), *, archive_base=False, apply=False):
    """Chỉ xóa nguồn base sau khi lưu trữ đầy đủ và small đã sẵn sàng."""
    source, models = Path(source), Path(models)
    config, weights = source.parent/'config.json', source.parent/'pytorch_model.bin'
    metadata_path = sidecar(source)
    record = json.loads(metadata_path.read_text())
    identity = dict(origin='huggingface', model_id=MODEL_ID, resolved_revision=REVISION, **SMALL_HASHES)
    if any(record.get(k) != v for k, v in identity.items()):
        raise ValueError('Official small provenance mismatch')
    for path, key in ((source, 'native_sha256'), (config, 'config_sha256'), (weights, 'source_sha256')):
        verify(path, SMALL_HASHES[key])
    metadata_hash = digest(metadata_path)
    target = models/'phowhisper-small.pt'
    archive = models/'archive/phowhisper-base'
    if not archive_base:
        raise ValueError('Explicit --archive-base authorization required')
    for directory in (models, models/'archive', archive):
        if directory.is_symlink():
            raise ValueError(f'Symlink directory refused: {directory}')
    for path, expected in ((target, SMALL_HASHES['native_sha256']), (sidecar(target), metadata_hash)):
        if path.exists() or path.is_symlink():
            verify(path, expected)
    entries = []
    for name, expected in BASE_HASHES.items():
        old, saved = models/name, archive/name
        if saved.exists() or saved.is_symlink():
            verify(saved, expected)
        if old.exists() or old.is_symlink():
            if name == 'config.json' and digest(old) == SMALL_HASHES['config_sha256']:
                verify(saved, expected)
            else:
                verify(old, expected)
        else:
            verify(saved, expected)
        entries.append(dict(source=str(old), destination=str(saved), source_sha256=expected, destination_sha256=expected))
    manifest = {'model': 'phowhisper-base', 'files': entries}
    manifest_path = archive/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Archive manifest conflict')
    if not apply:
        return manifest
    archive.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        saved = Path(entry['destination'])
        if not saved.exists():
            publish(Path(entry['source']), saved, entry['source_sha256'])
    # Không xóa nguồn ngay cả khi bước công bố small thất bại.
    with tempfile.TemporaryDirectory(dir=models) as staging:
        staged = Path(staging)/'manifest.json'
        staged.write_text(json.dumps(manifest, indent=2)+'\n')
        publish(staged, manifest_path, digest(staged))
        publish(source, target, SMALL_HASHES['native_sha256'])
        publish(metadata_path, sidecar(target), metadata_hash)
        publish(config, models/'config.json', SMALL_HASHES['config_sha256'], replace=True)
    for name, expected in BASE_HASHES.items():
        verify(archive/name, expected)
    verify(target, SMALL_HASHES['native_sha256'])
    verify(sidecar(target), metadata_hash)
    for name in ('phowhisper-base.pt', 'pytorch_model.bin'):
        old = models/name
        if old.exists():
            verify(old, BASE_HASHES[name])
            old.unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Pinned official native.pt')
    parser.add_argument('--archive-base', action='store_true')
    parser.add_argument('--apply', action='store_true', help='Write files; otherwise verify only')
    args = parser.parse_args()
    activate(args.source, archive_base=args.archive_base, apply=args.apply)
    print('Applied' if args.apply else 'Verified dry check; no files changed')


if __name__ == '__main__':
    main()
