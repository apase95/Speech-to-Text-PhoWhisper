"""Build a readable, self-contained Kaggle Python script from an explicit allowlist."""
from pprint import pformat
from pathlib import Path


def build(root):
    names = ['training/metadata.py', 'training/kaggle/train.py', 'training/kaggle/pilot.py',
             'training/kaggle/build_pilot_source.py',
             'training/kaggle/test_smoke.py', 'training/kaggle/test_pilot.py', 'benchmarks/metrics.py',
             'src/phowhisper/text/normalization.py']
    payload = {name: (root / name).read_text(encoding='utf-8') for name in names}
    payload.update({'training/__init__.py': '', 'training/kaggle/__init__.py': '',
                    'kaggle_training/__init__.py': 'from training import kaggle\n__path__ = kaggle.__path__\n',
                    'benchmarks/__init__.py': '', 'src/__init__.py': '',
                    'src/phowhisper/__init__.py': '', 'src/phowhisper/text/__init__.py': ''})
    for name in ('viet_bud500_1000.csv', 'viet_bud500_dev_250.csv'):
        payload['training/kaggle/exclusions/' + name] = (
            root / 'benchmarks/manifests' / name).read_text(encoding='utf-8')
        payload['benchmarks/manifests/' + name] = payload['training/kaggle/exclusions/' + name]
    return '''# Generated self-contained PhoWhisper pilot1000; no repository checkout required.
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat() + ' artifact startup start', flush=True)
import argparse
import os
import json
import time
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tempfile
import importlib.util

class Progress:''' + payload['training/kaggle/train.py'].split('class Progress:', 1)[1].split('\ndef normalize', 1)[0] + '''
BOOT = Progress('/kaggle/working/outputs/benchmarks/current-progress.json' if __name__ == '__main__' and '--verify-only' not in sys.argv else None)
BOOT.update('artifact_startup', 'start')
PAYLOAD = ''' + pformat(payload, width=110, sort_dicts=True) + '''

def main():
    BOOT.update('artifact_startup', 'done')
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/kaggle/input/datasets/tuandaodev/viet-bud500-formatted')
    parser.add_argument('--output', default='/kaggle/working/outputs/benchmarks')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    # Mask a T4 x2 image to one GPU before any torch import; never launch DDP.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '60')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '60')
    if not args.verify_only:
        BOOT.current = Path(args.output) / 'current-progress.json'
    with tempfile.TemporaryDirectory(prefix='phowhisper-pilot1000-') as directory:
        root = Path(directory)
        with BOOT.stage('extraction'):
            for name, text in PAYLOAD.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + str(root / 'src'), PYTHONUNBUFFERED='1')
        with BOOT.stage('unittest'):
            subprocess.run([sys.executable, '-u', '-m', 'unittest',
                'training.kaggle.test_smoke', 'training.kaggle.test_pilot', '-v'],
                cwd=root, env=env, check=True)
        if not args.verify_only:
            # Validate mounted metadata/audio directory before provisioning or HF access.
            with BOOT.stage('layout_validation'):
                subprocess.run([sys.executable, '-u', '-c',
                    'from pathlib import Path; from training.kaggle.pilot import validate_layout; '
                    'import sys; validate_layout(Path(sys.argv[1]))', args.root], cwd=root, env=env, check=True)
            with BOOT.stage('dependency_installation'):
                if importlib.util.find_spec('jiwer') is None:
                    subprocess.run([sys.executable, '-u', '-m', 'pip', 'install',
                        '--timeout', '60', '--retries', '2', 'jiwer==3.0.5'], env=env, check=True, timeout=600)
            with BOOT.stage('pilot_subprocess'):
                subprocess.run([sys.executable, '-u', '-m', 'training.kaggle.pilot',
                    '--root', args.root, '--output', args.output], cwd=root, env=env, check=True)

if __name__ == '__main__':
    main()
'''


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    source = build(root)
    compile(source, 'pilot1000_source.py', 'exec')
    artifact = root / 'training/kaggle/pilot1000_source.py'
    artifact.write_text(source, encoding='utf-8')
    print(artifact)
