"""Offline, metadata-only verification of the bridged benchmark version lock.

Never print raw freeze output: direct URLs or editable paths could contain secrets.
No imports of model libraries, network requests, installation or model execution.
"""
import importlib.metadata
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = re.compile(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+*-]+)')


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def main():
    started = time.monotonic()
    if platform.python_version() != '3.14.8':
        raise ValueError('Python version differs from captured Python 3.14.8')
    expected = {}
    for line in (ROOT / 'requirements-benchmark-effective.lock').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        match = PIN.fullmatch(line)
        if not match or normalized(match[1]) in expected:
            raise ValueError('Invalid or duplicate lock pin')
        expected[normalized(match[1])] = match[2]
    process = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                             capture_output=True, text=True, timeout=10)
    if process.returncode:
        raise ValueError('pip freeze failed; raw output withheld')
    actual = {}
    for line in process.stdout.splitlines():
        if not line or line.startswith('#'):
            continue
        match = PIN.fullmatch(line)
        if not match:
            raise ValueError('Non-pin freeze entry; raw entry withheld for credential safety')
        # Local overlay wins even if pip reports a shadowed inherited distribution.
        version = importlib.metadata.version(match[1])
        if not PIN.fullmatch(match[1] + '==' + version):
            raise ValueError('Unsafe metadata version; raw entry withheld')
        actual[normalized(match[1])] = version
    if actual != expected:
        raise ValueError('Effective dependency versions differ from captured lock')
    bridge = ROOT / '.venv-benchmark/lib/python3.14/site-packages/project-torch-bridge.pth'
    if bridge.read_text().strip() != str(ROOT / '.venv/lib/python3.14/site-packages'):
        raise ValueError('Unexpected bridge provenance')
    print(f'PASS: Python3.14.8; {len(actual)} effective version pins; explicit .venv bridge; '
          f'{time.monotonic() - started:.3f}s; no model imports')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.TimeoutExpired):
        sys.exit('Environment/lock validation failed; check Python, pins and bridge (raw output withheld).')
