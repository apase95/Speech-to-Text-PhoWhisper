"""Generate the private, input-free phowhisper3 setup notebook; never publish it."""
import importlib.util
import inspect
import json
import sys
from pathlib import Path

IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f'


def missing_dependencies():
    if importlib.util.find_spec('torch') is None:
        raise RuntimeError('Missing critical torch: select a working Kaggle GPU image; do not pip install torch.')
    packages = {
        'transformers': 'transformers>=4.46,<5',
        'accelerate': 'accelerate>=1.1,<2',
        'huggingface_hub': 'huggingface-hub>=0.26,<1',
        'soundfile': 'soundfile>=0.13,<1',
        'numpy': 'numpy>=2.1,<3',
        'scipy': 'scipy>=1.15,<2',
        'jiwer': 'jiwer==3.0.5',
        'rapidfuzz': 'rapidfuzz>=3,<4',
    }
    return [spec for module, spec in packages.items() if importlib.util.find_spec(module) is None]


BOOTSTRAP = r'''import sys, os, json, time, threading, subprocess, queue, signal
import importlib.util, importlib.metadata
from datetime import datetime, timezone
from pathlib import Path
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT = Path('/kaggle/working/outputs/benchmarks') / ('phowhisper3-setup-' + STAMP)
OUT.mkdir(parents=True, exist_ok=False)
LOG = OUT / 'setup.log'
def emit(message):
    text = datetime.now(timezone.utc).isoformat() + ' ' + str(message)
    print(text, flush=True)
    with LOG.open('a') as handle:
        handle.write(text + '\n')
emit('START phowhisper3: setup only; no dataset inputs or checkpoint downloads')
REPORT = {'started_utc': STAMP, 'intent': 'setup-only', 'dataset_attached': False,
          'checkpoints_downloaded': False, 'blockers': [], 'dependencies_installed': []}
def save_report():
    (OUT / 'report.json').write_text(json.dumps(REPORT, indent=2))
save_report()
def run(args, timeout=120, cwd=None):
    env = dict(os.environ, PYTHONUNBUFFERED='1', GIT_TERMINAL_PROMPT='0')
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            start_new_session=True)
    lines, events = [], queue.Queue()
    def reader():
        for line in proc.stdout:
            events.put(line)
        events.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    start = last = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if now - start > timeout:
                raise TimeoutError('Command exceeded bounded timeout')
            if now - last >= 5:
                emit('HEARTBEAT subprocess running')
                last = now
            try:
                line = events.get(timeout=min(0.05, timeout))
            except queue.Empty:
                continue
            if line is None:
                break
            lines.append(line)
            emit(line.rstrip())
        code = proc.wait(timeout=max(0.01, timeout - (time.monotonic() - start)))
        if code:
            raise RuntimeError('Subprocess failed with exit code ' + str(code))
        return ''.join(lines).strip()
    except Exception as error:
        os.killpg(proc.pid, signal.SIGKILL) if proc.poll() is None else None
        proc.wait()
        REPORT['blockers'].append(str(error))
        save_report()
        raise
    finally:
        thread.join(timeout=1)
        proc.stdout.close()
def stage(function):
    try:
        return function()
    except Exception as error:
        REPORT['blockers'].append(str(error))
        save_report()
        emit('BLOCKED: ' + str(error))
        raise
'''

PREFLIGHT = r'''def preflight():
    emit('Python ' + sys.version)
    REPORT['python'] = sys.version
    run(['nvidia-smi'], timeout=30)
    run(['ffmpeg', '-version'], timeout=15)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: select a GPU session/image')
    tensor = torch.ones(4, device='cuda') * 2
    torch.cuda.synchronize()
    if not torch.isfinite(tensor).all().item():
        raise RuntimeError('Nonfinite CUDA smoke tensor')
    REPORT['torch'] = torch.__version__
    REPORT['gpu'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    save_report()
stage(preflight)
'''

CLONE = r'''def clone_repo():
    repo = Path('/kaggle/working/phowhisper-pytorch')
    origin = 'https://github.com/apase95/Speech-to-Text-PhoWhisper.git'
    if repo.exists():
        actual = run(['git', 'remote', 'get-url', 'origin'], cwd=repo)
        if actual != origin:
            raise RuntimeError('Existing repository origin mismatch; preserve directory and resolve manually')
        if run(['git', 'status', '--porcelain'], cwd=repo):
            raise RuntimeError('Existing repository is dirty; preserve edits and resolve manually')
        run(['git', 'fetch', '--depth', '1', 'origin', 'HEAD'], cwd=repo)
        run(['git', 'checkout', '--detach', 'FETCH_HEAD'], cwd=repo)
    else:
        run(['git', 'clone', '--depth', '1', origin, str(repo)], timeout=120)
    REPORT['repo_sha'] = run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    REPORT['repo_url'] = origin
    save_report()
stage(clone_repo)
'''

DEPENDENCIES = r'''
def dependencies():
    missing = missing_dependencies()
    # --no-deps prevents resolver replacement of Kaggle CUDA torch or existing packages.
    # Missing transitive dependencies fail the import check with an actionable blocker.
    if missing:
        run([sys.executable, '-u', '-m', 'pip', 'install', '--no-deps', *missing], timeout=180)
        REPORT['dependencies_installed'] = missing
        save_report()
        importlib.invalidate_caches()
    versions = {}
    for module in ('torch', 'transformers', 'accelerate', 'huggingface_hub', 'soundfile', 'numpy', 'scipy', 'jiwer', 'rapidfuzz'):
        try:
            imported = __import__(module)
            versions[module] = importlib.metadata.version(module.replace('_', '-'))
        except Exception as error:
            raise RuntimeError('Import failed for ' + module + '; repair GPU image/dependencies and restart; no automatic torch replacement') from error
    REPORT['versions'] = versions
    save_report()
stage(dependencies)
'''

NOTES = '''SETUP ONLY — no dataset inputs attached. Optional NEXT attachment, only after suitability review:
https://www.kaggle.com/datasets/lnt0821/vietnamese-accent-3-regions
Verified metadata: version 12, 18,176,281,455 bytes, 46,207 WAV files and one 873,547-byte CSV.
Description: accent/region/gender classification, original + expanded 16,156 samples.
Possible columns: path, split, region, region_id, gender_id. Transcripts and speaker IDs are NOT verified; license Unknown.
Audit CSV transcripts, speaker IDs, license, actual region counts, official split grouping, source/channel leakage
and duplication across vocal/song/phone variants before attachment or any ASR adapter.
Central/South focus is an intent, not verified counts or suitability. Northern bias in PhoWhisper/VietBUD500
is a user hypothesis, not established evidence. No quality improvement promises.
No training, pseudo-labels, audio/model downloads, or openai-whisper installation in this setup.
Checkpoints are not downloaded; src.check_environment is intentionally not invoked.
'''


def build(destination):
    destination.mkdir(parents=True, exist_ok=True)
    sources = [BOOTSTRAP, PREFLIGHT, CLONE,
               inspect.getsource(missing_dependencies) + DEPENDENCIES,
               'emit(' + repr(NOTES) + ")\nREPORT['status'] = 'setup_complete'\nsave_report()\nemit('Report: ' + str(OUT / 'report.json'))\n"]
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'cell_type': 'code', 'id': 'setup-' + str(i), 'metadata': {},
                           'execution_count': None, 'outputs': [], 'source': source.splitlines(True)}
                          for i, source in enumerate(sources)]}
    metadata = {'id': 'hngthiduy/phowhisper3', 'title': 'phowhisper3', 'code_file': 'phowhisper3.ipynb',
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': True,
                'docker_image': IMAGE, 'machine_shape': 'NvidiaTeslaT4',
                **{key: [] for key in ('dataset_sources', 'kernel_sources', 'competition_sources', 'model_sources')}}
    for name, content in [('phowhisper3.ipynb', notebook), ('kernel-metadata.json', metadata)]:
        (destination / name).write_text(json.dumps(content, indent=2) + '\n')


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[1] / 'temp_session_kaggle/phowhisper3-setup')
