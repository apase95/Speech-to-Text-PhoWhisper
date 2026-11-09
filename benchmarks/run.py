"""CLI đánh giá PhoWhisper: legacy greedy or frozen public fast/rescore.

Đọc manifest, kiểm tra tệp, nạp model cache rồi chạy từng mẫu tuần tự.
Mỗi lần chạy tạo thư mục mới chứa predictions.jsonl và summary.json với
WER/CER, thời gian, bộ nhớ, cấu hình và hash phục vụ truy vết. Đường dẫn
âm thanh được hiểu theo cwd, không theo thư mục chứa manifest.
"""

import argparse
from contextlib import ExitStack
from functools import lru_cache
from unittest.mock import patch
import csv
import hashlib
import json
import math
import platform
import os
import random
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
import jiwer

from benchmarks.metrics import corpus_error_rates, normalize_vietnamese
from src.audio_utils import apply_time_stretch
from src.inference import get_model, model_path


def validate_public_audio(paths):
    for path in paths:
        if not 0 < sf.info(path).duration <= 30:
            raise ValueError(f'Public comparison requires audio <=30 seconds: {path}')


class PublicMode(ExitStack):
    """Instrument the real public API, rejecting its warning-based LM fallback."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.details = {}
        self.lm_load_seconds = 0

    def __enter__(self):
        super().__enter__()
        try:
            if self.mode == 'rescore':
                from phowhisper import rescorer
                from benchmarks.runners.compare_vietmed_modes import pinned_loader, Diagnostics, FailWarnings, LM
                self.diagnostic = Diagnostics()
                handler = FailWarnings()
                rescorer.logger.addHandler(handler)
                self.callback(rescorer.logger.removeHandler, handler)
                cached = lru_cache(maxsize=1)(pinned_loader)
                torch.cuda.synchronize()
                start = time.perf_counter()
                cached()
                torch.cuda.synchronize()
                self.lm_load_seconds = time.perf_counter() - start
                original = rescorer.NBestDecodingTask.run_nbest
                def observe(task, mel):
                    result = original(task, mel)
                    self.diagnostic.candidates = result[0]
                    return result
                def lm(name=LM):
                    if name != LM:
                        raise ValueError('Unexpected LM')
                    self.diagnostic.lm_calls += 1
                    return cached()
                self.enter_context(patch.object(rescorer.NBestDecodingTask, 'run_nbest', observe))
                self.enter_context(patch.object(rescorer, 'get_lm_model', lm))
            return self
        except BaseException:
            self.close()
            raise

    def transcribe(self, path):
        from phowhisper import pipeline
        self.details = {}
        if self.mode == 'rescore':
            self.diagnostic.reset()
        text = pipeline.transcribe(path, rescore=self.mode == 'rescore')
        if self.mode == 'rescore':
            self.details = self.diagnostic.finish(text)
        return text


def cuda_peaks():
    return dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None


def add_silence_padding(audio: np.ndarray, milliseconds: int) -> np.ndarray:
    """Đệm số 0 hai đầu waveform 16 kHz; thời lượng âm gây ValueError."""
    if milliseconds < 0:
        raise ValueError("Silence padding must be non-negative")
    padding_samples = round(whisper.audio.SAMPLE_RATE * milliseconds / 1000)
    return np.pad(audio, (padding_samples, padding_samples))


def file_sha256(path: Path) -> str:
    """Đọc tệp theo khối 1 MiB và trả SHA-256; lỗi I/O truyền ra ngoài."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
    """Lấy HEAD của cwd qua git, trả None khi không gọi được hoặc git lỗi.

    Không ghi nhận diff của working tree, nên revision không chứng minh cây sạch.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_manifest(path: Path, limit: int | None) -> list[dict[str, str]]:
    """Đọc CSV UTF-8 có id/audio_path/reference/split rồi cắt rows[:limit].

    Manifest rỗng hoặc thiếu cột gây ValueError; không kiểm tra nội dung
    từng ô và không tự phân chia tập theo cột split.
    """
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "audio_path", "reference", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if any(not row['id'] or not row['reference'].strip() for row in rows):
        raise ValueError("IDs and references must be nonempty")
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError("Duplicate manifest IDs")
    for row in rows:
        actual = file_sha256(Path(row['audio_path']))
        if row.get('sha256') and actual != row['sha256']:
            raise ValueError(f"Audio hash mismatch: {row['id']}")
        row['sha256'] = actual
    return rows[:limit]


def source_state():
    diff = subprocess.run(['git', 'diff', 'HEAD', '--binary'], capture_output=True, check=True).stdout
    status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, check=True).stdout
    return {'revision': git_revision(), 'dirty': bool(status),
            'benchmark_source_sha256': {
                name: file_sha256(Path(name)) for name in (
                    'benchmarks/run.py', 'benchmarks/metrics.py',
                    'benchmarks/runners/compare_model_sizes.py',
                    'src/phowhisper/pipeline.py')},
            'tracked_diff_sha256': hashlib.sha256(diff).hexdigest(),
            'status_sha256': hashlib.sha256(status).hexdigest()}


def main() -> None:
    """Chạy benchmark và in tiến độ cùng summary ra stdout.

    Mặc định xuất vào outputs/benchmarks, không đệm im lặng, rate=1,
    without_timestamps=True và condition_on_previous_text=False; beam_size
    chỉ truyền khi được chỉ định. CUDA được đồng bộ để đo thời gian.
    Lỗi trong tiền xử lý/giải mã từng mẫu được lưu cùng dự đoán rỗng và
    vẫn tính vào metrics. Lỗi đọc metadata, ghi kết quả hay tổng hợp nằm
    ngoài khối bắt lỗi đó, có thể dừng lượt chạy và để lại đầu ra một phần.
    """
    parser = argparse.ArgumentParser(description="Benchmark PhoWhisper on a fixed manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument('--mode', choices=('greedy', 'fast', 'rescore'), default='greedy')
    parser.add_argument('--warmup-audio', type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--name", default="phowhisper-small")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--without-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--silence-padding-ms", type=int, default=0)
    parser.add_argument("--time-stretch-rate", type=float, default=1.0)
    args = parser.parse_args()
    public = args.mode != 'greedy'
    if public:
        if (args.beam_size is not None or not args.without_timestamps or args.condition_on_previous_text
                or args.silence_padding_ms or args.time_stretch_rate != 1):
            parser.error('Public modes require frozen public defaults; decoding overrides forbidden')
        if not torch.cuda.is_available():
            raise RuntimeError('Public benchmark requires CUDA; no CPU fallback')
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
        torch.set_num_threads(4)
    if args.warmup < 0:
        parser.error('--warmup must be nonnegative')
    if args.checkpoint:
        os.environ['PHOWHISPER_MODEL_PATH'] = str(args.checkpoint.resolve())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.silence_padding_ms < 0:
        parser.error("--silence-padding-ms must be non-negative")
    if args.time_stretch_rate <= 0:
        parser.error("--time-stretch-rate must be positive")

    # Kiểm tra đầu vào trước khi tạo thư mục kết quả và nạp checkpoint.
    rows = read_manifest(args.manifest, args.limit)
    if public:
        validate_public_audio([r['audio_path'] for r in rows] + ([str(args.warmup_audio)] if args.warmup_audio else []))
    missing = [row["audio_path"] for row in rows if not Path(row["audio_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} audio files; first: {missing[0]}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"{args.name}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = run_dir / "predictions.jsonl"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    load_started = time.perf_counter()
    model = get_model()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    public_mode = PublicMode(args.mode)
    if public:
        public_mode.__enter__()
    load_peaks = cuda_peaks()

    decode_options = {
        "language": "vi",
        "task": "transcribe",
        "fp16": torch.cuda.is_available(),
        "condition_on_previous_text": args.condition_on_previous_text,
        "without_timestamps": args.without_timestamps,
        "silence_padding_ms": args.silence_padding_ms,
        "time_stretch_rate": args.time_stretch_rate,
        "temperature": 0,
        "beam_size": None,
        "best_of": None,
    }
    if args.beam_size is not None:
        decode_options["beam_size"] = args.beam_size
    if public:
        from benchmarks.runners.compare_vietmed_modes import PROTOCOL
        decode_options = dict(public_api=True, fast=PROTOCOL['fast'],
                              rescoring=PROTOCOL['rescoring'] if args.mode == 'rescore' else None)

    references: list[str] = []
    hypotheses: list[str] = []
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0
    failures = 0
    latencies = []
    warm_options = {k: v for k, v in decode_options.items()
                    if k not in ('silence_padding_ms', 'time_stretch_rate')}
    warm_started = time.perf_counter()
    for _ in range(args.warmup):
        path = str(args.warmup_audio) if args.warmup_audio else rows[0]['audio_path']
        if public:
            public_mode.transcribe(path)
        else:
            model.transcribe(path, **warm_options)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warm_started
    if public:
        torch.cuda.reset_peak_memory_stats()

    # Thời gian từng mẫu gồm tiền xử lý nếu bật, không gồm đọc sf.info phía trước.
    with predictions_path.open("w", encoding="utf-8") as output:
        for index, row in enumerate(rows, start=1):
            audio_path = Path(row["audio_path"])
            duration = sf.info(audio_path).duration
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                transcription_input = str(audio_path)
                model_options = decode_options.copy()
                model_options.pop("silence_padding_ms", None)
                model_options.pop("time_stretch_rate", None)
                needs_stretch = not math.isclose(args.time_stretch_rate, 1.0, rel_tol=1e-5)
                if args.silence_padding_ms or needs_stretch:
                    audio = whisper.load_audio(str(audio_path))
                    if needs_stretch:
                        audio = apply_time_stretch(audio, args.time_stretch_rate)
                    if args.silence_padding_ms:
                        audio = add_silence_padding(audio, args.silence_padding_ms)
                    transcription_input = audio
                if public:
                    prediction = public_mode.transcribe(transcription_input)
                else:
                    prediction = model.transcribe(transcription_input, **model_options)["text"].strip()
                error = None
            except Exception as exc:
                prediction = ""
                error = f"{type(exc).__name__}: {exc}"
                failures += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)

            references.append(row["reference"])
            hypotheses.append(prediction)
            total_audio_seconds += duration
            total_inference_seconds += elapsed
            output.write(json.dumps({
                "id": row["id"],
                "audio_path": str(audio_path),
                "audio_sha256": row['sha256'],
                "split": row["split"],
                "reference": row["reference"],
                "hypothesis": prediction,
                "normalized_reference": normalize_vietnamese(row["reference"]),
                "normalized_hypothesis": normalize_vietnamese(prediction),
                "audio_seconds": duration,
                "inference_seconds": elapsed,
                "rtf": elapsed / duration,
                "error": error,
                **({'diagnostics': public_mode.details} if public else {}),
                "word_edits": edit_counts(row['reference'], prediction, False),
                "character_edits": edit_counts(row['reference'], prediction, True),
            }, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] {row['id']} - {elapsed:.2f}s", flush=True)

    # RTF dùng thời lượng tệp gốc; hash ghi nhận manifest và checkpoint hiện tại.
    rates = corpus_error_rates(references, hypotheses)
    checkpoint = model_path()
    public_mode.close()
    summary = {
        "run_id": run_id,
        "name": args.name,
        "samples": len(rows),
        "failures": failures,
        "protocol": {'decode': decode_options, 'seed': args.seed, 'warmup': args.warmup,
                     'rescore': args.mode == 'rescore', 'normalization': 'NFC-lower-punctuation-whitespace',
                     **({'mode': args.mode, 'warmup_audio': str(args.warmup_audio) if args.warmup_audio else rows[0]['audio_path'],
                         'warmup_audio_sha256': file_sha256(args.warmup_audio or Path(rows[0]['audio_path']))} if public else {})},
        **({'mode': args.mode, 'lm_load_seconds': public_mode.lm_load_seconds,
            'load_cuda_peak_bytes': load_peaks, 'measured_cuda_peak_bytes': cuda_peaks(),
            'warmup_seconds': warmup_seconds,
            'final_process_rss_bytes': int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')} if public else {}),
        "pilot": args.limit is not None,
        "source": source_state(),
        "latency_median_seconds": float(np.median(latencies)),
        "latency_p95_seconds": float(np.percentile(latencies, 95)),
        "cold_first_sample_seconds": latencies[0] if not args.warmup else None,
        "word_edits": edit_counts(references, hypotheses, False),
        "character_edits": edit_counts(references, hypotheses, True),
        "wer": rates["wer"],
        "cer": rates["cer"],
        "total_audio_seconds": total_audio_seconds,
        "total_inference_seconds": total_inference_seconds,
        "rtf": total_inference_seconds / total_audio_seconds,
        "model_load_seconds": load_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
        "cuda_total_capacity_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "decode_options": decode_options,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "whisper": getattr(whisper, "__version__", "unknown"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_revision": git_revision(),
        },
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def edit_counts(references, hypotheses, characters=False):
    if isinstance(references, str):
        references, hypotheses = [references], [hypotheses]
    processor = jiwer.process_characters if characters else jiwer.process_words
    result = processor([normalize_vietnamese(x) for x in references],
                       [normalize_vietnamese(x) for x in hypotheses])
    return {k: getattr(result, k) for k in ('substitutions', 'deletions', 'insertions', 'hits')}


if __name__ == "__main__":
    main()
