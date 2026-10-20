import argparse
import csv
import hashlib
import json
import math
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper

from benchmarks.metrics import corpus_error_rates, normalize_vietnamese
from src.audio_utils import apply_time_stretch
from src.inference import get_model, model_path


def add_silence_padding(audio: np.ndarray, milliseconds: int) -> np.ndarray:
    if milliseconds < 0:
        raise ValueError("Silence padding must be non-negative")
    padding_samples = round(whisper.audio.SAMPLE_RATE * milliseconds / 1000)
    return np.pad(audio, (padding_samples, padding_samples))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> str | None:
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
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"id", "audio_path", "reference", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PhoWhisper on a fixed manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--name", default="phowhisper-base")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--beam-size", type=int)
    parser.add_argument("--without-timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--condition-on-previous-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--silence-padding-ms", type=int, default=0)
    parser.add_argument("--time-stretch-rate", type=float, default=1.0)
    args = parser.parse_args()

    if args.silence_padding_ms < 0:
        parser.error("--silence-padding-ms must be non-negative")
    if args.time_stretch_rate <= 0:
        parser.error("--time-stretch-rate must be positive")

    rows = read_manifest(args.manifest, args.limit)
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

    decode_options = {
        "language": "vi",
        "task": "transcribe",
        "fp16": torch.cuda.is_available(),
        "condition_on_previous_text": args.condition_on_previous_text,
        "without_timestamps": args.without_timestamps,
        "silence_padding_ms": args.silence_padding_ms,
        "time_stretch_rate": args.time_stretch_rate,
    }
    if args.beam_size is not None:
        decode_options["beam_size"] = args.beam_size

    references: list[str] = []
    hypotheses: list[str] = []
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0
    failures = 0

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
                model_options.pop("silence_padding_ms")
                model_options.pop("time_stretch_rate")
                needs_stretch = not math.isclose(args.time_stretch_rate, 1.0, rel_tol=1e-5)
                if args.silence_padding_ms or needs_stretch:
                    audio = whisper.load_audio(str(audio_path))
                    if needs_stretch:
                        audio = apply_time_stretch(audio, args.time_stretch_rate)
                    if args.silence_padding_ms:
                        audio = add_silence_padding(audio, args.silence_padding_ms)
                    transcription_input = audio
                prediction = model.transcribe(transcription_input, **model_options)["text"].strip()
                error = None
            except Exception as exc:
                prediction = ""
                error = f"{type(exc).__name__}: {exc}"
                failures += 1
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            references.append(row["reference"])
            hypotheses.append(prediction)
            total_audio_seconds += duration
            total_inference_seconds += elapsed
            output.write(json.dumps({
                "id": row["id"],
                "audio_path": str(audio_path),
                "split": row["split"],
                "reference": row["reference"],
                "hypothesis": prediction,
                "normalized_reference": normalize_vietnamese(row["reference"]),
                "normalized_hypothesis": normalize_vietnamese(prediction),
                "audio_seconds": duration,
                "inference_seconds": elapsed,
                "rtf": elapsed / duration,
                "error": error,
            }, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(rows)}] {row['id']} - {elapsed:.2f}s", flush=True)

    rates = corpus_error_rates(references, hypotheses)
    checkpoint = model_path()
    summary = {
        "run_id": run_id,
        "name": args.name,
        "samples": len(rows),
        "failures": failures,
        "wer": rates["wer"],
        "cer": rates["cer"],
        "total_audio_seconds": total_audio_seconds,
        "total_inference_seconds": total_inference_seconds,
        "rtf": total_inference_seconds / total_audio_seconds,
        "model_load_seconds": load_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
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


if __name__ == "__main__":
    main()
