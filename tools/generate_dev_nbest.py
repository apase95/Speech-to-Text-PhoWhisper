"""Generate N-best ASR candidates on a fixed benchmark manifest and compute Oracle WER."""

import argparse
import csv
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import soundfile as sf
import torch
import whisper
from jiwer import cer, process_words, wer

from benchmarks.metrics import (
    corpus_error_rates,
    harmonize_vietnamese_tone_placement,
    normalize_vietnamese,
    normalize_vietnamese_harmonized,
)
from src.inference import get_model, model_path


def extract_standard_mel(audio_path: str | Path, model: whisper.Whisper) -> torch.Tensor:
    """Extract standard mel spectrogram matching whisper.transcribe exactly."""
    audio = whisper.load_audio(str(audio_path))
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    mel_full = whisper.log_mel_spectrogram(audio, model.dims.n_mels, padding=whisper.audio.N_SAMPLES)
    content_frames = mel_full.shape[-1] - whisper.audio.N_FRAMES
    segment_size = min(whisper.audio.N_FRAMES, content_frames)
    mel_segment = mel_full[:, 0:segment_size]
    return whisper.pad_or_trim(mel_segment, whisper.audio.N_FRAMES).to(model.device).to(dtype)


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


class NBestDecodingTask(whisper.decoding.DecodingTask):
    """Whisper decoding task that collects and exports all N-best candidates from beam search."""

    def run_nbest(self, mel: torch.Tensor) -> list[list[dict[str, Any]]]:
        self.decoder.reset()
        tokenizer = self.tokenizer
        n_audio = mel.shape[0]

        audio_features = self._get_audio_features(mel)
        tokens = torch.tensor([self.initial_tokens]).repeat(n_audio, 1)
        tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)

        tokens, sum_logprobs, _ = self._main_loop(audio_features, tokens)

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)

        results: list[list[dict[str, Any]]] = []
        for a_idx in range(n_audio):
            cands: list[dict[str, Any]] = []
            for t_seq, lp in zip(tokens[a_idx], sum_logprobs[a_idx]):
                eot_matches = (t_seq == tokenizer.eot).nonzero()
                eot_idx = (
                    eot_matches[0, 0].item()
                    if len(eot_matches) > 0
                    else len(t_seq)
                )
                cand_tokens = t_seq[self.sample_begin : eot_idx].tolist()
                text = tokenizer.decode(cand_tokens).strip()
                avg_lp = float(lp / (len(cand_tokens) + 1))
                cands.append(
                    {
                        "text": text,
                        "tokens": cand_tokens,
                        "sum_logprob": float(lp),
                        "avg_logprob": avg_lp,
                    }
                )

            # Sort by avg_logprob descending
            cands.sort(key=lambda x: x["avg_logprob"], reverse=True)
            for rank, c in enumerate(cands, start=1):
                c["rank"] = rank
            results.append(cands)
        return results


def calculate_word_distance(ref_norm: str, hyp_norm: str) -> tuple[int, int, int, int]:
    """Return (total_errors, substitutions, deletions, insertions) between ref and hyp."""
    out = process_words(ref_norm, hyp_norm)
    return (
        out.substitutions + out.deletions + out.insertions,
        out.substitutions,
        out.deletions,
        out.insertions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate N-best candidates and compute Oracle WER")
    parser.add_argument("--manifest", type=Path, default=Path("benchmarks/manifests/viet_bud500_dev_250.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--name", default="dev250-nbest")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if args.limit:
        rows = rows[: args.limit]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"{args.name}-beam{args.beam_size}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    predictions_path = run_dir / "nbest_predictions.jsonl"

    print(f"Loading PhoWhisper model from {model_path()}...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    load_start = time.perf_counter()
    model = get_model()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_time = time.perf_counter() - load_start

    options = whisper.DecodingOptions(
        language="vi",
        task="transcribe",
        beam_size=args.beam_size,
        without_timestamps=True,
        fp16=torch.cuda.is_available(),
    )
    task = NBestDecodingTask(model, options)

    records: list[dict[str, Any]] = []
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0

    print(f"Generating {args.beam_size}-best candidates for {len(rows)} samples...")
    with predictions_path.open("w", encoding="utf-8") as out_file:
        for index, row in enumerate(rows, start=1):
            audio_path = Path(row["audio_path"])
            duration = sf.info(audio_path).duration
            total_audio_seconds += duration

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()

            mel = extract_standard_mel(audio_path, model)

            with torch.no_grad():
                cands = task.run_nbest(mel.unsqueeze(0))[0]

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            total_inference_seconds += elapsed

            ref_raw = row["reference"]
            ref_norm_std = normalize_vietnamese(ref_raw)
            ref_norm_harm = normalize_vietnamese_harmonized(ref_raw)

            # Evaluate each candidate
            best_std_errors = float("inf")
            oracle_cand_std = None

            best_harm_errors = float("inf")
            oracle_cand_harm = None

            for c in cands:
                c_norm_std = normalize_vietnamese(c["text"])
                c_norm_harm = normalize_vietnamese_harmonized(c["text"])
                c["normalized_text"] = c_norm_std
                c["normalized_harmonized_text"] = c_norm_harm

                std_errs, s_std, d_std, i_std = calculate_word_distance(ref_norm_std, c_norm_std)
                c["word_errors_std"] = {"total": std_errs, "s": s_std, "d": d_std, "i": i_std}

                harm_errs, s_h, d_h, i_h = calculate_word_distance(ref_norm_harm, c_norm_harm)
                c["word_errors_harmonized"] = {"total": harm_errs, "s": s_h, "d": d_h, "i": i_h}

                if std_errs < best_std_errors:
                    best_std_errors = std_errs
                    oracle_cand_std = c

                if harm_errs < best_harm_errors:
                    best_harm_errors = harm_errs
                    oracle_cand_harm = c

            record = {
                "id": row["id"],
                "audio_path": str(audio_path),
                "audio_seconds": duration,
                "inference_seconds": elapsed,
                "reference": ref_raw,
                "normalized_reference_std": ref_norm_std,
                "normalized_reference_harmonized": ref_norm_harm,
                "top1_candidate": cands[0],
                "oracle_candidate_std": oracle_cand_std,
                "oracle_candidate_harmonized": oracle_cand_harm,
                "all_candidates": cands,
            }
            records.append(record)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if index % 25 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] {row['id']} - {elapsed:.2f}s", flush=True)

    # Compute Corpus-level Metrics
    # 1. Top-1 Metrics
    top1_hyps_std = [r["top1_candidate"]["normalized_text"] for r in records]
    top1_hyps_harm = [r["top1_candidate"]["normalized_harmonized_text"] for r in records]
    refs_std = [r["normalized_reference_std"] for r in records]
    refs_harm = [r["normalized_reference_harmonized"] for r in records]

    top1_words_std = process_words(refs_std, top1_hyps_std)
    top1_words_harm = process_words(refs_harm, top1_hyps_harm)

    top1_exact_std = sum(1 for r, h in zip(refs_std, top1_hyps_std) if r == h)
    top1_exact_harm = sum(1 for r, h in zip(refs_harm, top1_hyps_harm) if r == h)

    # 2. Oracle Metrics (Standard Normalization)
    oracle_hyps_std = [r["oracle_candidate_std"]["normalized_text"] for r in records]
    oracle_words_std = process_words(refs_std, oracle_hyps_std)
    oracle_exact_std = sum(1 for r, h in zip(refs_std, oracle_hyps_std) if r == h)

    # 3. Oracle Metrics (Harmonized Normalization)
    oracle_hyps_harm = [r["oracle_candidate_harmonized"]["normalized_harmonized_text"] for r in records]
    oracle_words_harm = process_words(refs_harm, oracle_hyps_harm)
    oracle_exact_harm = sum(1 for r, h in zip(refs_harm, oracle_hyps_harm) if r == h)

    summary = {
        "run_id": run_id,
        "name": args.name,
        "samples": len(rows),
        "beam_size": args.beam_size,
        "total_audio_seconds": total_audio_seconds,
        "total_inference_seconds": total_inference_seconds,
        "rtf": total_inference_seconds / total_audio_seconds,
        "model_load_seconds": load_time,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "checkpoint": str(model_path()),
        "checkpoint_sha256": file_sha256(model_path()),
        "metrics_comparison": {
            "top1_standard": {
                "wer": top1_words_std.wer,
                "cer": cer(refs_std, top1_hyps_std),
                "substitutions": top1_words_std.substitutions,
                "deletions": top1_words_std.deletions,
                "insertions": top1_words_std.insertions,
                "exact_matches": top1_exact_std,
                "exact_match_ratio": top1_exact_std / len(rows),
            },
            "top1_harmonized": {
                "wer": top1_words_harm.wer,
                "cer": cer(refs_harm, top1_hyps_harm),
                "substitutions": top1_words_harm.substitutions,
                "deletions": top1_words_harm.deletions,
                "insertions": top1_words_harm.insertions,
                "exact_matches": top1_exact_harm,
                "exact_match_ratio": top1_exact_harm / len(rows),
            },
            "oracle_standard": {
                "wer": oracle_words_std.wer,
                "cer": cer(refs_std, oracle_hyps_std),
                "substitutions": oracle_words_std.substitutions,
                "deletions": oracle_words_std.deletions,
                "insertions": oracle_words_std.insertions,
                "exact_matches": oracle_exact_std,
                "exact_match_ratio": oracle_exact_std / len(rows),
            },
            "oracle_harmonized": {
                "wer": oracle_words_harm.wer,
                "cer": cer(refs_harm, oracle_hyps_harm),
                "substitutions": oracle_words_harm.substitutions,
                "deletions": oracle_words_harm.deletions,
                "insertions": oracle_words_harm.insertions,
                "exact_matches": oracle_exact_harm,
                "exact_match_ratio": oracle_exact_harm / len(rows),
            },
        },
        "headroom": {
            "standard_wer_reduction": top1_words_std.wer - oracle_words_std.wer,
            "harmonized_wer_reduction": top1_words_harm.wer - oracle_words_harm.wer,
            "additional_exact_matches_standard": oracle_exact_std - top1_exact_std,
            "additional_exact_matches_harmonized": oracle_exact_harm - top1_exact_harm,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "whisper": getattr(whisper, "__version__", "unknown"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_revision": git_revision(),
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved N-best results and summary to {run_dir}")
    print(json.dumps(summary["metrics_comparison"], indent=2))
    print(json.dumps(summary["headroom"], indent=2))


if __name__ == "__main__":
    main()
