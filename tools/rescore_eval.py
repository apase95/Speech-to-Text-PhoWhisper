"""Cross-validation on dev set, parameter lock, and one-time test evaluation for PhoWhisper N-best LM rescoring."""

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
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.metrics import (
    harmonize_vietnamese_tone_placement,
    normalize_vietnamese,
    normalize_vietnamese_harmonized,
    paired_bootstrap_test,
)
from src.inference import get_model, model_path
from tools.generate_dev_nbest import (
    NBestDecodingTask,
    calculate_word_distance,
    extract_standard_mel,
    file_sha256,
    git_revision,
)


def score_text_with_lm(text: str, model: Any, tokenizer: Any, device: str) -> tuple[float, float]:
    """Return (lm_loss, lm_logprob_per_token)."""
    if not text.strip():
        return 20.0, -20.0
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = float(outputs.loss.item())
    return loss, -loss


def compute_metrics_dict(refs: list[str], hyps: list[str]) -> dict[str, Any]:
    words_out = process_words(refs, hyps)
    exact = sum(1 for r, h in zip(refs, hyps) if r == h)
    return {
        "wer": words_out.wer,
        "cer": cer(refs, hyps),
        "substitutions": words_out.substitutions,
        "deletions": words_out.deletions,
        "insertions": words_out.insertions,
        "hits": words_out.hits,
        "exact_matches": exact,
        "exact_match_ratio": exact / len(refs) if refs else 0.0,
    }


def select_rescored_candidate(
    candidates: list[dict[str, Any]],
    alpha: float,
    beta: float,
    gamma: float = -float("inf"),
    hyp_key: str = "normalized_text",
) -> dict[str, Any]:
    """Select candidate with optional acoustic safety margin gamma."""
    top1 = candidates[0]
    top1_ac = top1["avg_logprob"]
    best_cand = top1
    best_score = top1_ac + alpha * top1.get("lm_logprob_per_token", 0.0) + beta * len(top1[hyp_key].split())

    for c in candidates[1:]:
        score = c["avg_logprob"] + alpha * c.get("lm_logprob_per_token", 0.0) + beta * len(c[hyp_key].split())
        if score > best_score and (c["avg_logprob"] - top1_ac) >= gamma:
            best_score = score
            best_cand = c

    return best_cand


def run_dev_cross_validation(
    dev_enriched_path: Path,
    k_folds: int = 5,
    fixed_gamma: float | None = None,
) -> dict[str, Any]:
    print(f"\n--- Running {k_folds}-Fold Cross Validation on Dev 250 (Conservative Rescoring) ---")
    lines = [json.loads(l) for l in dev_enriched_path.read_text(encoding="utf-8").strip().split("\n")]
    n_samples = len(lines)
    fold_size = n_samples // k_folds
    folds = [lines[i * fold_size : (i + 1) * fold_size] for i in range(k_folds)]

    alphas = np.linspace(0.05, 0.25, 21)   # step 0.01
    betas = np.linspace(-0.02, 0.01, 16)   # step 0.002
    gammas = [fixed_gamma] if fixed_gamma is not None else [-float("inf"), -0.30, -0.20, -0.15, -0.10, -0.05]

    def eval_candidates(samples: list[dict[str, Any]], alpha: float, beta: float, gamma: float, harmonized: bool = False) -> float:
        refs, hyps = [], []
        ref_key = "normalized_reference_harmonized" if harmonized else "normalized_reference_std"
        hyp_key = "normalized_harmonized_text" if harmonized else "normalized_text"
        for s in samples:
            refs.append(s[ref_key])
            best_c = select_rescored_candidate(s["all_candidates"], alpha, beta, gamma, hyp_key)
            hyps.append(best_c[hyp_key])
        return wer(refs, hyps)

    fold_results = []
    oof_predictions = []

    for k in range(k_folds):
        val_fold = folds[k]
        train_fold = [item for j in range(k_folds) if j != k for item in folds[j]]

        # Optimize alpha, beta, gamma on train fold
        best_train_wer = float("inf")
        best_alpha, best_beta, best_gamma = 0.0, 0.0, -float("inf")
        for g in gammas:
            for a in alphas:
                for b in betas:
                    w = eval_candidates(train_fold, a, b, g, harmonized=False)
                    if w < best_train_wer:
                        best_train_wer = w
                        best_alpha, best_beta, best_gamma = a, b, g

        # Evaluate on validation fold
        val_refs_std = [s["normalized_reference_std"] for s in val_fold]
        val_refs_harm = [s["normalized_reference_harmonized"] for s in val_fold]

        val_hyps_top1_std = [s["top1_candidate"]["normalized_text"] for s in val_fold]
        val_hyps_rescored_std = []
        val_hyps_rescored_harm = []

        for s in val_fold:
            best_c = select_rescored_candidate(
                s["all_candidates"],
                best_alpha,
                best_beta,
                best_gamma,
                "normalized_text",
            )
            val_hyps_rescored_std.append(best_c["normalized_text"])
            val_hyps_rescored_harm.append(best_c["normalized_harmonized_text"])

            oof_predictions.append({
                "id": s["id"],
                "fold": k + 1,
                "reference_std": s["normalized_reference_std"],
                "reference_harmonized": s["normalized_reference_harmonized"],
                "top1_text_std": s["top1_candidate"]["normalized_text"],
                "rescored_text_std": best_c["normalized_text"],
                "rescored_text_harmonized": best_c["normalized_harmonized_text"],
                "chosen_rank": best_c["rank"],
                "alpha_used": best_alpha,
                "beta_used": best_beta,
                "gamma_used": best_gamma,
            })

        val_wer_top1 = wer(val_refs_std, val_hyps_top1_std)
        val_wer_rescored = wer(val_refs_std, val_hyps_rescored_std)

        fold_results.append({
            "fold": k + 1,
            "train_samples": len(train_fold),
            "val_samples": len(val_fold),
            "best_alpha": float(best_alpha),
            "best_beta": float(best_beta),
            "best_gamma": float(best_gamma),
            "train_wer": float(best_train_wer),
            "val_wer_top1": float(val_wer_top1),
            "val_wer_rescored": float(val_wer_rescored),
            "val_wer_reduction": float(val_wer_top1 - val_wer_rescored),
        })
        print(f"  Fold {k+1}: best alpha={best_alpha:.3f}, beta={best_beta:.3f}, gamma={best_gamma:.2f} | Val Top-1: {val_wer_top1*100:.2f}% -> Val Rescored: {val_wer_rescored*100:.2f}%")

    # Aggregate Out-of-Fold (OOF)
    oof_refs_std = [p["reference_std"] for p in oof_predictions]
    oof_refs_harm = [p["reference_harmonized"] for p in oof_predictions]
    oof_top1_std = [p["top1_text_std"] for p in oof_predictions]
    oof_rescored_std = [p["rescored_text_std"] for p in oof_predictions]
    oof_rescored_harm = [p["rescored_text_harmonized"] for p in oof_predictions]

    oof_metrics_top1_std = compute_metrics_dict(oof_refs_std, oof_top1_std)
    oof_metrics_rescored_std = compute_metrics_dict(oof_refs_std, oof_rescored_std)
    oof_metrics_rescored_harm = compute_metrics_dict(oof_refs_harm, oof_rescored_harm)

    # Calculate median locked parameters
    locked_alpha = float(np.median([fr["best_alpha"] for fr in fold_results]))
    locked_beta = float(np.median([fr["best_beta"] for fr in fold_results]))
    locked_gamma = float(np.median([fr["best_gamma"] for fr in fold_results]))

    # Compute OOF Paired Bootstrap Test
    oof_bootstrap_std = paired_bootstrap_test(oof_refs_std, oof_top1_std, oof_rescored_std, n_bootstraps=2000, seed=42)

    print(f"\nOOF Top-1 WER:    {oof_metrics_top1_std['wer']*100:.2f}%")
    print(f"OOF Rescored WER: {oof_metrics_rescored_std['wer']*100:.2f}% (Harmonized: {oof_metrics_rescored_harm['wer']*100:.2f}%)")
    print(f"Locked Parameters: alpha={locked_alpha:.3f}, beta={locked_beta:.3f}, gamma={locked_gamma:.2f}")
    print(f"OOF Bootstrap 95% CI: [{oof_bootstrap_std['ci_lower']:.3f}%, {oof_bootstrap_std['ci_upper']:.3f}%], p-value: {oof_bootstrap_std['p_value']:.5f}")

    return {
        "k_folds": k_folds,
        "fold_results": fold_results,
        "locked_parameters": {
            "alpha": locked_alpha,
            "beta": locked_beta,
            "gamma": locked_gamma,
            "selection_method": "median_across_cv_folds",
        },
        "oof_metrics": {
            "top1_standard": oof_metrics_top1_std,
            "rescored_standard": oof_metrics_rescored_std,
            "rescored_harmonized": oof_metrics_rescored_harm,
            "wer_reduction_standard": oof_metrics_top1_std["wer"] - oof_metrics_rescored_std["wer"],
            "exact_match_gain": oof_metrics_rescored_std["exact_matches"] - oof_metrics_top1_std["exact_matches"],
        },
        "oof_bootstrap_standard": oof_bootstrap_std,
        "oof_predictions": oof_predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Locked PhoWhisper N-Best Conservative Rescoring Evaluation")
    parser.add_argument(
        "--dev-nbest",
        type=Path,
        default=Path("outputs/benchmarks/dev250-nbest-standard-beam5-20261020T182818Z/nbest_predictions_with_lm.jsonl"),
    )
    parser.add_argument("--test-manifest", type=Path, default=Path("benchmarks/manifests/viet_bud500_1000.csv"))
    parser.add_argument("--test-predictions-input", type=Path, help="Pre-scored test predictions file to fast re-evaluate with locked parameters")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--lm-model", default="NlpHUST/gpt2-vietnamese")
    parser.add_argument("--fixed-gamma", type=float, default=None, help="Force a fixed acoustic safety margin instead of grid-searching CV")
    parser.add_argument("--skip-test", action="store_true", help="Only run dev CV without test evaluation")
    parser.add_argument("--limit-test", type=int, help="Limit number of test samples for dry run")
    args = parser.parse_args()

    # Step 1: Run 5-Fold Cross-Validation on Dev 250
    cv_report = run_dev_cross_validation(args.dev_nbest, k_folds=5, fixed_gamma=args.fixed_gamma)

    if args.skip_test:
        print("\nSkipping test evaluation as requested.")
        return

    # Check stability condition before proceeding to Test
    oof_wer_reduction = cv_report["oof_metrics"]["wer_reduction_standard"]
    if oof_wer_reduction <= 0:
        print(f"\n[ABORT] Out-of-fold WER did not improve on dev (change: {oof_wer_reduction:+.4f}). Aborting test run.")
        return

    print(f"\n[DECISION: GO TO TEST] Dev OOF WER improved by {oof_wer_reduction*100:.2f}%. Proceeding to test set evaluation!")

    # Step 2: Lock Configuration
    locked_alpha = cv_report["locked_parameters"]["alpha"]
    locked_beta = cv_report["locked_parameters"]["beta"]
    locked_gamma = cv_report["locked_parameters"]["gamma"]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"test1000-conservative-rescored-beam{args.beam_size}-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Save Dev CV summary
    (run_dir / "dev_cross_validation.json").write_text(json.dumps(cv_report, indent=2, ensure_ascii=False), encoding="utf-8")

    test_predictions_path = run_dir / "test_predictions.jsonl"
    test_records = []

    if args.test_predictions_input and args.test_predictions_input.is_file():
        print(f"\nRe-evaluating from pre-scored test predictions: {args.test_predictions_input}")
        total_audio_seconds = 0.0
        total_inference_seconds = 0.0
        total_load_seconds = 0.0

        with test_predictions_path.open("w", encoding="utf-8") as out_file:
            for line in args.test_predictions_input.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                record = json.loads(line)
                cands = record["all_candidates"]
                total_audio_seconds += record.get("audio_seconds", 0.0)
                total_inference_seconds += record.get("inference_seconds", 0.0)

                for c in cands:
                    c["composite_score"] = c["avg_logprob"] + locked_alpha * c.get("lm_logprob_per_token", 0.0) + locked_beta * len(c["normalized_text"].split())

                rescored_cand = select_rescored_candidate(cands, locked_alpha, locked_beta, locked_gamma, "normalized_text")
                record["rescored_candidate"] = rescored_cand
                test_records.append(record)
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        # Step 3: Run Inference on Test Set (1.000 samples)
        with args.test_manifest.open(encoding="utf-8", newline="") as stream:
            test_rows = list(csv.DictReader(stream))
        if args.limit_test:
            test_rows = test_rows[: args.limit_test]

        print(f"\nLoading Whisper model and LM ({args.lm_model}) on GPU...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        load_start = time.perf_counter()
        whisper_model = get_model()
        lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model)
        lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model).to(device)
        lm_model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_load_seconds = time.perf_counter() - load_start

        options = whisper.DecodingOptions(
            language="vi",
            task="transcribe",
            beam_size=args.beam_size,
            without_timestamps=True,
            fp16=torch.cuda.is_available(),
        )
        task = NBestDecodingTask(whisper_model, options)

        total_audio_seconds = 0.0
        total_inference_seconds = 0.0

        print(f"\nProcessing {len(test_rows)} test samples with locked parameters (alpha={locked_alpha}, beta={locked_beta}, gamma={locked_gamma})...")
        with test_predictions_path.open("w", encoding="utf-8") as out_file:
            for index, row in enumerate(test_rows, start=1):
                audio_path = Path(row["audio_path"])
                duration = sf.info(audio_path).duration
                total_audio_seconds += duration

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()

                mel = extract_standard_mel(audio_path, whisper_model)

                with torch.no_grad():
                    cands = task.run_nbest(mel.unsqueeze(0))[0]

                # LM scoring for candidates
                for c in cands:
                    text = c["text"]
                    _, lm_lp = score_text_with_lm(text, lm_model, lm_tokenizer, device)
                    c["lm_logprob_per_token"] = lm_lp
                    c["normalized_text"] = normalize_vietnamese(text)
                    c["normalized_harmonized_text"] = normalize_vietnamese_harmonized(text)
                    c["composite_score"] = c["avg_logprob"] + locked_alpha * lm_lp + locked_beta * len(c["normalized_text"].split())

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                total_inference_seconds += elapsed

                ref_raw = row["reference"]
                ref_norm_std = normalize_vietnamese(ref_raw)
                ref_norm_harm = normalize_vietnamese_harmonized(ref_raw)

                top1_cand = cands[0]
                rescored_cand = select_rescored_candidate(cands, locked_alpha, locked_beta, locked_gamma, "normalized_text")
                oracle_cand_std = min(cands, key=lambda c: calculate_word_distance(ref_norm_std, c["normalized_text"])[0])
                oracle_cand_harm = min(cands, key=lambda c: calculate_word_distance(ref_norm_harm, c["normalized_harmonized_text"])[0])

                record = {
                    "id": row["id"],
                    "audio_path": str(audio_path),
                    "audio_seconds": duration,
                    "inference_seconds": elapsed,
                    "reference": ref_raw,
                    "normalized_reference_std": ref_norm_std,
                    "normalized_reference_harmonized": ref_norm_harm,
                    "top1_candidate": top1_cand,
                    "rescored_candidate": rescored_cand,
                    "oracle_candidate_std": oracle_cand_std,
                    "oracle_candidate_harmonized": oracle_cand_harm,
                    "all_candidates": cands,
                }
                test_records.append(record)
                out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                if index % 50 == 0 or index == len(test_rows):
                    print(f"[{index}/{len(test_rows)}] {row['id']} - {elapsed:.2f}s", flush=True)

    # Compute Final Test Corpus Metrics
    refs_std = [r["normalized_reference_std"] for r in test_records]
    refs_harm = [r["normalized_reference_harmonized"] for r in test_records]

    top1_std = [r["top1_candidate"]["normalized_text"] for r in test_records]
    top1_harm = [r["top1_candidate"]["normalized_harmonized_text"] for r in test_records]

    rescored_std = [r["rescored_candidate"]["normalized_text"] for r in test_records]
    rescored_harm = [r["rescored_candidate"]["normalized_harmonized_text"] for r in test_records]

    oracle_std = [r["oracle_candidate_std"]["normalized_text"] for r in test_records]
    oracle_harm = [r["oracle_candidate_harmonized"]["normalized_harmonized_text"] for r in test_records]

    test_metrics = {
        "top1_standard": compute_metrics_dict(refs_std, top1_std),
        "top1_harmonized": compute_metrics_dict(refs_harm, top1_harm),
        "rescored_standard": compute_metrics_dict(refs_std, rescored_std),
        "rescored_harmonized": compute_metrics_dict(refs_harm, rescored_harm),
        "oracle_standard": compute_metrics_dict(refs_std, oracle_std),
        "oracle_harmonized": compute_metrics_dict(refs_harm, oracle_harm),
    }

    # Compute Paired Bootstrap Significance Tests
    bootstrap_std = paired_bootstrap_test(refs_std, top1_std, rescored_std, n_bootstraps=2000, seed=42)
    bootstrap_harm = paired_bootstrap_test(refs_harm, top1_harm, rescored_harm, n_bootstraps=2000, seed=42, harmonized=True)

    test_summary = {
        "run_id": run_id,
        "name": "test1000-conservative-rescored-beam5",
        "samples": len(test_records),
        "beam_size": args.beam_size,
        "total_audio_seconds": total_audio_seconds,
        "total_inference_seconds": total_inference_seconds,
        "rtf": total_inference_seconds / total_audio_seconds if total_audio_seconds > 0 else 0.0,
        "model_load_seconds": total_load_seconds,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "locked_parameters": {
            "lm_model": args.lm_model,
            "alpha": locked_alpha,
            "beta": locked_beta,
            "gamma": locked_gamma,
            "provenance": "median_across_5_fold_cross_validation_on_dev250",
        },
        "manifest": str(args.test_manifest),
        "manifest_sha256": file_sha256(args.test_manifest),
        "checkpoint": str(model_path()),
        "checkpoint_sha256": file_sha256(model_path()),
        "test_metrics": test_metrics,
        "paired_bootstrap_standard": bootstrap_std,
        "paired_bootstrap_harmonized": bootstrap_harm,
        "gains_vs_top1": {
            "wer_reduction_standard": test_metrics["top1_standard"]["wer"] - test_metrics["rescored_standard"]["wer"],
            "wer_reduction_harmonized": test_metrics["top1_harmonized"]["wer"] - test_metrics["rescored_harmonized"]["wer"],
            "exact_matches_gain": test_metrics["rescored_standard"]["exact_matches"] - test_metrics["top1_standard"]["exact_matches"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "whisper": getattr(whisper, "__version__", "unknown"),
            "transformers": getattr(sys.modules.get("transformers"), "__version__", "unknown"),
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_revision": git_revision(),
        },
    }

    (run_dir / "summary.json").write_text(json.dumps(test_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFinal Test Results saved to {run_dir}")
    print(json.dumps(test_summary["test_metrics"], indent=2))
    print(json.dumps(test_summary["gains_vs_top1"], indent=2))
    print("Paired Bootstrap Standard:")
    print(json.dumps(test_summary["paired_bootstrap_standard"], indent=2))


if __name__ == "__main__":
    main()
