"""Benchmark PhoBERT (vinai/phobert-base-v2) as a Challenger for N-best ASR Rescoring."""

import argparse
import csv
import hashlib
import json
import platform
import resource
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
import torch.nn.functional as F
from jiwer import cer, process_words, wer
from pyvi import ViTokenizer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from benchmarks.metrics import (
    harmonize_vietnamese_tone_placement,
    normalize_vietnamese,
    normalize_vietnamese_harmonized,
    paired_bootstrap_test,
)
from src.inference import model_path
from tools.generate_dev_nbest import file_sha256, git_revision
from tools.rescore_eval import compute_metrics_dict, select_rescored_candidate


def score_text_with_pll(
    text: str,
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    device: str,
    max_chunk_size: int = 16,
) -> tuple[float, float]:
    """Compute Pseudo-Log-Likelihood (PLL) and PLL-per-token using PhoBERT with chunked masking."""
    if not text.strip():
        return -20.0, -20.0

    segmented = ViTokenizer.tokenize(text.strip())
    max_seq_len = getattr(tokenizer, "model_max_length", 256)
    if max_seq_len > 512:
        max_seq_len = 256

    input_ids = tokenizer.encode(
        segmented,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)[0]  # shape: (L,)

    L = len(input_ids)
    if L <= 2:  # only <s> and </s>
        return 0.0, 0.0

    positions = torch.arange(1, L - 1, device=device)
    target_ids = input_ids[positions]
    T = len(positions)
    token_lps = []

    for start in range(0, T, max_chunk_size):
        end = min(start + max_chunk_size, T)
        pos = positions[start:end]
        tgt = target_ids[start:end]
        k = len(pos)

        chunk_batch = input_ids.unsqueeze(0).repeat(k, 1)
        chunk_batch[torch.arange(k, device=device), pos] = tokenizer.mask_token_id

        with torch.no_grad():
            sub_logits = model(chunk_batch).logits
        m_logits = sub_logits[torch.arange(k, device=device), pos]
        token_lps.append(F.log_softmax(m_logits, dim=-1)[torch.arange(k, device=device), tgt])

    all_lps = torch.cat(token_lps)
    pll = float(all_lps.sum().item())
    pll_per_token = float(all_lps.mean().item())
    return pll, pll_per_token


def enrich_samples_with_phobert(
    samples: list[dict[str, Any]],
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    device: str,
    desc: str = "Dev",
) -> tuple[list[dict[str, Any]], float]:
    """Enrich N-best candidates with PhoBERT PLL scores."""
    print(f"\nComputing PhoBERT PLL scores for {len(samples)} {desc} samples (5 candidates each)...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for idx, s in enumerate(samples, start=1):
        for c in s["all_candidates"]:
            if "phobert_pll_per_token" not in c:
                norm_text = c.get("normalized_text", normalize_vietnamese(c["text"]))
                c["normalized_text"] = norm_text
                pll, pll_pt = score_text_with_pll(norm_text, model, tokenizer, device)
                c["phobert_pll"] = pll
                c["phobert_pll_per_token"] = pll_pt
                c["lm_logprob_per_token"] = pll_pt  # for compatibility with select_rescored_candidate

        if idx % 50 == 0 or idx == len(samples):
            print(f"  [{idx}/{len(samples)}] processed", flush=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"Done computing PhoBERT PLL for {desc} in {elapsed:.2f}s ({elapsed/len(samples)*1000:.1f} ms/sample).")
    return samples, elapsed


def run_phobert_cross_validation(
    dev_samples: list[dict[str, Any]],
    k_folds: int = 5,
) -> dict[str, Any]:
    print(f"\n--- Running {k_folds}-Fold Cross Validation with PhoBERT PLL on Dev 250 ---")
    n_samples = len(dev_samples)
    fold_size = n_samples // k_folds
    folds = [dev_samples[i * fold_size : (i + 1) * fold_size] for i in range(k_folds)]

    alphas = np.linspace(0.02, 0.20, 19)   # step 0.01
    betas = np.linspace(-0.02, 0.02, 21)   # step 0.002
    gammas = [-float("inf"), -0.30, -0.20, -0.15, -0.10, -0.05]

    def eval_candidates(samples: list[dict[str, Any]], alpha: float, beta: float, gamma: float) -> float:
        refs, hyps = [], []
        for s in samples:
            refs.append(s["normalized_reference_std"])
            best_c = select_rescored_candidate(s["all_candidates"], alpha, beta, gamma, "normalized_text")
            hyps.append(best_c["normalized_text"])
        return wer(refs, hyps)

    fold_results = []
    oof_predictions = []

    for k in range(k_folds):
        val_fold = folds[k]
        train_fold = [item for j in range(k_folds) if j != k for item in folds[j]]

        best_train_wer = float("inf")
        best_alpha, best_beta, best_gamma = 0.0, 0.0, -float("inf")
        for g in gammas:
            for a in alphas:
                for b in betas:
                    w = eval_candidates(train_fold, a, b, g)
                    if w < best_train_wer:
                        best_train_wer = w
                        best_alpha, best_beta, best_gamma = a, b, g

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
            val_hyps_rescored_harm.append(best_c.get("normalized_harmonized_text", normalize_vietnamese_harmonized(best_c["text"])))

            oof_predictions.append({
                "id": s["id"],
                "fold": k + 1,
                "reference_std": s["normalized_reference_std"],
                "reference_harmonized": s["normalized_reference_harmonized"],
                "top1_text_std": s["top1_candidate"]["normalized_text"],
                "rescored_text_std": best_c["normalized_text"],
                "chosen_rank": best_c.get("rank", 1),
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

    oof_refs_std = [p["reference_std"] for p in oof_predictions]
    oof_refs_harm = [p["reference_harmonized"] for p in oof_predictions]
    oof_top1_std = [p["top1_text_std"] for p in oof_predictions]
    oof_rescored_std = [p["rescored_text_std"] for p in oof_predictions]

    oof_metrics_top1 = compute_metrics_dict(oof_refs_std, oof_top1_std)
    oof_metrics_rescored = compute_metrics_dict(oof_refs_std, oof_rescored_std)

    locked_alpha = float(np.median([fr["best_alpha"] for fr in fold_results]))
    locked_beta = float(np.median([fr["best_beta"] for fr in fold_results]))
    locked_gamma = float(np.median([fr["best_gamma"] for fr in fold_results]))

    oof_bootstrap = paired_bootstrap_test(oof_refs_std, oof_top1_std, oof_rescored_std, n_bootstraps=2000, seed=42)

    print(f"\nPhoBERT OOF Top-1 WER:    {oof_metrics_top1['wer']*100:.2f}%")
    print(f"PhoBERT OOF Rescored WER: {oof_metrics_rescored['wer']*100:.2f}%")
    print(f"Locked Parameters: alpha={locked_alpha:.3f}, beta={locked_beta:.3f}, gamma={locked_gamma:.2f}")
    print(f"OOF Bootstrap 95% CI: [{oof_bootstrap['ci_lower']:.3f}%, {oof_bootstrap['ci_upper']:.3f}%], p-value: {oof_bootstrap['p_value']:.5f}")

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
            "top1_standard": oof_metrics_top1,
            "rescored_standard": oof_metrics_rescored,
            "wer_reduction_standard": oof_metrics_top1["wer"] - oof_metrics_rescored["wer"],
            "exact_match_gain": oof_metrics_rescored["exact_matches"] - oof_metrics_top1["exact_matches"],
        },
        "oof_bootstrap_standard": oof_bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PhoBERT as Challenger for ASR N-best Rescoring")
    parser.add_argument(
        "--dev-input",
        type=Path,
        default=Path("outputs/benchmarks/dev250-nbest-standard-beam5-20261020T182818Z/nbest_predictions.jsonl"),
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=Path("outputs/benchmarks/test1000-rescored-beam5-20261020T183131Z/test_predictions.jsonl"),
    )
    parser.add_argument("--model-name", default="vinai/phobert-base-v2")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_name} on {device}...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_time = time.perf_counter() - t0
    print(f"Model loaded in {model_load_time:.2f}s.")

    # 1. Dev 250 evaluation
    dev_samples = [json.loads(l) for l in args.dev_input.read_text(encoding="utf-8").strip().split("\n")]
    dev_enriched, dev_scoring_time = enrich_samples_with_phobert(dev_samples, model, tokenizer, device, desc="Dev 250")

    # Run 5-fold CV on dev
    cv_report = run_phobert_cross_validation(dev_enriched, k_folds=5)

    locked_alpha = cv_report["locked_parameters"]["alpha"]
    locked_beta = cv_report["locked_parameters"]["beta"]
    locked_gamma = cv_report["locked_parameters"]["gamma"]

    # 2. Test 1000 evaluation
    test_samples = [json.loads(l) for l in args.test_input.read_text(encoding="utf-8").strip().split("\n")]
    test_enriched, test_scoring_time = enrich_samples_with_phobert(test_samples, model, tokenizer, device, desc="Test 1000")

    # Apply locked configuration
    refs_std = [s["normalized_reference_std"] for s in test_enriched]
    refs_harm = [s["normalized_reference_harmonized"] for s in test_enriched]
    top1_std = [s["top1_candidate"]["normalized_text"] for s in test_enriched]
    top1_harm = [s["top1_candidate"]["normalized_harmonized_text"] for s in test_enriched]

    gpt2_std = [s["rescored_candidate"]["normalized_text"] for s in test_enriched]
    gpt2_harm = [s["rescored_candidate"]["normalized_harmonized_text"] for s in test_enriched]

    phobert_std = []
    phobert_harm = []

    for s in test_enriched:
        best_c = select_rescored_candidate(
            s["all_candidates"],
            locked_alpha,
            locked_beta,
            locked_gamma,
            "normalized_text",
        )
        phobert_std.append(best_c["normalized_text"])
        phobert_harm.append(best_c.get("normalized_harmonized_text", normalize_vietnamese_harmonized(best_c["text"])))

    # Metrics
    m_top1 = compute_metrics_dict(refs_std, top1_std)
    m_gpt2 = compute_metrics_dict(refs_std, gpt2_std)
    m_phobert = compute_metrics_dict(refs_std, phobert_std)

    m_top1_harm = compute_metrics_dict(refs_harm, top1_harm)
    m_gpt2_harm = compute_metrics_dict(refs_harm, gpt2_harm)
    m_phobert_harm = compute_metrics_dict(refs_harm, phobert_harm)

    # Paired Bootstrap Tests:
    # 1. PhoBERT vs Top-1 Baseline
    bs_vs_top1 = paired_bootstrap_test(refs_std, top1_std, phobert_std, n_bootstraps=2000, seed=42)
    # 2. PhoBERT vs GPT-2
    bs_vs_gpt2 = paired_bootstrap_test(refs_std, gpt2_std, phobert_std, n_bootstraps=2000, seed=42)

    peak_cuda = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / f"test1000-phobert-challenger-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "run_id": run_id,
        "model_name": args.model_name,
        "model_load_seconds": model_load_time,
        "dev_samples": len(dev_samples),
        "test_samples": len(test_samples),
        "dev_scoring_seconds": dev_scoring_time,
        "test_scoring_seconds": test_scoring_time,
        "test_latency_per_sample_ms": (test_scoring_time / len(test_samples)) * 1000,
        "peak_cuda_memory_bytes": peak_cuda,
        "peak_process_rss_bytes": peak_rss,
        "locked_parameters": cv_report["locked_parameters"],
        "dev_cv_report": cv_report,
        "test_metrics": {
            "top1_standard": m_top1,
            "top1_harmonized": m_top1_harm,
            "gpt2_standard": m_gpt2,
            "gpt2_harmonized": m_gpt2_harm,
            "phobert_standard": m_phobert,
            "phobert_harmonized": m_phobert_harm,
        },
        "bootstrap_vs_top1": bs_vs_top1,
        "bootstrap_vs_gpt2": bs_vs_gpt2,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print("PHOBERT CHALLENGER EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Baseline Top-1 WER:      {m_top1['wer']*100:.2f}% (Exact: {m_top1['exact_matches']}/1000, S={m_top1['substitutions']}, D={m_top1['deletions']})")
    print(f"GPT-2 Rescored WER:      {m_gpt2['wer']*100:.2f}% (Exact: {m_gpt2['exact_matches']}/1000, S={m_gpt2['substitutions']}, D={m_gpt2['deletions']})")
    print(f"PhoBERT Rescored WER:    {m_phobert['wer']*100:.2f}% (Exact: {m_phobert['exact_matches']}/1000, S={m_phobert['substitutions']}, D={m_phobert['deletions']})")
    print("-" * 70)
    print(f"PhoBERT vs Top-1 Delta:  {bs_vs_top1['mean_diff']:+.3f}% [95% CI: {bs_vs_top1['ci_lower']:+.3f}%, {bs_vs_top1['ci_upper']:+.3f}%], p={bs_vs_top1['p_value']:.5f}")
    print(f"PhoBERT vs GPT-2 Delta:  {bs_vs_gpt2['mean_diff']:+.3f}% [95% CI: {bs_vs_gpt2['ci_lower']:+.3f}%, {bs_vs_gpt2['ci_upper']:+.3f}%], p={bs_vs_gpt2['p_value']:.5f}")
    print(f"Scoring Latency (Test):  {test_scoring_time:.2f}s ({test_scoring_time/len(test_samples)*1000:.1f} ms/sample)")
    print(f"Peak VRAM:               {peak_cuda / (1024**2):.1f} MiB" if peak_cuda else "Peak VRAM: N/A")
    print(f"Saved to:                {run_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
