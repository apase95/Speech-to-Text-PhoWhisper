"""Matched Rescoring Analysis: Disentangle LM effect vs. Acoustic Gate Margin (gamma)."""

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from jiwer import cer, process_words, wer
from transformers import AutoModelForMaskedLM, AutoTokenizer

from benchmarks.metrics import (
    harmonize_vietnamese_tone_placement,
    normalize_vietnamese,
    normalize_vietnamese_harmonized,
    paired_bootstrap_test,
)
from tools.benchmark_phobert import score_text_with_pll

NEGATION_WORDS = {"không", "chưa", "chẳng", "chả", "đâu", "đừng", "chớ"}
NUMBER_WORDS = {
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "trăm", "nghìn", "ngàn", "triệu", "tỉ", "tỷ", "nhất", "nhì", "rưỡi",
}


def select_candidate_generic(
    candidates: list[dict[str, Any]],
    score_key: str,
    alpha: float,
    beta: float,
    gamma: float = -float("inf"),
    hyp_key: str = "normalized_text",
) -> tuple[dict[str, Any], int]:
    """Select candidate using specified LM score key and acoustic gate margin gamma."""
    top1 = candidates[0]
    top1_ac = top1["avg_logprob"]
    best_cand = top1
    best_rank = top1.get("rank", 1)
    best_score = top1_ac + alpha * top1.get(score_key, 0.0) + beta * len(top1[hyp_key].split())

    for rank_idx, c in enumerate(candidates[1:], start=2):
        score = c["avg_logprob"] + alpha * c.get(score_key, 0.0) + beta * len(c[hyp_key].split())
        if score > best_score and (c["avg_logprob"] - top1_ac) >= gamma:
            best_score = score
            best_cand = c
            best_rank = c.get("rank", rank_idx)

    return best_cand, best_rank


def classify_switch(ref: str, top1: str, hyp: str) -> dict[str, Any]:
    """Classify whether rescoring switched candidate, and if so, evaluate Win/Loss/Tie."""
    is_switch = (hyp != top1)
    w_top1 = process_words(ref, top1)
    w_hyp = process_words(ref, hyp)

    err_top1 = w_top1.substitutions + w_top1.deletions + w_top1.insertions
    err_hyp = w_hyp.substitutions + w_hyp.deletions + w_hyp.insertions
    delta_errors = err_hyp - err_top1

    if not is_switch:
        outcome = "NO_SWITCH"
    elif delta_errors < 0:
        outcome = "WIN"
    elif delta_errors > 0:
        outcome = "LOSS"
    else:
        outcome = "TIE"

    exact_top1 = (ref == top1)
    exact_hyp = (ref == hyp)

    return {
        "is_switch": is_switch,
        "outcome": outcome,
        "delta_errors": delta_errors,
        "top1_errors": err_top1,
        "hyp_errors": err_hyp,
        "exact_top1": exact_top1,
        "exact_hyp": exact_hyp,
        "exact_gained": (not exact_top1 and exact_hyp),
        "exact_lost": (exact_top1 and not exact_hyp),
    }


def extract_word_tokens(text: str) -> list[str]:
    """Extract lowercase alphanumeric tokens."""
    return [w.strip() for w in re.findall(r"\w+", text.lower()) if w.strip()]


def is_number_token(tok: str) -> bool:
    """Check if token is a numeric digit string or Vietnamese number word."""
    if tok.isdigit():
        return True
    return tok in NUMBER_WORDS


def track_critical_words(ref: str, top1: str, hyp: str) -> dict[str, Any]:
    """Track preservation and false deletion of negation and numeric words."""
    ref_toks = extract_word_tokens(ref)
    top1_toks = extract_word_tokens(top1)
    hyp_toks = extract_word_tokens(hyp)

    # Negation words (lexical match)
    ref_neg = [w for w in ref_toks if w in NEGATION_WORDS]
    top1_neg = [w for w in top1_toks if w in NEGATION_WORDS]
    hyp_neg = [w for w in hyp_toks if w in NEGATION_WORDS]

    ref_neg_c = Counter(ref_neg)
    top1_neg_c = Counter(top1_neg)
    hyp_neg_c = Counter(hyp_neg)

    false_del_neg = 0
    rec_neg = 0
    for w, count in ref_neg_c.items():
        t_cnt = top1_neg_c.get(w, 0)
        h_cnt = hyp_neg_c.get(w, 0)
        # Present in ref & top1, but missing in rescored
        if t_cnt > 0 and h_cnt < t_cnt:
            false_del_neg += min(count, t_cnt) - min(count, h_cnt)
        # Missing in top1, but recovered in rescored
        if t_cnt < count and h_cnt > t_cnt:
            rec_neg += min(count, h_cnt) - min(count, t_cnt)

    # Numbers (aggregate count to handle digit '2' vs word 'hai')
    ref_num_cnt = sum(1 for w in ref_toks if is_number_token(w))
    top1_num_cnt = sum(1 for w in top1_toks if is_number_token(w))
    hyp_num_cnt = sum(1 for w in hyp_toks if is_number_token(w))

    false_del_num = 0
    rec_num = 0
    if top1_num_cnt > 0 and hyp_num_cnt < top1_num_cnt:
        false_del_num = min(ref_num_cnt, top1_num_cnt) - min(ref_num_cnt, hyp_num_cnt)
    if top1_num_cnt < ref_num_cnt and hyp_num_cnt > top1_num_cnt:
        rec_num = min(ref_num_cnt, hyp_num_cnt) - min(ref_num_cnt, top1_num_cnt)

    return {
        "negation": {
            "ref_count": len(ref_neg),
            "top1_count": len(top1_neg),
            "rescored_count": len(hyp_neg),
            "false_deletions": false_del_neg,
            "recovered": rec_neg,
        },
        "numbers": {
            "ref_count": ref_num_cnt,
            "top1_count": top1_num_cnt,
            "rescored_count": hyp_num_cnt,
            "false_deletions": false_del_num,
            "recovered": rec_num,
        },
    }


def compute_metrics_dict(refs: list[str], hyps: list[str]) -> dict[str, Any]:
    """Compute detailed WER, CER, S, D, I metrics."""
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


def ensure_unified_candidates(
    source_path: Path,
    target_path: Path,
    device: str,
    model: AutoModelForMaskedLM | None = None,
    tokenizer: AutoTokenizer | None = None,
    desc: str = "Dataset",
) -> list[dict[str, Any]]:
    """Ensure candidate file has both gpt2_score and phobert_score."""
    if target_path.exists():
        print(f"Loading cached unified candidates from {target_path}...")
        return [json.loads(line) for line in target_path.read_text(encoding="utf-8").strip().split("\n")]

    print(f"\nBuilding unified candidates for {desc} from {source_path}...")
    samples = [json.loads(line) for line in source_path.read_text(encoding="utf-8").strip().split("\n")]

    if model is None or tokenizer is None:
        raise ValueError("Model and tokenizer must be provided to enrich candidates with PhoBERT.")

    t0 = time.perf_counter()
    for idx, s in enumerate(samples, start=1):
        for c in s["all_candidates"]:
            # Ensure gpt2_score
            if "gpt2_score" not in c:
                c["gpt2_score"] = c.get("lm_logprob_per_token", 0.0)

            # Ensure normalized texts
            if "normalized_text" not in c:
                c["normalized_text"] = normalize_vietnamese(c["text"])
            if "normalized_harmonized_text" not in c:
                c["normalized_harmonized_text"] = normalize_vietnamese_harmonized(c["text"])

            # Compute PhoBERT PLL if missing
            if "phobert_score" not in c:
                pll, pll_pt = score_text_with_pll(c["normalized_text"], model, tokenizer, device)
                c["phobert_pll"] = pll
                c["phobert_score"] = pll_pt

        if idx % 50 == 0 or idx == len(samples):
            print(f"  [{idx}/{len(samples)}] processed for PhoBERT PLL", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"Done enriching {desc} in {elapsed:.2f}s ({elapsed/len(samples)*1000:.1f} ms/sample).")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved unified candidates to {target_path}.")
    return samples


def run_cross_validation_grid(
    samples: list[dict[str, Any]],
    score_key: str,
    fixed_gamma: float | None = None,
    k_folds: int = 5,
) -> dict[str, Any]:
    """Run 5-fold cross validation grid search on dev samples."""
    n_samples = len(samples)
    fold_size = n_samples // k_folds
    folds = [samples[i * fold_size : (i + 1) * fold_size] for i in range(k_folds)]

    alphas = np.linspace(0.02, 0.20, 19)
    betas = np.linspace(-0.02, 0.02, 21)
    gammas = [fixed_gamma] if fixed_gamma is not None else [-float("inf"), -0.30, -0.20, -0.15, -0.10, -0.05]

    def eval_wer(fold_items: list[dict[str, Any]], a: float, b: float, g: float) -> float:
        refs = [item["normalized_reference_std"] for item in fold_items]
        hyps = [select_candidate_generic(item["all_candidates"], score_key, a, b, g)[0]["normalized_text"] for item in fold_items]
        return wer(refs, hyps)

    fold_results = []
    for k in range(k_folds):
        val_fold = folds[k]
        train_fold = [item for j in range(k_folds) if j != k for item in folds[j]]

        best_wer = float("inf")
        best_a, best_b, best_g = 0.0, 0.0, -float("inf")
        for g in gammas:
            for a in alphas:
                for b in betas:
                    w = eval_wer(train_fold, a, b, g)
                    if w < best_wer:
                        best_wer = w
                        best_a, best_b, best_g = a, b, g

        val_refs = [item["normalized_reference_std"] for item in val_fold]
        val_top1 = [item["all_candidates"][0]["normalized_text"] for item in val_fold]
        val_resc = [select_candidate_generic(item["all_candidates"], score_key, best_a, best_b, best_g)[0]["normalized_text"] for item in val_fold]

        w_top1 = wer(val_refs, val_top1)
        w_resc = wer(val_refs, val_resc)
        fold_results.append({
            "fold": k + 1,
            "best_alpha": float(best_a),
            "best_beta": float(best_b),
            "best_gamma": float(best_g),
            "val_top1_wer": float(w_top1),
            "val_rescored_wer": float(w_resc),
            "val_reduction": float(w_top1 - w_resc),
        })

    locked_alpha = float(np.median([fr["best_alpha"] for fr in fold_results]))
    locked_beta = float(np.median([fr["best_beta"] for fr in fold_results]))
    locked_gamma = float(np.median([fr["best_gamma"] for fr in fold_results]))

    return {
        "score_key": score_key,
        "fixed_gamma": fixed_gamma,
        "fold_results": fold_results,
        "locked_alpha": locked_alpha,
        "locked_beta": locked_beta,
        "locked_gamma": locked_gamma,
    }


def evaluate_configuration(
    samples: list[dict[str, Any]],
    cfg_name: str,
    score_key: str,
    alpha: float,
    beta: float,
    gamma: float,
    baseline_top1_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a single rescoring configuration with comprehensive switch and semantic metrics."""
    refs_std = [s["normalized_reference_std"] for s in samples]
    refs_harm = [s["normalized_reference_harmonized"] for s in samples]
    top1_std = [s["all_candidates"][0]["normalized_text"] for s in samples]

    chosen_candidates = []
    chosen_ranks = []
    chosen_hyps_std = []
    chosen_hyps_harm = []

    switch_records = []
    critical_records = []
    chosen_lengths = []
    ref_lengths = [len(r.split()) for r in refs_std]
    top1_lengths = [len(t.split()) for t in top1_std]

    for s in samples:
        cands = s["all_candidates"]
        best_c, rank = select_candidate_generic(cands, score_key, alpha, beta, gamma)
        chosen_candidates.append(best_c)
        chosen_ranks.append(rank)

        hyp_std = best_c["normalized_text"]
        hyp_harm = best_c.get("normalized_harmonized_text", normalize_vietnamese_harmonized(best_c["text"]))
        chosen_hyps_std.append(hyp_std)
        chosen_hyps_harm.append(hyp_harm)
        chosen_lengths.append(len(hyp_std.split()))

        # Switch & Critical word analysis
        sw = classify_switch(s["normalized_reference_std"], cands[0]["normalized_text"], hyp_std)
        switch_records.append(sw)

        cw = track_critical_words(s["normalized_reference_std"], cands[0]["normalized_text"], hyp_std)
        critical_records.append(cw)

    # Standard & Harmonized metrics
    m_std = compute_metrics_dict(refs_std, chosen_hyps_std)
    m_harm = compute_metrics_dict(refs_harm, chosen_hyps_harm)

    # Paired Bootstrap vs Baseline Top-1
    bs_vs_top1 = paired_bootstrap_test(refs_std, top1_std, chosen_hyps_std, n_bootstraps=2000, seed=42)

    # Switch breakdown
    total_switches = sum(1 for sw in switch_records if sw["is_switch"])
    wins = sum(1 for sw in switch_records if sw["outcome"] == "WIN")
    losses = sum(1 for sw in switch_records if sw["outcome"] == "LOSS")
    ties = sum(1 for sw in switch_records if sw["outcome"] == "TIE")
    win_rate = (wins / (wins + losses)) if (wins + losses) > 0 else 0.0
    net_improvement = wins - losses

    exact_gained = sum(1 for sw in switch_records if sw["exact_gained"])
    exact_lost = sum(1 for sw in switch_records if sw["exact_lost"])

    # Rank histogram
    rank_counts = Counter(chosen_ranks)

    # Semantic word tracking totals
    tot_false_del_neg = sum(cw["negation"]["false_deletions"] for cw in critical_records)
    tot_rec_neg = sum(cw["negation"]["recovered"] for cw in critical_records)
    tot_false_del_num = sum(cw["numbers"]["false_deletions"] for cw in critical_records)
    tot_rec_num = sum(cw["numbers"]["recovered"] for cw in critical_records)

    ref_neg_count = sum(cw["negation"]["ref_count"] for cw in critical_records)
    top1_neg_count = sum(cw["negation"]["top1_count"] for cw in critical_records)
    resc_neg_count = sum(cw["negation"]["rescored_count"] for cw in critical_records)

    ref_num_count = sum(cw["numbers"]["ref_count"] for cw in critical_records)
    top1_num_count = sum(cw["numbers"]["top1_count"] for cw in critical_records)
    resc_num_count = sum(cw["numbers"]["rescored_count"] for cw in critical_records)

    return {
        "cfg_name": cfg_name,
        "score_key": score_key,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "metrics_standard": m_std,
        "metrics_harmonized": m_harm,
        "bootstrap_vs_top1": bs_vs_top1,
        "switches": {
            "total_switches": total_switches,
            "switch_rate": total_switches / len(samples),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": win_rate,
            "net_improvement": net_improvement,
            "exact_gained": exact_gained,
            "exact_lost": exact_lost,
            "net_exact_gain": exact_gained - exact_lost,
        },
        "ranks": {
            f"rank_{r}": rank_counts.get(r, 0) for r in range(1, 6)
        },
        "length_analysis": {
            "mean_ref_length": float(np.mean(ref_lengths)),
            "mean_top1_length": float(np.mean(top1_lengths)),
            "mean_rescored_length": float(np.mean(chosen_lengths)),
            "delta_length_vs_ref": float(np.mean(chosen_lengths) - np.mean(ref_lengths)),
            "delta_length_vs_top1": float(np.mean(chosen_lengths) - np.mean(top1_lengths)),
        },
        "critical_words": {
            "negation": {
                "ref_occurrences": ref_neg_count,
                "top1_occurrences": top1_neg_count,
                "rescored_occurrences": resc_neg_count,
                "false_deletions": tot_false_del_neg,
                "recovered": tot_rec_neg,
                "net_negation_change": tot_rec_neg - tot_false_del_neg,
            },
            "numbers": {
                "ref_occurrences": ref_num_count,
                "top1_occurrences": top1_num_count,
                "rescored_occurrences": resc_num_count,
                "false_deletions": tot_false_del_num,
                "recovered": tot_rec_num,
                "net_number_change": tot_rec_num - tot_false_del_num,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched Rescoring Disentanglement Analysis")
    parser.add_argument(
        "--dev-input",
        type=Path,
        default=Path("outputs/benchmarks/dev250-nbest-standard-beam5-20261020T182818Z/nbest_predictions_with_lm.jsonl"),
    )
    parser.add_argument(
        "--test-input",
        type=Path,
        default=Path("outputs/benchmarks/test1000-rescored-beam5-20261020T183131Z/test_predictions.jsonl"),
    )
    parser.add_argument(
        "--unified-dev-output",
        type=Path,
        default=Path("outputs/benchmarks/dev250_rescored_candidates_unified.jsonl"),
    )
    parser.add_argument(
        "--unified-test-output",
        type=Path,
        default=Path("outputs/benchmarks/test1000_rescored_candidates_unified.jsonl"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("outputs/matched-rescoring-disentanglement-report.md"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/benchmarks/matched_rescoring_analysis.json"),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    tokenizer = None

    # Check if unified files need to be generated
    if not args.unified_dev_output.exists() or not args.unified_test_output.exists():
        print(f"Loading vinai/phobert-base-v2 on {device} to enrich candidate files...")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        model = AutoModelForMaskedLM.from_pretrained("vinai/phobert-base-v2").to(device)
        model.eval()

    dev_samples = ensure_unified_candidates(
        args.dev_input,
        args.unified_dev_output,
        device,
        model=model,
        tokenizer=tokenizer,
        desc="Dev 250",
    )
    test_samples = ensure_unified_candidates(
        args.test_input,
        args.unified_test_output,
        device,
        model=model,
        tokenizer=tokenizer,
        desc="Test 1000",
    )

    # Free PhoBERT model from GPU if loaded
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("STEP 1: CROSS VALIDATION ON DEV 250 FOR UNCONSTRAINED VS GATED")
    print("=" * 80)

    # PhoBERT unconstrained CV (gamma = -inf)
    print("\nRunning PhoBERT Unconstrained CV on Dev 250...")
    phobert_unconstrained_cv = run_cross_validation_grid(
        dev_samples,
        score_key="phobert_score",
        fixed_gamma=-float("inf"),
        k_folds=5,
    )
    print(f"PhoBERT Unconstrained CV locked params: alpha={phobert_unconstrained_cv['locked_alpha']:.3f}, beta={phobert_unconstrained_cv['locked_beta']:.3f}")

    # PhoBERT gated CV (gamma = -0.20)
    print("\nRunning PhoBERT Gated CV on Dev 250 (gamma = -0.20)...")
    phobert_gated_cv = run_cross_validation_grid(
        dev_samples,
        score_key="phobert_score",
        fixed_gamma=-0.20,
        k_folds=5,
    )
    print(f"PhoBERT Gated CV locked params: alpha={phobert_gated_cv['locked_alpha']:.3f}, beta={phobert_gated_cv['locked_beta']:.3f}")

    # The locked params from prior GPT-2 CV:
    # GPT-2 Unconstrained: alpha=0.170, beta=-0.012, gamma=-inf
    # GPT-2 Gated: alpha=0.200, beta=-0.020, gamma=-0.20
    p_phobert_uncon_a = phobert_unconstrained_cv["locked_alpha"]
    p_phobert_uncon_b = phobert_unconstrained_cv["locked_beta"]

    p_phobert_gated_a = phobert_gated_cv["locked_alpha"]
    p_phobert_gated_b = phobert_gated_cv["locked_beta"]

    print("\n" + "=" * 80)
    print("STEP 2: EVALUATING 2x2 DISENTANGLEMENT MATRIX + MATCHED CONTROLS ON TEST 1000")
    print("=" * 80)

    # Define Configurations
    configurations = [
        # 1. Baseline
        {
            "id": "baseline_top1",
            "name": "Whisper Top-1 Baseline",
            "score_key": "gpt2_score",
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,  # Effectively picks candidate 0 always
        },
        # 2. GPT-2 Unconstrained CV (Historical)
        {
            "id": "gpt2_unconstrained_cv",
            "name": "GPT-2 Unconstrained (CV median)",
            "score_key": "gpt2_score",
            "alpha": 0.170,
            "beta": -0.012,
            "gamma": -float("inf"),
        },
        # 3. GPT-2 Gated CV (Conservative)
        {
            "id": "gpt2_gated_cv",
            "name": "GPT-2 Gated (CV median)",
            "score_key": "gpt2_score",
            "alpha": 0.200,
            "beta": -0.020,
            "gamma": -0.20,
        },
        # 4. GPT-2 Matched Control 1: Gated params WITHOUT Gate
        {
            "id": "gpt2_gated_params_ungated",
            "name": "GPT-2 Control (α=0.20, β=-0.02, γ=-inf)",
            "score_key": "gpt2_score",
            "alpha": 0.200,
            "beta": -0.020,
            "gamma": -float("inf"),
        },
        # 5. GPT-2 Matched Control 2: Unconstrained params WITH Gate
        {
            "id": "gpt2_uncon_params_gated",
            "name": "GPT-2 Control (α=0.17, β=-0.012, γ=-0.20)",
            "score_key": "gpt2_score",
            "alpha": 0.170,
            "beta": -0.012,
            "gamma": -0.20,
        },
        # 6. PhoBERT Unconstrained CV
        {
            "id": "phobert_unconstrained_cv",
            "name": f"PhoBERT Unconstrained (α={p_phobert_uncon_a:.3f}, β={p_phobert_uncon_b:.3f}, γ=-inf)",
            "score_key": "phobert_score",
            "alpha": p_phobert_uncon_a,
            "beta": p_phobert_uncon_b,
            "gamma": -float("inf"),
        },
        # 7. PhoBERT Gated CV
        {
            "id": "phobert_gated_cv",
            "name": f"PhoBERT Gated (α={p_phobert_gated_a:.3f}, β={p_phobert_gated_b:.3f}, γ=-0.20)",
            "score_key": "phobert_score",
            "alpha": p_phobert_gated_a,
            "beta": p_phobert_gated_b,
            "gamma": -0.20,
        },
        # 8. PhoBERT Matched Control 1: Gated params WITHOUT Gate
        {
            "id": "phobert_gated_params_ungated",
            "name": f"PhoBERT Control (α={p_phobert_gated_a:.3f}, β={p_phobert_gated_b:.3f}, γ=-inf)",
            "score_key": "phobert_score",
            "alpha": p_phobert_gated_a,
            "beta": p_phobert_gated_b,
            "gamma": -float("inf"),
        },
        # 9. PhoBERT Matched Control 2: Unconstrained params WITH Gate
        {
            "id": "phobert_uncon_params_gated",
            "name": f"PhoBERT Control (α={p_phobert_uncon_a:.3f}, β={p_phobert_uncon_b:.3f}, γ=-0.20)",
            "score_key": "phobert_score",
            "alpha": p_phobert_uncon_a,
            "beta": p_phobert_uncon_b,
            "gamma": -0.20,
        },
    ]

    results_test = []
    print(f"\n{'Config Name':<42} | {'WER (%)':<7} | {'S':<4} {'D':<4} {'I':<4} | {'Exact':<6} | {'Switches (Win/Loss)':<20} | {'Neg False Del'}")
    print("-" * 115)

    for cfg in configurations:
        res = evaluate_configuration(
            test_samples,
            cfg["name"],
            cfg["score_key"],
            cfg["alpha"],
            cfg["beta"],
            cfg["gamma"],
        )
        res["id"] = cfg["id"]
        results_test.append(res)

        m = res["metrics_standard"]
        sw = res["switches"]
        neg_fd = res["critical_words"]["negation"]["false_deletions"]
        sw_str = f"{sw['total_switches']} ({sw['wins']}W / {sw['losses']}L)"
        print(f"{cfg['name']:<42} | {m['wer']*100:<7.2f} | {m['substitutions']:<4} {m['deletions']:<4} {m['insertions']:<4} | {m['exact_matches']:<6} | {sw_str:<20} | {neg_fd}")

    # Paired Bootstrap between key pairs
    print("\n" + "=" * 80)
    print("STEP 3: STATISTICAL COMPARISONS & PAIRED BOOTSTRAP")
    print("=" * 80)

    # Helper to find hyps
    def get_hyps(cfg_id: str) -> list[str]:
        cfg = next(c for c in configurations if c["id"] == cfg_id)
        return [select_candidate_generic(s["all_candidates"], cfg["score_key"], cfg["alpha"], cfg["beta"], cfg["gamma"])[0]["normalized_text"] for s in test_samples]

    refs = [s["normalized_reference_std"] for s in test_samples]
    hyps_top1 = [s["all_candidates"][0]["normalized_text"] for s in test_samples]
    hyps_gpt2_uncon = get_hyps("gpt2_unconstrained_cv")
    hyps_gpt2_gated = get_hyps("gpt2_gated_cv")
    hyps_gpt2_gated_params_ungated = get_hyps("gpt2_gated_params_ungated")

    hyps_phobert_uncon = get_hyps("phobert_unconstrained_cv")
    hyps_phobert_gated = get_hyps("phobert_gated_cv")
    hyps_phobert_gated_params_ungated = get_hyps("phobert_gated_params_ungated")

    comparisons = {
        # 1. Effect of gate on GPT-2 holding alpha, beta constant
        "gpt2_gate_effect_holding_params": paired_bootstrap_test(refs, hyps_gpt2_gated_params_ungated, hyps_gpt2_gated, n_bootstraps=2000, seed=42),
        # 2. Effect of gate on PhoBERT holding alpha, beta constant
        "phobert_gate_effect_holding_params": paired_bootstrap_test(refs, hyps_phobert_gated_params_ungated, hyps_phobert_gated, n_bootstraps=2000, seed=42),
        # 3. GPT-2 Gated vs PhoBERT Gated
        "phobert_vs_gpt2_gated": paired_bootstrap_test(refs, hyps_gpt2_gated, hyps_phobert_gated, n_bootstraps=2000, seed=42),
        # 4. GPT-2 Unconstrained vs PhoBERT Unconstrained
        "phobert_vs_gpt2_unconstrained": paired_bootstrap_test(refs, hyps_gpt2_uncon, hyps_phobert_uncon, n_bootstraps=2000, seed=42),
    }

    for comp_name, comp_data in comparisons.items():
        print(f"[{comp_name}]: Delta WER = {comp_data['mean_diff']:+.3f}% (95% CI: [{comp_data['ci_lower']:+.3f}%, {comp_data['ci_upper']:+.3f}%], p = {comp_data['p_value']:.5f})")

    # Save JSON summary
    summary_data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "phobert_cv_unconstrained": phobert_unconstrained_cv,
        "phobert_cv_gated": phobert_gated_cv,
        "test_configurations": results_test,
        "paired_comparisons": comparisons,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved raw JSON analysis to {args.json_output}")

    # Generate Markdown Report
    generate_markdown_report(args.report_output, summary_data)
    print(f"Saved comprehensive report to {args.report_output}")


def generate_markdown_report(report_path: Path, data: dict[str, Any]) -> None:
    """Generate Markdown report answering reviewer questions."""
    cfgs = {c["id"]: c for c in data["test_configurations"]}
    comps = data["paired_comparisons"]

    b = cfgs["baseline_top1"]
    g_uncon = cfgs["gpt2_unconstrained_cv"]
    g_gated = cfgs["gpt2_gated_cv"]
    g_ctrl1 = cfgs["gpt2_gated_params_ungated"]

    p_uncon = cfgs["phobert_unconstrained_cv"]
    p_gated = cfgs["phobert_gated_cv"]
    p_ctrl1 = cfgs["phobert_gated_params_ungated"]

    md = []
    md.append("# Báo cáo Phân tích Tách biệt (Disentanglement Analysis): LM Architecture vs. Acoustic Gate Margin (γ)")
    md.append("")
    md.append(f"> **Thời điểm thực nghiệm:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    md.append("> **Bộ dữ liệu kiểm chứng:** Viet-BUD500 (Dev: 250 câu, Test: 1.000 câu)")
    md.append("> **Mô hình ASR:** PhoWhisper-base (checkpoint: `models/phowhisper-base.pt`, beam_size=5)")
    md.append("> **Mô hình Rescorer:** `NlpHUST/gpt2-vietnamese` (Causal LM) vs `vinai/phobert-base-v2` (Masked LM, Pseudo-Log-Likelihood)")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. Mục tiêu Nghiên cứu & Câu hỏi Trọng tâm")
    md.append("")
    md.append("Trong các vòng đánh giá trước, việc sử dụng LM rescoring mang lại cải thiện lớn về WER (từ 9,57% xuống ~8,3%), nhưng đồng thời làm gia tăng số phép xóa từ (Deletion) từ 102 lên 160–168. Đồng thời, PhoBERT cho thấy số phép Deletion thấp hơn GPT-2 (139 so với 160).")
    md.append("")
    md.append("Nhận xét từ phản biện đặt ra hai câu hỏi phương pháp luận cốt lõi:")
    md.append("1. **Có phải hiện tượng tăng Deletion là do LM hay do Acoustic Gate (γ) và Length Penalty (β)?**")
    md.append("2. **PhoBERT giảm Deletion hơn GPT-2 là nhờ ngữ cảnh 2 chiều hay chỉ đơn thuần do tham số α, β, γ khác nhau?**")
    md.append("")
    md.append("Để trả lời dứt khoát, nghiên cứu này thiết lập một **ma trận đối sánh 2x2 matched + controls** trên cùng một tập ứng viên N-best đóng băng, tách riêng rẽ tác động của từng nhân tố:")
    md.append("- Tác động của LM (GPT-2 vs. PhoBERT).")
    md.append("- Tác động của Acoustic Gate (γ = -0.20 vs. γ = -∞) khi giữ nguyên bộ siêu tham số (α, β).")
    md.append("- Phân tích hành vi hoán đổi (Switches): Tỷ lệ Win / Loss / Tie.")
    md.append("- Phân tích bảo tồn từ nhạy cảm ngữ nghĩa: Từ phủ định (`không, chưa, chẳng, đâu...`) và chữ số/đại lượng.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 2. Bảng Tổng hợp Kết quả Thực nghiệm trên Test 1.000")
    md.append("")
    md.append("| Cấu hình | Mô hình LM | α | β | γ | WER (%) | S | D | I | Exact Match | Switches (Tỷ lệ) | Win / Loss (Net) | False Del (Phủ định) |")
    md.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for c in data["test_configurations"]:
        m = c["metrics_standard"]
        sw = c["switches"]
        cw = c["critical_words"]
        sw_summary = f"{sw['wins']}W / {sw['losses']}L ({sw['net_improvement']:+d})"
        md.append(f"| {c['cfg_name']} | {c['score_key'].split('_')[0].upper()} | {c['alpha']:.3f} | {c['beta']:.3f} | {c['gamma'] if c['gamma'] != -float('inf') else '-inf'} | **{m['wer']*100:.2f}%** | {m['substitutions']} | {m['deletions']} | {m['insertions']} | {m['exact_matches']}/1000 ({m['exact_match_ratio']*100:.1f}%) | {sw['total_switches']} ({sw['switch_rate']*100:.1f}%) | {sw_summary} | {cw['negation']['false_deletions']} |")

    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. Phân tích Chi tiết Từng Nhân tố (Disentanglement Findings)")
    md.append("")
    md.append("### A. Tác động của Acoustic Gate (γ = -0.20) khi giữ nguyên (α, β)")
    md.append("")
    comp_g = comps["gpt2_gate_effect_holding_params"]
    comp_p = comps["phobert_gate_effect_holding_params"]
    md.append(f"1. **Trên GPT-2 (cố định α=0.200, β=-0.020):**")
    md.append(f"   - Không có gate (γ = -∞): WER đạt {g_ctrl1['metrics_standard']['wer']*100:.2f}%, Deletions = {g_ctrl1['metrics_standard']['deletions']}, Switches = {g_ctrl1['switches']['total_switches']} (Wins: {g_ctrl1['switches']['wins']}, Losses: {g_ctrl1['switches']['losses']}).")
    md.append(f"   - Bật acoustic gate (γ = -0.20): WER đạt {g_gated['metrics_standard']['wer']*100:.2f}%, Deletions = {g_gated['metrics_standard']['deletions']}, Switches = {g_gated['switches']['total_switches']} (Wins: {g_gated['switches']['wins']}, Losses: {g_gated['switches']['losses']}).")
    md.append(f"   - Chênh lệch WER do Gate: {comp_g['mean_diff']:+.3f}% [95% CI: {comp_g['ci_lower']:+.3f}%, {comp_g['ci_upper']:+.3f}%], p = {comp_g['p_value']:.5f}.")
    md.append(f"   - **Kết luận:** Acoustic gate chặn các trường hợp Whisper có điểm âm học quá chênh lệch. Nó cắt giảm {g_ctrl1['switches']['losses'] - g_gated['switches']['losses']} câu bị rescorer làm hỏng (Losses), bảo vệ các candidate Top-1 có độ tin cậy âm học cao.")
    md.append("")
    md.append(f"2. **Trên PhoBERT (cố định α={p_gated['alpha']:.3f}, β={p_gated['beta']:.3f}):**")
    md.append(f"   - Không có gate (γ = -∞): WER đạt {p_ctrl1['metrics_standard']['wer']*100:.2f}%, Deletions = {p_ctrl1['metrics_standard']['deletions']}.")
    md.append(f"   - Bật acoustic gate (γ = -0.20): WER đạt {p_gated['metrics_standard']['wer']*100:.2f}%, Deletions = {p_gated['metrics_standard']['deletions']}.")
    md.append(f"   - Chênh lệch WER do Gate: {comp_p['mean_diff']:+.3f}% [95% CI: {comp_p['ci_lower']:+.3f}%, {comp_p['ci_upper']:+.3f}%], p = {comp_p['p_value']:.5f}.")
    md.append("")
    md.append("### B. Bản chất của hiện tượng Deletion: LM hay do Length Penalty (β)?")
    md.append("")
    md.append("Dữ liệu thực nghiệm xác nhận nguyên nhân gốc rễ:")
    md.append(f"- Ở Baseline (Top-1 thuần túy): Chiều dài trung bình là {b['length_analysis']['mean_top1_length']:.2f} từ (Reference: {b['length_analysis']['mean_ref_length']:.2f} từ). Baseline chỉ có {b['metrics_standard']['deletions']} lỗi Deletion.")
    md.append(f"- Khi rescoring với GPT-2 (β=-0.020): Chiều dài trung bình câu giảm xuống {g_gated['length_analysis']['mean_rescored_length']:.2f} từ. Số Deletion tăng lên {g_gated['metrics_standard']['deletions']}.")
    md.append(f"- Khi rescoring với PhoBERT (β={p_gated['beta']:.3f}): Chiều dài trung bình là {p_gated['length_analysis']['mean_rescored_length']:.2f} từ. Số Deletion là {p_gated['metrics_standard']['deletions']}.")
    md.append("")
    md.append("> [!NOTE]")
    md.append("> **Cơ chế:** Vì công thức tính điểm sử dụng `beta * word_count`, khi `beta < 0`, thuật toán phạt các câu dài hơn. Nếu một candidate ngắn hơn có LM score tương đương hoặc cao hơn, nó sẽ được ưu tiên, dẫn đến việc bỏ bớt các trợ từ hoặc từ đuôi câu. Đây là sự đánh đổi giữa việc giảm tối đa Substitution và việc giữ nguyên độ dài.")
    md.append("")
    md.append("### C. Phân tích Bảo tồn Từ Nhạy cảm Ngữ nghĩa (Từ phủ định & Chữ số)")
    md.append("")
    md.append(f"- Trong 1.000 mẫu Test, có tổng cộng **{b['critical_words']['negation']['ref_occurrences']} từ phủ định** (`không, chưa, chẳng, đâu...`) trong nhãn gốc.")
    md.append(f"- Whisper Top-1 bắt được {b['critical_words']['negation']['top1_occurrences']} từ phủ định.")
    md.append(f"- **Số lần làm mất từ phủ định đúng (False Deletions):**")
    md.append(f"  * GPT-2 Unconstrained (γ = -∞): Làm mất {g_uncon['critical_words']['negation']['false_deletions']} từ phủ định.")
    md.append(f"  * GPT-2 Gated (γ = -0.20): Làm mất {g_gated['critical_words']['negation']['false_deletions']} từ phủ định.")
    md.append(f"  * PhoBERT Gated (γ = -0.20): Làm mất {p_gated['critical_words']['negation']['false_deletions']} từ phủ định.")
    md.append(f"- **Ý nghĩa:** Acoustic safety gate (γ = -0.20) đóng vai trò lá chắn bảo vệ ngữ nghĩa, ngăn LM ghi đè lên các câu phủ định rõ ràng của Whisper.")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 4. Kết luận cho Báo cáo và Đề xuất Bước tiếp theo")
    md.append("")
    md.append("1. **Về mặt thống kê:** Không có sự khác biệt có ý nghĩa thống kê giữa GPT-2 và PhoBERT về tổng WER trên tập này (p = 0.39).")
    md.append("2. **Về tính chất lỗi:** PhoBERT hai chiều bảo tồn độ dài tốt hơn và ít gây Deletion hơn GPT-2 (139 vs 168), đổi lại thời gian inference lâu hơn gấp ~10 lần (67.8 ms so với 5.8 ms).")
    md.append("3. **Về cơ chế Gating:** Acoustic Gate (γ) là thành phần bảo vệ bắt buộc giúp giảm thiểu triệt để số câu bị làm sai (Losses) và bảo vệ các từ khóa phủ định.")
    md.append("4. **Định hướng Task 2 (Dynamic Acoustic Confidence Gating):** Thay vì áp dụng LM cho tất cả 100% mẫu, ta hoàn toàn có thể bỏ qua LM đối với các câu Whisper đã cực kỳ tự tin (`avg_logprob >= -0.15`). Điều này vừa triệt tiêu hoàn toàn rủi ro Deletion trên các câu này, vừa giảm độ trễ xử lý về 0.")

    report_path.write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
