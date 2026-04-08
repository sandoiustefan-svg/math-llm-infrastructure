from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def token_probability_confidence(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    answer: str,
    device: str = "cuda",
) -> dict:
    """
    Compute per-token probability of the answer tokens given the prompt.

    For each generated answer token, this computes softmax over the entire
    vocabulary and takes the probability assigned to the actual token chosen.
    This is a fine-grained confidence signal: high probability = model was
    certain about that token; low probability = model was uncertain.

    This is complementary to answer-level entropy:
    - answer_entropy: measures disagreement across N full generations
    - token_probability_confidence: measures how peaked the distribution is
      at each individual decoding step, without needing multiple passes

    Returns:
        token_probs      : list of per-token probabilities (one per answer token)
        mean_confidence  : geometric mean of token probs = exp(mean log prob)
                           Range [0, 1]; 1 = model was certain at every step
        perplexity       : exp(-mean log prob); lower = more confident
        min_token_prob   : probability of the least-certain token (weakest link)
    """
    full_text = prompt + answer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    n_prompt = prompt_ids.shape[1]
    n_answer = full_ids.shape[1] - n_prompt
    if n_answer <= 0:
        return {"token_probs": [], "mean_confidence": float("nan"),
                "perplexity": float("nan"), "min_token_prob": float("nan")}

    outputs = model(input_ids=full_ids[:, :-1])
    logits = outputs.logits  # (1, seq_len-1, vocab_size)

    # Positions in logits that predict answer tokens start at n_prompt - 1
    answer_logits = logits[0, n_prompt - 1: n_prompt - 1 + n_answer, :]
    answer_token_ids = full_ids[0, n_prompt:]  # (n_answer,)

    probs = F.softmax(answer_logits, dim=-1)  # (n_answer, vocab_size)
    token_probs = probs[range(n_answer), answer_token_ids].tolist()

    log_probs = [math.log(p) for p in token_probs if p > 0]
    mean_log_prob = sum(log_probs) / len(log_probs) if log_probs else float("-inf")
    mean_confidence = math.exp(mean_log_prob) if math.isfinite(mean_log_prob) else 0.0
    perplexity = math.exp(-mean_log_prob) if math.isfinite(mean_log_prob) else float("inf")
    min_token_prob = min(token_probs) if token_probs else float("nan")

    return {
        "token_probs": token_probs,
        "mean_confidence": round(mean_confidence, 6),
        "perplexity": round(perplexity, 4),
        "min_token_prob": round(min_token_prob, 6),
    }


def answer_entropy(answers: list[str]) -> float:
    """
    Predictive entropy over a discrete answer distribution.

    H = -sum_a p(a) * log2(p(a))

    Returns 0.0 for a unanimous set of answers, log2(N) for a perfectly
    uniform distribution over N distinct answers.
    """
    n = len(answers)
    if n == 0:
        return 0.0
    counts = Counter(answers)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def expected_calibration_error(results: list[dict], n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Bins predictions by confidence, computes |accuracy - confidence| per bin,
    and returns the weighted average. Lower is better; 0 = perfectly calibrated.

    Args:
        results: list of dicts with "confidence" (float) and "correct" (bool).
                 Items with correct=None are skipped.
        n_bins:  Number of equal-width confidence bins in [0, 1].
    """
    labeled = [r for r in results if r["correct"] is not None]
    if not labeled:
        return float("nan")

    bins = [[] for _ in range(n_bins)]
    for r in labeled:
        idx = min(int(r["confidence"] * n_bins), n_bins - 1)
        bins[idx].append(r)

    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc = sum(1 for r in b if r["correct"]) / len(b)
        conf = sum(r["confidence"] for r in b) / len(b)
        ece += (len(b) / len(labeled)) * abs(acc - conf)

    return ece


def plot_reliability_diagram(results: list[dict], output_path: str, n_bins: int = 10) -> None:
    """
    Reliability diagram: mean confidence vs accuracy per bin.

    A perfectly calibrated model lies on the diagonal. Bars above the diagonal
    indicate under-confidence; bars below indicate over-confidence.
    """
    labeled = [r for r in results if r["correct"] is not None]
    if not labeled:
        return

    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        b = [r for r in labeled if lo <= r["confidence"] < hi]
        if not b:
            continue
        bin_accs.append(sum(1 for r in b if r["correct"]) / len(b))
        bin_confs.append(sum(r["confidence"] for r in b) / len(b))
        bin_sizes.append(len(b))

    ece = expected_calibration_error(results, n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_confs, bin_accs, width=1 / n_bins, align="center", alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram  (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarise(results: list[dict]) -> dict:
    """
    Print-friendly summary of UQ evaluation results.

    Returns:
        dict with accuracy, mean_confidence, mean_entropy, ece,
        and coverage at 0.8/0.9 confidence thresholds.
    """
    labeled = [r for r in results if r["correct"] is not None]
    n_total = len(results)
    n_labeled = len(labeled)

    accuracy = sum(1 for r in labeled if r["correct"]) / n_labeled if n_labeled else float("nan")
    mean_conf = sum(r["confidence"] for r in results) / n_total if n_total else float("nan")
    mean_entropy = sum(r["entropy"] for r in results) / n_total if n_total else float("nan")
    ece = expected_calibration_error(results)

    coverage_80 = sum(1 for r in results if r["confidence"] >= 0.8) / n_total if n_total else float("nan")
    coverage_90 = sum(1 for r in results if r["confidence"] >= 0.9) / n_total if n_total else float("nan")

    token_confs = [r["token_mean_confidence"] for r in results if "token_mean_confidence" in r]
    token_ppls = [r["token_perplexity"] for r in results if "token_perplexity" in r]
    mean_token_conf = sum(token_confs) / len(token_confs) if token_confs else float("nan")
    mean_token_ppl = sum(token_ppls) / len(token_ppls) if token_ppls else float("nan")

    return {
        "n_problems": n_total,
        "n_labeled": n_labeled,
        "accuracy": round(accuracy, 4),
        # Answer-level UQ (agreement across passes/members)
        "mean_answer_confidence": round(mean_conf, 4),
        "mean_answer_entropy": round(mean_entropy, 4),
        "ece": round(ece, 4),
        "coverage_at_0.8_confidence": round(coverage_80, 4),
        "coverage_at_0.9_confidence": round(coverage_90, 4),
        # Token-level UQ (vocabulary probability at each decoding step)
        "mean_token_confidence": round(mean_token_conf, 6),
        "mean_token_perplexity": round(mean_token_ppl, 4),
    }
