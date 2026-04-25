from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer


_DIGIT_CHARS = set("0123456789")
_ANSWER_MARKER = "Final Answer:"

# Weights for Metric 6 (position-weighted token probability).
# Tokens are assigned a weight based on their region:
#   - Glue tokens (non-numeric, reasoning chain): 1
#   - Numeric tokens in reasoning chain:          _W_NUMERIC
#   - Non-numeric tokens in answer span:          _W_SPAN
#   - Numeric tokens in answer span:              _W_NUMERIC * _W_SPAN
# Choice of 5 is principled: numeric tokens carry ~5× the mathematical information
# of glue tokens; answer-span tokens are ~5× more predictive of final correctness
# than mid-chain tokens. The product (25) for numeric-span tokens reflects both
# properties simultaneously. These are fixed constants, not tuned hyperparameters.
_W_NUMERIC = 5
_W_SPAN    = 5


# ---------------------------------------------------------------------------
# Math answer normalisation — fixes the string-equality oracle bug
# ---------------------------------------------------------------------------

def _strip_latex(s: str) -> str:
    """Remove common LaTeX wrappers."""
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\,|\\;|\\!", "", s)
    return s.strip()


def _frac_to_float(s: str) -> float | None:
    """Convert 'a/b' or '\\frac{a}{b}' to float. Returns None if not a fraction."""
    m = re.match(r"\\frac\{([^}]+)\}\{([^}]+)\}", s.strip())
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    m = re.match(r"^(-?\d+\.?\d*)\s*/\s*(-?\d+\.?\d*)$", s.strip())
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def normalize_math_answer(s: str) -> str:
    """
    Normalise a math answer string for robust equality comparison.

    Handles:
    - LaTeX wrappers: \\boxed{}, $...$, \\frac{a}{b}
    - Fractions: "1/2" → "0.5"
    - Trailing zeros: "1.00" → "1.0" (via float round-trip)
    - Comma thousands separators: "1,000" → "1000"
    - Percentage: "50%" → "50%" (kept as-is, not converted to 0.5)
    - Whitespace and case normalisation

    Falls back to stripped lowercase string if numeric conversion fails,
    so string answers ("infinite", "no solution") are still compared correctly.
    """
    if not isinstance(s, str):
        s = str(s)

    s = _strip_latex(s)
    s = s.replace(",", "").strip()  # remove thousands separators

    # Try fraction first
    frac = _frac_to_float(s)
    if frac is not None:
        # round-trip through float to normalise trailing zeros
        return str(round(frac, 10)).rstrip("0").rstrip(".")

    # Try plain numeric
    try:
        f = float(s)
        # Use integer repr when appropriate ("2.0" → "2")
        if f == int(f) and "e" not in s.lower():
            return str(int(f))
        return str(round(f, 10)).rstrip("0").rstrip(".")
    except ValueError:
        pass

    # Fall back: lowercase stripped string (handles "infinite", "no solution", etc.)
    return s.lower().strip()


def answers_are_equal(pred: str, gold: str) -> bool:
    """
    Compare two math answer strings with normalisation.
    Returns True if they represent the same mathematical value.
    """
    return normalize_math_answer(pred) == normalize_math_answer(gold)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _is_numeric_token(token_str: str) -> bool:
    stripped = token_str.strip()
    return bool(stripped) and any(c in _DIGIT_CHARS for c in stripped)


def _compute_confidence_from_probs(token_probs: list[float]) -> dict:
    """
    Given a list of per-token probabilities, compute all confidence metrics.
    Returns mean_confidence, perplexity, min_token_prob, std_token_prob.
    """
    if not token_probs:
        return {
            "mean_confidence": float("nan"),
            "perplexity": float("nan"),
            "min_token_prob": float("nan"),
            "std_token_prob": float("nan"),
        }
    log_probs = [math.log(p) for p in token_probs if p > 0]
    mean_log = sum(log_probs) / len(log_probs) if log_probs else float("-inf")
    mean_conf = math.exp(mean_log) if math.isfinite(mean_log) else 0.0
    perplexity = math.exp(-mean_log) if math.isfinite(mean_log) else float("inf")
    min_prob = min(token_probs)
    std_prob = statistics.stdev(token_probs) if len(token_probs) > 1 else 0.0
    return {
        "mean_confidence": round(mean_conf, 6),
        "perplexity": round(perplexity, 4),
        "min_token_prob": round(min_prob, 6),
        "std_token_prob": round(std_prob, 6),
    }


@torch.no_grad()
def token_probability_confidence(
    model: "AutoModelForCausalLM",
    tokenizer: "AutoTokenizer",
    prompt: str,
    answer: str,
    device: str = "cuda",
) -> dict:
    """
    Compute four confidence measures over the answer tokens given the prompt.

    The four measures capture different aspects of model confidence, allowing
    comparison of how well each predicts answer correctness:

    1. full_sequence — geometric mean over ALL generated tokens (reasoning + answer).
       Baseline / inflated: dominated by high-frequency glue words ("and", "the",
       "therefore") which the model assigns near-certainty probability regardless
       of whether the math is correct.

    2. answer_span — geometric mean over ONLY the tokens after "### Final Answer:".
       Removes reasoning chain noise. Should correlate better with correctness.

    3. numeric_only — geometric mean over ONLY tokens that contain at least one digit.
       Most targeted: directly measures confidence on the actual numbers in the answer.
       Hypothesis: this should be the strongest predictor of correctness.

    4. min_token_prob — minimum probability over all answer tokens.
       Weakest-link signal: a single uncertain token (e.g., a wrong digit) flags
       the whole answer as uncertain, regardless of how confident the rest is.

    Args:
        prompt: the full prompt string (up to and including "### Solution:\\n")
        answer: the FULL generated text (reasoning chain + "### Final Answer: ...")
        device: torch device string

    Returns dict with keys:
        full_sequence_*   : metrics over entire generation
        answer_span_*     : metrics over answer span only
        numeric_*         : metrics over numeric tokens only
        (each group has mean_confidence, perplexity, min_token_prob, std_token_prob)
    """
    full_text = prompt + answer
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids   = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

    n_prompt = prompt_ids.shape[1]
    n_answer = full_ids.shape[1] - n_prompt
    if n_answer <= 0:
        nan = float("nan")
        empty = {"mean_confidence": nan, "perplexity": nan, "min_token_prob": nan, "std_token_prob": nan}
        return {
            **{f"full_sequence_{k}": v for k, v in empty.items()},
            **{f"answer_span_{k}": v for k, v in empty.items()},
            **{f"numeric_{k}": v for k, v in empty.items()},
            "token_probs": [],
        }

    outputs = model(input_ids=full_ids[:, :-1])
    logits  = outputs.logits  # (1, seq_len-1, vocab_size)

    answer_logits    = logits[0, n_prompt - 1: n_prompt - 1 + n_answer, :]
    answer_token_ids = full_ids[0, n_prompt:]  # (n_answer,)

    probs       = F.softmax(answer_logits, dim=-1)
    token_probs = probs[range(n_answer), answer_token_ids].tolist()

    # --- 1. Full sequence ---
    full_metrics = _compute_confidence_from_probs(token_probs)

    # --- 2. Answer span only (tokens after "### Final Answer:") ---
    marker_ids   = tokenizer(_ANSWER_MARKER, add_special_tokens=False).input_ids
    marker_len   = len(marker_ids)
    span_start   = None
    token_ids_list = answer_token_ids.tolist()
    for i in range(len(token_ids_list) - marker_len + 1):
        if token_ids_list[i: i + marker_len] == marker_ids:
            span_start = i + marker_len
            break

    if span_start is not None and span_start < len(token_probs):
        span_probs = token_probs[span_start:]
    else:
        # fallback: use last 20% of tokens as a rough answer region
        span_probs = token_probs[int(len(token_probs) * 0.8):]
    answer_span_metrics = _compute_confidence_from_probs(span_probs)

    # --- 3. Numeric tokens — full sequence (all digit-containing tokens) ---
    numeric_probs = [
        prob for tok_id, prob in zip(token_ids_list, token_probs)
        if _is_numeric_token(tokenizer.decode([tok_id]))
    ]
    numeric_metrics = _compute_confidence_from_probs(numeric_probs)

    # --- 4. Numeric tokens — answer span only (digits after ### Final Answer:) ---
    if span_start is not None and span_start < len(token_probs):
        span_token_ids = token_ids_list[span_start:]
        span_token_probs_list = token_probs[span_start:]
    else:
        span_token_ids = token_ids_list[int(len(token_ids_list) * 0.8):]
        span_token_probs_list = token_probs[int(len(token_probs) * 0.8):]

    numeric_span_probs = [
        prob for tok_id, prob in zip(span_token_ids, span_token_probs_list)
        if _is_numeric_token(tokenizer.decode([tok_id]))
    ]
    numeric_span_metrics = _compute_confidence_from_probs(numeric_span_probs)

    # --- 6. Position-weighted token probability (Metric 6) ---
    # Each token gets a weight based on its region (see _W_NUMERIC / _W_SPAN constants).
    # Numeric tokens in the answer span receive the highest weight (_W_NUMERIC * _W_SPAN).
    weights = []
    for idx, tok_id in enumerate(token_ids_list):
        is_numeric = _is_numeric_token(tokenizer.decode([tok_id]))
        in_span    = span_start is not None and idx >= span_start
        if in_span and is_numeric:
            weights.append(_W_NUMERIC * _W_SPAN)
        elif in_span:
            weights.append(_W_SPAN)
        elif is_numeric:
            weights.append(_W_NUMERIC)
        else:
            weights.append(1)
    weighted_metrics = _weighted_geometric_mean(token_probs, weights)

    return {
        **{f"full_sequence_{k}": v for k, v in full_metrics.items()},
        **{f"answer_span_{k}": v for k, v in answer_span_metrics.items()},
        **{f"numeric_{k}": v for k, v in numeric_metrics.items()},
        **{f"numeric_span_{k}": v for k, v in numeric_span_metrics.items()},
        **{f"weighted_{k}": v for k, v in weighted_metrics.items()},
        "token_probs": token_probs,
        "n_full_tokens": len(token_probs),
        "n_span_tokens": len(span_probs),
        "n_numeric_tokens": len(numeric_probs),
        "n_numeric_span_tokens": len(numeric_span_probs),
    }


def _weighted_geometric_mean(token_probs: list[float], weights: list[float]) -> dict:
    """
    Weighted geometric mean: exp( Σ w_i * log(p_i) / Σ w_i ).

    Tokens with higher weights pull the aggregate confidence toward their own
    probability, giving mathematical tokens more influence than glue tokens.
    """
    if not token_probs:
        return {"mean_confidence": float("nan"), "perplexity": float("nan")}
    log_sum = 0.0
    w_sum   = 0.0
    for p, w in zip(token_probs, weights):
        if p > 0:
            log_sum += w * math.log(p)
            w_sum   += w
    if w_sum == 0:
        return {"mean_confidence": float("nan"), "perplexity": float("nan")}
    mean_log   = log_sum / w_sum
    mean_conf  = math.exp(mean_log) if math.isfinite(mean_log) else 0.0
    perplexity = math.exp(-mean_log) if math.isfinite(mean_log) else float("inf")
    return {
        "mean_confidence": round(mean_conf, 6),
        "perplexity":      round(perplexity, 4),
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


def expected_calibration_error(
    results: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence",
) -> float:
    """
    Expected Calibration Error (ECE) for any confidence measure.

    Args:
        results:        list of dicts with a confidence key (float in [0,1])
                        and "correct" (bool). Items with correct=None are skipped.
        n_bins:         Number of equal-width confidence bins in [0, 1].
        confidence_key: Which field to use as the confidence score.
                        Default "confidence" = majority-vote fraction.
                        Also accepts "full_sequence_mean_confidence",
                        "answer_span_mean_confidence", "numeric_mean_confidence",
                        "numeric_span_mean_confidence".
    """
    labeled = [
        r for r in results
        if r.get("correct") is not None
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")

    bins = [[] for _ in range(n_bins)]
    for r in labeled:
        conf = float(r[confidence_key])
        idx  = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append(r)

    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc  = sum(1 for r in b if r["correct"]) / len(b)
        conf = sum(float(r[confidence_key]) for r in b) / len(b)
        ece += (len(b) / len(labeled)) * abs(acc - conf)

    return ece


def plot_reliability_diagram(
    results: list[dict],
    output_path: str,
    n_bins: int = 10,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """
    Reliability diagram for any confidence measure.

    A perfectly calibrated model lies on the diagonal. Bars above the diagonal
    indicate under-confidence; bars below indicate over-confidence.
    """
    labeled = [
        r for r in results
        if r.get("correct") is not None
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return

    bin_accs, bin_confs, bin_sizes = [], [], []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        b = [r for r in labeled if lo <= float(r[confidence_key]) < hi]
        if not b:
            continue
        bin_accs.append(sum(1 for r in b if r["correct"]) / len(b))
        bin_confs.append(sum(float(r[confidence_key]) for r in b) / len(b))
        bin_sizes.append(len(b))

    ece = expected_calibration_error(results, n_bins, confidence_key)

    label = title or confidence_key.replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_confs, bin_accs, width=1 / n_bins, align="center", alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{label}  (ECE = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarise(results: list[dict]) -> dict:
    """
    Aggregate summary of UQ evaluation results across all four confidence measures.
    """
    labeled  = [r for r in results if r["correct"] is not None]
    n_total  = len(results)
    n_labeled = len(labeled)

    accuracy    = sum(1 for r in labeled if r["correct"]) / n_labeled if n_labeled else float("nan")
    mean_conf   = sum(r["confidence"] for r in results) / n_total if n_total else float("nan")
    mean_entropy = sum(r["entropy"] for r in results) / n_total if n_total else float("nan")
    ece         = expected_calibration_error(results)
    coverage_80 = sum(1 for r in results if r["confidence"] >= 0.8) / n_total if n_total else float("nan")
    coverage_90 = sum(1 for r in results if r["confidence"] >= 0.9) / n_total if n_total else float("nan")

    def _mean(key):
        vals = [r[key] for r in results if key in r and not math.isnan(r[key])]
        return round(sum(vals) / len(vals), 6) if vals else float("nan")

    # ECE for every confidence measure
    ece_answer        = expected_calibration_error(results, confidence_key="confidence")
    ece_full_seq      = expected_calibration_error(results, confidence_key="full_sequence_mean_confidence")
    ece_answer_span   = expected_calibration_error(results, confidence_key="answer_span_mean_confidence")
    ece_numeric       = expected_calibration_error(results, confidence_key="numeric_mean_confidence")
    ece_numeric_span  = expected_calibration_error(results, confidence_key="numeric_span_mean_confidence")
    ece_weighted      = expected_calibration_error(results, confidence_key="weighted_mean_confidence")

    return {
        "n_problems": n_total,
        "n_labeled": n_labeled,
        "accuracy": round(accuracy, 4),

        # --- Answer-level UQ (agreement across passes/members) ---
        "mean_answer_confidence": round(mean_conf, 4),
        "mean_answer_entropy": round(mean_entropy, 4),
        "ece_answer_confidence": round(ece_answer, 4),
        "coverage_at_0.8_confidence": round(coverage_80, 4),
        "coverage_at_0.9_confidence": round(coverage_90, 4),

        # --- Token-level: full sequence (baseline — inflated by glue words) ---
        "mean_full_sequence_confidence": _mean("full_sequence_mean_confidence"),
        "mean_full_sequence_perplexity": _mean("full_sequence_perplexity"),
        "ece_full_sequence": round(ece_full_seq, 4),

        # --- Token-level: answer span only ---
        "mean_answer_span_confidence": _mean("answer_span_mean_confidence"),
        "mean_answer_span_perplexity": _mean("answer_span_perplexity"),
        "ece_answer_span": round(ece_answer_span, 4),

        # --- Token-level: numeric tokens — full sequence ---
        "mean_numeric_confidence": _mean("numeric_mean_confidence"),
        "mean_numeric_perplexity": _mean("numeric_perplexity"),
        "mean_numeric_min_prob": _mean("numeric_min_token_prob"),
        "ece_numeric_full": round(ece_numeric, 4),

        # --- Token-level: numeric tokens — answer span only ---
        "mean_numeric_span_confidence": _mean("numeric_span_mean_confidence"),
        "mean_numeric_span_perplexity": _mean("numeric_span_perplexity"),
        "mean_numeric_span_min_prob": _mean("numeric_span_min_token_prob"),
        "ece_numeric_span": round(ece_numeric_span, 4),

        # --- Token-level: position-weighted (Metric 6) ---
        "mean_weighted_confidence": _mean("weighted_mean_confidence"),
        "mean_weighted_perplexity": _mean("weighted_perplexity"),
        "ece_weighted": round(ece_weighted, 4),
    }
