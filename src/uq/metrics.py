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
_RANK_INT   = {"low": 0, "medium": 1, "high": 2}   # ordinal encoding for similarity ranks
_JUDGE_SCORE = {"good": 1.0, "medium": 0.5, "bad": 0.0}  # numeric encoding for judge labels

# Weights for the weighted geometric mean confidence.
# All prompts use <<expr=result>> annotations, so arithmetic-critical tokens are
# explicitly identifiable rather than heuristically detected:
#   - Result tokens inside <<expr=result>>: _W_ARITH_RESULT
#   - Tokens in Final Answer span:          _W_FINAL_ANSWER
#   - All other tokens:                     1
_W_ARITH_RESULT  = 10
_W_FINAL_ANSWER  = 25

# Matches <<expr=result>> annotations; group 1 = expr, group 2 = result portion.
_ARITH_ANNOTATION = re.compile(r'<<([^=\n]+)=([^>\n]+)>>')
# Allowed characters in a safe arithmetic expression.
_SAFE_EXPR_RE     = re.compile(r'[^0-9+\-*/().\s]')


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

    2. answer_span — geometric mean over ONLY the tokens after "Final Answer:".
       Removes reasoning chain noise. Should correlate better with correctness.

    3. numeric_only — geometric mean over ONLY tokens that contain at least one digit.
       Most targeted: directly measures confidence on the actual numbers in the answer.
       Hypothesis: this should be the strongest predictor of correctness.

    4. min_token_prob — minimum probability over all answer tokens.
       Weakest-link signal: a single uncertain token (e.g., a wrong digit) flags
       the whole answer as uncertain, regardless of how confident the rest is.

    Args:
        prompt: the full prompt string from apply_chat_template (up to assistant header)
        answer: the FULL generated text (reasoning chain + "Final Answer: ...")
        device: torch device string

    Returns dict with keys:
        full_sequence_*   : metrics over entire generation
        answer_span_*     : metrics over answer span only
        numeric_*         : metrics over numeric tokens only
        (each group has mean_confidence, perplexity, min_token_prob, std_token_prob)
    """
    full_text = prompt + answer
    # add_special_tokens=False: prompt already contains <|begin_of_text|>
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids   = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

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

    # --- 2. Answer span only (tokens after "Final Answer:") ---
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

    # --- 4. Numeric tokens — answer span only (digits after "Final Answer:") ---
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

    # Weighted geometric mean: result tokens inside <<expr=result>> get _W_ARITH_RESULT×,
    # tokens in the Final Answer span get _W_FINAL_ANSWER×, all others get 1×.
    # Build a char→token index map so we can identify result-span tokens by regex.
    char_to_token: list[int] = []
    reconstructed = ""
    for idx, tok_id in enumerate(token_ids_list):
        tok_str = tokenizer.decode([tok_id])
        char_to_token.extend([idx] * len(tok_str))
        reconstructed += tok_str

    arith_token_set: set[int] = set()
    for m in _ARITH_ANNOTATION.finditer(reconstructed):
        for ci in range(m.start(2), min(m.end(2), len(char_to_token))):
            arith_token_set.add(char_to_token[ci])

    weights = []
    for idx in range(len(token_ids_list)):
        in_span  = span_start is not None and idx >= span_start
        in_arith = idx in arith_token_set
        if in_span:
            weights.append(_W_FINAL_ANSWER)
        elif in_arith:
            weights.append(_W_ARITH_RESULT)
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


def auroc(results: list[dict], confidence_key: str = "confidence") -> float:
    """Area Under the ROC Curve — how well confidence discriminates correct from incorrect."""
    labeled = [
        (float(r[confidence_key]), int(bool(r["correct"])))
        for r in results
        if r.get("correct") is not None
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")
    n_pos = sum(c for _, c in labeled)
    n_neg = len(labeled) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    labeled.sort(key=lambda x: -x[0])
    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auc = 0.0
    for _, correct in labeled:
        if correct:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tpr, prev_fpr = tpr, fpr
    return round(auc, 4)


def _spearman_from_lists(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation between two equal-length lists."""
    n = len(xs)
    if n < 2:
        return float("nan")

    def _ranks(lst: list) -> list[float]:
        order = sorted(range(n), key=lambda i: lst[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and lst[order[j + 1]] == lst[order[j]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((r - mx) ** 2 for r in rx) *
        sum((r - my) ** 2 for r in ry)
    )
    return round(num / den, 4) if den else float("nan")


def spearman_correlation(results: list[dict], confidence_key: str = "confidence") -> float:
    """Spearman rank correlation between confidence and binary correctness."""
    labeled = [
        (float(r[confidence_key]), int(bool(r["correct"])))
        for r in results
        if r.get("correct") is not None
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if len(labeled) < 2:
        return float("nan")
    return _spearman_from_lists([x[0] for x in labeled], [x[1] for x in labeled])


def overconfidence_analysis(
    results: list[dict],
    confidence_key: str = "confidence",
    threshold: float = 0.8,
) -> dict:
    """
    Quadrant analysis: (high/low confidence) × (correct/wrong).
    The high-confidence + wrong cell is the epistemic uncertainty gap —
    cases where the model has no internal signal that it is wrong.
    """
    labeled = [
        r for r in results
        if r.get("correct") is not None
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return {}
    hc_right = sum(1 for r in labeled if float(r[confidence_key]) >= threshold and r["correct"])
    hc_wrong = sum(1 for r in labeled if float(r[confidence_key]) >= threshold and not r["correct"])
    lc_right = sum(1 for r in labeled if float(r[confidence_key]) <  threshold and r["correct"])
    lc_wrong = sum(1 for r in labeled if float(r[confidence_key]) <  threshold and not r["correct"])
    n_high = hc_right + hc_wrong
    return {
        "high_conf_correct":   hc_right,
        "high_conf_wrong":     hc_wrong,
        "low_conf_correct":    lc_right,
        "low_conf_wrong":      lc_wrong,
        "overconfidence_rate": round(hc_wrong / n_high, 4) if n_high else float("nan"),
    }


def auroc_sim(results: list[dict], confidence_key: str = "confidence") -> float:
    """AUROC using similarity_rank == 'high' as the positive class (vs low + medium)."""
    labeled = [
        (float(r[confidence_key]), int(r.get("similarity_rank") == "high"))
        for r in results
        if r.get("similarity_rank") in _RANK_INT
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")
    n_pos = sum(c for _, c in labeled)
    n_neg = len(labeled) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    labeled.sort(key=lambda x: -x[0])
    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auc = 0.0
    for _, pos in labeled:
        if pos:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tpr, prev_fpr = tpr, fpr
    return round(auc, 4)


def auroc_arith(
    results: list[dict],
    confidence_key: str = "confidence",
    threshold: float = 0.8,
) -> float:
    """AUROC where arith_step_score >= threshold is the positive class."""
    labeled = [
        (float(r[confidence_key]), int(r.get("arith_step_score", float("nan")) >= threshold))
        for r in results
        if not math.isnan(r.get("arith_step_score", float("nan")))
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")
    n_pos = sum(c for _, c in labeled)
    n_neg = len(labeled) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    labeled.sort(key=lambda x: -x[0])
    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auc = 0.0
    for _, pos in labeled:
        if pos:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tpr, prev_fpr = tpr, fpr
    return round(auc, 4)


def auroc_judge(results: list[dict], confidence_key: str = "confidence") -> float:
    """AUROC using judge_rank == 'good' as the positive class (vs medium + bad)."""
    labeled = [
        (float(r[confidence_key]), int(r.get("judge_rank") == "good"))
        for r in results
        if r.get("judge_rank") in _JUDGE_SCORE
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")
    n_pos = sum(c for _, c in labeled)
    n_neg = len(labeled) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    labeled.sort(key=lambda x: -x[0])
    tp = fp = 0
    prev_tpr = prev_fpr = 0.0
    auc = 0.0
    for _, pos in labeled:
        if pos:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
        prev_tpr, prev_fpr = tpr, fpr
    return round(auc, 4)


def expected_calibration_error_judge(
    results: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence",
) -> float:
    """ECE-style calibration against judge score (good=1.0, medium=0.5, bad=0.0)."""
    labeled = [
        r for r in results
        if r.get("judge_rank") in _JUDGE_SCORE
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")

    bins: list[list] = [[] for _ in range(n_bins)]
    for r in labeled:
        idx = min(int(float(r[confidence_key]) * n_bins), n_bins - 1)
        bins[idx].append(r)

    ece = 0.0
    for b in bins:
        if not b:
            continue
        mean_score = sum(_JUDGE_SCORE[r["judge_rank"]] for r in b) / len(b)
        mean_conf  = sum(float(r[confidence_key]) for r in b) / len(b)
        ece += (len(b) / len(labeled)) * abs(mean_score - mean_conf)
    return round(ece, 4)


def plot_reliability_diagram_judge(
    results: list[dict],
    output_path: str,
    n_bins: int = 10,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """Reliability diagram where y-axis shows mean judge score (good=1, medium=0.5, bad=0) per bin."""
    labeled = [
        r for r in results
        if r.get("judge_rank") in _JUDGE_SCORE
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return

    bin_scores, bin_confs = [], []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        b = [r for r in labeled if lo <= float(r[confidence_key]) < hi]
        if not b:
            continue
        bin_scores.append(sum(_JUDGE_SCORE[r["judge_rank"]] for r in b) / len(b))
        bin_confs.append(sum(float(r[confidence_key]) for r in b) / len(b))

    ece_judge = expected_calibration_error_judge(results, n_bins, confidence_key)
    label = title or confidence_key.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_confs, bin_scores, width=1 / n_bins, align="center", alpha=0.7,
           color="steelblue", label="Mean judge score")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Mean judge score  (good=1, medium=0.5, bad=0)")
    ax.set_title(f"{label}  (ECE-judge = {ece_judge:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def expected_calibration_error_arith(
    results: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence",
) -> float:
    """ECE-style calibration against arith_step_score instead of binary accuracy."""
    labeled = [
        r for r in results
        if not math.isnan(r.get("arith_step_score", float("nan")))
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")

    bins: list[list] = [[] for _ in range(n_bins)]
    for r in labeled:
        idx = min(int(float(r[confidence_key]) * n_bins), n_bins - 1)
        bins[idx].append(r)

    ece = 0.0
    for b in bins:
        if not b:
            continue
        mean_arith = sum(r["arith_step_score"] for r in b) / len(b)
        mean_conf  = sum(float(r[confidence_key]) for r in b) / len(b)
        ece += (len(b) / len(labeled)) * abs(mean_arith - mean_conf)
    return round(ece, 4)


def plot_reliability_diagram_arith(
    results: list[dict],
    output_path: str,
    n_bins: int = 10,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """
    Reliability diagram where the y-axis shows mean arithmetic step score per bin.
    Answers: does higher confidence correspond to more arithmetically correct intermediate steps?
    """
    labeled = [
        r for r in results
        if not math.isnan(r.get("arith_step_score", float("nan")))
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return

    bin_scores, bin_confs = [], []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        b = [r for r in labeled if lo <= float(r[confidence_key]) < hi]
        if not b:
            continue
        bin_scores.append(sum(r["arith_step_score"] for r in b) / len(b))
        bin_confs.append(sum(float(r[confidence_key]) for r in b) / len(b))

    ece_arith = expected_calibration_error_arith(results, n_bins, confidence_key)
    label = title or confidence_key.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_confs, bin_scores, width=1 / n_bins, align="center", alpha=0.7,
           color="steelblue", label="Mean arith step score")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Mean arithmetic step score")
    ax.set_title(f"{label}  (ECE-arith = {ece_arith:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def spearman_sim(results: list[dict], confidence_key: str = "confidence") -> float:
    """Spearman correlation between confidence and ordinal similarity rank (low=0/medium=1/high=2)."""
    labeled = [
        (float(r[confidence_key]), _RANK_INT[r["similarity_rank"]])
        for r in results
        if r.get("similarity_rank") in _RANK_INT
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if len(labeled) < 2:
        return float("nan")
    return _spearman_from_lists([x[0] for x in labeled], [x[1] for x in labeled])


def expected_calibration_error_sim(
    results: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence",
) -> float:
    """
    ECE-style calibration against mean_raw_similarity instead of binary accuracy.
    Measures whether a model's confidence matches how semantically correct its output is.
    """
    labeled = [
        r for r in results
        if not math.isnan(r.get("mean_raw_similarity", float("nan")))
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return float("nan")

    bins: list[list] = [[] for _ in range(n_bins)]
    for r in labeled:
        idx = min(int(float(r[confidence_key]) * n_bins), n_bins - 1)
        bins[idx].append(r)

    ece = 0.0
    for b in bins:
        if not b:
            continue
        mean_sim  = sum(r["mean_raw_similarity"] for r in b) / len(b)
        mean_conf = sum(float(r[confidence_key]) for r in b) / len(b)
        ece += (len(b) / len(labeled)) * abs(mean_sim - mean_conf)
    return round(ece, 4)


def plot_reliability_diagram_sim(
    results: list[dict],
    output_path: str,
    n_bins: int = 10,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """
    Reliability diagram where the y-axis shows mean embedding similarity per bin
    instead of binary accuracy. Diagnoses whether confidence tracks semantic correctness.
    """
    labeled = [
        r for r in results
        if not math.isnan(r.get("mean_raw_similarity", float("nan")))
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return

    bin_sims, bin_confs = [], []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        b = [r for r in labeled if lo <= float(r[confidence_key]) < hi]
        if not b:
            continue
        bin_sims.append(sum(r["mean_raw_similarity"] for r in b) / len(b))
        bin_confs.append(sum(float(r[confidence_key]) for r in b) / len(b))

    ece_sim = expected_calibration_error_sim(results, n_bins, confidence_key)
    label = title or confidence_key.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_confs, bin_sims, width=1 / n_bins, align="center", alpha=0.7,
           color="steelblue", label="Mean similarity")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Mean embedding similarity")
    ax.set_title(f"{label}  (ECE-sim = {ece_sim:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confidence_distribution_3way(
    results: list[dict],
    output_path: str,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """
    Three overlapping confidence histograms split by similarity rank (low/medium/high).
    Well-separated distributions indicate the confidence measure discriminates all three levels.
    """
    bins = [i / 20 for i in range(21)]
    rank_confs: dict[str, list[float]] = {"low": [], "medium": [], "high": []}
    for r in results:
        rank = r.get("similarity_rank")
        if rank not in rank_confs:
            continue
        if confidence_key not in r or math.isnan(float(r[confidence_key])):
            continue
        rank_confs[rank].append(float(r[confidence_key]))

    if not any(rank_confs.values()):
        return

    colours = {"low": "tomato", "medium": "gold", "high": "steelblue"}
    label_map = {"low": "Low (<0.3)", "medium": "Medium (0.3–0.7)", "high": "High (>0.7)"}

    fig, ax = plt.subplots(figsize=(7, 4))
    for rank in ("low", "medium", "high"):
        if rank_confs[rank]:
            ax.hist(rank_confs[rank], bins=bins, alpha=0.55,
                    label=label_map[rank], color=colours[rank], density=True)

    display_label = title or confidence_key.replace("_", " ").title()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence by Similarity Rank — {display_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


_ROC_CONF_KEYS = [
    ("confidence",               "Majority Vote"),
    ("weighted_mean_confidence", "Weighted"),
]


def plot_roc_curve(
    results: list[dict],
    output_path: str,
    label_fn,
    title: str | None = None,
) -> None:
    """
    ROC curve overlaying both confidence measures on one plot.

    label_fn(r) -> bool | None  — returns the positive/negative label for each
    result dict, or None to skip that problem. Caller defines what "positive" means
    (binary correct, arith step score >= threshold, similarity rank == high, etc.).
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUROC = 0.50)")

    for conf_key, display_label in _ROC_CONF_KEYS:
        labeled = []
        for r in results:
            lbl = label_fn(r)
            if lbl is None:
                continue
            if conf_key not in r or math.isnan(float(r[conf_key])):
                continue
            labeled.append((float(r[conf_key]), bool(lbl)))

        if not labeled:
            continue
        n_pos = sum(c for _, c in labeled)
        n_neg = len(labeled) - n_pos
        if n_pos == 0 or n_neg == 0:
            continue

        labeled.sort(key=lambda x: -x[0])
        fprs, tprs = [0.0], [0.0]
        tp = fp = 0
        for _, correct in labeled:
            if correct:
                tp += 1
            else:
                fp += 1
            fprs.append(fp / n_neg)
            tprs.append(tp / n_pos)
        fprs.append(1.0)
        tprs.append(1.0)

        auc = sum(
            (fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2
            for i in range(len(fprs) - 1)
        )
        ax.plot(fprs, tprs, label=f"{display_label} (AUROC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or "ROC Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _safe_eval_expr(expr: str) -> float | None:
    """Evaluate a simple arithmetic expression string, returning None on failure."""
    expr = expr.replace("×", "*").replace("÷", "/").replace("^", "**").strip()
    if _SAFE_EXPR_RE.search(expr):
        return None
    try:
        return float(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        return None


def _parse_numeric(s: str) -> float | None:
    """Extract the first numeric value from a string."""
    m = re.search(r"-?\d+\.?\d*", s.replace(",", ""))
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def arithmetic_step_correctness(raw: str) -> dict:
    """
    Parse all <<expr=result>> annotations from a model output and verify each.

    Correctness is purely internal — no reference solution required.
    Returns arith_steps_total, arith_steps_correct, arith_step_score, arith_steps_detail.
    """
    detail = []
    n_correct = 0
    for expr_str, claimed_str in _ARITH_ANNOTATION.findall(raw):
        actual  = _safe_eval_expr(expr_str)
        claimed = _parse_numeric(claimed_str)
        if actual is None or claimed is None:
            correct = False
        else:
            tol = max(1e-6, 1e-6 * abs(actual))
            correct = abs(actual - claimed) <= tol
        if correct:
            n_correct += 1
        detail.append({
            "expr":    expr_str.strip(),
            "claimed": round(claimed, 6) if claimed is not None else None,
            "actual":  round(actual,  6) if actual  is not None else None,
            "correct": correct,
        })
    n_total = len(detail)
    return {
        "arith_steps_total":   n_total,
        "arith_steps_correct": n_correct,
        "arith_step_score":    round(n_correct / n_total, 4) if n_total else float("nan"),
        "arith_steps_detail":  detail,
    }


def _extract_ref_values(text: str) -> set[float]:
    """Extract the numeric result values from all <<expr=result>> annotations in text."""
    values: set[float] = set()
    for _, claimed_str in _ARITH_ANNOTATION.findall(text):
        v = _parse_numeric(claimed_str)
        if v is not None:
            values.add(v)
    return values


def reference_alignment(raw: str, reference_solution: str) -> dict:
    """
    Compare model output <<expr=result>> result values against ground truth values.

    Extracts the numeric result (value after = and before >>) from every
    <<expr=result>> annotation in both the model output and the reference solution,
    then computes set-based coverage:

        ref_alignment_score = |ref_values ∩ model_values| / |ref_values|

    Only result values are compared — expressions are ignored — so a model that
    writes <<96/4=24>> matches a reference <<48/2=24>> because both arrive at 24.

    Returns ref_alignment_score = NaN when the reference has no <<>> annotations
    (e.g. MATH dataset without preprocessing).
    """
    ref_values   = _extract_ref_values(reference_solution)
    model_values = _extract_ref_values(raw)

    if not ref_values:
        return {
            "ref_values_total":    0,
            "ref_values_matched":  0,
            "ref_alignment_score": float("nan"),
        }

    matched = len(ref_values & model_values)
    return {
        "ref_values_total":    len(ref_values),
        "ref_values_matched":  matched,
        "ref_alignment_score": round(matched / len(ref_values), 4),
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


def plot_selective_prediction(
    results: list[dict],
    output_path: str,
) -> None:
    """
    Accuracy vs coverage curves for all confidence measures on one plot.
    A useful UQ signal produces a curve that rises steeply as coverage decreases
    (high-confidence predictions are more often correct).
    """
    conf_keys = {
        "Majority-vote": "confidence",
        "Weighted":      "weighted_mean_confidence",
    }
    labeled = [r for r in results if r.get("correct") is not None]
    if not labeled:
        return
    n = len(labeled)
    baseline = sum(bool(r["correct"]) for r in labeled) / n

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, key in conf_keys.items():
        valid = [
            (float(r[key]), bool(r["correct"]))
            for r in labeled
            if key in r and not math.isnan(float(r[key]))
        ]
        if not valid:
            continue
        valid.sort(key=lambda x: -x[0])
        coverages, accuracies, correct_so_far = [], [], 0
        for i, (_, correct) in enumerate(valid):
            if correct:
                correct_so_far += 1
            coverages.append((i + 1) / n)
            accuracies.append(correct_so_far / (i + 1))
        ax.plot(coverages, accuracies, label=label)

    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Overall accuracy ({baseline:.2f})")
    ax.set_xlabel("Coverage (fraction of problems included)")
    ax.set_ylabel("Accuracy on included problems")
    ax.set_title("Selective Prediction: Accuracy vs Coverage")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_confidence_distribution(
    results: list[dict],
    output_path: str,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """
    Confidence histograms split by correctness.
    Well-separated distributions indicate the measure discriminates well.
    """
    labeled = [
        r for r in results
        if r.get("correct") is not None
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return
    correct_confs   = [float(r[confidence_key]) for r in labeled if r["correct"]]
    incorrect_confs = [float(r[confidence_key]) for r in labeled if not r["correct"]]
    bins = [i / 20 for i in range(21)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(correct_confs,   bins=bins, alpha=0.6, label="Correct",   density=True)
    ax.hist(incorrect_confs, bins=bins, alpha=0.6, label="Incorrect", density=True)
    label = title or confidence_key.replace("_", " ").title()
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Confidence Distribution — {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarise(results: list[dict]) -> dict:
    """
    Aggregate summary focused on confidence-correctness alignment.
    Reports calibration (ECE), discrimination (AUROC), correlation (Spearman),
    and the overconfidence quadrant for each confidence measure.
    """
    labeled   = [r for r in results if r["correct"] is not None]
    n_total   = len(results)
    n_labeled = len(labeled)

    accuracy     = sum(1 for r in labeled if r["correct"]) / n_labeled if n_labeled else float("nan")
    mean_conf    = sum(r["confidence"] for r in results) / n_total if n_total else float("nan")
    mean_entropy = sum(r["entropy"] for r in results) / n_total if n_total else float("nan")

    def _mean(key):
        vals = [r[key] for r in results if key in r and not math.isnan(r[key])]
        return round(sum(vals) / len(vals), 6) if vals else float("nan")

    oc = overconfidence_analysis(results)

    summary = {
        "n_problems": n_total,
        "n_labeled":  n_labeled,
        "accuracy":   round(accuracy, 4),

        # --- Answer-level UQ ---
        "mean_answer_confidence": round(mean_conf, 4),
        "mean_answer_entropy":    round(mean_entropy, 4),
        "mean_weighted_confidence": _mean("weighted_mean_confidence"),

        # --- Overconfidence quadrant (majority-vote, threshold=0.8) ---
        "overconf_high_conf_correct": oc.get("high_conf_correct"),
        "overconf_high_conf_wrong":   oc.get("high_conf_wrong"),
        "overconf_low_conf_correct":  oc.get("low_conf_correct"),
        "overconf_low_conf_wrong":    oc.get("low_conf_wrong"),
        "overconf_rate":              oc.get("overconfidence_rate"),

        # --- Calibration: ECE (binary correctness) ---
        "ece_confidence": round(expected_calibration_error(results, confidence_key="confidence"), 4),
        "ece_weighted":   round(expected_calibration_error(results, confidence_key="weighted_mean_confidence"), 4),

        # --- Discrimination: AUROC (binary correctness) ---
        "auroc_confidence": auroc(results, confidence_key="confidence"),
        "auroc_weighted":   auroc(results, confidence_key="weighted_mean_confidence"),

        # --- Embedding similarity aggregate ---
        "mean_raw_similarity":     _mean("mean_raw_similarity"),
        "mean_std_raw_similarity": _mean("std_raw_similarity"),
        "sim_rank_low":    sum(1 for r in results if r.get("similarity_rank") == "low"),
        "sim_rank_medium": sum(1 for r in results if r.get("similarity_rank") == "medium"),
        "sim_rank_high":   sum(1 for r in results if r.get("similarity_rank") == "high"),

        # --- Calibration: ECE (embedding similarity) ---
        "ece_sim_confidence": expected_calibration_error_sim(results, confidence_key="confidence"),
        "ece_sim_weighted":   expected_calibration_error_sim(results, confidence_key="weighted_mean_confidence"),

        # --- Discrimination: AUROC (embedding similarity — high vs low+medium) ---
        "auroc_sim_confidence": auroc_sim(results, confidence_key="confidence"),
        "auroc_sim_weighted":   auroc_sim(results, confidence_key="weighted_mean_confidence"),

        # --- Arithmetic step correctness aggregate ---
        "mean_arith_step_score": _mean("arith_step_score"),
        "arith_steps_total":   sum(r.get("arith_steps_total",   0) for r in results),
        "arith_steps_correct": sum(r.get("arith_steps_correct", 0) for r in results),

        # --- Calibration: ECE (arithmetic step correctness) ---
        "ece_arith_confidence": expected_calibration_error_arith(results, confidence_key="confidence"),
        "ece_arith_weighted":   expected_calibration_error_arith(results, confidence_key="weighted_mean_confidence"),

        # --- Discrimination: AUROC (arithmetic step correctness) ---
        "auroc_arith_confidence": auroc_arith(results, confidence_key="confidence"),
        "auroc_arith_weighted":   auroc_arith(results, confidence_key="weighted_mean_confidence"),

        # --- Reference alignment aggregate (GSM8K only, when enabled) ---
        "mean_ref_alignment_score": _mean("ref_alignment_score"),
        "ref_values_total":   sum(r.get("ref_values_total",   0) for r in results),
        "ref_values_matched": sum(r.get("ref_values_matched", 0) for r in results),
    }

    # --- LLM judge aggregate (present only after llm_judge enrichment) ---
    # (conditional — not populated until run_llm_judge has been run)
    if any(r.get("judge_rank") in _JUDGE_SCORE for r in results):
        summary.update({
            "judge_rank_good":   sum(1 for r in results if r.get("judge_rank") == "good"),
            "judge_rank_medium": sum(1 for r in results if r.get("judge_rank") == "medium"),
            "judge_rank_bad":    sum(1 for r in results if r.get("judge_rank") == "bad"),
            "ece_judge_confidence": expected_calibration_error_judge(results, confidence_key="confidence"),
            "ece_judge_weighted":   expected_calibration_error_judge(results, confidence_key="weighted_mean_confidence"),
            "auroc_judge_confidence": auroc_judge(results, confidence_key="confidence"),
            "auroc_judge_weighted":   auroc_judge(results, confidence_key="weighted_mean_confidence"),
        })

    return summary
