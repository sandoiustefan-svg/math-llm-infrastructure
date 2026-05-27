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


_ANSWER_MARKER = "Final Answer:"
_JUDGE_SCORE   = {"good": 1.0, "medium": 0.5, "bad": 0.0}

# Weights for the weighted geometric mean confidence.
# All prompts use <<expr=result>> annotations, so arithmetic-critical tokens are
# explicitly identifiable rather than heuristically detected:
#   - Result tokens inside <<expr=result>>: _W_ARITH_RESULT
#   - Tokens in Final Answer span:          _W_FINAL_ANSWER
#   - All other tokens:                     1
_W_ARITH_RESULT = 10
_W_FINAL_ANSWER = 25

# Matches <<expr=result>> annotations; group 1 = expr, group 2 = result portion.
# Retained to identify arithmetic result tokens for weighted_mean_confidence.
_ARITH_ANNOTATION = re.compile(r'<<([^=\n]+)=([^>\n]+)>>')

# Three confidence measures used across all plots and metrics.
_ROC_CONF_KEYS = [
    ("confidence",       "Majority Vote"),
    ("consistency_rate", "Consistency Rate"),
]

_UNCERTAINTY_KEYS = [
    ("entropy",                    "Answer Entropy"),
    ("n_unique_answers",           "Unique Answers"),
    ("std_log_prob",               "Std Log-Prob"),
    ("std_numeric_span_log_prob",  "Std Numeric Span Log-Prob"),
]


# ---------------------------------------------------------------------------
# Math answer normalisation
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> str | None:
    """Extract the content of the outermost \\boxed{...}, handling nested braces."""
    idx = text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None


def _strip_latex(s: str) -> str:
    inner = _extract_boxed(s)
    if inner is not None:
        s = inner
    s = re.sub(r"\\\$", "", s)   # \$ (LaTeX dollar sign) → nothing, before bare-$ strip
    s = re.sub(r"\$+", "", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\,|\\;|\\!", "", s)
    return s.strip()


def _frac_to_float(s: str) -> float | None:
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
    Handles LaTeX wrappers, fractions, trailing zeros, comma separators.
    Falls back to stripped lowercase string for non-numeric answers.
    """
    if not isinstance(s, str):
        s = str(s)
    s = _strip_latex(s)
    s = s.replace(",", "").strip()
    frac = _frac_to_float(s)
    if frac is not None:
        return str(round(frac, 10)).rstrip("0").rstrip(".")
    try:
        f = float(s)
        if math.isfinite(f) and f == int(f) and "e" not in s.lower():
            return str(int(f))
        return str(round(f, 10)).rstrip("0").rstrip(".")
    except ValueError:
        pass
    return s.lower().strip()


def answers_are_equal(pred: str, gold: str) -> bool:
    """
    Compare two math answer strings with normalisation.
    Falls back to sympy symbolic equivalence for MATH dataset answers
    where string normalisation is insufficient (e.g. equivalent expressions).
    """
    if normalize_math_answer(pred) == normalize_math_answer(gold):
        return True
    try:
        from sympy import sympify, simplify
        pred_expr = sympify(pred, evaluate=True)
        gold_expr = sympify(gold, evaluate=True)
        # simplify can hang on complex MATH expressions; bail out quickly
        return simplify(pred_expr - gold_expr, rational=False) == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NLG baselines (BLEU / ROUGE / METEOR)
# ---------------------------------------------------------------------------

def _normalize_for_nlg(text: str) -> str:
    """Expand common LaTeX to plain text before NLG scoring."""
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", text)
    text = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\cdot|\\times", "*", text)
    text = re.sub(r"\\div", "/", text)
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\boxed\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\$+", "", text)
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_nlg_scores(output: str, reference: str) -> dict:
    """
    Compute ROUGE-L and METEOR between one output and the reference.
    Both texts are LaTeX-normalised before scoring.
    Returns NaN for both metrics when either text is empty.
    """
    _nan = {"rougeL": float("nan"), "meteor": float("nan")}
    out_norm = _normalize_for_nlg(output).strip()
    ref_norm = _normalize_for_nlg(reference).strip()
    if not out_norm or not ref_norm:
        return _nan

    try:
        from rouge_score import rouge_scorer as _rouge_scorer
        import nltk
        from nltk.translate.meteor_score import meteor_score as _meteor_fn
    except ImportError as e:
        raise ImportError(
            f"NLG metric libraries missing: {e}. "
            "Run: pip install rouge-score nltk"
        ) from e

    scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge = scorer.score(ref_norm, out_norm)

    ref_tok = ref_norm.split()
    hyp_tok = out_norm.split()
    meteor = _meteor_fn([ref_tok], hyp_tok) if ref_tok and hyp_tok else 0.0

    return {
        "rougeL": round(rouge["rougeL"].fmeasure, 6),
        "meteor": round(float(meteor),            6),
    }


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_DIGIT_CHARS = set("0123456789")


def _is_numeric_token(token_str: str) -> bool:
    stripped = token_str.strip()
    return bool(stripped) and any(c in _DIGIT_CHARS for c in stripped)


def _compute_confidence_from_probs(token_probs: list[float]) -> dict:
    if not token_probs:
        return {
            "mean_confidence": float("nan"),
            "perplexity":      float("nan"),
            "min_token_prob":  float("nan"),
            "std_token_prob":  float("nan"),
        }
    log_probs = [math.log(p) for p in token_probs if p > 0]
    mean_log  = sum(log_probs) / len(log_probs) if log_probs else float("-inf")
    mean_conf = math.exp(mean_log) if math.isfinite(mean_log) else 0.0
    perplexity = math.exp(-mean_log) if math.isfinite(mean_log) else float("inf")
    return {
        "mean_confidence": round(mean_conf, 6),
        "perplexity":      round(perplexity, 4),
        "min_token_prob":  round(min(token_probs), 6),
        "std_token_prob":  round(statistics.stdev(token_probs) if len(token_probs) > 1 else 0.0, 6),
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
    Compute confidence measures over the answer tokens given the prompt.

    1. full_sequence — geometric mean over ALL generated tokens.
       Baseline: inflated by high-frequency glue tokens.
    2. answer_span — geometric mean over tokens after "Final Answer:".
    3. numeric_only — geometric mean over digit-containing tokens.
    4. weighted — geometric mean with <<expr=result>> result tokens at 10×
       and Final Answer span tokens at 25×.
    """
    full_text  = prompt + answer
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
            **{f"numeric_span_{k}": v for k, v in empty.items()},
            "weighted_mean_confidence": nan,
            "weighted_perplexity":      nan,
            "token_probs": [],
        }

    outputs = model(input_ids=full_ids[:, :-1])
    logits  = outputs.logits

    answer_logits    = logits[0, n_prompt - 1: n_prompt - 1 + n_answer, :]
    answer_token_ids = full_ids[0, n_prompt:]

    probs       = F.softmax(answer_logits, dim=-1)
    token_probs = probs[range(n_answer), answer_token_ids].tolist()

    full_metrics = _compute_confidence_from_probs(token_probs)

    marker_ids     = tokenizer(_ANSWER_MARKER, add_special_tokens=False).input_ids
    marker_len     = len(marker_ids)
    span_start     = None
    token_ids_list = answer_token_ids.tolist()
    for i in range(len(token_ids_list) - marker_len + 1):
        if token_ids_list[i: i + marker_len] == marker_ids:
            span_start = i + marker_len
            break

    span_probs = token_probs[span_start:] if span_start is not None and span_start < len(token_probs) \
        else token_probs[int(len(token_probs) * 0.8):]
    answer_span_metrics = _compute_confidence_from_probs(span_probs)

    numeric_probs = [
        prob for tok_id, prob in zip(token_ids_list, token_probs)
        if _is_numeric_token(tokenizer.decode([tok_id]))
    ]
    numeric_metrics = _compute_confidence_from_probs(numeric_probs)

    if span_start is not None and span_start < len(token_probs):
        span_token_ids       = token_ids_list[span_start:]
        span_token_probs_lst = token_probs[span_start:]
    else:
        span_token_ids       = token_ids_list[int(len(token_ids_list) * 0.8):]
        span_token_probs_lst = token_probs[int(len(token_probs) * 0.8):]

    numeric_span_probs = [
        prob for tok_id, prob in zip(span_token_ids, span_token_probs_lst)
        if _is_numeric_token(tokenizer.decode([tok_id]))
    ]
    numeric_span_metrics = _compute_confidence_from_probs(numeric_span_probs)

    # Weighted geometric mean
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
        "token_probs":           token_probs,
        "n_full_tokens":         len(token_probs),
        "n_span_tokens":         len(span_probs),
        "n_numeric_tokens":      len(numeric_probs),
        "n_numeric_span_tokens": len(numeric_span_probs),
    }


def _weighted_geometric_mean(token_probs: list[float], weights: list[float]) -> dict:
    if not token_probs:
        return {"mean_confidence": float("nan"), "perplexity": float("nan")}
    log_sum = w_sum = 0.0
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


# ---------------------------------------------------------------------------
# Core calibration / discrimination metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(
    results: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence",
) -> float:
    """ECE against binary correctness for any confidence measure."""
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
        idx = min(int(float(r[confidence_key]) * n_bins), n_bins - 1)
        bins[idx].append(r)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        acc  = sum(1 for r in b if r["correct"]) / len(b)
        conf = sum(float(r[confidence_key]) for r in b) / len(b)
        ece += (len(b) / len(labeled)) * abs(acc - conf)
    return round(ece, 4)


def auroc(results: list[dict], confidence_key: str = "confidence") -> float:
    """AUROC — confidence discriminates binary correct from wrong."""
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


def _auroc_from_pairs(labeled: list[tuple[float, int]]) -> float:
    """Generic AUROC from (confidence, binary_label) pairs."""
    if not labeled:
        return float("nan")
    n_pos = sum(c for _, c in labeled)
    n_neg = len(labeled) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    labeled = sorted(labeled, key=lambda x: -x[0])
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


def auroc_uncertainty(results: list[dict], uncertainty_key: str) -> float:
    """AUROC for an uncertainty measure — higher value means more uncertain (negated for ranking)."""
    labeled = [
        (-float(r[uncertainty_key]), int(bool(r["correct"])))
        for r in results
        if r.get("correct") is not None
        and uncertainty_key in r
        and not math.isnan(float(r.get(uncertainty_key, float("nan"))))
    ]
    return _auroc_from_pairs(labeled)


def auroc_judge(results: list[dict], confidence_key: str = "confidence") -> float:
    """AUROC — judge_rank == 'good' as positive class (vs medium + bad)."""
    labeled = [
        (float(r[confidence_key]), int(r.get("judge_rank") == "good"))
        for r in results
        if r.get("judge_rank") in _JUDGE_SCORE
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    return _auroc_from_pairs(labeled)


def auroc_rougeL(
    results: list[dict],
    confidence_key: str = "confidence",
    threshold: float = 0.4,
) -> float:
    """AUROC — mean_rougeL >= threshold as positive class (NLG baseline)."""
    labeled = [
        (float(r[confidence_key]), int(r.get("mean_rougeL", float("nan")) >= threshold))
        for r in results
        if not math.isnan(r.get("mean_rougeL", float("nan")))
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    return _auroc_from_pairs(labeled)


def expected_calibration_error_judge(
    results: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence",
) -> float:
    """ECE against judge score (good=1.0, medium=0.5, bad=0.0)."""
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


def expected_calibration_error_rougeL(
    results: list[dict],
    n_bins: int = 10,
    confidence_key: str = "confidence",
) -> float:
    """ECE against mean ROUGE-L score (NLG baseline)."""
    labeled = [
        r for r in results
        if not math.isnan(r.get("mean_rougeL", float("nan")))
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
        mean_score = sum(r["mean_rougeL"] for r in b) / len(b)
        mean_conf  = sum(float(r[confidence_key]) for r in b) / len(b)
        ece += (len(b) / len(labeled)) * abs(mean_score - mean_conf)
    return round(ece, 4)


def overconfidence_analysis(
    results: list[dict],
    confidence_key: str = "confidence",
    threshold: float = 0.8,
) -> dict:
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


def answer_entropy(answers: list[str]) -> float:
    """Shannon entropy over the answer distribution. 0 = unanimous, max = log2(N)."""
    n = len(answers)
    if n == 0:
        return 0.0
    counts = Counter(answers)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


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

    n = len(labeled)

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

    xs = [x[0] for x in labeled]
    ys = [x[1] for x in labeled]
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(
        sum((r - mx) ** 2 for r in rx) *
        sum((r - my) ** 2 for r in ry)
    )
    return round(num / den, 4) if den else float("nan")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability_diagram(
    results: list[dict],
    output_path: str,
    n_bins: int = 10,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """Reliability diagram against binary correctness."""
    labeled = [
        r for r in results
        if r.get("correct") is not None
        and confidence_key in r
        and not math.isnan(float(r[confidence_key]))
    ]
    if not labeled:
        return
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        b = [r for r in labeled if lo <= float(r[confidence_key]) < hi]
        if not b:
            continue
        bin_accs.append(sum(1 for r in b if r["correct"]) / len(b))
        bin_confs.append(sum(float(r[confidence_key]) for r in b) / len(b))
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


def plot_reliability_diagram_judge(
    results: list[dict],
    output_path: str,
    n_bins: int = 10,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """Reliability diagram — y-axis is mean judge score (good=1, medium=0.5, bad=0)."""
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
    ece = expected_calibration_error_judge(results, n_bins, confidence_key)
    label = title or confidence_key.replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_confs, bin_scores, width=1 / n_bins, align="center", alpha=0.7,
           color="steelblue", label="Mean judge score")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Mean judge score  (good=1, medium=0.5, bad=0)")
    ax.set_title(f"{label}  (ECE-judge = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_reliability_diagram_rougeL(
    results: list[dict],
    output_path: str,
    n_bins: int = 10,
    confidence_key: str = "confidence",
    title: str | None = None,
) -> None:
    """Reliability diagram — y-axis is mean ROUGE-L score per bin (NLG baseline)."""
    labeled = [
        r for r in results
        if not math.isnan(r.get("mean_rougeL", float("nan")))
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
        bin_scores.append(sum(r["mean_rougeL"] for r in b) / len(b))
        bin_confs.append(sum(float(r[confidence_key]) for r in b) / len(b))
    ece = expected_calibration_error_rougeL(results, n_bins, confidence_key)
    label = title or confidence_key.replace("_", " ").title()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_confs, bin_scores, width=1 / n_bins, align="center", alpha=0.7,
           color="coral", label="Mean ROUGE-L")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Mean ROUGE-L F1")
    ax.set_title(f"{label}  (ECE-rougeL = {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(
    results: list[dict],
    output_path: str,
    label_fn,
    title: str | None = None,
) -> None:
    """
    ROC curve overlaying all three confidence measures on one plot.
    label_fn(r) -> bool | None — positive/negative label, or None to skip.
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


def plot_selective_prediction(
    results: list[dict],
    output_path: str,
) -> None:
    """Accuracy vs coverage for all three confidence measures on one plot."""
    conf_keys = {
        "Majority-vote": "confidence",
        "Consistency Rate": "consistency_rate",
    }
    labeled = [r for r in results if r.get("correct") is not None]
    if not labeled:
        return
    n        = len(labeled)
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
    """Confidence histograms split by correctness."""
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


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(results: list[dict]) -> dict:
    """
    Aggregate summary focused on confidence-correctness alignment.
    Reports calibration (ECE), discrimination (AUROC), and overconfidence
    for all three confidence measures against all correctness signals.
    """
    labeled   = [r for r in results if r["correct"] is not None]
    n_total   = len(results)
    n_labeled = len(labeled)

    accuracy     = sum(1 for r in labeled if r["correct"]) / n_labeled if n_labeled else float("nan")
    mean_conf    = sum(r["confidence"] for r in results) / n_total if n_total else float("nan")
    mean_entropy = sum(r["entropy"] for r in results) / n_total if n_total else float("nan")

    def _mean(key: str) -> float:
        vals = [r[key] for r in results if key in r and not math.isnan(float(r.get(key, float("nan"))))]
        return round(sum(vals) / len(vals), 6) if vals else float("nan")

    oc = overconfidence_analysis(results)

    _conf_keys = [
        ("confidence",       "confidence"),
        ("consistency_rate", "consistency"),
    ]

    _uncertainty_keys = [
        ("entropy",                    "entropy"),
        ("n_unique_answers",           "n_unique"),
        ("std_log_prob",               "std_log_prob"),
        ("std_numeric_span_log_prob",  "std_numeric_span"),
    ]

    summary: dict = {
        "n_problems": n_total,
        "n_labeled":  n_labeled,
        "accuracy":   round(accuracy, 4),

        "mean_answer_confidence":        round(mean_conf, 4),
        "mean_answer_entropy":           round(mean_entropy, 4),
        "mean_consistency_rate":         _mean("consistency_rate"),
        "mean_entropy":                  _mean("entropy"),
        "mean_n_unique_answers":         _mean("n_unique_answers"),
        "mean_std_log_prob":             _mean("std_log_prob"),
        "mean_std_numeric_span_log_prob": _mean("std_numeric_span_log_prob"),

        "overconf_high_conf_correct": oc.get("high_conf_correct"),
        "overconf_high_conf_wrong":   oc.get("high_conf_wrong"),
        "overconf_low_conf_correct":  oc.get("low_conf_correct"),
        "overconf_low_conf_wrong":    oc.get("low_conf_wrong"),
        "overconf_rate":              oc.get("overconfidence_rate"),
    }

    # ECE and AUROC against binary correctness (all three confidence measures)
    for ck, short in _conf_keys:
        summary[f"ece_{short}"]   = expected_calibration_error(results, confidence_key=ck)
        summary[f"auroc_{short}"] = auroc(results, confidence_key=ck)

    # AUROC for uncertainty measures (negated — higher uncertainty = more likely wrong)
    for uk, short in _uncertainty_keys:
        summary[f"auroc_{short}"] = auroc_uncertainty(results, uk)

    # NLG baselines aggregate
    for metric in ("rougeL", "meteor"):
        summary[f"mean_{metric}"] = _mean(f"mean_{metric}")

    # ECE and AUROC against ROUGE-L (NLG baseline — all three confidence measures)
    for ck, short in _conf_keys:
        summary[f"ece_rougeL_{short}"]   = expected_calibration_error_rougeL(results, confidence_key=ck)
        summary[f"auroc_rougeL_{short}"] = auroc_rougeL(results, confidence_key=ck)

    # LLM judge — conditional on judge enrichment having been run
    if any(r.get("judge_rank") in _JUDGE_SCORE for r in results):
        summary.update({
            "judge_rank_good":   sum(1 for r in results if r.get("judge_rank") == "good"),
            "judge_rank_medium": sum(1 for r in results if r.get("judge_rank") == "medium"),
            "judge_rank_bad":    sum(1 for r in results if r.get("judge_rank") == "bad"),
        })
        for ck, short in _conf_keys:
            summary[f"ece_judge_{short}"]   = expected_calibration_error_judge(results, confidence_key=ck)
            summary[f"auroc_judge_{short}"] = auroc_judge(results, confidence_key=ck)

    return summary
