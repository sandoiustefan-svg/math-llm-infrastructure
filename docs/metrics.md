# UQ Metrics

All metrics are computed in `src/uq/metrics.py` and aggregated by `summarise()`.
Per-problem raw values are saved to `results.json`; aggregate values to `summary.json`.

## Experiment design

The evaluation is a **2×2 factorial design**:

| | Zero-shot | CoT |
|---|---|---|
| **Llama 3.2-1B (LoRA rank 16)** | accuracy + UQ | accuracy + UQ |
| **Llama 3.1-8B (LoRA rank 64)** | accuracy + UQ | accuracy + UQ |

Each cell runs MC Dropout (20 stochastic passes) on two test sets: GSM8K and MATH.

## Correctness signals

Two complementary correctness signals are used per problem:

- **Binary correctness** — `answers_are_equal()` string match on the extracted final answer. Fast and unambiguous, but ignores the reasoning process.
- **Embedding similarity** — cosine similarity between the full raw output and the expected answer, using `all-MiniLM-L6-v2`. Captures process-level alignment even when the final answer string differs.

## Central research question

**How well does a model's confidence align with the actual correctness of its outputs?**

Metrics are organised around two ways of answering that question — calibration and
discrimination — plus a direct analysis of the overconfidence failure mode.
Both correctness signals are used.

---

## Confidence measures

Two confidence measures are computed per problem, representing two different
sources of uncertainty signal:

| Key in `results.json` | What it measures |
|---|---|
| `confidence` | Majority-vote fraction across 20 passes — answer-level signal |
| `weighted_mean_confidence` | Weighted geometric mean token prob (numeric answer-span tokens = 25×) — token-level signal |

`confidence` captures epistemic uncertainty through *disagreement across passes*.
`weighted_mean_confidence` captures it through *token-level probability*, with
extra weight on the tokens that matter most (digits in the answer span).

---

## Answer-level metrics

### Majority-vote confidence

```
confidence = count(majority_answer) / num_passes
```

Fraction of the 20 passes that agreed on the most common answer.
`confidence = 1.0` means all passes gave the same answer.
`confidence = 0.05` means each pass gave a different answer.

### Answer entropy

```
H = -Σ p(a) * log₂(p(a))
```

Shannon entropy over the answer distribution across passes.
`H = 0` when all passes agree; maximum when all passes give different answers.
Reported in `summary.json` as `mean_answer_entropy`.

---

## Confidence-correctness alignment metrics

### Calibration — Expected Calibration Error (ECE)

Measures whether stated confidence equals empirical accuracy.
The confidence range is split into 10 bins:

```
ECE = Σ_b (|b| / N) * |accuracy(b) - mean_confidence(b)|
```

ECE = 0 → perfect calibration. ECE > 0.15 → poor calibration.

Computed for both confidence measures against **both correctness signals**:
- `ece_confidence`, `ece_weighted` — against binary correctness
- `ece_sim_confidence`, `ece_sim_weighted` — against mean embedding similarity

---

### Discrimination — AUROC

Measures whether confidence *ranks* correct problems above incorrect ones,
independently of absolute calibration.

```
AUROC = 0.5  →  no better than random
AUROC = 1.0  →  perfect discrimination
```

Computed for both confidence measures against **both correctness signals**:
- `auroc_confidence`, `auroc_weighted` — binary correct vs wrong
- `auroc_sim_confidence`, `auroc_sim_weighted` — high similarity rank vs low+medium

---

## Overconfidence analysis

The most important failure mode: the model is highly confident but wrong.

The test set is partitioned at a confidence threshold of 0.8:

```
                     Correct          Wrong
High confidence  │  knows it      │  doesn't know       ← epistemic gap
(≥ 0.8)          │  knows  ✓      │  it doesn't  ✗
─────────────────┼────────────────┼─────────────────
Low confidence   │  knows but     │  knows it
(< 0.8)          │  unsure        │  doesn't know  ✓
```

The **overconfidence rate** = high_conf_wrong / (high_conf_correct + high_conf_wrong).

A high overconfidence rate means confidence is not a reliable signal of correctness —
the model commits to wrong answers just as confidently as right ones.

---

## Embedding similarity (process-level correctness)

Binary correctness only checks the final answer string. Embedding similarity
captures whether the reasoning process itself is on the right track.

`all-MiniLM-L6-v2` encodes both the full raw output and the expected answer into
a 384-dimensional vector. Cosine similarity is computed for all 20 MC Dropout passes.

```
sim(raw, expected_answer) = dot(embed(raw), embed(expected_answer))   # L2-normalised
```

Four fields per problem in `results.json`:

| Key | What it stores |
|---|---|
| `raw_similarities` | List of 20 cosine similarity scores (one per pass) |
| `mean_raw_similarity` | Mean across all 20 passes |
| `std_raw_similarity` | Std across passes — high value means reasoning varies between passes |
| `similarity_rank` | `low` (< 0.3) / `medium` (0.3–0.7) / `high` (> 0.7) |

**How to use alongside binary correctness**

```
                     Binary correct      Binary wrong
High similarity  │  genuine understanding │  near-miss / format error
Low similarity   │  lucky guess           │  completely off
```

---

## Plots

Each run produces 5 plots in three subdirectories:

```
results/mc_dropout/<label>/<source>/<prompt>/
  confidence/
    selective_prediction.png              ← 1 plot
  binary_correctness/
    reliability_confidence.png            ← 1 plot
    reliability_weighted_mean_confidence.png  ← 1 plot
  embedding_similarity/
    reliability_sim_confidence.png            ← 1 plot
    reliability_sim_weighted_mean_confidence.png  ← 1 plot
```

---

### `confidence/selective_prediction.png`

Accuracy vs coverage for both confidence measures on one plot.

**Coverage** = fraction of problems answered after filtering out everything below
a confidence threshold. As the threshold rises, fewer problems are included but
those included should be more accurate.

- Curve rising above the dashed baseline → confidence is a useful abstention signal
- Flat curve → confidence adds no value for selective answering
- Dashed line = overall accuracy at coverage 1.0 (no filtering)

---

### `binary_correctness/reliability_<measure>.png` — Reliability diagrams (×2)

Problems are binned into 10 confidence buckets. Each bar shows the actual fraction
of correct problems in that bin. The diagonal is perfect calibration.

- Bar on the diagonal → confidence equals accuracy in that bucket
- Bar below the diagonal → over-confidence
- Bar above the diagonal → under-confidence

The ECE in the title is the same value reported in `summary.json`.

---

### `embedding_similarity/reliability_sim_<measure>.png` — Similarity reliability diagrams (×2)

Same structure as the binary reliability diagram but the y-axis shows **mean
embedding similarity** per confidence bin instead of fraction correct.

Answers: does higher confidence correspond to semantically closer outputs,
not just formally correct final answers?

The ECE-sim in the title measures calibration against process quality rather
than binary correctness.

---

## Reading `summary.json`

```json
{
  "n_problems": 500,
  "n_labeled":  500,
  "accuracy":   0.312,

  "mean_answer_confidence": 0.41,
  "mean_answer_entropy":    2.84,
  "mean_weighted_confidence": 0.51,

  "overconf_high_conf_correct": 48,
  "overconf_high_conf_wrong":   62,
  "overconf_low_conf_correct":  108,
  "overconf_low_conf_wrong":    282,
  "overconf_rate":              0.563,

  "ece_confidence": 0.115,
  "ece_weighted":   0.08,

  "auroc_confidence": 0.71,
  "auroc_weighted":   0.70,

  "mean_raw_similarity":     0.61,
  "mean_std_raw_similarity": 0.04,
  "sim_rank_low":    82,
  "sim_rank_medium": 241,
  "sim_rank_high":   177,

  "ece_sim_confidence": 0.09,
  "ece_sim_weighted":   0.07,

  "auroc_sim_confidence": 0.74,
  "auroc_sim_weighted":   0.72
}
```

**What to look at:**

1. `overconf_rate` — fraction of high-confidence predictions that are wrong. The direct answer to the epistemic uncertainty gap question.
2. `auroc_confidence` vs `auroc_weighted` — which measure better discriminates correct from incorrect?
3. `ece_confidence` vs `ece_sim_confidence` — does calibration improve when correctness is defined by process quality instead of final answer string?
4. `auroc_*` vs `auroc_sim_*` — does the ranking hold when correctness is redefined?
5. `sim_rank_low/medium/high` — mostly-high on GSM8K vs mostly-medium on MATH would show reasoning degrades on harder problems.
