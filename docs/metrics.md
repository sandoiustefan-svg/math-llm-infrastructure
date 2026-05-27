# UQ Metrics

All metrics are computed in `src/uq/metrics.py` and aggregated by `summarise()`.
Per-problem raw values are saved to `results.json`; aggregate values to `summary.json`.

For how epistemic uncertainty and confidence are estimated (MC Dropout, majority-vote,
consistency rate, answer entropy, std log-prob) see `docs/uncertainty_methods.md`.

---

## Correctness signals

Three correctness signals are used per problem:

- **Binary correctness** — `answers_are_equal()` string match on the extracted final
  answer, with sympy symbolic equivalence as a fallback for MATH (handles cases like
  `\frac{1}{2}` vs `0.5`). Fast and unambiguous, but ignores the reasoning process.
- **LLM-as-judge rank** — a post-processing step that assigns `good / medium / bad`
  to the majority answer using an external LLM judge. Captures semantic correctness
  and partial credit that binary correctness misses. See [LLM-as-judge rank](#llm-as-judge-rank-semantic-correctness) below.
- **NLG baselines (ROUGE-L / METEOR)** — standard sequence-overlap metrics computed
  between each MC Dropout pass output and the full `reference_solution`. Included as
  weak baselines to demonstrate their inadequacy for mathematical reasoning.
  See [NLG baselines](#nlg-baselines) below.

---

## Central research question

**Does MC Dropout produce reliable epistemic uncertainty estimates for LoRA fine-tuned
LLMs on mathematical reasoning, and what factors affect that reliability?**

Metrics are organised around two ways of answering that question — calibration and
discrimination — applied to two confidence scores and four uncertainty measures.
Binary correctness and the LLM judge are the primary correctness signals.

---

## Confidence scores and uncertainty measures

Two confidence scores (used for ECE and AUROC):

| Key | What it measures |
|---|---|
| `confidence` | Majority-vote fraction — fraction of 20 passes agreeing on the final answer |
| `consistency_rate` | Σ p(a)² — sum of squared answer frequencies, considers full distribution |

Four uncertainty measures (used for AUROC only, negated for ranking):

| Key | What it measures |
|---|---|
| `entropy` | Shannon entropy over answer distribution across 20 passes |
| `n_unique_answers` | Count of distinct answers across 20 passes |
| `std_log_prob` | Std of per-pass avg log-prob across 20 passes — token-level epistemic variance |
| `std_numeric_span_log_prob` | Same but restricted to numeric tokens in the Final Answer span |

---

## Confidence-correctness alignment metrics

### Calibration — Expected Calibration Error (ECE)

Measures whether stated confidence equals empirical accuracy.
The confidence range is split into 10 bins:

```
ECE = Σ_b (|b| / N) * |accuracy(b) - mean_confidence(b)|
```

ECE = 0 → perfect calibration. ECE > 0.15 → poor calibration.

Computed for both confidence scores against all correctness signals:
- `ece_confidence`, `ece_consistency` — against binary correctness
- `ece_judge_confidence`, `ece_judge_consistency` — against LLM judge score (good=1, medium=0.5, bad=0)
- `ece_rougeL_confidence`, `ece_rougeL_consistency` — against mean ROUGE-L score (NLG baseline)

---

### Discrimination — AUROC

Measures whether confidence *ranks* correct problems above incorrect ones,
independently of absolute calibration.

```
AUROC = 0.5  →  no better than random
AUROC = 1.0  →  perfect discrimination
```

Computed for confidence scores against all correctness signals:
- `auroc_confidence`, `auroc_consistency` — binary correct vs wrong
- `auroc_judge_confidence`, `auroc_judge_consistency` — judge `good` vs `medium+bad`
- `auroc_rougeL_confidence`, `auroc_rougeL_consistency` — ROUGE-L above threshold vs below

Computed for uncertainty measures (negated — higher uncertainty = more likely wrong):
- `auroc_entropy`, `auroc_n_unique`, `auroc_std_log_prob`, `auroc_std_numeric_span`

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

---

## LLM-as-judge rank (semantic correctness)

A post-processing step applied to `results.json` after inference. An external LLM
judge is given the problem and the model's majority answer and returns one of three
labels, stored as `judge_rank`:

| Label | Meaning |
|---|---|
| `good` | Correct final answer and sound reasoning |
| `medium` | Wrong final answer but correct approach / partial credit |
| `bad` | Wrong answer and flawed or irrelevant reasoning |

**Running the judge**

```bash
python -m src.uq.llm_judge \
    --results results/mc_dropout/1b/seed42/gsm8k/zero_shot_aligned/results.json \
    --provider anthropic --model claude-haiku-4-5-20251001
```

The module reads `results.json`, calls the judge API, and writes `results_judged.json`,
`summary_judged.json`, and plots under `judge_correctness/`.

**Cost estimate** (500 problems, ~630 input tokens, ~10 output tokens per call):

| Judge model | Approx. cost |
|---|---|
| GPT-4o-mini | ~$0.15 |
| Claude Haiku 4.5 | ~$0.80 |

---

## NLG baselines

ROUGE-L and METEOR computed between each MC Dropout pass output and the
`reference_solution`, then averaged across passes. Included as weak baselines to
demonstrate inadequacy for mathematical reasoning — n-gram overlap is sensitive to
surface form, not mathematical meaning.

| Key | What it stores |
|---|---|
| `mean_rougeL` | Mean ROUGE-L F1 across all 20 passes |
| `mean_meteor` | Mean METEOR score across all 20 passes |

---

## Plots

Each run produces plots under `results/mc_dropout/<model>/<seed>/<source>/<prompt>/`:

```
binary_correctness/
  reliability_confidence.png
  reliability_consistency_rate.png
  roc_binary.png
nlg_baselines/
  reliability_rougeL_confidence.png
  reliability_rougeL_consistency_rate.png
  roc_rougeL.png
confidence/
  selective_prediction.png
judge_correctness/               ← produced by src/uq/llm_judge.py
  reliability_judge_confidence.png
  reliability_judge_consistency_rate.png
  roc_judge.png
```

---

## Reading `summary.json`

```json
{
  "n_problems": 500,
  "n_labeled":  500,
  "accuracy":   0.388,

  "mean_consistency_rate":          0.712,
  "mean_entropy":                   1.24,
  "mean_n_unique_answers":          2.1,
  "mean_std_log_prob":              0.043,
  "mean_std_numeric_span_log_prob": 0.081,

  "overconf_high_conf_correct": 109,
  "overconf_high_conf_wrong":   27,
  "overconf_low_conf_correct":  85,
  "overconf_low_conf_wrong":    279,
  "overconf_rate":              0.198,

  "ece_confidence":   0.128,
  "ece_consistency":  0.134,

  "auroc_confidence":   0.846,
  "auroc_consistency":  0.831,

  "auroc_entropy":           0.821,
  "auroc_n_unique":          0.798,
  "auroc_std_log_prob":      0.743,
  "auroc_std_numeric_span":  0.761,

  "mean_rougeL":  0.264,
  "mean_meteor":  0.301,

  "ece_rougeL_confidence":   0.266,
  "ece_rougeL_consistency":  0.271,

  "auroc_rougeL_confidence":   0.878,
  "auroc_rougeL_consistency":  0.862
}
```

**What to look at:**

1. `overconf_rate` — fraction of high-confidence predictions that are wrong.
2. `ece_confidence` vs `ece_consistency` — which confidence score is better calibrated?
3. `auroc_confidence` vs `auroc_entropy` vs `auroc_std_log_prob` — do answer-level and token-level epistemic signals discriminate equally well?
4. ECE and AUROC across `zero_shot_aligned` vs `cot` — does prompt-induced distribution shift degrade calibration?
5. ECE and AUROC across 1B vs 8B — does model scale improve epistemic UQ reliability?
6. ECE and AUROC across GSM8K vs MATH — does domain shift degrade epistemic UQ?
