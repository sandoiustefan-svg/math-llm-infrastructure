# UQ Metrics

All metrics are computed in `src/uq/metrics.py` and aggregated by `summarise()`.
Per-problem raw values are saved to `results.json`; aggregate values to `summary.json`.

For how epistemic uncertainty and confidence are estimated (MC Dropout, majority-vote,
weighted/unweighted geometric mean, answer entropy) see `docs/uncertainty_methods.md`.

## Correctness signals

Three correctness signals are used per problem:

- **Binary correctness** — `answers_are_equal()` string match on the extracted final
  answer, with sympy symbolic equivalence as a fallback for MATH (handles cases like
  `\frac{1}{2}` vs `0.5`). Fast and unambiguous, but ignores the reasoning process.
- **LLM-as-judge rank** — a post-processing step that assigns `good / medium / bad`
  to the majority answer using an external LLM judge. Captures semantic correctness
  and partial credit that binary correctness misses. See [LLM-as-judge rank](#llm-as-judge-rank-semantic-correctness) below.
- **NLG baselines (BLEU / ROUGE / METEOR)** — standard sequence-overlap metrics
  computed between each MC Dropout pass output and the full `reference_solution`.
  Included as weak baselines to demonstrate their inadequacy for mathematical reasoning.
  See [NLG baselines](#nlg-baselines-bleu--rouge--meteor) below.

## Central research question

**How well does a model's confidence align with the actual correctness of its outputs,
and does structured prompting improve that alignment?**

Metrics are organised around two ways of answering that question — calibration and
discrimination — plus a direct analysis of the overconfidence failure mode.
Binary correctness and the LLM judge are used as the primary correctness signals
against all three confidence measures (`confidence`, `full_sequence_mean_confidence`,
and `weighted_mean_confidence`); the NLG baselines are retained to show they carry
weaker calibration signal. See `docs/uncertainty_methods.md` for how the confidence
measures are computed.

---

## Confidence-correctness alignment metrics

### Calibration — Expected Calibration Error (ECE)

Measures whether stated confidence equals empirical accuracy.
The confidence range is split into 10 bins:

```
ECE = Σ_b (|b| / N) * |accuracy(b) - mean_confidence(b)|
```

ECE = 0 → perfect calibration. ECE > 0.15 → poor calibration.

Computed for all three confidence measures against all correctness signals:
- `ece_confidence`, `ece_full_sequence`, `ece_weighted` — against binary correctness
- `ece_judge_confidence`, `ece_judge_full_sequence`, `ece_judge_weighted` — against LLM judge score (good=1, medium=0.5, bad=0)
- `ece_rougeL_confidence`, `ece_rougeL_full_sequence`, `ece_rougeL_weighted` — against mean ROUGE-L score (NLG baseline)

---

### Discrimination — AUROC

Measures whether confidence *ranks* correct problems above incorrect ones,
independently of absolute calibration.

```
AUROC = 0.5  →  no better than random
AUROC = 1.0  →  perfect discrimination
```

Computed for all three confidence measures against all correctness signals:
- `auroc_confidence`, `auroc_full_sequence`, `auroc_weighted` — binary correct vs wrong
- `auroc_judge_confidence`, `auroc_judge_full_sequence`, `auroc_judge_weighted` — judge `good` vs `medium+bad`
- `auroc_rougeL_confidence`, `auroc_rougeL_full_sequence`, `auroc_rougeL_weighted` — ROUGE-L above threshold vs below (NLG baseline)

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

## LLM-as-judge rank (semantic correctness)

A post-processing step applied to `results.json` after inference. An external LLM
judge is given the problem and the model's majority answer and returns one of three
labels, stored as `judge_rank`:

| Label | Meaning |
|---|---|
| `good` | Correct final answer and sound reasoning |
| `medium` | Wrong final answer but correct approach / partial credit |
| `bad` | Wrong answer and flawed or irrelevant reasoning |

This is the primary reasoning-quality signal:

- **ECE**: `ece_judge_confidence`, `ece_judge_weighted` — calibration against judge rank
- **AUROC**: `auroc_judge_confidence`, `auroc_judge_weighted` — `good` vs `medium+bad`
- **Reliability diagrams**: mean judge score (good=1, medium=0.5, bad=0) per confidence bin

**Why three labels instead of binary**

The `medium` bucket captures the case binary correctness misses: the model sets up the
problem correctly and executes valid reasoning but makes an error at the final
extraction or simplification step. This is distinct from a fully wrong answer and
useful for separating prompt variants that improve reasoning quality from those that
only improve answer formatting.

**Running the judge**

```bash
python -m src.uq.llm_judge \
    --results results/mc_dropout/seed42/gsm8k/cot/results.json \
    --model gpt-4o-mini

# Anthropic judge
python -m src.uq.llm_judge \
    --results results/mc_dropout/seed42/gsm8k/cot/results.json \
    --provider anthropic --model claude-haiku-4-5-20251001
```

The module reads `results.json`, calls the judge API on the first raw output for each
problem, and writes `results_judged.json` (with `judge_rank` and `judge_score` added),
`summary_judged.json`, and plots under `judge_correctness/`.
Inference and judging are fully decoupled — only the JSON file is needed, not the model.

**Cost**

At 500 problems × 2 datasets × 3 prompts × 2 models = 6,000 judgments,
with ~630 input tokens and ~10 output tokens per call:

| Judge model | Approx. total cost |
|---|---|
| GPT-4o-mini | ~$0.90 |
| GPT-4o | ~$13 |
| Claude Haiku 4.5 | ~$5 |

---

## NLG baselines (BLEU / ROUGE / METEOR)

Standard sequence-overlap metrics computed between each MC Dropout pass output and
the full `reference_solution`. All three are retained as explicit weak baselines to
demonstrate their inadequacy for mathematical reasoning.

**Computation**

Each metric is computed independently for every MC Dropout pass output against the
`reference_solution` text, then averaged across all passes per problem.

For the MATH dataset, both the model output and `reference_solution` are
LaTeX-normalised before scoring: commands like `\frac`, `\sqrt`, `\cdot` are
expanded to plain text equivalents and whitespace is standardised. This removes
formatting differences that would otherwise penalise equivalent expressions.

| Metric | Library | What it measures |
|---|---|---|
| BLEU | `sacrebleu` (sentence-level) | Precision-weighted n-gram overlap (n=1..4) |
| ROUGE-1 / ROUGE-2 / ROUGE-L | `rouge-score` | Unigram / bigram / LCS recall-focused overlap |
| METEOR | `nltk` | Unigram F-score with stemming and synonym matching |

**Fields in `results.json` per problem**

| Key | What it stores |
|---|---|
| `bleu_scores` | List of BLEU scores, one per MC Dropout pass |
| `mean_bleu` | Mean across all passes |
| `rouge1_scores` | List of ROUGE-1 F1 scores, one per pass |
| `mean_rouge1` | Mean across all passes |
| `rouge2_scores` | List of ROUGE-2 F1 scores, one per pass |
| `mean_rouge2` | Mean across all passes |
| `rougeL_scores` | List of ROUGE-L F1 scores, one per pass |
| `mean_rougeL` | Mean across all passes |
| `meteor_scores` | List of METEOR scores, one per pass |
| `mean_meteor` | Mean across all passes |

**Why these are weak baselines for math**

n-gram overlap is sensitive to surface form, not mathematical meaning. Two solutions
that take different but equally valid paths, or express the same calculation with
different notation, will score low despite being semantically equivalent. ECE and
AUROC against these metrics are expected to be substantially worse than against the
LLM judge, making this a clear negative result that motivates the judge approach.

---

## Plots

Each run produces plots in subdirectories under `results/<method>/<label>/<source>/<prompt>/`:

```
results/mc_dropout/<label>/<source>/<prompt>/
  confidence/
    selective_prediction.png
  binary_correctness/
    reliability_confidence.png
    reliability_full_sequence_confidence.png
    reliability_weighted_mean_confidence.png
    roc_binary.png
  nlg_baselines/
    reliability_rougeL_confidence.png
    reliability_rougeL_full_sequence_confidence.png
    reliability_rougeL_weighted_mean_confidence.png
    roc_rougeL.png
  judge_correctness/               ← produced by src/uq/llm_judge.py
    reliability_judge_confidence.png
    reliability_judge_full_sequence_confidence.png
    reliability_judge_weighted_mean_confidence.png
    roc_judge.png
```

---

### `confidence/selective_prediction.png`

Accuracy vs coverage for both confidence measures on one plot.

**Coverage** = fraction of problems answered after filtering out everything below
a confidence threshold. As the threshold rises, fewer problems are included but
those included should be more accurate.

- Curve rising above the dashed baseline → confidence is a useful abstention signal
- Flat curve → confidence adds no value for selective answering

---

### `binary_correctness/reliability_<measure>.png` — Reliability diagrams (×3)

Problems are binned into 10 confidence buckets. Each bar shows the actual fraction
of correct problems in that bin. The diagonal is perfect calibration.
One diagram per confidence measure: `confidence`, `full_sequence_confidence`, `weighted_mean_confidence`.

---

### `nlg_baselines/reliability_rougeL_<measure>.png` — ROUGE-L reliability diagrams (×3)

Same structure but y-axis shows mean ROUGE-L score per bin.
ROUGE-L is used as the representative NLG metric in reliability diagrams since it
captures longest common subsequence rather than fixed-n n-gram overlap.
Expected to show flat or weakly correlated bars — the intended negative result.
One diagram per confidence measure.

---

### `judge_correctness/reliability_judge_<measure>.png` — Judge reliability diagrams (×3)

Same structure but y-axis shows mean judge score per bin (good=1, medium=0.5, bad=0).
Produced by `src/uq/llm_judge.py` after inference; not present in the initial run output.
One diagram per confidence measure.

---

### `judge_correctness/roc_judge.png` — ROC curve for judge rank

`good` is the positive class; `medium + bad` is negative. Overlays all three confidence
measures on one plot, same format as the other ROC curves.

---

## Reading `summary.json`

```json
{
  "n_problems": 500,
  "n_labeled":  500,
  "accuracy":   0.312,

  "mean_answer_confidence":          0.41,
  "mean_answer_entropy":             2.84,
  "mean_full_sequence_confidence":   0.87,
  "mean_weighted_confidence":        0.51,

  "overconf_high_conf_correct": 48,
  "overconf_high_conf_wrong":   62,
  "overconf_low_conf_correct":  108,
  "overconf_low_conf_wrong":    282,
  "overconf_rate":              0.563,

  "ece_confidence":    0.115,
  "ece_full_sequence": 0.19,
  "ece_weighted":      0.08,

  "auroc_confidence":    0.71,
  "auroc_full_sequence": 0.61,
  "auroc_weighted":      0.70,

  "mean_bleu":    0.08,
  "mean_rouge1":  0.31,
  "mean_rouge2":  0.14,
  "mean_rougeL":  0.27,
  "mean_meteor":  0.22,

  "ece_rougeL_confidence":    0.24,
  "ece_rougeL_full_sequence": 0.25,
  "ece_rougeL_weighted":      0.23,

  "auroc_rougeL_confidence":    0.54,
  "auroc_rougeL_full_sequence": 0.52,
  "auroc_rougeL_weighted":      0.53,

  // present only after src/uq/llm_judge.py has been run (summary_judged.json)
  "judge_rank_good":   201,
  "judge_rank_medium": 87,
  "judge_rank_bad":    212,

  "ece_judge_confidence":    0.11,
  "ece_judge_full_sequence": 0.17,
  "ece_judge_weighted":      0.09,

  "auroc_judge_confidence":    0.74,
  "auroc_judge_full_sequence": 0.64,
  "auroc_judge_weighted":      0.73
}
```

**What to look at:**

1. `overconf_rate` — fraction of high-confidence predictions that are wrong.
2. `ece_confidence` vs `ece_full_sequence` vs `ece_weighted` — does focusing token probability on `<<expr=result>>` and Final Answer tokens give better calibration than the unweighted baseline?
3. `auroc_confidence` vs `auroc_full_sequence` vs `auroc_weighted` — which confidence measure best discriminates correct from incorrect answers?
4. `ece_judge_*` vs `ece_confidence` — does the judge give better-calibrated ECE than binary correctness across all three confidence measures?
5. `auroc_judge_*` vs `auroc_rougeL_*` — does the judge signal give substantially higher AUROC than NLG overlap metrics, confirming they are inadequate for math reasoning?
6. `judge_rank_medium` count — how many problems had correct reasoning but wrong final answer? Compare across prompt variants to show whether structured prompting improves reasoning quality independently of answer extraction.
7. `mean_rougeL` / `mean_bleu` / `mean_meteor` across prompt variants — expected to be weakly correlated with judge rank, confirming the negative baseline result.
