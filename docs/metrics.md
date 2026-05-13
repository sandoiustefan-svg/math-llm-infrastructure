# UQ Metrics

All metrics are computed in `src/uq/metrics.py` and aggregated by `summarise()`.
Per-problem raw values are saved to `results.json`; aggregate values to `summary.json`.

For how epistemic uncertainty and confidence are estimated (MC Dropout, majority-vote,
weighted/unweighted geometric mean, answer entropy) see `docs/uncertainty_methods.md`.

## Correctness signals

Four complementary correctness signals are used per problem:

- **Binary correctness** — `answers_are_equal()` string match on the extracted final
  answer. Fast and unambiguous, but ignores the reasoning process.
- **Embedding similarity** — cosine similarity between the full raw output and the
  `reference_solution`, using `all-MiniLM-L6-v2`. Retained as a baseline; known to
  be a weak signal for mathematical text.
- **Arithmetic step correctness** — fraction of `<<expr=result>>` annotations in the
  model's output that evaluate correctly: `eval(expr) ≈ result` within a numeric
  tolerance. Measures whether the model's stated arithmetic is internally consistent,
  independent of the reference solution. See [Arithmetic step correctness](#arithmetic-step-correctness) below.
- **LLM-as-judge rank** — a post-processing step that assigns `good / medium / bad`
  to the majority answer using an external LLM judge. Captures semantic correctness
  and partial credit that binary correctness misses. See [LLM-as-judge rank](#llm-as-judge-rank-semantic-correctness) below.

## Central research question

**How well does a model's confidence align with the actual correctness of its outputs,
and does structured prompting improve that alignment?**

Metrics are organised around two ways of answering that question — calibration and
discrimination — plus a direct analysis of the overconfidence failure mode.
All four correctness signals are used against both confidence measures
(`confidence` and `weighted_mean_confidence`); see `docs/uncertainty_methods.md`
for how those measures are computed.

---

## Confidence-correctness alignment metrics

### Calibration — Expected Calibration Error (ECE)

Measures whether stated confidence equals empirical accuracy.
The confidence range is split into 10 bins:

```
ECE = Σ_b (|b| / N) * |accuracy(b) - mean_confidence(b)|
```

ECE = 0 → perfect calibration. ECE > 0.15 → poor calibration.

Computed for both confidence measures against **all four correctness signals**:
- `ece_confidence`, `ece_weighted` — against binary correctness
- `ece_sim_confidence`, `ece_sim_weighted` — against mean embedding similarity
- `ece_arith_confidence`, `ece_arith_weighted` — against arithmetic step correctness
- `ece_judge_confidence`, `ece_judge_weighted` — against LLM judge score (good=1, medium=0.5, bad=0)

---

### Discrimination — AUROC

Measures whether confidence *ranks* correct problems above incorrect ones,
independently of absolute calibration.

```
AUROC = 0.5  →  no better than random
AUROC = 1.0  →  perfect discrimination
```

Computed for both confidence measures against **all four correctness signals**:
- `auroc_confidence`, `auroc_weighted` — binary correct vs wrong
- `auroc_sim_confidence`, `auroc_sim_weighted` — high similarity rank vs low+medium
- `auroc_arith_confidence`, `auroc_arith_weighted` — high arithmetic correctness vs low
- `auroc_judge_confidence`, `auroc_judge_weighted` — judge `good` vs `medium+bad`

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

## Arithmetic step correctness

The arithmetic step metric has two components that can be used independently or together:

1. **Internal consistency** — always active. Verifies the model's own arithmetic.
2. **Reference alignment** — optional augmentation. Compares the model's intermediate
   values against the ground truth's. Requires `<<expr=result>>` annotations in the
   `reference_solution` field. Currently applicable to **GSM8K only**.

### Internal consistency

Each `<<expr=result>>` annotation in the model's output is extracted by regex and
verified independently:

```
eval(expr) ≈ result   (within tolerance 1e-6 for floats, exact for integers)
```

This check requires no reference solution — it is purely about whether the model's
own stated arithmetic is self-consistent. A model can score 1.0 here while still
reasoning incorrectly (e.g. wrong problem setup, correct arithmetic on wrong values).

### Reference alignment (optional augmentation)

When enabled, the `<<expr=result>>` annotations in the **model output** are compared
against the `<<expr=result>>` annotations in the **ground truth `reference_solution`**.
Both sides use the same pattern — the result values (the number after `=` and before
`>>`) are extracted from each and compared as sets.

**Concrete example**

```
Ground truth reference_solution:
  "Natalia sold 48/2 = <<48/2=24>>24 clips in May.
   She sold 48+24 = <<48+24=72>>72 clips in total."

  → ref_values = {24, 72}

Model output (correct path):
  "Half of 48 is 48/2 = <<48/2=24>>24.
   Total: 48+24 = <<48+24=72>>72."

  → model_values = {24, 72}
  → ref_alignment_score = |{24,72} ∩ {24,72}| / |{24,72}| = 2/2 = 1.0

Model output (wrong path):
  "Natalia sold 48 clips in April and 48 more in May.
   Total: 48+48 = <<48+48=96>>96."

  → model_values = {96}
  → ref_alignment_score = |{24,72} ∩ {96}| / |{24,72}| = 0/2 = 0.0
```

**Alignment strategy — value matching (no step ordering required)**

Rather than aligning steps positionally (which requires a sequence alignment
algorithm and is brittle when model and reference take different paths), reference
alignment uses **set-based value matching**:

```
ref_values   = set of result values from reference_solution <<expr=result>> annotations
model_values = set of result values from model output <<expr=result>> annotations

ref_alignment_score = |ref_values ∩ model_values| / |ref_values|
```

This asks: *what fraction of the reference's key intermediate checkpoints does the
model also arrive at?* A model that takes a different but valid path and arrives at
the same intermediate numbers still scores high. A model that sets up the problem
incorrectly (different numbers entirely) scores low even if its arithmetic is clean.
This is an acknowledged limitation on problems with multiple valid solution paths,
and less of an issue on GSM8K where problems typically have one natural path.

**Activation**

Reference alignment is disabled by default and activated per-run by passing
`reference_alignment=True` to the evaluator. When the `reference_solution` field
is absent or contains no `<<expr=result>>` annotations, the metric silently falls
back to internal consistency only and records `ref_alignment_score: null`.

### Fields in `results.json`

| Key | Always present | What it stores |
|---|---|---|
| `arith_steps_total` | ✓ | Number of `<<expr=result>>` annotations in model output |
| `arith_steps_correct` | ✓ | Annotations where `eval(expr) ≈ result` |
| `arith_step_score` | ✓ | `arith_steps_correct / arith_steps_total` (NaN if 0) |
| `arith_steps_detail` | ✓ | List of `{expr, claimed, actual, correct}` per annotation |
| `ref_values_total` | when enabled | Number of intermediate values in reference solution |
| `ref_values_matched` | when enabled | Reference values also found in model output |
| `ref_alignment_score` | when enabled | `ref_values_matched / ref_values_total` (null if unavailable) |

**How to read internal consistency alongside binary correctness**

```
                     Binary correct        Binary wrong
High arith score │  correct reasoning   │  correct steps, wrong final answer
                 │  correct answer  ✓   │  (setup/extraction error)
─────────────────┼──────────────────────┼───────────────────────────────────
Low arith score  │  correct answer      │  wrong reasoning
                 │  despite errors      │  wrong answer  ✗
                 │  (lucky guess)       │
```

**How to read reference alignment alongside internal consistency**

```
                         High ref alignment    Low ref alignment
High internal score  │  correct path        │  different path,
                     │  correct arithmetic  │  correct arithmetic
─────────────────────┼──────────────────────┼──────────────────────
Low internal score   │  right path,         │  wrong path,
                     │  arithmetic errors   │  arithmetic errors
```

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

This is structurally identical to `similarity_rank` and slots into the same metric
infrastructure:

- **ECE**: `ece_judge_confidence`, `ece_judge_weighted` — calibration against judge rank
- **AUROC**: `auroc_judge_confidence`, `auroc_judge_weighted` — `good` vs `medium+bad`
- **Reliability diagrams**: mean judge score (good=1, medium=0.5, bad=0) per confidence bin

**Why three labels instead of binary**

The `medium` bucket captures the case binary correctness misses: the model sets up the
problem correctly and executes valid arithmetic but makes an error at the final
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

**Comparison with embedding similarity**

| | Embedding similarity | LLM-as-judge rank |
|---|---|---|
| Signal strength for math | Weak (ECE-sim ≈ 0.28) | Strong (semantic understanding) |
| Cost | Free (local model) | ~$1 API cost |
| Partial credit | No | Yes (`medium` label) |
| Speed | Fast (batch encode) | Slow (serial API calls) |
| Interpretability | Opaque cosine score | Human-readable label |

Embedding similarity is retained as a baseline to explicitly demonstrate its
limitations relative to the judge rank.

---

## Embedding similarity (process-level correctness)

Binary correctness only checks the final answer string. Embedding similarity
captures whether the reasoning process itself is on the right track.

`all-MiniLM-L6-v2` encodes both the full raw output and the `reference_solution`
(complete step-by-step reasoning from the dataset) into a 384-dimensional vector.
Cosine similarity is computed for all 20 MC Dropout passes.

```
sim(raw, reference_solution) = dot(embed(raw), embed(reference_solution))   # L2-normalised
```

Four fields per problem in `results.json`:

| Key | What it stores |
|---|---|
| `raw_similarities` | List of 20 cosine similarity scores (one per pass) |
| `mean_raw_similarity` | Mean across all 20 passes |
| `std_raw_similarity` | Std across passes — high value means reasoning varies between passes |
| `similarity_rank` | `low` (< 0.3) / `medium` (0.3–0.7) / `high` (> 0.7) |

Note: embedding similarity is known to be a weak correctness signal for mathematical
text — semantically similar solutions can be arithmetically wrong, and vice versa.
It is retained as a baseline to explicitly demonstrate this limitation.

---

## Plots

Each run produces plots in subdirectories under `results/<method>/<label>/<source>/<prompt>/`:

```
results/mc_dropout/<label>/<source>/<prompt>/
  confidence/
    selective_prediction.png
  binary_correctness/
    reliability_confidence.png
    reliability_weighted_mean_confidence.png
    roc_binary.png
  embedding_similarity/
    reliability_sim_confidence.png
    reliability_sim_weighted_mean_confidence.png
    roc_sim.png
  arithmetic_correctness/
    reliability_arith_confidence.png
    reliability_arith_weighted_mean_confidence.png
    roc_arith.png
  judge_correctness/               ← produced by src/uq/llm_judge.py
    reliability_judge_confidence.png
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

### `binary_correctness/reliability_<measure>.png` — Reliability diagrams (×2)

Problems are binned into 10 confidence buckets. Each bar shows the actual fraction
of correct problems in that bin. The diagonal is perfect calibration.

---

### `embedding_similarity/reliability_sim_<measure>.png` — Similarity reliability diagrams (×2)

Same structure but y-axis shows mean embedding similarity per bin.
Retained as a baseline; expected to show weak calibration signal.

---

### `arithmetic_correctness/reliability_arith_<measure>.png` — Arithmetic reliability diagrams (×2)

Same structure but y-axis shows mean arithmetic step score per bin.
Answers: does higher confidence correspond to more arithmetically correct intermediate steps?

---

### `judge_correctness/reliability_judge_<measure>.png` — Judge reliability diagrams (×2)

Same structure but y-axis shows mean judge score per bin (good=1, medium=0.5, bad=0).
Produced by `src/uq/llm_judge.py` after inference; not present in the initial run output.

---

### `judge_correctness/roc_judge.png` — ROC curve for judge rank

`good` is the positive class; `medium + bad` is negative. Overlays both confidence
measures on one plot, same format as the other ROC curves.

---

## Reading `summary.json`

```json
{
  "n_problems": 500,
  "n_labeled":  500,
  "accuracy":   0.312,

  "mean_answer_confidence":   0.41,
  "mean_answer_entropy":      2.84,
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

  "mean_raw_similarity":     0.21,
  "mean_std_raw_similarity": 0.04,
  "sim_rank_low":    312,
  "sim_rank_medium": 188,
  "sim_rank_high":   0,

  "ece_sim_confidence": 0.28,
  "ece_sim_weighted":   0.27,

  "auroc_sim_confidence": 0.52,
  "auroc_sim_weighted":   0.51,

  "mean_arith_step_score":    0.74,
  "arith_steps_total":        3241,
  "arith_steps_correct":      2398,

  "ece_arith_confidence": 0.09,
  "ece_arith_weighted":   0.07,

  "auroc_arith_confidence": 0.68,
  "auroc_arith_weighted":   0.66,

  "mean_ref_alignment_score": 0.61,
  "ref_values_total":         1847,
  "ref_values_matched":       1127,

  // present only after src/uq/llm_judge.py has been run (summary_judged.json)
  "judge_rank_good":   201,
  "judge_rank_medium": 87,
  "judge_rank_bad":    212,

  "ece_judge_confidence": 0.11,
  "ece_judge_weighted":   0.09,

  "auroc_judge_confidence": 0.74,
  "auroc_judge_weighted":   0.73
}
```

**What to look at:**

1. `overconf_rate` — fraction of high-confidence predictions that are wrong.
2. `auroc_confidence` vs `auroc_weighted` — which measure better discriminates correct from incorrect?
3. `ece_arith_*` vs `ece_confidence` / `ece_sim_*` — does arithmetic step score give better-calibrated ECE than embedding similarity?
4. Compare `mean_arith_step_score` across prompt variants — does step-by-step prompting improve intermediate arithmetic correctness?
5. `auroc_arith_*` — does confidence rank problems with correct intermediate steps above those with errors?
6. `mean_ref_alignment_score` vs `mean_arith_step_score` (GSM8K only) — gap between these two reveals how often the model does correct arithmetic on the wrong reasoning path.
7. `judge_rank_medium` count — how many problems had correct reasoning but wrong final answer? Compares directly across prompt variants to show whether structured prompting improves reasoning quality independently of answer extraction.
8. `auroc_judge_*` vs `auroc_confidence` — does the judge signal give higher AUROC than binary correctness, indicating it is a more informative correctness signal for calibration analysis?
