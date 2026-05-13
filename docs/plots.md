# Plots

All plots are produced under `results/mc_dropout/<label>/<source>/<prompt>/` by
`scripts/python/run_uq_eval.py`. The `judge_correctness/` subdirectory is produced
separately by `src/uq/llm_judge.py` after inference.

For metric definitions (ECE, AUROC, correctness signals, confidence measures) see
`docs/metrics.md`. For how confidence is estimated see `docs/uncertainty_methods.md`.

---

## Directory layout

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

Where:
- `<label>` = `seed42` / `seed123` / ...
- `<source>` = `gsm8k` / `math`
- `<prompt>` = `zero_shot` / `cot` / `cot_step_by_step`

---

## Confidence measures

Two confidence measures appear across all plots:

| Name | Key in results.json | What it measures |
|---|---|---|
| Majority Vote | `confidence` | Fraction of 20 MC Dropout passes that agreed on the same answer |
| Weighted | `weighted_mean_confidence` | Geometric mean token probability, with `<<expr=result>>` result tokens at 10× and Final Answer span at 25× |

The weighted measure is the hypothesis: by down-weighting glue tokens and focusing on
arithmetic-critical tokens, it should correlate better with correctness than the
unweighted baseline.

---

## `confidence/selective_prediction.png`

**What it shows:** Accuracy vs coverage as the confidence threshold is swept from 1
down to 0. Both confidence measures are overlaid on one plot.

**How to read it:**
- **Coverage** (x-axis) = fraction of problems that pass the threshold. At coverage=1.0
  all problems are included; at coverage=0.1 only the top 10% by confidence are included.
- **Accuracy** (y-axis) = fraction correct among included problems.
- The dashed horizontal line = overall accuracy (no filtering).

**What good looks like:** A curve that rises steeply as coverage decreases — high-confidence
predictions are more often correct, so filtering by confidence improves accuracy. The
steeper and higher above the baseline, the more useful confidence is as an abstention signal.

**What bad looks like:** A flat curve at or near the baseline — confidence gives no
useful signal for deciding which problems to answer and which to abstain on.

**Comparison across prompts:** If cot_step_by_step produces a steeper curve than zero_shot,
structured prompting has improved the reliability of confidence as an abstention signal.

---

## Reliability diagrams

A reliability diagram bins problems by confidence score (10 equal-width bins) and
plots the actual correctness signal for each bin. The diagonal is perfect calibration.

### Reading a reliability diagram

```
y-axis (signal)
1.0 │           ╔═══╗
    │      ╔════╝   ║
    │  ╔═══╝        ║
0.5 │  ║            ╚═══╗
    │  ║                ╚═══╗
0.0 └──┴───┴───┴───┴───┴───┴──
    0.0  0.2  0.4  0.6  0.8  1.0
                x-axis (confidence)
```

- **Bars on the diagonal** → well calibrated: stated confidence matches actual signal.
- **Bars below the diagonal** → overconfident: the model claims more certainty than it has.
- **Bars above the diagonal** → underconfident: the model is more often right than it thinks.
- **ECE** (shown in the title) = weighted average gap between bars and diagonal. Lower is better.

There are two reliability diagrams per correctness signal — one for each confidence measure.

---

### `binary_correctness/reliability_confidence.png`
### `binary_correctness/reliability_weighted_mean_confidence.png`

**Y-axis:** Fraction of problems with the correct final answer in each confidence bin.

**What to look for:** Do higher-confidence bins also have higher accuracy? Are the bars
close to the diagonal? The ECE shown in the title is the primary calibration metric.

**Expected pattern:** Bars clustered at high confidence with some below the diagonal
(overconfidence). Step-by-step prompting should pull bars closer to the diagonal.

---

### `embedding_similarity/reliability_sim_confidence.png`
### `embedding_similarity/reliability_sim_weighted_mean_confidence.png`

**Y-axis:** Mean cosine similarity between the model output and `reference_solution`
(using `all-MiniLM-L6-v2`) in each confidence bin.

**What to look for:** Does higher confidence correspond to reasoning that is semantically
closer to the ground truth? ECE-sim measures calibration against this signal.

**Expected pattern:** A nearly flat curve regardless of confidence — embedding similarity
is a known weak signal for mathematical text. This plot is included as a baseline to
demonstrate that limitation explicitly. ECE-sim ≈ 0.28 in pilot experiments.

---

### `arithmetic_correctness/reliability_arith_confidence.png`
### `arithmetic_correctness/reliability_arith_weighted_mean_confidence.png`

**Y-axis:** Mean arithmetic step score in each confidence bin. The arithmetic step score
is the fraction of `<<expr=result>>` annotations in the model output where `eval(expr) ≈ result`.

**What to look for:** Does higher confidence correspond to more arithmetically correct
intermediate steps? ECE-arith measures calibration against this signal.

**Expected pattern:** A stronger signal than embedding similarity (ECE-arith < ECE-sim)
because the metric is computed on the exact tokens the model was instructed to produce.
Step-by-step prompting should raise the mean score across all bins.

---

### `judge_correctness/reliability_judge_confidence.png`
### `judge_correctness/reliability_judge_weighted_mean_confidence.png`

**Y-axis:** Mean judge score in each confidence bin, where `good=1.0`, `medium=0.5`,
`bad=0.0`. This is the numeric encoding of the LLM judge's three-way label.

**What to look for:** Does higher confidence correspond to semantically better answers?
ECE-judge measures calibration against this signal.

**Expected pattern:** The strongest reliability signal of the four — the judge
understands mathematical reasoning and awards partial credit. Bars should rise more
consistently with confidence than the other signals, giving lower ECE.

The `medium` bin is the most informative: problems the judge rates `medium` are cases
where the model reasoned correctly but extracted the final answer incorrectly. A higher
proportion of `medium` vs `bad` in the low-confidence region suggests the model
*knows* when it has made an extraction error (good epistemic awareness).

---

## ROC curves

ROC curves measure discrimination: can confidence rank correct problems above incorrect
ones, independently of absolute calibration? Both confidence measures are overlaid on
one plot for direct comparison.

```
TPR (True Positive Rate)
1.0 │              ╭────────
    │         ╭────╯
    │    ╭────╯
0.5 │────╯              ← random baseline (AUROC = 0.50)
    │
0.0 └──────────────────────
    0.0       0.5       1.0
         FPR (False Positive Rate)
```

- **AUROC = 1.0** → perfect discrimination (high confidence always means correct).
- **AUROC = 0.5** → no better than random (confidence is uninformative).
- **AUROC < 0.5** → confidence is inversely correlated with correctness (pathological).

The AUROC for each measure is shown in the legend.

---

### `binary_correctness/roc_binary.png`

**Positive class:** binary correct (`correct = True`).
**Negative class:** binary wrong.

Primary discrimination metric — how well does confidence separate right from wrong answers?

---

### `embedding_similarity/roc_sim.png`

**Positive class:** `similarity_rank == "high"` (cosine similarity > 0.7).
**Negative class:** `similarity_rank` in `"low"` or `"medium"`.

Expected to show AUROC near 0.5 (baseline; similarity rank is a weak signal).

---

### `arithmetic_correctness/roc_arith.png`

**Positive class:** `arith_step_score >= 0.8`.
**Negative class:** `arith_step_score < 0.8`.

Does high confidence predict that the model's arithmetic annotations are internally
consistent? Expected to be stronger than the similarity ROC.

---

### `judge_correctness/roc_judge.png`

**Positive class:** `judge_rank == "good"`.
**Negative class:** `judge_rank` in `"medium"` or `"bad"`.

Expected to be the strongest ROC signal — the judge makes a holistic correctness
judgement. Comparing AUROC-judge vs AUROC-binary reveals whether the judge signal
provides additional discriminative power beyond exact answer matching.

---

## Comparing plots across experimental cells

The 2 × 3 factorial design (2 models × 3 prompts) on 2 datasets produces
12 sets of plots. The key comparisons:

| Comparison | What to look at |
|---|---|
| zero_shot vs cot vs cot_step_by_step | Reliability diagrams: do bars move closer to diagonal? |
| zero_shot vs cot vs cot_step_by_step | `mean_arith_step_score` and ROC-arith: does structured prompting improve intermediate arithmetic? |
| 1B vs 8B | Overall accuracy and ECE: does scale improve calibration independently of prompting? |
| GSM8K vs MATH | Drop in accuracy and AUROC from in-distribution to OOD |
| majority-vote vs weighted confidence | Which measure gives lower ECE and higher AUROC consistently? |
| judge_rank_medium across prompts | Does step-by-step prompting reduce `medium` → `bad` transitions (improve answer extraction)? |
