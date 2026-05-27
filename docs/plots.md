# Plots

All plots are produced under `results/mc_dropout/<model>/<seed>/<source>/<prompt>/` by
`scripts/python/run_uq_eval.py`. The `judge_correctness/` subdirectory is produced
separately by `src/uq/llm_judge.py` after inference.

For metric definitions (ECE, AUROC, correctness signals, confidence scores, uncertainty
measures) see `docs/metrics.md`. For how confidence and uncertainty are estimated see
`docs/uncertainty_methods.md`.

---

## Directory layout

```
results/mc_dropout/<model>/<seed>/<source>/<prompt>/
  confidence/
    selective_prediction.png
  binary_correctness/
    reliability_confidence.png
    reliability_consistency_rate.png
    roc_binary.png
  nlg_baselines/
    reliability_rougeL_confidence.png
    reliability_rougeL_consistency_rate.png
    roc_rougeL.png
  judge_correctness/               ← produced by src/uq/llm_judge.py
    reliability_judge_confidence.png
    reliability_judge_consistency_rate.png
    roc_judge.png
```

Where:
- `<model>` = `1b` / `8b`
- `<seed>` = `seed42` / ...
- `<source>` = `gsm8k` / `math`
- `<prompt>` = `zero_shot_aligned` / `cot`

---

## Confidence scores

Two confidence scores appear across all plots:

| Name | Key in results.json | What it measures |
|---|---|---|
| Majority Vote | `confidence` | Fraction of 20 MC Dropout passes agreeing on the same final answer |
| Consistency Rate | `consistency_rate` | Σ p(a)² — considers the full answer distribution, not just the top answer |

Uncertainty measures (entropy, n_unique_answers, std_log_prob, std_numeric_span_log_prob)
appear as AUROC numbers in `summary.json` but do not produce separate plots — they are
uncertainty signals, not confidence scores, so reliability diagrams do not apply.

---

## `confidence/selective_prediction.png`

**What it shows:** Accuracy vs coverage as the confidence threshold is swept from 1
down to 0. Both confidence scores are overlaid on one plot.

**How to read it:**
- **Coverage** (x-axis) = fraction of problems that pass the threshold. At coverage=1.0
  all problems are included; at coverage=0.1 only the top 10% by confidence are included.
- **Accuracy** (y-axis) = fraction correct among included problems.
- The dashed horizontal line = overall accuracy (no filtering).

**What good looks like:** A curve that rises steeply as coverage decreases — high-confidence
predictions are more often correct, so filtering by confidence improves accuracy.

**What bad looks like:** A flat curve at the baseline — confidence gives no useful signal
for deciding which problems to answer and which to abstain on.

---

## Reliability diagrams

A reliability diagram bins problems by confidence score (10 equal-width bins) and
plots the actual correctness signal for each bin. The diagonal is perfect calibration.

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

- **Bars on the diagonal** → well calibrated.
- **Bars below the diagonal** → overconfident.
- **Bars above the diagonal** → underconfident.
- **ECE** (shown in the title) = weighted average gap between bars and diagonal. Lower is better.

---

### `binary_correctness/reliability_confidence.png`
### `binary_correctness/reliability_consistency_rate.png`

**Y-axis:** Fraction of problems with the correct final answer in each confidence bin.

**What to look for:** Do higher-confidence bins have higher accuracy? Are bars close to
the diagonal? The ECE in the title is the primary calibration metric.

**Expected pattern:** Bars clustered at high confidence with some below the diagonal
(overconfidence). The 8B model should show better calibration than 1B. The `cot` prompt
should show worse calibration than `zero_shot_aligned` due to distribution shift.

---

### `nlg_baselines/reliability_rougeL_confidence.png`
### `nlg_baselines/reliability_rougeL_consistency_rate.png`

**Y-axis:** Mean ROUGE-L F1 score between the model output and `reference_solution`
in each confidence bin.

**Expected pattern:** A nearly flat curve — n-gram overlap is a weak signal for
mathematical reasoning. Included as an explicit negative baseline.

---

### `judge_correctness/reliability_judge_confidence.png`
### `judge_correctness/reliability_judge_consistency_rate.png`

**Y-axis:** Mean judge score per bin (good=1.0, medium=0.5, bad=0.0).

**Expected pattern:** The strongest reliability signal. Bars should rise more consistently
with confidence than NLG baselines, giving lower ECE.

---

## ROC curves

ROC curves measure discrimination: can confidence rank correct problems above incorrect
ones? Both confidence scores are overlaid on one plot for direct comparison.

```
TPR
1.0 │              ╭────────
    │         ╭────╯
    │    ╭────╯
0.5 │────╯              ← random baseline (AUROC = 0.50)
0.0 └──────────────────────
    0.0       0.5       1.0  FPR
```

---

### `binary_correctness/roc_binary.png`

**Positive class:** binary correct. Primary discrimination plot.

---

### `nlg_baselines/roc_rougeL.png`

**Positive class:** `mean_rougeL >= 0.4`. Expected AUROC near 0.5 — negative baseline.

---

### `judge_correctness/roc_judge.png`

**Positive class:** `judge_rank == "good"`. Expected to be the strongest ROC signal.

---

## Key comparisons across experimental cells

| Comparison | What to look at |
|---|---|
| zero_shot_aligned vs cot | ECE and AUROC: does OOD prompt degrade calibration? |
| 1B vs 8B | ECE: does scale improve epistemic UQ reliability? |
| GSM8K vs MATH | AUROC drop: does domain shift degrade discrimination? |
| majority-vote vs consistency_rate | Which confidence score is better calibrated? |
| auroc_confidence vs auroc_entropy vs auroc_std_log_prob | Answer-level vs token-level epistemic signals |
| ECE binary vs ECE judge | Does semantic correctness show better-calibrated confidence? |
