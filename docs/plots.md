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

Where:
- `<label>` = `seed42` / `seed123` / ...
- `<source>` = `gsm8k` / `math`
- `<prompt>` = `zero_shot` / `cot` / `cot_step_by_step`

---

## Confidence measures

Three confidence measures appear across all plots:

| Name | Key in results.json | What it measures |
|---|---|---|
| Majority Vote | `confidence` | Fraction of 20 MC Dropout passes that agreed on the same answer — answer-level signal |
| Unweighted | `full_sequence_mean_confidence` | Geometric mean of all token probabilities equally weighted — token-level baseline, inflated by glue tokens |
| Weighted | `weighted_mean_confidence` | Geometric mean token probability, with `<<expr=result>>` result tokens at 10× and Final Answer span at 25× |

The progression: `full_sequence_mean_confidence` is the naive token-level baseline (expected to
be poorly calibrated due to glue token inflation); `confidence` avoids that by operating at
answer level; `weighted_mean_confidence` is the hypothesis that focusing token probability
on arithmetic-critical tokens beats both.

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

There are three reliability diagrams per correctness signal — one for each confidence measure.

---

### `binary_correctness/reliability_confidence.png`
### `binary_correctness/reliability_weighted_mean_confidence.png`

**Y-axis:** Fraction of problems with the correct final answer in each confidence bin.

**What to look for:** Do higher-confidence bins also have higher accuracy? Are the bars
close to the diagonal? The ECE shown in the title is the primary calibration metric.

**Expected pattern:** Bars clustered at high confidence with some below the diagonal
(overconfidence). Step-by-step prompting should pull bars closer to the diagonal.

---

### `nlg_baselines/reliability_rougeL_confidence.png`
### `nlg_baselines/reliability_rougeL_weighted_mean_confidence.png`

**Y-axis:** Mean ROUGE-L F1 score between the model output and `reference_solution`
in each confidence bin. ROUGE-L is used as the representative NLG metric (LCS-based,
less sensitive to exact n-gram matches than ROUGE-1/2).

**What to look for:** Does higher confidence correspond to outputs that overlap more
with the reference solution text? ECE-rougeL measures calibration against this signal.

**Expected pattern:** A nearly flat curve regardless of confidence — n-gram overlap
is a known weak signal for mathematical reasoning. This plot is the negative baseline:
it is included to explicitly demonstrate that surface-form similarity carries little
calibration information. ECE-rougeL is expected to be substantially higher than
ECE-judge.

---

### `judge_correctness/reliability_judge_confidence.png`
### `judge_correctness/reliability_judge_weighted_mean_confidence.png`

**Y-axis:** Mean judge score in each confidence bin, where `good=1.0`, `medium=0.5`,
`bad=0.0`. This is the numeric encoding of the LLM judge's three-way label.

**What to look for:** Does higher confidence correspond to semantically better answers?
ECE-judge measures calibration against this signal.

**Expected pattern:** The strongest reliability signal — the judge understands
mathematical reasoning and awards partial credit. Bars should rise more consistently
with confidence than the NLG baselines, giving lower ECE.

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

All three confidence measures are overlaid on one plot. The AUROC for each is shown in the legend.

---

### `binary_correctness/roc_binary.png`

**Positive class:** binary correct (`correct = True`).
**Negative class:** binary wrong.

Primary discrimination metric — how well does confidence separate right from wrong answers?

---

### `nlg_baselines/roc_rougeL.png`

**Positive class:** `mean_rougeL >= 0.4` (above-median overlap with reference).
**Negative class:** `mean_rougeL < 0.4`.

Expected to show AUROC near 0.5 — surface-form overlap with the reference solution
is a poor proxy for correctness, so confidence should not discriminate it well.
Included as the NLG negative baseline; compare directly against `roc_judge.png`.

---

### `judge_correctness/roc_judge.png`

**Positive class:** `judge_rank == "good"`.
**Negative class:** `judge_rank` in `"medium"` or `"bad"`.

Expected to be the strongest ROC signal — the judge makes a holistic correctness
judgement. Comparing AUROC-judge vs AUROC-rougeL reveals the discriminative gap
between semantic understanding and surface-form overlap for math reasoning.

---

## Comparing plots across experimental cells

The 2 × 3 factorial design (2 models × 3 prompts) on 2 datasets produces
12 sets of plots. The key comparisons:

| Comparison | What to look at |
|---|---|
| zero_shot vs cot vs cot_step_by_step | Reliability diagrams: do bars move closer to diagonal? |
| zero_shot vs cot vs cot_step_by_step | `judge_rank_medium` count: does structured prompting reduce reasoning errors? |
| 1B vs 8B | Overall accuracy and ECE: does scale improve calibration independently of prompting? |
| GSM8K vs MATH | Drop in accuracy and AUROC from in-distribution to OOD |
| majority-vote vs weighted confidence | Which measure gives lower ECE and higher AUROC consistently? |
| judge AUROC vs rougeL AUROC | Quantifies how much better semantic understanding is than n-gram overlap for calibration |
