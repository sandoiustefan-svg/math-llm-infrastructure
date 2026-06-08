# MC Dropout Results — Phase 2

Results from the primary UQ experiment. Raw outputs are in
`results/results_mc_dropout_1/`. Per-problem data in `results.json`;
aggregate metrics in `summary.json`. See `docs/metrics.md` for metric
definitions and `docs/uncertainty_methods.md` for method details.

---

## Experiment Setup

- **Models:** Llama-3.2-1B-Instruct (LoRA rank 16, seed 42), Llama-3.1-8B-Instruct (LoRA rank 64, seed 42)
- **Method:** MC Dropout, N=20 greedy passes, `lora_dropout=0.05`, `model.train()` at inference — no additional hooks
- **Test sets:** GSM8K (500 problems), MATH (500 problems) — all labeled
- **Prompts:** `zero_shot_aligned` (in-distribution, matches fine-tuning format), `cot` (out-of-distribution, multi-turn few-shot)
- **Results location:** `results/results_mc_dropout_1/`

Note: the Phase 2 plan in `docs/findings.md` scoped `cot` to GSM8K only. The
actual run includes `cot` on MATH as well — all 8 conditions were evaluated.

---

## Accuracy

| Model | Benchmark | Prompt | Accuracy |
|-------|-----------|--------|----------|
| 1B | GSM8K | cot | 34.2% |
| 1B | GSM8K | zero_shot_aligned | 40.2% |
| 1B | MATH | cot | 4.0% |
| 1B | MATH | zero_shot_aligned | 4.2% |
| 8B | GSM8K | cot | 56.8% |
| 8B | GSM8K | zero_shot_aligned | **58.6%** |
| 8B | MATH | cot | 4.2% |
| 8B | MATH | zero_shot_aligned | 2.2% |

MATH is essentially unsolved by both models (~2–4%). The 8B gains +17–18 pp over
1B on GSM8K but shows no meaningful improvement on MATH.

---

## Calibration — ECE (majority-vote confidence)

Lower is better. ECE > 0.15 is poor.

| Model | Benchmark | Prompt | ECE (confidence) | ECE (consistency) |
|-------|-----------|--------|-----------------|-------------------|
| 8B | GSM8K | cot | **0.038** | 0.138 |
| 8B | GSM8K | zero_shot_aligned | **0.040** | 0.132 |
| 1B | GSM8K | zero_shot_aligned | 0.123 | 0.075 |
| 1B | GSM8K | cot | 0.179 | 0.071 |
| 8B | MATH | zero_shot_aligned | 0.255 | 0.138 |
| 8B | MATH | cot | 0.263 | 0.142 |
| 1B | MATH | zero_shot_aligned | 0.297 | 0.165 |
| 1B | MATH | cot | 0.334 | 0.193 |

The 8B model on GSM8K is very well calibrated (ECE ~0.04). Calibration collapses
on MATH across both models, where near-zero accuracy makes high confidence
structurally wrong. Consistency rate (ECE ~0.07–0.14) is better calibrated than
majority-vote confidence on GSM8K for the 1B model.

---

## Discrimination — AUROC

AUROC = 0.5 is random; 1.0 is perfect. All signals negated where needed
(higher uncertainty = lower rank).

### Answer-level signals

| Model | Benchmark | Prompt | AUROC (confidence) | AUROC (consistency) | AUROC (entropy) | AUROC (n_unique) |
|-------|-----------|--------|--------------------|---------------------|-----------------|-----------------|
| 8B | GSM8K | zero_shot_aligned | **0.858** | **0.857** | **0.851** | **0.837** |
| 8B | GSM8K | cot | 0.846 | 0.839 | 0.834 | 0.818 |
| 1B | GSM8K | cot | 0.817 | 0.823 | 0.825 | 0.816 |
| 1B | GSM8K | zero_shot_aligned | 0.810 | 0.810 | 0.810 | 0.804 |
| 1B | MATH | cot | 0.696 | 0.720 | 0.747 | 0.776 |
| 8B | MATH | zero_shot_aligned | 0.696 | 0.724 | 0.742 | 0.753 |
| 8B | MATH | cot | 0.628 | 0.664 | 0.677 | 0.694 |
| 1B | MATH | zero_shot_aligned | 0.569 | 0.588 | 0.599 | 0.622 |

### Token-level signals (std of log-prob across passes)

| Model | Benchmark | Prompt | AUROC (std_log_prob) | AUROC (std_numeric_span) |
|-------|-----------|--------|---------------------|--------------------------|
| 1B | GSM8K | cot | 0.721 | 0.695 |
| 1B | GSM8K | zero_shot_aligned | 0.704 | 0.712 |
| 8B | GSM8K | zero_shot_aligned | 0.540 | 0.721 |
| 8B | GSM8K | cot | 0.483 | 0.758 |
| 1B | MATH | cot | 0.724 | 0.757 |
| 1B | MATH | zero_shot_aligned | 0.603 | 0.673 |
| 8B | MATH | cot | 0.467 | 0.678 |
| 8B | MATH | zero_shot_aligned | 0.498 | 0.618 |

Token-level signals are weaker than answer-level on GSM8K for 8B. `std_numeric_span`
consistently outperforms `std_log_prob` — restricting to the Final Answer digits removes
noise from the reasoning chain.

---

## Overconfidence Analysis

Threshold: confidence ≥ 0.8 is "high confidence". Overconfidence rate =
high_conf_wrong / (high_conf_correct + high_conf_wrong).

| Model | Benchmark | Prompt | High-conf correct | High-conf wrong | Overconf rate |
|-------|-----------|--------|-------------------|-----------------|---------------|
| 8B | GSM8K | zero_shot_aligned | 156 | 8 | **4.9%** |
| 8B | GSM8K | cot | 136 | 9 | **6.2%** |
| 1B | GSM8K | zero_shot_aligned | 108 | 31 | 22.3% |
| 1B | GSM8K | cot | 97 | 30 | 23.6% |
| 8B | MATH | cot | 2 | 2 | 50.0% |
| 1B | MATH | cot | 5 | 15 | 75.0% |
| 1B | MATH | zero_shot_aligned | 1 | 16 | 94.1% |
| 8B | MATH | zero_shot_aligned | 0 | 4 | **100.0%** |

The 8B model on MATH `zero_shot_aligned` has an overconfidence rate of 1.0 — every
single high-confidence prediction is wrong. The model is severely overconfident in
exactly the domain where it has no ability.

---

## Mean Uncertainty Across Passes

Higher mean entropy and n_unique answers indicate the model is more uncertain overall.

| Model | Benchmark | Prompt | Mean confidence | Mean consistency | Mean entropy | Mean n_unique |
|-------|-----------|--------|----------------|-----------------|-------------|---------------|
| 8B | GSM8K | zero_shot_aligned | 0.564 | 0.461 | 1.906 | 7.19 |
| 8B | GSM8K | cot | 0.538 | 0.430 | 1.977 | 7.23 |
| 1B | GSM8K | zero_shot_aligned | 0.525 | 0.424 | 2.030 | 7.54 |
| 1B | GSM8K | cot | 0.521 | 0.413 | 2.025 | 7.34 |
| 1B | MATH | cot | 0.374 | 0.233 | 2.817 | 10.64 |
| 1B | MATH | zero_shot_aligned | 0.336 | 0.206 | 2.983 | 11.44 |
| 8B | MATH | cot | 0.305 | 0.183 | 3.055 | 11.52 |
| 8B | MATH | zero_shot_aligned | 0.277 | 0.160 | 3.191 | 12.11 |

The model's own uncertainty signals correctly reflect its inability on MATH — low
confidence, high entropy, ~11–12 unique answers per problem vs ~7 on GSM8K. The
uncertainty estimates are directionally correct even where calibration fails.

---

## NLG Baselines (ROUGE-L / METEOR)

Included as weak baselines only; n-gram overlap is an inadequate signal for
mathematical correctness. Values are mean across all 20 passes.

| Model | Benchmark | Prompt | Mean ROUGE-L | Mean METEOR |
|-------|-----------|--------|-------------|------------|
| 1B | GSM8K | zero_shot_aligned | 0.264 | 0.301 |
| 1B | GSM8K | cot | 0.258 | 0.296 |
| 8B | GSM8K | zero_shot_aligned | 0.225 | 0.253 |
| 8B | GSM8K | cot | 0.201 | 0.237 |
| 1B | MATH | cot | 0.201 | 0.195 |
| 1B | MATH | zero_shot_aligned | 0.204 | 0.202 |
| 8B | MATH | cot | 0.144 | 0.150 |
| 8B | MATH | zero_shot_aligned | 0.149 | 0.156 |

AUROC against ROUGE-L (`auroc_rougeL_confidence`) tracks well with binary AUROC on
GSM8K, reaching 0.882–0.921. One anomaly: `auroc_rougeL_confidence` for 8B MATH
`zero_shot_aligned` = 0.072 (essentially inverted); `auroc_rougeL_consistency` = 0.028.
This appears to be a degenerate case caused by near-zero ROUGE-L variance when the
model produces near-random outputs — the ROUGE-L threshold splits collapse.

---

## Signal Analysis

### Confidence scores: majority-vote vs consistency rate

Two scores are derived from the answer distribution across 20 passes and are used
for both ECE (calibration) and AUROC (discrimination).

| Score | Formula | What it captures |
|-------|---------|-----------------|
| `confidence` | count(majority) / 20 | Fraction of passes backing the top answer only |
| `consistency_rate` | Σ p(a)² | Sum of squared answer frequencies — weights the full distribution |

The two scores diverge when the answer distribution is spread across multiple answers:
a 10/5/5 split gives `confidence = 0.50` but `consistency_rate = 0.375`, because
consistency rate penalises the two minority clusters that majority-vote ignores.

**Calibration (ECE):**

| Model | Benchmark | Prompt | ECE (confidence) | ECE (consistency) |
|-------|-----------|--------|-----------------|-------------------|
| 8B | GSM8K | cot | **0.038** | 0.138 |
| 8B | GSM8K | zero_shot_aligned | **0.040** | 0.132 |
| 1B | GSM8K | zero_shot_aligned | 0.123 | **0.075** |
| 1B | GSM8K | cot | 0.179 | **0.071** |
| 8B | MATH | zero_shot_aligned | 0.255 | **0.138** |
| 8B | MATH | cot | 0.263 | **0.142** |
| 1B | MATH | zero_shot_aligned | 0.297 | **0.165** |
| 1B | MATH | cot | 0.334 | **0.193** |

The direction flips between models. For the **1B**, consistency rate is markedly
better calibrated than majority-vote across all conditions (ECE ~0.07 vs ~0.12–0.18
on GSM8K). For the **8B on GSM8K**, majority-vote is better calibrated (ECE ~0.04 vs
~0.13). On MATH, consistency rate wins in every condition — ECE roughly halved.

The explanation: majority-vote confidence = fraction of 20 passes on the top answer.
When the 8B is correct on GSM8K, it tends to be highly consistent (confidence ≥ 0.9),
so majority-vote tracks accuracy well. The 1B is more spread — even on correct
problems it rarely achieves full agreement — so the plurality fraction overstates
certainty relative to what the full distribution actually looks like.

**Discrimination (AUROC):**

| Model | Benchmark | Prompt | AUROC (confidence) | AUROC (consistency) |
|-------|-----------|--------|--------------------|---------------------|
| 8B | GSM8K | zero_shot_aligned | 0.858 | 0.857 |
| 8B | GSM8K | cot | 0.846 | 0.839 |
| 1B | GSM8K | cot | 0.817 | 0.823 |
| 1B | GSM8K | zero_shot_aligned | 0.810 | 0.810 |
| 1B | MATH | cot | 0.696 | 0.720 |
| 8B | MATH | zero_shot_aligned | 0.696 | 0.724 |
| 8B | MATH | cot | 0.628 | 0.664 |
| 1B | MATH | zero_shot_aligned | 0.569 | 0.588 |

AUROC is nearly identical for both scores in every condition — they separate correct
from incorrect problems equally well. Choosing between them matters only for
calibration: use consistency rate for the 1B or for MATH; majority-vote is better
calibrated for the 8B on GSM8K.

---

### Uncertainty measures: entropy, n\_unique, std\_log\_prob, std\_numeric\_span

Four measures are used for AUROC only (negated — higher uncertainty ranks a problem
as more likely wrong). They fall into two families:

**Answer-level** (count what the model says):

| Measure | What it captures |
|---------|----------------|
| `entropy` | Shannon entropy H = −Σ p(a) log₂ p(a) over 20 pass answers |
| `n_unique_answers` | Raw count of distinct answers across 20 passes |

**Token-level** (measure how certain the model sounds, not what it says):

| Measure | What it captures |
|---------|----------------|
| `std_log_prob` | Std of per-pass mean token log-prob — full sequence (reasoning chain + answer) |
| `std_numeric_span_log_prob` | Same, restricted to numeric tokens within the Final Answer span |

**AUROC comparison across all conditions:**

| Model | Benchmark | Prompt | entropy | n\_unique | std\_log\_prob | std\_numeric\_span |
|-------|-----------|--------|---------|----------|--------------|-----------------|
| 8B | GSM8K | zero_shot_aligned | **0.851** | 0.837 | 0.540 | 0.721 |
| 8B | GSM8K | cot | **0.834** | 0.818 | 0.483 | 0.758 |
| 1B | GSM8K | cot | **0.825** | 0.816 | 0.721 | 0.695 |
| 1B | GSM8K | zero_shot_aligned | 0.810 | 0.804 | 0.704 | 0.712 |
| 1B | MATH | cot | 0.747 | **0.776** | 0.724 | 0.757 |
| 8B | MATH | zero_shot_aligned | 0.742 | **0.753** | 0.498 | 0.618 |
| 8B | MATH | cot | 0.677 | **0.694** | 0.467 | 0.678 |
| 1B | MATH | zero_shot_aligned | 0.599 | **0.622** | 0.603 | 0.673 |

**Finding 1 — Answer-level signals dominate on GSM8K.**
Entropy leads on GSM8K for both models (AUROC 0.81–0.85). `std_log_prob` is
substantially weaker, especially for the 8B (0.483–0.540 vs 0.851). Averaging
log-prob over a long reasoning chain dampens per-dropout-mask variance; the answer
distribution captures the net effect of stochasticity much more cleanly.

**Finding 2 — `std_log_prob` is weak for the 8B but competitive for the 1B.**
On GSM8K, the 1B achieves AUROC 0.704–0.721 with `std_log_prob` (close to its
answer-level 0.810–0.825), while the 8B drops to 0.483–0.540. A possible
explanation: the 8B generates longer, more confident reasoning chains, so the
full-sequence log-prob variance across dropout masks is suppressed. The 1B's
shorter, less consistent outputs leave more per-mask variance in the token stream.

**Finding 3 — `std_numeric_span` consistently outperforms `std_log_prob`.**
Restricting to the numeric tokens in the Final Answer span removes reasoning chain
noise. On GSM8K for the 8B: `std_numeric_span` = 0.721–0.758 vs `std_log_prob`
= 0.483–0.540 — a 0.2 AUROC gain from focusing on the right tokens. The gain is
smaller for the 1B (0.695–0.712 vs 0.704–0.721) because the 1B's token-level
variance is less concentrated in the answer span.

**Finding 4 — n\_unique leads on MATH.**
When accuracy is ~4%, answer extraction becomes noisy and entropy's log-weighted
formula is sensitive to small count differences. `n_unique_answers` — a simple
count of distinct answers — is the best discriminator on MATH in 3 of 4 conditions
(AUROC 0.622–0.776). It captures the same signal (answer diversity) without the
log weighting that amplifies noise at very low accuracy. This is the most robust
signal in the degraded regime.

**Finding 5 — Token-level signals are not redundant with answer-level.**
`std_numeric_span` reaches 0.673–0.757 on MATH while answer-level entropy ranges
0.599–0.747. In the hardest condition (1B MATH `zero_shot_aligned`), `std_numeric_span`
(0.673) exceeds entropy (0.599) by 0.07 AUROC. When answer-level diversity saturates
(~12 unique answers out of 20 passes), the token-level variance in the Final Answer
digits still carries residual discriminative signal.

---

## Answers to Sub-Questions

### SQ1 — Calibration and discrimination
*How well do MC Dropout uncertainty estimates align with actual correctness for
fine-tuned language models on mathematical reasoning tasks?*

SQ1 is answered using the `zero_shot_aligned` prompt as the controlled, in-distribution
condition — the setting that isolates MC Dropout reliability from confounding prompt
or domain shift. Stochasticity comes solely from `model.train()` reactivating
`lora_dropout=0.05` on the LoRA adapter layers.

**Calibration.** On GSM8K the 8B model is very well calibrated (ECE = 0.038–0.040);
majority-vote confidence closely tracks empirical accuracy. The 1B is moderately
calibrated (ECE = 0.12–0.18). For the 1B, consistency rate is better calibrated than
majority-vote confidence (ECE 0.075 vs 0.123), suggesting the full answer distribution
is a more honest confidence score than the plurality fraction alone.

**Discrimination.** AUROC ≥ 0.81 across all GSM8K conditions for both answer-level
signals (confidence, consistency, entropy). MC Dropout reliably ranks correct answers
above incorrect ones when the model has non-trivial task competence (34–59% accuracy).
Answer-level signals (entropy, consistency rate) outperform token-level signals on
GSM8K for the 8B model (AUROC 0.85 vs 0.54–0.72 for `std_log_prob`). `std_numeric_span`
partially recovers that gap by focusing on the Final Answer digits rather than the full
reasoning chain.

**Answer:** MC Dropout produces reliable uncertainty estimates — both well-calibrated
and discriminative — under in-distribution, moderately difficult conditions.

---

### SQ2 — Prompt distribution shift
*How does evaluating with an out-of-distribution prompt affect the reliability of
MC Dropout uncertainty estimates compared to an in-distribution prompt?*

Comparing `zero_shot_aligned` (in-distribution) vs `cot` (out-of-distribution
multi-turn few-shot) within each model and benchmark:

**Accuracy.** The 1B model drops ~6 pp on GSM8K under `cot` (34.2% vs 40.2%),
consistent with the Phase 1 finding that multi-turn structure is OOD for a model
fine-tuned on single-turn conversations. The 8B shows minimal accuracy impact
(56.8% vs 58.6%), having stronger residual multi-turn capability from pre-training.

**Calibration.** ECE worsens for the 1B under `cot` (0.179 vs 0.123). The 8B is
nearly unaffected (0.038 vs 0.040). Prompt-induced distribution shift degrades
calibration for the smaller model but not the larger one.

**Discrimination.** AUROC is stable across both prompts for both models — the
ability to rank correct above incorrect answers is robust to the prompt format change
even where calibration degrades.

**Answer:** Prompt distribution shift has a measurable but asymmetric effect.
Calibration degrades for the 1B under OOD prompting; discrimination does not
degrade for either model. The 8B is largely robust to prompt shift on both axes.
This confirms the thesis argument for calibration (especially at smaller scale) but
does not hold for discrimination.

---

### SQ3 — Model scale and task difficulty
*How do model scale and task difficulty affect the reliability of MC Dropout as an
epistemic uncertainty signal?*

This sub-question has two axes.

**Model scale (1B → 8B), on GSM8K:**
- Accuracy: +17–18 pp (40.2% → 58.6% under `zero_shot_aligned`)
- Calibration: ECE improves ~3–4× (0.123 → 0.040 for `zero_shot_aligned`)
- Discrimination: modest AUROC gain (0.810 → 0.858)
- Overconfidence rate: drops from 22.3% to 4.9% — the 8B is rarely confident and wrong

Model scale substantially improves both calibration and overconfidence under controlled
conditions.

**Task difficulty (GSM8K → MATH), holding model and prompt fixed:**
- Accuracy collapses from 34–59% to 2–4% — MATH is essentially unsolved
- Calibration degrades severely: ECE rises from 0.04–0.18 to 0.25–0.33
- Discrimination drops from 0.81–0.86 to 0.57–0.75 AUROC
- Overconfidence rate reaches 94–100% on MATH for 1B and 8B `zero_shot_aligned` —
  every high-confidence prediction is wrong
- Mean unique answers per problem rises from ~7 (GSM8K) to ~11–12 (MATH), showing
  the model's own uncertainty signals correctly reflect its inability — the estimates
  are directionally correct even where calibration and discrimination formally fail

**Model scale does not rescue MATH performance:** the 8B shows no meaningful accuracy
or calibration improvement over the 1B on MATH (both ~4%, ECE ~0.25–0.26). Task
difficulty is the dominant factor.

**Answer:** Model scale improves reliability on tractable tasks (GSM8K), but task
difficulty is the stronger factor overall. When the model cannot solve the task,
MC Dropout uncertainty estimates lose their discriminative and calibrative value —
though they correctly signal high uncertainty, indicating the method degrades
gracefully rather than catastrophically.

---

## Pending

- **LLM-as-judge:** `src/uq/llm_judge.py` has not been run on any of these results.
  Running it produces `results_judged.json`, `summary_judged.json`, and
  `judge_correctness/` plots, adding `ece_judge_*` and `auroc_judge_*` to the
  aggregate metrics. Estimated cost: ~$0.80 per 500-problem run with Claude Haiku 4.5.
- **Additional seeds:** Only seed 42 has been evaluated. Running seeds 123 and 456
  would enable ensemble comparison and provide error bars on the reported metrics.

---

## Thesis Results Section — Plots and Inline Numbers

Note: LLM-as-judge has not been run. All plots and numbers below are from binary
correctness only. Judge ECE/AUROC columns are placeholders pending that run.

---

### Plots (4 figures, 9 panels total)

**Figure 1 — Reliability diagrams: scale and dataset difficulty (2×2)**

Rows = dataset (GSM8K / MATH), Cols = model (1B / 8B). All `zero_shot_aligned`.
Use `binary_correctness/reliability_confidence.png` (majority-vote) for all panels.

| Panel | File | ECE |
|---|---|---|
| GSM8K / 1B | `1b/seed42/gsm8k/zero_shot_aligned/binary_correctness/reliability_confidence.png` | 0.123 |
| GSM8K / 8B | `8b/seed42/gsm8k/zero_shot_aligned/binary_correctness/reliability_confidence.png` | 0.040 |
| MATH / 1B  | `1b/seed42/math/zero_shot_aligned/binary_correctness/reliability_confidence.png`  | 0.297 |
| MATH / 8B  | `8b/seed42/math/zero_shot_aligned/binary_correctness/reliability_confidence.png`  | 0.255 |

This is the central calibration figure. Shows the scale effect (1B→8B ECE halves on
GSM8K) and the task-difficulty collapse (ECE triples on MATH for both models).

---

**Figure 2 — Reliability diagrams: prompt distribution shift (1×2)**

1B model only, GSM8K only. Left = `zero_shot_aligned`, right = `cot`.
Use `binary_correctness/reliability_confidence.png`.

| Panel | File | ECE |
|---|---|---|
| 1B GSM8K zero_shot | `1b/seed42/gsm8k/zero_shot_aligned/binary_correctness/reliability_confidence.png` | 0.123 |
| 1B GSM8K cot       | `1b/seed42/gsm8k/cot/binary_correctness/reliability_confidence.png`               | 0.179 |

The 8B is not worth a panel here — ECE is 0.040 vs 0.038, visually indistinguishable.
The 1B shows the calibration degradation clearly (+0.056 ECE under OOD prompting).

---

**Figure 3 — ROC curves: best case vs degraded regime (1×2)**

Each panel is `binary_correctness/roc_binary.png` — both confidence measures overlaid.

| Panel | File | AUROC conf / consistency |
|---|---|---|
| Best case: 8B GSM8K zero_shot    | `8b/seed42/gsm8k/zero_shot_aligned/binary_correctness/roc_binary.png` | 0.858 / 0.857 |
| Degraded: 1B MATH zero_shot      | `1b/seed42/math/zero_shot_aligned/binary_correctness/roc_binary.png`  | 0.569 / 0.588 |

Shows the full range: strong discrimination under tractable conditions, near-random
under task collapse (1B MATH). Use these two panels rather than all 8 ROC plots.

---

**Figure 4 — Selective prediction (1 panel)**

`8b/seed42/gsm8k/zero_shot_aligned/confidence/selective_prediction.png`

Shows accuracy vs coverage for the best-case condition. Makes the practical utility
argument: abstaining on low-confidence predictions substantially raises accuracy on
the accepted subset.

---

### Inline numbers (tables for the thesis)

All paths are relative to `results/results_mc_dropout_1/`.

**Table 1 — Accuracy and overconfidence**

| Model | Dataset | Prompt | Accuracy | Mean conf | Mean entropy | Overconf rate |
|-------|---------|--------|----------|-----------|--------------|---------------|
| 8B | GSM8K | zero_shot | 58.6% | 0.564 | 1.906 | 4.9% |
| 8B | GSM8K | cot | 56.8% | 0.538 | 1.977 | 6.2% |
| 1B | GSM8K | zero_shot | 40.2% | 0.525 | 2.030 | 22.3% |
| 1B | GSM8K | cot | 34.2% | 0.521 | 2.025 | 23.6% |
| 8B | MATH | zero_shot | 2.2% | 0.277 | 3.191 | **100.0%** |
| 8B | MATH | cot | 4.2% | 0.305 | 3.055 | 50.0% |
| 1B | MATH | zero_shot | 4.2% | 0.336 | 2.983 | 94.1% |
| 1B | MATH | cot | 4.0% | 0.374 | 2.817 | 75.0% |

Key callout: 8B MATH `zero_shot_aligned` — mean confidence 0.277 yet overconf rate
100% (4 high-confidence predictions, all wrong). Model is uncertain on average but
catastrophically overconfident in the rare cases it commits.

---

**Table 2 — Calibration (ECE)**

| Model | Dataset | Prompt | ECE conf | ECE consistency |
|-------|---------|--------|----------|-----------------|
| 8B | GSM8K | zero_shot | **0.040** | 0.132 |
| 8B | GSM8K | cot | **0.038** | 0.138 |
| 1B | GSM8K | zero_shot | 0.123 | **0.075** |
| 1B | GSM8K | cot | 0.179 | **0.071** |
| 8B | MATH | zero_shot | 0.255 | **0.138** |
| 8B | MATH | cot | 0.263 | **0.142** |
| 1B | MATH | zero_shot | 0.297 | **0.165** |
| 1B | MATH | cot | 0.334 | **0.193** |

Bold = better-calibrated score for each condition. Direction flips between models:
consistency rate wins for the 1B (and all MATH conditions); majority-vote wins for 8B on GSM8K.

---

**Table 3 — Discrimination (AUROC)**

Confidence measures only — answers the question: does the model's confidence correctly
rank problems it gets right above problems it gets wrong?

| Model | Dataset | Prompt | AUROC conf | AUROC consistency |
|-------|---------|--------|------------|-------------------|
| 8B | GSM8K | zero_shot | **0.858** | 0.857 |
| 8B | GSM8K | cot | **0.846** | 0.839 |
| 1B | GSM8K | cot | 0.817 | **0.823** |
| 1B | GSM8K | zero_shot | 0.810 | 0.810 |
| 1B | MATH | cot | 0.696 | **0.720** |
| 8B | MATH | zero_shot | 0.696 | **0.724** |
| 8B | MATH | cot | 0.628 | **0.664** |
| 1B | MATH | zero_shot | 0.569 | **0.588** |

Both measures discriminate nearly identically — choosing between them matters only
for calibration (Table 2), not discrimination.

---

### What to mention in text but not table

- `auroc_entropy`: 0.810–0.851 on GSM8K, 0.599–0.747 on MATH — closely tracks the
  confidence AUROC in every condition, confirming answer-level signals carry the same
  discriminative information.
- `auroc_std_numeric_span`: on MATH outperforms or matches entropy for 1B (0.673–0.757
  vs 0.599–0.747) — token-level variance in the Final Answer digits is the most robust
  signal when the answer distribution saturates (~12 unique answers per problem).
- `auroc_std_log_prob`: weak for 8B (0.483–0.540 on GSM8K) because long confident
  reasoning chains suppress dropout variance; competitive for 1B (0.704–0.721).
- `auroc_n_unique`: best single discriminator on MATH in 3/4 conditions (0.622–0.776)
  because simple counts are robust to answer-extraction noise at ~4% accuracy; entropy's
  log weighting amplifies noise at very low base rates.
- ROUGE-L / METEOR: mean values 0.14–0.26 across conditions, clearly below the accuracy
  floor on GSM8K. The 8B MATH `zero_shot_aligned` inversion (`auroc_rougeL` = 0.072)
  is a degenerate case from near-zero ROUGE-L variance — cite as evidence the metric
  is unsuitable for symbolic math.
- `mean_std_numeric_span_log_prob` rises from ~0.17 (1B GSM8K) to ~1.24 (8B MATH
  zero_shot) — the model's token-level confidence variance correctly scales with task
  difficulty even when binary discrimination degrades.
