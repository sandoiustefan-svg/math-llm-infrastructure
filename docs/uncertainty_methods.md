# Uncertainty Quantification Methods

This document describes how epistemic uncertainty is estimated during inference.
The MC Dropout evaluation loop lives in `src/uq/mc_dropout.py`; confidence
measures and uncertainty measures are computed in `src/uq/metrics.py`.

---

## Experiment design

The evaluation uses MC Dropout across two models and two prompt variants:

| | zero_shot_aligned | cot |
|---|---|---|
| **Llama 3.2-1B (LoRA rank 16)** | GSM8K + MATH | GSM8K only |
| **Llama 3.1-8B (LoRA rank 64)** | GSM8K + MATH | GSM8K only |

`zero_shot_aligned` matches the fine-tuning format exactly (in-distribution anchor).
`cot` uses a multi-turn few-shot structure that is intentionally out-of-distribution,
isolating the effect of prompt-induced distribution shift on UQ reliability.

---

## MC Dropout

MC Dropout re-activates the LoRA dropout (`lora_dropout=0.1`) at inference time by
calling `model.train()`. The same dropout that regularised training is reactivated —
no additional hooks or dropout layers are added. Each of the `num_passes=20` greedy
forward passes produces a different stochastic prediction due to the active LoRA dropout.

This is principled because the model learned its weights in the presence of this
dropout, giving a Bayesian approximation (Gal & Ghahramani, 2016) over the LoRA
adapter weights specifically. Uncertainty is estimated from the adapter — the
task-specific component — while the frozen base model knowledge is fixed.

Epistemic uncertainty is estimated from **disagreement and variance across passes**:
if all 20 passes give the same answer and consistent token confidence the model is
highly certain; if answers and confidence vary the model has high epistemic uncertainty.

---

## Confidence scores

Confidence scores are in [0, 1] where higher means more confident. They are used
for ECE (calibration) and AUROC (discrimination).

### 1. Majority-vote confidence

```
confidence = count(majority_answer) / num_passes
```

Fraction of the 20 passes that agreed on the most common extracted final answer.
- `confidence = 1.0` — all passes gave the same answer (certain)
- `confidence = 0.05` — each pass gave a different answer (maximally uncertain)

Captures epistemic uncertainty through answer-level disagreement across passes.
This is the primary UQ signal — it directly measures how consistently the model
commits to an answer under stochastic dropout.

### 2. Consistency rate

```
consistency_rate = Σ_a p(a)²
```

Sum of squared answer frequencies across 20 passes, equivalent to 1 − Gini impurity.
Considers the full answer distribution rather than only the top answer.
- `consistency_rate = 1.0` — all passes gave the same answer
- `consistency_rate → 0` — passes spread uniformly across many distinct answers

Complements majority-vote by weighting all answer clusters, not just the plurality.

---

## Uncertainty measures

Uncertainty measures are higher when the model is more uncertain. They are used
for AUROC only (negated for ranking — higher uncertainty = lower likelihood of correct).
They cannot be used directly for ECE because ECE requires a confidence direction.

### 3. Answer entropy

```
H = -Σ_a p(a) * log₂(p(a))
```

Shannon entropy over the answer distribution across the 20 passes.
- `H = 0` — all passes agree (certain)
- `H = log₂(20) ≈ 4.32` — all 20 passes give different answers (maximally uncertain)

Captures the full shape of the answer distribution. A model that splits 10/10 between
two answers has the same majority-vote confidence (0.5) as one that gives 20 different
answers, but very different entropy. Entropy distinguishes these cases.

### 4. Number of unique answers

```
n_unique_answers = |{distinct answers across 20 passes}|
```

Raw count of distinct extracted answers. Simple and interpretable — 1 means full
agreement, 20 means every pass gave a different answer. Cruder than entropy but
directly interpretable without log-probability reasoning.

### 5. Std of avg log-prob across passes (full sequence)

```
std_log_prob = std( avg_log_prob_1, ..., avg_log_prob_20 )
where avg_log_prob_i = (1/L_i) * Σ_j log(p_ij)
```

Standard deviation of the per-pass average token log-probability across the 20 passes.
Captures token-level epistemic variance: does the model's overall token confidence
fluctuate between dropout masks? High std means the LoRA adapter weights are sensitive
to dropout — the model is epistemically uncertain at the token level.

Complements answer entropy: a model could consistently predict the same final answer
(low entropy) but with highly variable confidence in the reasoning steps (high std_log_prob),
revealing uncertainty in the chain of thought that answer-level signals miss.

### 6. Std of numeric span avg log-prob across passes

```
std_numeric_span_log_prob = std( avg_log_prob_numeric_span_1, ..., avg_log_prob_numeric_span_20 )
```

Same as `std_log_prob` but restricted to numeric tokens within the Final Answer span.
More focused than the full-sequence version — captures epistemic variance specifically
in the digits the model writes as its answer, rather than the full reasoning chain.
