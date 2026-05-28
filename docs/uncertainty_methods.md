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

MC Dropout re-activates the LoRA dropout (`lora_dropout=0.05`) at inference time by
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

## Toy example

All six measures below are illustrated on the same scenario.

**Problem**: "A box holds 4 apples. How many apples are in 5 boxes?"
**Expected answer**: 20

**20 MC Dropout passes — extracted final answers**:

| Passes | Answer | Count | p(a) |
|--------|--------|-------|------|
| 1–14   | "20"   | 14    | 0.70 |
| 15–18  | "24"   | 4     | 0.20 |
| 19–20  | "15"   | 2     | 0.10 |

**Per-pass average token log-probability** (full sequence, sample of 5 shown):

| Pass | avg_log_prob | avg_log_prob (numeric span only) |
|------|-------------|----------------------------------|
| 1    | −0.82       | −0.31                            |
| 2    | −0.95       | −0.67                            |
| 3    | −0.88       | −0.42                            |
| 4    | −1.13       | −0.89                            |
| 5    | −0.79       | −0.28                            |
| …    | …           | …                                |

The full-sequence values are averaged over the entire reasoning chain (many tokens),
so individual pass variance is dampened. The numeric-span values cover only the digits
in "Final Answer: 20", so they are more sensitive to dropout mask changes.

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

**Toy example**:
```
majority_answer = "20"   (14 out of 20 passes)
confidence = 14 / 20 = 0.70
```

### 2. Consistency rate

```
consistency_rate = Σ_a p(a)²
```

Sum of squared answer frequencies across 20 passes, equivalent to 1 − Gini impurity.
Considers the full answer distribution rather than only the top answer.
- `consistency_rate = 1.0` — all passes gave the same answer
- `consistency_rate → 0` — passes spread uniformly across many distinct answers

Complements majority-vote by weighting all answer clusters, not just the plurality.

**Toy example**:
```
p("20") = 0.70,  p("24") = 0.20,  p("15") = 0.10

consistency_rate = 0.70² + 0.20² + 0.10²
                 = 0.49  + 0.04  + 0.01
                 = 0.54
```

Note: majority-vote gives 0.70 and consistency rate gives 0.54 — they differ because
consistency rate penalises the two minority answers (0.20 and 0.10) that majority-vote
ignores beyond identifying "20" as the winner.

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

**Toy example**:
```
H = −(0.70 × log₂(0.70)) − (0.20 × log₂(0.20)) − (0.10 × log₂(0.10))
  = −(0.70 × −0.515)     − (0.20 × −2.322)      − (0.10 × −3.322)
  =   0.361               +  0.464               +  0.332
  =   1.16 bits
```

Compare to a perfectly split 10/10 case: H = −2 × (0.5 × log₂(0.5)) = 1.0 bit.
The three-way split here (14/4/2) gives slightly higher entropy (1.16 bits) because
the answer distribution is more spread out than a clean 50/50.

### 4. Number of unique answers

```
n_unique_answers = |{distinct answers across 20 passes}|
```

Raw count of distinct extracted answers. Simple and interpretable — 1 means full
agreement, 20 means every pass gave a different answer. Cruder than entropy but
directly interpretable without log-probability reasoning.

**Toy example**:
```
answers seen: {"20", "24", "15"}
n_unique_answers = 3
```

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

**Toy example**:
```
avg_log_prob values across 20 passes:
  −0.82, −0.95, −0.88, −1.13, −0.79, −0.86, −0.91, −1.02,
  −0.84, −0.97, −0.83, −1.08, −0.90, −0.85, −0.94, −1.01,
  −0.88, −0.92, −0.87, −0.96

std_log_prob = std(above 20 values) ≈ 0.086
```

Each value is the mean token log-prob for one full response (reasoning chain + answer).
The std across 20 passes reflects how much dropout masks change the model's token-level
confidence — not what it says, but how sure it sounds saying it.

### 6. Std of numeric span avg log-prob across passes

```
std_numeric_span_log_prob = std( avg_log_prob_numeric_span_1, ..., avg_log_prob_numeric_span_20 )
```

Same as `std_log_prob` but restricted to numeric tokens within the Final Answer span.
More focused than the full-sequence version — captures epistemic variance specifically
in the digits the model writes as its answer, rather than the full reasoning chain.

**Toy example**:
```
avg_log_prob restricted to "Final Answer: 20" tokens, across 20 passes:
  −0.31, −0.67, −0.42, −0.89, −0.28, −0.71, −0.35, −0.83,
  −0.29, −0.74, −0.33, −0.91, −0.45, −0.30, −0.68, −0.87,
  −0.38, −0.72, −0.40, −0.65

std_numeric_span_log_prob = std(above 20 values) ≈ 0.212
```

The numeric span std (0.212) is much higher than the full-sequence std (0.086) because
averaging over the long reasoning chain dampens the per-token variance. The numeric span
isolates the answer digits — the point where epistemic uncertainty about the final result
is most directly expressed.
