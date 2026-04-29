# Thesis: Confidence Calibration in Mathematical Reasoning LLMs — A Comparative Study of Token Probability Measures and Ensemble-Based Uncertainty Quantification

**Research Question:** Does model confidence predict answer correctness in mathematical reasoning — and which confidence measure is the most reliable predictor?

**Student:** Bogdan Sandoiu
**Last Updated:** 2026-04-09
**Status:** Living document — update after every significant result

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [The Core Problem: Token Probability Inflation](#2-the-core-problem-token-probability-inflation)
3. [Methodology](#3-methodology)
4. [Experimental Design](#4-experimental-design)
5. [Expected Findings](#5-expected-findings)
6. [Known Weaknesses and Mitigations](#6-known-weaknesses-and-mitigations)
7. [Thesis Chapter Outline](#7-thesis-chapter-outline)
8. [Critical Self-Assessment](#8-critical-self-assessment)
9. [Essential Reading List](#9-essential-reading-list)

---

## 1. Motivation

Large language models deployed in high-stakes domains — mathematics education, tutoring systems, automated grading, scientific computation — need to do more than produce answers. They need to communicate when they are likely to be wrong.

Mathematical reasoning is a particularly demanding test case for uncertainty quantification (UQ). Unlike open-ended generation tasks, math problems have definite correct answers. This creates a clean evaluation protocol: you can directly measure whether a model's expressed confidence correlates with its probability of being right. A model that says "I'm 90% confident" should be correct roughly 90% of the time. When it isn't, the model is miscalibrated, and downstream systems (teachers, students, other AI components) cannot trust its self-reports.

The problem has become more pressing as chain-of-thought prompting has become standard practice. Models now produce long reasoning chains before emitting a final answer. This creates an apparent richness of information — the full token sequence — but it also creates a trap: most tokens in a reasoning chain are structural glue ("therefore", "we can see that", "substituting into") that carry near-certainty probabilities regardless of whether the underlying mathematics is correct. Naively aggregating token probabilities over the full sequence produces a confidence estimate that is systematically inflated and weakly correlated with actual correctness.

This thesis investigates whether more targeted token selection — restricting confidence estimation to answer-span tokens, or further to only numeric tokens — meaningfully improves the calibration of token-probability-based confidence. It also compares token-probability approaches against distribution-level approaches (answer entropy from MC Dropout passes or ensemble members), which sidestep the token selection problem entirely by treating confidence as agreement across multiple predictions.

The practical stakes are clear: a reliably calibrated confidence measure enables abstention (the model declines to answer when uncertain), selective grading (flag low-confidence answers for human review), and reinforcement learning with verifiable rewards. An uncalibrated confidence measure is worse than useless — it creates false trust.

---

## 2. The Core Problem: Token Probability Inflation

### 2.1 The Mechanism

When an autoregressive language model generates text, it assigns a probability to each token conditioned on all prior tokens. The geometric mean of these probabilities over a sequence is the standard measure of sequence-level model confidence — equivalently, the exponential of the negative average log-probability (i.e., the inverse of perplexity).

For mathematical reasoning, this measure has a structural flaw. Consider a generated solution of 200 tokens. The vast majority — connective phrases, punctuation, formatting tokens, common mathematical English — will receive probabilities close to 1.0 because these tokens are nearly deterministic given context. The model has seen millions of instances of "... therefore ..." and "... which gives us ..." during training. The uncertainty about whether the mathematics is correct is concentrated in a small number of tokens: the intermediate numerical values and the final answer.

Suppose 180 of 200 tokens have probability 0.99, and the 20 numeric tokens have average probability 0.6. The full-sequence geometric mean is approximately:

```
exp(180/200 * ln(0.99) + 20/200 * ln(0.6)) ≈ exp(-0.009 - 0.102) ≈ 0.895
```

The model reports 89.5% confidence, but the actual numerical certainty is only 60%. This is not noise — it is a systematic upward bias introduced by including tokens whose probability carries no information about mathematical correctness.

### 2.2 Why This Matters

Calibration literature typically assumes that confidence scores are measured on a comparable scale. If full-sequence token probability is systematically inflated for all examples, then ECE (Expected Calibration Error) may still appear low if the bias is uniform — but the reliability diagram will show a characteristic pattern: all bars concentrated in the 0.8–1.0 confidence range, with actual accuracy much lower than the confidence bins suggest. This is overconfidence, and it is precisely what we expect to observe for Measure 1.

### 2.3 The Proposed Solution

The thesis tests a hierarchy of increasingly targeted token selection:

1. **Full sequence** (baseline): all generated tokens — maximally inflated
2. **Answer span only**: tokens after "### Final Answer:" — removes reasoning chain glue
3. **Numeric tokens, full sequence**: all digit-containing tokens across reasoning chain — targets numbers but includes intermediate computation
4. **Numeric tokens, answer span only**: digits in the final answer only — most targeted
5. **Answer entropy**: agreement across N passes/members — bypasses token selection entirely

The hypothesis is that moving down this list produces progressively better calibration. The actual ordering of 3 and 4 is an open empirical question: answer-span numerics may be better (more focused) or worse (fewer tokens means higher variance in the estimate).

---

## 3. Methodology

### 3.1 Confidence Measure Definitions

All six measures are implemented in `src/uq/metrics.py` and `src/uq/mc_dropout.py` / `src/uq/ensemble.py`.

**Measure 1 — Full-Sequence Token Probability**

For a generated sequence of tokens $t_1, \ldots, t_N$ given prompt $x$:

$$\text{conf}_{\text{full}} = \exp\!\left(\frac{1}{N}\sum_{i=1}^{N} \log p(t_i \mid x, t_{<i})\right)$$

This is the geometric mean probability, equivalently $\text{perplexity}^{-1}$ on the generated sequence. Implemented as `full_sequence_mean_confidence`.

**Measure 2 — Answer-Span Token Probability**

Let $s$ be the index of the first token after the "### Final Answer:" marker. Then:

$$\text{conf}_{\text{span}} = \exp\!\left(\frac{1}{N - s}\sum_{i=s}^{N} \log p(t_i \mid x, t_{<i})\right)$$

If the marker is not found, the implementation falls back to the last 20% of tokens as an approximation — a known fallback that may introduce noise when the model fails to produce the expected format. Implemented as `answer_span_mean_confidence`.

**Measure 3 — Numeric Tokens, Full Sequence**

Let $\mathcal{N}$ be the set of token indices where the decoded token contains at least one digit character. Then:

$$\text{conf}_{\text{num}} = \exp\!\left(\frac{1}{|\mathcal{N}|}\sum_{i \in \mathcal{N}} \log p(t_i \mid x, t_{<i})\right)$$

This may be undefined (NaN) for problems with no numeric tokens in the reasoning chain, which will reduce the effective sample size for this measure. Implemented as `numeric_mean_confidence`.

**Measure 4 — Numeric Tokens, Answer Span Only**

Intersection of the answer span (Measure 2) and numeric tokens (Measure 3). Implemented as `numeric_span_mean_confidence`. This is the most targeted measure and will have the smallest effective token count, making it highest-variance but potentially best-calibrated.

**Measure 5 — Answer Entropy**

For N passes (MC Dropout) or K members (ensemble), collect the set of extracted final answers $\{a_1, \ldots, a_N\}$. Compute the empirical distribution over distinct normalized answers:

$$H = -\sum_{a} \hat{p}(a) \log_2 \hat{p}(a)$$

where $\hat{p}(a) = \text{count}(a) / N$. This is Shannon entropy in bits. $H = 0$ means unanimous agreement; $H = \log_2 N$ means maximal disagreement.

The derived confidence is the majority-vote fraction: $\text{conf}_{\text{entropy}} = \max_a \hat{p}(a)$.

Answer normalization via `normalize_math_answer()` handles LaTeX wrappers, fraction-to-decimal conversion, and trailing zeros to avoid spurious disagreement from equivalent representations.

**Measure 6 — Position-Weighted Token Probability**

Instead of a binary include/exclude decision per token, assign each token a weight based on its region, then take the weighted geometric mean:

$$\text{conf}_{\text{weighted}} = \exp\!\left(\frac{\sum_{i=1}^{N} w_i \log p(t_i \mid x, t_{<i})}{\sum_{i=1}^{N} w_i}\right)$$

Weights are assigned by region:

| Token region | Weight |
|---|---|
| Non-numeric, reasoning chain (glue words) | $1$ |
| Numeric token in reasoning chain | $\alpha = 5$ |
| Non-numeric token in answer span | $\beta = 5$ |
| Numeric token in answer span | $\alpha \cdot \beta = 25$ |

The weight scheme is **composable**: numeric × span = 25 because a token that is both numeric and in the answer span satisfies both criteria independently. The values $\alpha = \beta = 5$ are fixed constants motivated by the information asymmetry between glue and mathematical tokens, not tuned on evaluation data.

This measure is a continuous interpolation between Measure 1 (all $w_i = 1$) and the hard-filter measures (Measures 2–4). The central hypothesis is that soft-weighting captures more signal than hard filtering because it retains glue-token coherence as a weak signal while emphasising the tokens most predictive of correctness. Implemented as `weighted_mean_confidence`.

### 3.2 Evaluation Metrics

**Expected Calibration Error (ECE):** Partition confidence scores into 10 equal-width bins. For each bin, compute $|{\rm acc}(b) - \overline{\rm conf}(b)|$. ECE is the weighted average of these gaps. Implemented in `metrics.py::expected_calibration_error()`. Lower is better. ECE = 0 means perfect calibration.

**Reliability Diagram:** Visual complement to ECE. Bars represent empirical accuracy per confidence bin; the diagonal is perfect calibration. Systematic patterns (bars below diagonal = overconfidence, above = underconfidence) are more interpretable than ECE alone.

**AUROC (to be added):** Area Under the ROC curve treating confidence as a binary classifier for correctness. Measures discrimination: can the model distinguish correct from incorrect answers using confidence alone? This is orthogonal to calibration — a model can have high AUROC (good discrimination) but poor ECE (poor calibration magnitude).

**Coverage-Accuracy Trade-off:** At what confidence threshold does abstaining achieve 90% accuracy? The fraction of questions answered at that threshold is the coverage. This is the operationally most important metric for real-world deployment.

### 3.3 Oracle: Answer Correctness

Correctness is determined by `answers_are_equal(pred, gold)` which calls `normalize_math_answer()` on both strings. This handles the most common equivalence failures (LaTeX, fractions, trailing zeros), but will still fail on:

- Mixed symbolic/numeric answers ("2x" vs "2*x")
- Units ("15 meters" vs "15")
- Multi-part answers
- Expressions that require simplification ("sqrt(4)" vs "2")

This oracle is not perfect. Any errors in the oracle propagate directly into ECE and accuracy measurements. Spot-checking oracle decisions on a sample of results is essential before drawing conclusions.

---

## 3.5 All Metrics Explained in Plain Language

This section explains every metric used in this thesis without assuming prior machine learning knowledge. The goal is that anyone — including a thesis committee member unfamiliar with NLP — can understand what each number means, why it is computed, and what a good or bad value looks like.

---

### How a Language Model Generates Text

Before understanding the metrics, it helps to understand the mechanism. A language model generates text one token (roughly one word or word-piece) at a time. At each step, it looks at everything written so far and outputs a **probability distribution** over every word in its vocabulary (32,000 words in this project). It then picks the next token based on that distribution.

For example, after "2 + 2 =", the model might assign:
- "4" → 72% probability
- "3" → 8% probability
- "five" → 5% probability
- ... (32,000 other tokens share the remaining 15%)

These probabilities are the raw material for every confidence metric in this thesis.

---

### Metric 1 — Full-Sequence Token Probability (Geometric Mean)

**What it is:** For every token the model generated (the entire reasoning chain + final answer), we look at how confident the model was when it chose that token. We take the geometric mean of all those probabilities.

**Geometric mean, explained:** If you have probabilities [0.9, 0.8, 0.7], the regular average (arithmetic mean) is (0.9+0.8+0.7)/3 = 0.8. The geometric mean is (0.9 × 0.8 × 0.7)^(1/3) ≈ 0.793. For probabilities, geometric mean is used because multiplying small probabilities together (which is how sequence probability works) is better captured by geometric mean than arithmetic mean.

**The problem (token inflation):** Consider the sentence "Therefore, we can see that the answer is 4." Most tokens here — "Therefore", "we", "can", "see", "that", "the", "answer", "is" — are extremely common English words the model has seen millions of times. It assigns them probabilities close to 1.0 regardless of whether the math is right. Only "4" (the actual answer) carries mathematical uncertainty. Averaging over all tokens drowns out the signal from "4" in a sea of near-certainty probabilities from glue words.

**Key in code:** `full_sequence_mean_confidence`
**Range:** 0 to 1. Closer to 1 = model was confident. Closer to 0 = model was uncertain.
**Expected behaviour:** Systematically high (overconfident) for all problems because of glue word inflation.

---

### Metric 2 — Answer-Span Token Probability

**What it is:** Same as Metric 1, but we only look at the tokens that appear after the "### Final Answer:" marker in the model's output. Everything before that marker (the reasoning chain) is ignored.

**Why this is better:** The reasoning chain is full of structural language ("First, let us", "Substituting into the equation", "Therefore") that inflates Metric 1. The answer span — typically just the final number or expression — contains fewer glue words and more mathematical content.

**Example:** If the full generation is "We solve the equation step by step. First... [200 tokens]... ### Final Answer: 42", Metric 2 only uses the probabilities for the token(s) "42".

**Limitation:** If the model never produces "### Final Answer:", the code falls back to using the last 20% of tokens — a rough approximation. This is a known limitation noted in the methodology.

**Key in code:** `answer_span_mean_confidence`
**Range:** 0 to 1.
**Expected behaviour:** Lower and more variable than Metric 1 (fewer tokens, less inflation from glue words), hopefully better correlated with correctness.

---

### Metric 3 — Numeric Token Probability (Full Sequence)

**What it is:** We scan the entire generated text and keep only the tokens that contain at least one digit (0–9). We compute the geometric mean probability only over those tokens.

**Why this is interesting:** Mathematical correctness depends on getting numbers right. A token like "therefore" being highly probable tells us nothing about whether the math is correct. A token like "47" being low-probability (the model was unsure between "47" and "48") is a strong signal that the answer might be wrong.

**What it captures:** Confidence across ALL numerical appearances in the generation — both intermediate calculations ("First we compute 3 × 4 = 12...") and the final answer. This is the "all numerical appearances" version.

**Example:** In "We compute 3 × 4 = 12, then 12 + 5 = 17. ### Final Answer: 17", the numeric tokens are "3", "4", "12", "12", "5", "17", "17". The metric averages confidence over all of them.

**Key in code:** `numeric_mean_confidence`
**Range:** 0 to 1. NaN (undefined) if no numeric tokens appear.
**Expected behaviour:** Lower than Metric 1 (numeric tokens are harder to predict), potentially better correlated with correctness than full sequence.

---

### Metric 4 — Numeric Token Probability (Answer Span Only)

**What it is:** Intersection of Metrics 2 and 3. We only look at tokens that are (a) in the answer span after "### Final Answer:", AND (b) contain at least one digit.

**Why this is the most targeted measure:** The final answer number is the single most important piece of text for correctness. If the model is confident about "42" as the final answer, that directly predicts whether "42" is correct. All intermediate reasoning-chain numbers, formatting tokens, and glue words are excluded.

**Example:** In the same example above, Metric 4 only uses the probabilities for the very last "17" — the one in the answer span.

**Limitation:** This produces very few tokens (sometimes just 1–3 tokens). With fewer tokens, the estimate is more noisy — a single unlucky token can dominate. This is the highest-variance measure.

**Key in code:** `numeric_span_mean_confidence`
**Range:** 0 to 1. NaN if the answer span contains no numeric tokens.
**Expected behaviour:** Most directly targeted at the answer, but noisiest due to small token count.

---

### Metric 5 — Answer Entropy (Agreement Across Passes)

**What it is:** Instead of looking at token probabilities, we run the model N times (MC Dropout) or with K different model versions (Ensemble) and collect N or K independent answer predictions. We then measure how much they agree.

**Entropy, explained:** Entropy is a measure of disorder or spread. Zero entropy = perfect agreement (all N passes say "42"). Maximum entropy = complete disagreement (every pass gives a different answer). Measured in bits: 0 = fully certain, log₂(N) = fully uncertain.

**Example with N=4 passes:**
- Passes say: "42", "42", "42", "41" → 3/4 agree on "42". Confidence = 75%. Entropy is low.
- Passes say: "42", "43", "41", "44" → no majority. Confidence = 25%. Entropy is high.

**The derived confidence:** The fraction of passes that agree on the majority answer. This is what appears in the results as `confidence` (the primary field).

**Why this avoids the inflation problem:** This measure never looks at token probabilities at all. It only cares about the final answer. A model can produce a fluent, grammatically perfect chain-of-thought with high token probabilities all the way through, and still show high entropy if half the passes arrive at "42" and the other half arrive at "43". This makes it fundamentally harder to fool than token probability measures.

**Key in code:** `confidence` (majority vote fraction), `entropy` (Shannon entropy in bits)
**Range:** Confidence 0 to 1. Entropy 0 to log₂(N) bits.
**Expected behaviour:** Best predictor of correctness — hardest to inflate.

---

### Metric 6 — Position-Weighted Token Probability

**What it is:** A refinement of Metric 1 that gives different tokens different levels of importance when computing the overall confidence, instead of treating every token equally.

**The problem it solves:** Metrics 2–4 work by hard selection — a token is either included or excluded. This throws away all information from the discarded tokens. For example, Metric 4 might use only 2–3 tokens (the final answer digits), which is very noisy. Metric 6 instead keeps all tokens but turns the volume up on the ones that matter and down on the ones that don't.

**How the weights work:**

| Token type | Where in the generation | Weight |
|---|---|---|
| "therefore", "we", "the", etc. | Reasoning chain | 1 (baseline) |
| "12", "47", "3" | Reasoning chain | 5 |
| "Let us", "thus", "equals" | After "### Final Answer:" | 5 |
| "42", "0.5", "17" | After "### Final Answer:" | 25 |

A numeric token in the answer span gets weight 25, meaning it has 25× more influence on the final confidence score than a glue token. The total confidence is the weighted geometric mean: a weighted average of the log-probabilities, then exponentiated back.

**Concrete example:** Suppose the generation has 100 tokens: 80 glue tokens (weight 1 each), 15 numeric reasoning tokens (weight 5 each), and 5 answer-span numeric tokens (weight 25 each). The total weight is 80×1 + 15×5 + 5×25 = 330. The 5 answer-span numeric tokens, despite being only 5% of tokens, contribute 125/330 = 38% of the weight. The 80 glue tokens, despite being 80% of tokens, only contribute 24% of the weight.

**Why the weights are 5 and 25 (not arbitrary):** The values are chosen to be principled rather than tuned. The idea is that numeric tokens carry approximately one order of magnitude more mathematical signal than glue tokens. Within numerics, answer-span tokens are approximately one order of magnitude more predictive than reasoning-chain tokens. Using the same multiplier (5) for both dimensions creates a clean, composable system: numeric × answer-span = 5 × 5 = 25. Crucially, these weights were **not adjusted based on results** — fixing them upfront avoids circular reasoning.

**How it differs from Metrics 1–4:**
- Measure 1: all weights = 1 (glue words dominate by count)
- Measure 4: answer-span numeric tokens weight = 1, everything else = 0 (hard filter, noisy)
- Measure 6: smooth gradient between these extremes (keeps weak signal from all tokens, amplifies strong signal from numeric/span tokens)

**Key in code:** `weighted_mean_confidence`, `weighted_perplexity`
**Range:** 0 to 1. Same interpretation as other confidence measures.
**Expected behaviour:** Lower than Metric 1 (numeric tokens pull it down), higher than Metric 4 (glue tokens pull it up slightly). Potentially better ECE than both because it avoids the noise floor of Metric 4 while avoiding the inflation ceiling of Metric 1.

---

### Perplexity

**What it is:** Perplexity = exp(−mean log probability) = 1 / geometric_mean_probability. It is the inverse of mean confidence, expressed differently.

**Intuition:** Perplexity answers "how many equally-likely choices was the model facing at each step on average?" A perplexity of 2 means the model was effectively choosing between 2 equally-likely options at every step. A perplexity of 100 means the model was deeply uncertain at every step.

**Example:** If geometric mean confidence is 0.8, perplexity = 1/0.8 = 1.25. If confidence is 0.01, perplexity = 100.

**Why both are reported:** Confidence and perplexity are mathematically equivalent (one is the inverse of the other), but researchers in different fields prefer different units. Both are reported for completeness.

**Key in code:** `full_sequence_perplexity`, `answer_span_perplexity`, `numeric_perplexity`, `numeric_span_perplexity`
**Range:** 1 to ∞. Lower = more confident. Perplexity of 1 = perfect certainty.

---

### Minimum Token Probability (min_token_prob)

**What it is:** The single lowest probability assigned to any token in the span being measured.

**Why it matters:** The geometric mean can hide a single catastrophically uncertain token. If a model assigns probability 0.99 to 99 tokens and 0.01 to one critical token (the final answer digit), the geometric mean is ≈ 0.96 — looks confident. But the model was 99% uncertain about the one token that mattered. `min_token_prob` catches this.

**Analogy:** This is the "weakest link" in the chain. A chain is only as strong as its weakest link. A generation is only as reliable as its least-certain token.

**Key in code:** `full_sequence_min_token_prob`, `answer_span_min_token_prob`, `numeric_min_token_prob`, `numeric_span_min_token_prob`
**Range:** 0 to 1. Higher = even the most uncertain token was still reasonably confident.

---

### Standard Deviation of Token Probabilities (std_token_prob)

**What it is:** A measure of how much the per-token confidence varies throughout the generation.

**Intuition:** A generation with std=0.01 means the model was similarly confident (or uncertain) about every token — a flat distribution of certainty. A generation with std=0.3 means some tokens were very certain (prob ≈ 0.99) and others were very uncertain (prob ≈ 0.1) — a spiky distribution.

**Why it's useful:** High standard deviation combined with low minimum probability signals a specific failure mode: the model was confident through most of the reasoning chain but choked at a specific mathematical step. This is different from "uniformly uncertain throughout" and may be a better predictor of the type of error made.

**Key in code:** `full_sequence_std_token_prob`, `answer_span_std_token_prob`, `numeric_std_token_prob`, `numeric_span_std_token_prob`

---

### Expected Calibration Error (ECE)

**What it is:** A single number that measures how well a model's confidence matches its actual accuracy.

**How it is computed:**
1. Group all predictions into 10 buckets by confidence score (0–10%, 10–20%, ..., 90–100%).
2. For each bucket, compute: the average confidence in that bucket, and the actual accuracy (fraction of correct answers).
3. A perfectly calibrated model would have: all predictions in the 70–80% bucket are correct 75% of the time, all predictions in the 90–100% bucket are correct 95% of the time, etc.
4. ECE = weighted average of |accuracy − confidence| across all buckets.

**Example:**
- Bucket 80–90% confidence: model is correct 60% of the time → gap = 25%
- Bucket 90–100% confidence: model is correct 55% of the time → gap = 40%
- ECE would be high (overconfident)

**Range:** 0 to 1. ECE = 0 means perfect calibration. ECE = 0.1 is typical for overconfident models. ECE > 0.2 is poor.

**Why ECE is computed for all six measures:** Each confidence measure produces a different score for the same prediction. ECE computed for each measure tells us which measure is best calibrated — i.e., which measure's confidence scores best predict actual correctness.

**Keys in code:** `ece_answer_confidence`, `ece_full_sequence`, `ece_answer_span`, `ece_numeric_full`, `ece_numeric_span`

---

### Reliability Diagram

**What it is:** The visual version of ECE. A bar chart where:
- X-axis: confidence score (0 to 1)
- Y-axis: actual accuracy in that confidence range
- Diagonal line: perfect calibration (where the bars should reach)

**Reading the diagram:**
- Bars **below** the diagonal: model is overconfident (claims 80% but is only right 60% of the time)
- Bars **above** the diagonal: model is underconfident (claims 50% but is actually right 70% of the time)
- Bars **on** the diagonal: perfectly calibrated

**Expected pattern for Measure 1:** Almost all bars concentrated in the 80–100% range (because full-sequence confidence is inflated), with those bars well below the diagonal (because the model is not actually right 90% of the time just because it sounds fluent).

---

### AUROC (Area Under the ROC Curve) — To Be Added

**What it is:** A measure of how well you can use a confidence score to separate correct answers from incorrect ones.

**Intuition:** Imagine sorting all predictions by confidence, highest first. AUROC answers: if I scan down this sorted list, do correct answers tend to appear near the top? If confidence is a perfect predictor of correctness, all correct answers come first — AUROC = 1.0. If confidence is useless, correct answers are randomly scattered throughout — AUROC = 0.5.

**Key difference from ECE:** ECE measures calibration (do the numbers make sense?). AUROC measures discrimination (can we tell correct from incorrect?). A model can have good AUROC (great at ranking) but poor ECE (the actual numbers are all wrong), or vice versa.

**Range:** 0.5 (random) to 1.0 (perfect). AUROC > 0.7 is considered useful.

---

### Coverage-Accuracy Trade-off

**What it is:** If we only have the model answer questions where it is confident above some threshold, what accuracy do we get, and what fraction of questions does it answer?

**Example:** If we set the threshold at 80% confidence:
- The model answers 40% of questions (those where it is >80% confident)
- On those questions, it is correct 85% of the time
- 60% of questions are "abstained" (no answer given)

**Why it matters:** In a real tutoring or grading system, abstaining on uncertain questions is valuable. You would rather the AI say "I don't know" than confidently give a wrong answer. The coverage-accuracy trade-off tells you how useful the confidence measure is for this abstention task.

**Key in code:** `coverage_at_0.8_confidence`, `coverage_at_0.9_confidence`

---

## 4. Experimental Design

### 4.1 Models

**Model A — Scratch-Trained LLaMA (~1.1B)**

| Property | Value |
|---|---|
| Architecture | LlamaForCausalLM, 16 layers, 2048 hidden, 16 heads |
| Parameter count | ~1.1B |
| Tokenizer | mistralai/Mistral-7B-v0.1 |
| Training data | OpenMathInstruct-2, ~12M samples |
| Training infrastructure | DDP, custom trainer in `src/training/trainer.py` |
| Dropout | attention_dropout=0.1, hidden_dropout=0.1 (baked into weights) |
| UQ method | MC Dropout (N=20 passes, model.train() at inference) |
| Key risk | Model may be underfit — previous run reached loss 5.3 due to training bugs; currently retraining with fixes |

Note: The dropout rates are architectural parameters baked into the model at initialization (`src/model/llama_model.py`, lines 87–89). MC Dropout requires calling `model.train()` at inference time to keep these masks active. The implementation in `MCDropoutEvaluator.__init__()` does this correctly.

**Model B — Fine-Tuned Llama-3.2-1B**

| Property | Value |
|---|---|
| Architecture | meta-llama/Llama-3.2-1B (pretrained) |
| Training data | OpenMathInstruct-2 (same dataset, fine-tuning) |
| Seeds | 3 independent fine-tuning runs (seeds 42, 43, 44) |
| UQ method | Deep Ensembles (3 members, greedy decoding per member) |
| Key risk | 3 members is below the 5-member standard established by Lakshminarayanan et al. 2017 |

**Critical Design Confound:** Models A and B differ in (1) training procedure (scratch vs. fine-tuned), (2) base weights, (3) UQ method, and (4) tokenizer (Mistral vs. Llama-3.2). Any observed difference in UQ quality between the two models cannot be attributed to UQ method alone. This is a fundamental limitation of the current design. To isolate UQ method effects, you would need to apply both MC Dropout and Deep Ensembles to the same base model. This is feasible for Model B (fine-tune additional seeds, enable dropout) but has not been done.

**What this means for the thesis:** The comparison between Model A and Model B should be framed as "comparing two different systems" rather than "comparing MC Dropout vs. Deep Ensembles." The within-model comparison of the six confidence measures is the scientifically cleaner contribution.

### 4.2 Training Data

OpenMathInstruct-2 (NVIDIA, 2024): 14M (12M used) math instruction pairs. Each example formatted as:

```
### Problem:
{problem text}

### Solution:
{reasoning chain}

### Final Answer:
{answer}
```

Loss masking ensures only completion tokens (solution + answer) contribute to training loss — prompt tokens are masked. This is critical context for interpreting confidence measures: the model has never been trained to predict its own prompt, so token probabilities over the prompt are not meaningful.

### 4.3 In-Distribution Benchmarks

- **GSM8K** (Cobbe et al. 2021): 1,319 grade-school math problems. Standard calibration benchmark. ~8-step reasoning chains. Use the test split (1,319 problems).
- **MATH** (Hendrycks et al. 2021): 5,000 competition math problems across 7 subjects, 5 difficulty levels. Much harder than GSM8K. Expect lower accuracy and potentially different calibration patterns by difficulty level. Use the test split.

For calibration studies, aim for at least 500 labeled problems per benchmark per model. 1,000+ is preferable for reliable ECE estimates (10-bin ECE with N=1,000 gives ~100 samples per bin on average).

### 4.4 Out-of-Distribution Tiers

A key contribution of this thesis is evaluating whether UQ measures remain reliable under distribution shift. Three OOD tiers are planned:

| Dataset | Tier | Rationale |
|---|---|---|
| Orca-Math (Microsoft, 2024) | Near-ID | Similar style to OpenMathInstruct-2, same difficulty range, different source |
| AM-Thinking-v1 | Near-OOD | Long chain-of-thought reasoning, potentially different formatting style |
| MegaScience (science subset) | Far-OOD | Scientific calculation, different domain vocabulary, non-math problem structure |

**Expected pattern:** Confidence should remain predictive on near-ID data but degrade on far-OOD. If UQ measures remain well-calibrated on far-OOD data, that is a positive finding (robust uncertainty). If they degrade, that tells you what the model is actually representing with its confidence — familiarity with linguistic style rather than mathematical correctness.

**Note:** You need to verify the MegaScience science subset contains problems with definite correct answers (not open-ended questions). If it does not, the oracle cannot be applied and it cannot be used as a calibration benchmark — only as a distributional shift probe via confidence histograms.

### 4.5 Evaluation Infrastructure

- `scripts/python/evaluate_uq.py`: Entry point for batch evaluation. Supports both `--method mc_dropout` and `--method ensemble`.
- `src/uq/metrics.py`: ECE, reliability diagrams, `normalize_math_answer()`, `summarise()`.
- `src/uq/mc_dropout.py`: `MCDropoutEvaluator` — runs N stochastic passes, extracts answers, computes all six measures.
- `src/uq/ensemble.py`: `EnsembleEvaluator` — loads models sequentially (GPU memory aware), runs greedy decoding, computes all six measures.
- `app.py`: Streamlit demo UI for interactive exploration.
- `src/experiments/`: ExperimentRegistry tracking runs.

---

## 5. Expected Findings

### 5.1 Optimistic Scenario (What You Hope to Find)

1. **Measure 4 (numeric span) significantly outperforms Measure 1 (full sequence) on ECE.** This would confirm the token inflation hypothesis and validate the targeted selection approach.

2. **Answer entropy (Measure 5) has the best AUROC.** Agreement across passes/members captures something qualitatively different from token probability — it reflects output-level uncertainty rather than token-level uncertainty — and is expected to be a better binary discriminator for correct vs. incorrect.

3. **OOD degradation is measurable and measure-dependent.** Some measures (e.g., full-sequence) may degrade gracefully because glue words are also OOD-familiar; others (numeric span) may degrade more sharply because numeric patterns change across domains.

4. **Fine-tuned ensemble (Model B) is better calibrated than scratch MC Dropout (Model A).** This is expected if the scratch model is underfit — underfit models produce uniform token probability distributions that are neither confident nor well-calibrated.

### 5.2 Pessimistic Scenario (What Could Go Wrong)

1. **All token-probability measures have similar ECE.** If the model's numeric token probabilities are also inflated (e.g., if training data contains many similar numeric patterns), the targeted selection may not help. This would be a null result for the main hypothesis.

2. **Answer entropy is noisy due to small N.** With N=20 MC Dropout passes and K=3 ensemble members, the empirical answer distribution is estimated from few samples. On problems where the model is borderline, random variation in pass-to-pass answers may dominate. Bootstrap confidence intervals on entropy would reveal this.

3. **The scratch model is too underfit to draw conclusions.** If the retraining run does not achieve reasonable accuracy on GSM8K (>40%), all UQ measures will look similar — the model produces wrong answers with variable confidence, making calibration meaningless. This is the single biggest risk to the thesis.

4. **The answer oracle fails too often.** If `normalize_math_answer()` fails to match equivalent answers, "incorrect" labels will contaminate "correct" predictions and ECE will be inflated for all measures equally. Check oracle failure rate by manually reviewing ~50 cases where `pred != gold` but the answer looks right.

5. **No confidence measure is reliably predictive.** This is a valid scientific result but makes the thesis harder to write. Frame it as: "We find that token-probability-based confidence measures are not reliable predictors of mathematical correctness in small-scale LLMs, suggesting that calibration may require either larger models or explicit training objectives."

### 5.3 The Most Likely Scenario

Partial confirmation: Measures 2–4 will likely have lower ECE than Measure 1, but the differences may be modest. Answer entropy will likely have the best AUROC on in-distribution data but may not clearly outperform numeric span. OOD calibration will degrade for all measures. The fine-tuned ensemble will likely be better calibrated if the scratch model is adequately trained.

---

## 6. Known Weaknesses and How to Address Them

### 6.1 Ensemble Size (Critical)

**Problem:** 3 ensemble members is below the 5-member standard. Lakshminarayanan et al. (2017) showed that ECE and NLL continue to improve up to ~5 members with diminishing returns thereafter. With 3 members, confidence can only take values {1/3, 2/3, 1} — the confidence distribution is coarse, and ECE bins outside these three values will be empty. This discretization artifact will make the ensemble reliability diagram look unlike standard calibration results.

**Mitigation options (in order of preference):**
1. Train 2 more fine-tuned seeds (seeds 45, 46) to reach 5 members. This is the correct fix.
2. Report the discretization as a limitation and show what the reliability diagram looks like with 3 members vs. the theoretical expectation with 5.
3. Use temperature scaling to smooth ensemble confidence if adding members is not feasible.

### 6.2 No Comparison to Semantic Entropy (Critical)

**Problem:** Kuhn et al. (2023) "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation" is the current state-of-the-art for UQ in language generation. Semantic entropy clusters answers by meaning before computing entropy, avoiding penalizing equivalent answers stated differently. Your `normalize_math_answer()` is a domain-specific approximation of this idea, but you are not comparing against the full semantic entropy method.

**Mitigation options:**
1. Implement semantic entropy using your `normalize_math_answer()` as the equivalence function. This is essentially what you already have — the comparison is between raw entropy over string-exact answers vs. entropy over normalized answers. Add this as a sub-comparison.
2. Cite Kuhn et al. explicitly, acknowledge that your `normalize_math_answer()` approach is a domain-specific proxy for semantic clustering, and frame this as a limitation.
3. If time permits, compare your results to numbers from Kuhn et al. on GSM8K (they evaluate on NQ, not GSM8K, so direct comparison may not be possible).

### 6.3 No Statistical Significance Testing

**Problem:** Comparing ECE values without confidence intervals makes it impossible to know whether observed differences are meaningful. ECE itself is a biased estimator that depends on bin choice (you use 10 bins). Two models with ECE 0.08 vs. 0.11 may not be significantly different.

**Mitigation:**
1. Bootstrap confidence intervals on ECE: resample the evaluation set with replacement 1,000 times, compute ECE for each resample, report the 2.5th and 97.5th percentiles.
2. For AUROC, use the DeLong test for comparing two ROC curves from the same set of examples.
3. This is straightforward to add to `metrics.py` and should be done before writing the results chapter.

### 6.4 Scratch Model Training Quality

**Problem:** A previous training run reached loss 5.3 on the training set, indicating a training bug (correct loss for a well-trained ~1B model on math should be in the range 1.5–2.5 after sufficient training). The model is currently retraining. Until training completes with reasonable loss, you cannot trust any results from Model A.

**Mitigation:**
1. Do not run UQ evaluation on Model A until you have verified (a) training loss is reasonable, (b) GSM8K accuracy on the test set is >30% (GPT-2-level floor), and (c) the model produces formatted outputs matching "### Final Answer:" reliably.
2. Run a quick sanity check: evaluate on 100 GSM8K examples with greedy decoding before committing to the full evaluation run.
3. If the scratch model cannot be trained to reasonable quality within the thesis timeline, consider framing Model A as a "low-quality baseline" and focusing the main analysis on Model B.

### 6.5 Confounded Model Comparison

**Problem:** As noted in Section 4.1, Models A and B differ in training procedure, architecture initialization, tokenizer, and UQ method simultaneously. You cannot cleanly attribute any performance difference to any single factor.

**Mitigation:**
1. Add a third condition: fine-tuned Llama-3.2-1B with MC Dropout enabled (add `attention_dropout=0.1` during fine-tuning, evaluate with MC Dropout). This gives you a within-model UQ method comparison.
2. Alternatively, be explicit in the thesis framing: "We compare two full systems, not two UQ methods. Isolating UQ method effects is left for future work." This is honest and acceptable for a bachelor thesis.

### 6.6 Token Granularity: What Counts as "Numeric"

**Problem:** The current implementation in `metrics.py` (`_is_numeric_token`) classifies a token as numeric if it contains any digit character. This will include tokens like "1st", "2nd", "100%", and date fragments ("2023"). It will miss numeric tokens expressed as words ("forty-two", "zero"). The definition is a reasonable heuristic but has known failure modes.

**Mitigation:**
1. Report the average number of numeric tokens per problem (already available as `n_numeric_tokens` and `n_numeric_span_tokens` in the output). If this is very small (< 3 on average), the measure will be high-variance.
2. Ablate: try a stricter definition (token is numeric only if the entire stripped token is a digit sequence) and compare ECE.

### 6.7 MC Dropout Temperature

**Problem:** The current MC Dropout implementation uses `temperature=1.0` by default for sampling. MC Dropout with sampling introduces two sources of stochasticity: dropout mask variation and token sampling randomness. These are confounded. If you want pure MC Dropout uncertainty (epistemic), you should use greedy decoding (temperature=0) and let dropout be the only source of variation.

**Mitigation:**
1. Re-run MC Dropout evaluation with `temperature=0` (greedy, dropout only). Compare answer distributions to `temperature=1.0` runs.
2. Note this in the methodology section. The current implementation is arguably measuring a mix of epistemic (dropout) and aleatoric (sampling) uncertainty, which may or may not be desirable.

---

## 7. Thesis Chapter Outline

### Chapter 1: Introduction (8–12 pages)

- The calibration problem in deployed LLMs
- Why mathematical reasoning is a particularly clean test case
- The token probability inflation problem (your core insight — motivate it with a worked example)
- Research questions and hypotheses
- Contributions summary
- Thesis structure overview

### Chapter 2: Background and Related Work (15–20 pages)

**2.1 Uncertainty in Machine Learning**
- Epistemic vs. aleatoric uncertainty
- Calibration: the reliability diagram, ECE, Brier score
- Guo et al. (2017): modern neural networks are overconfident; temperature scaling

**2.2 Uncertainty Quantification for LLMs**
- MC Dropout (Gal & Ghahramani 2016): the theoretical basis
- Deep Ensembles (Lakshminarayanan et al. 2017): why ensembles work empirically
- Semantic Uncertainty (Kuhn et al. 2023): SOTA for generation UQ; why string-exact entropy fails

**2.3 Mathematical Reasoning in LLMs**
- Chain-of-thought prompting (Wei et al. 2022)
- GSM8K and MATH benchmarks
- Token probability as a proxy for answer confidence: prior work and failures

**2.4 Token Probability Inflation**
- This is where you introduce your core theoretical argument
- Connect to the literature on "verbosity bias" and "hedging tokens"
- This section is your intellectual contribution — spend time on it

### Chapter 3: Methodology (15–20 pages)

**3.1 Model Architecture and Training**
- Scratch LLaMA: architecture details, training procedure, data preprocessing
- Fine-tuned Llama-3.2-1B: fine-tuning procedure, 3 seeds
- Training data: OpenMathInstruct-2, formatting strategy, loss masking

**3.2 Confidence Measures**
- Formal definition of all six measures (reproduce Section 3.1 of this document, polished)
- Mathematical relationship between the measures (Measure 4 ⊆ Measure 2 ⊆ Measure 1; Measure 4 ⊆ Measure 3; Measure 6 is a continuous interpolation across Measures 1–4)
- Implementation details: marker detection, fallback behavior, NaN handling

**3.3 Evaluation Protocol**
- ECE, reliability diagrams, AUROC, coverage-accuracy trade-off
- Answer oracle: `normalize_math_answer()` with known limitations
- Bootstrap confidence intervals on ECE

**3.4 Benchmarks and OOD Datasets**
- GSM8K, MATH (ID)
- Orca-Math, AM-Thinking-v1, MegaScience (OOD tiers)
- Why these datasets, what distribution shift each represents

### Chapter 4: Results (20–25 pages)

**4.1 Model Quality**
- Training curves (loss vs. steps) for both models
- In-distribution accuracy: GSM8K and MATH
- Output format compliance rate (how often does the model produce "### Final Answer:"?)
- Sanity check: are confidence measures numerically stable? (No NaN epidemics)

**4.2 Calibration: In-Distribution**
- ECE table: all six measures × two models × two benchmarks
- Reliability diagrams for all six measures (one figure per measure, subplots per model)
- AUROC for all six measures
- Coverage-accuracy curves

**4.3 Calibration: Out-of-Distribution**
- ECE by OOD tier for the best-performing measure per model
- Confidence distribution shift under OOD (histograms)
- Does the model's confidence decrease under OOD? (It should, ideally)

**4.4 Analysis: What Makes a Good Confidence Measure?**
- Is the token inflation hypothesis confirmed? (Compare ECE Measure 1 vs. Measures 2–4)
- Within-model comparison across six measures: ranking and statistical significance
- Model A vs. Model B: system-level comparison with the confound caveat

### Chapter 5: Discussion (10–15 pages)

**5.1 The Token Inflation Effect**
- Quantify the inflation: report mean confidence for each measure, show the systematic upward bias for Measure 1
- Does targeting numeric tokens actually help? By how much?
- Practical implication: when does it matter?

**5.2 Answer Entropy vs. Token Probability**
- Qualitative comparison: what does each measure capture?
- Cases where they disagree (find examples where token probability is high but entropy is high, or vice versa)
- Which should a practitioner use?

**5.3 OOD Generalization of Confidence**
- Do confidence measures remain predictive under distribution shift?
- Connect to broader literature on calibration under covariate shift

**5.4 Limitations**
- Confounded model comparison (Section 6.5)
- 3-member ensemble (Section 6.1)
- Oracle failures (Section 6.4)
- No comparison to Semantic Entropy (Section 6.2)
- Scope: one domain (math), one model family, one training procedure

### Chapter 6: Conclusions and Future Work (5–8 pages)

- Summary of contributions
- Key findings (with honest qualification)
- Future work:
  - Apply MC Dropout to fine-tuned model (deconfound the comparison)
  - Expand to 5+ ensemble members
  - Implement Semantic Entropy (Kuhn et al. 2023) as a direct competitor
  - Calibration-aware fine-tuning (optimize ECE directly)
  - Selective prediction: deploy the best-calibrated measure in a real tutoring system

### Appendices

- A: Full ECE tables with bootstrap confidence intervals
- B: Reliability diagrams (all six measures, all benchmarks)
- C: Training configuration details
- D: Sample outputs (correct and incorrect examples with confidence scores)
- E: Oracle failure analysis

---

## 8. Critical Self-Assessment

### What This Work Does Well

**The core insight is real and underexplored.** Token probability inflation from glue words is a genuine problem that the literature has not systematically studied in the context of mathematical reasoning. The six-measure taxonomy is well-motivated and the implementation is clean. The `normalize_math_answer()` oracle is a genuine contribution — string-exact matching on math is broken, and your normalization function handles the most important cases (LaTeX, fractions, trailing zeros).

**The experimental infrastructure is solid.** Batch evaluation, reliability diagrams, ECE for multiple confidence keys, and the Streamlit demo are all implemented and working. The ExperimentRegistry adds reproducibility. This is not a paper-only project — there is real, runnable code.

**The OOD evaluation design is thoughtful.** Three tiers of distribution shift allow you to make graduated claims rather than just "ID vs. OOD." Few calibration papers for math LLMs include this level of distributional analysis.

### What Is Borderline

**The sample size may be too small for reliable ECE estimates.** With 10 bins and 500 evaluation examples, you get ~50 examples per bin on average — enough for a rough estimate but not for tight confidence intervals. 1,000+ examples per benchmark is strongly preferred.

**The within-model analysis (six measures on the same model) is the strongest contribution.** The between-model analysis (Model A vs. B) is confounded and should be downplayed. Make sure the thesis framing reflects this — lead with the six-measure comparison, not the model comparison.

**The OOD datasets need vetting.** Before including any dataset in the evaluation, manually inspect 20–30 examples to verify: (a) the dataset has definite correct answers, (b) the `normalize_math_answer()` oracle can handle the answer format, and (c) the problems are diverse enough to be informative.

### What Is Weak

**No comparison to Semantic Entropy.** This is the most significant gap relative to the current literature. You can partially address this by showing that your `normalize_math_answer()` normalization improves entropy estimates compared to string-exact entropy — this is essentially demonstrating the "semantic" aspect of your approach.

**The theoretical framework is informal.** The claim that full-sequence token probability is "inflated" by glue words is intuitive but not formally derived. A more rigorous treatment would define an "informativeness" criterion for tokens and show that glue words score low by this criterion. Even a short formal section (2–3 pages) would strengthen the thesis significantly.

**N=20 MC Dropout passes may be insufficient.** For a 1B parameter model on multi-step reasoning, 20 passes may not cover the space of plausible answers. Compare the answer distribution at N=10, N=20, N=50 on a subset to check convergence.

### Is This Good Enough for a Bachelor Thesis?

**Yes, if:**
- The scratch model trains to reasonable quality (loss < 2.5, GSM8K accuracy > 40%)
- The six-measure ECE comparison produces clear ordering with statistically significant differences
- Bootstrap confidence intervals are added to all ECE comparisons
- The thesis framing is honest about the confounded model comparison

**It would be clearly publishable (workshop paper level) if:**
- MC Dropout is also applied to the fine-tuned model, deconfounding the comparison
- Ensemble size is expanded to 5 members
- A formal statement of the token inflation effect is provided
- Comparison to string-exact entropy (as a proxy for Kuhn et al.) is added

**Current estimate:** With the existing infrastructure and the retraining completing successfully, this is a strong bachelor thesis. The core insight is original, the implementation is real, and the experimental design is thoughtful. The weaknesses are known and addressable. Do not wait for perfection — run the evaluation, report what you find honestly, and acknowledge what you could not do.

---

## 9. Essential Reading List

**Foundational Calibration**

1. **Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML.**
   Why it matters: Establishes that deep neural networks are overconfident after training; introduces temperature scaling; defines ECE and reliability diagrams as the standard evaluation protocol. This is the starting point for all calibration work.

2. **Niculescu-Mizil & Caruana (2005). "Predicting Good Probabilities with Supervised Learning." ICML.**
   Why it matters: Pre-deep-learning baseline for calibration methods (Platt scaling, isotonic regression); useful for understanding what calibration post-processing means.

**Uncertainty Quantification Methods**

3. **Gal & Ghahramani (2016). "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning." ICML.**
   Why it matters: The theoretical foundation for MC Dropout. Shows that a network with dropout is equivalent to a variational approximation to a Gaussian process. Required reading before claiming MC Dropout measures epistemic uncertainty.

4. **Lakshminarayanan, Pritzel & Blundell (2017). "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles." NeurIPS.**
   Why it matters: Establishes Deep Ensembles as a strong baseline (often better than Bayesian methods); shows 5 members is sufficient; demonstrates proper scoring rules for evaluation. Directly relevant to your ensemble setup.

5. **Kuhn, Gal & Farquhar (2023). "Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation." ICLR.**
   Why it matters: Current SOTA for UQ in generation tasks. Semantic entropy clusters semantically equivalent answers before computing entropy — directly addresses the string-matching problem you solve with `normalize_math_answer()`. This is the paper you are closest to and furthest from simultaneously.

**Mathematical Reasoning**

6. **Cobbe et al. (2021). "Training Verifiers to Solve Math Word Problems." arXiv.**
   Why it matters: Introduces GSM8K; establishes the chain-of-thought evaluation protocol for grade school math. You use this dataset — you must cite this paper.

7. **Hendrycks et al. (2021). "Measuring Mathematical Problem Solving with the MATH Dataset." NeurIPS.**
   Why it matters: Introduces the MATH benchmark; characterizes difficulty levels and subject areas. You use this dataset — you must cite this paper.

8. **Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS.**
   Why it matters: Establishes chain-of-thought as the dominant paradigm for math reasoning in LLMs. The formatted "Solution → Final Answer" structure in your training data is a direct application of this work.

**LLM-Specific Calibration**

9. **Kadavath et al. (2022). "Language Models (Mostly) Know What They Know." arXiv.**
   Why it matters: Shows that large LMs are reasonably well-calibrated in self-assessment ("Is this answer correct?") but this calibration breaks down for smaller models. Sets expectations for what calibration looks like at 1B scale.

10. **Xiong et al. (2024). "Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs." ICLR.**
    Why it matters: Systematic evaluation of verbal vs. numerical confidence elicitation in LLMs; shows that verbalized confidence is often poorly calibrated; directly relevant to why you use token probabilities rather than asking the model "how confident are you?"

11. **Tian et al. (2023). "Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback." EMNLP.**
    Why it matters: Shows that RLHF fine-tuning degrades calibration; relevant context for why your fine-tuned model (Model B) might be less calibrated than expected.

**Token Probability as Confidence**

12. **Malinin & Gales (2021). "Uncertainty Estimation in Autoregressive Structured Prediction." ICLR.**
    Why it matters: Formal treatment of uncertainty decomposition in sequence generation; distinguishes total, data, and knowledge uncertainty in a sequence probability decomposition. Provides mathematical grounding for why full-sequence probability is a poor uncertainty measure.

13. **Fomicheva et al. (2020). "Unsupervised Quality Estimation for Neural Machine Translation." TACL.**
    Why it matters: Shows that token probability and related measures (attention entropy, word-level probability) can estimate output quality in MT without ground truth labels. Conceptual predecessor to your token-probability-as-confidence approach.

**Training Data**

14. **Toshniwal et al. (2024). "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data." arXiv (NVIDIA).**
    Why it matters: The training dataset you used. Required citation; also useful for understanding the data distribution, formatting choices, and known biases.

---

*This document should be updated after every significant experimental result, model checkpoint, or design decision. The critical self-assessment section in particular should be revised as results come in — it is honest only relative to current knowledge.*
