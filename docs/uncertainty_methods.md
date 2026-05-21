# Uncertainty Quantification Methods

This document describes how epistemic uncertainty is estimated during inference.
The MC Dropout evaluation loop lives in `src/uq/mc_dropout.py`; confidence
measures are computed in `src/uq/metrics.py`.

---

## Experiment design

The evaluation is a **2 × 3 factorial design** using MC Dropout:

| | Zero-shot | Few-shot (CoT) | Few-shot + Step-by-step |
|---|---|---|---|
| **Llama 3.2-1B (LoRA rank 16)** | accuracy + UQ | accuracy + UQ | accuracy + UQ |
| **Llama 3.1-8B (LoRA rank 64)** | accuracy + UQ | accuracy + UQ | accuracy + UQ |

Each cell runs on two test sets: GSM8K (in-distribution) and MATH (out-of-distribution).
All three prompt variants instruct the model to use `<<expr=result>>` annotations,
enabling uniform metric computation across all cells.

---

## MC Dropout

MC Dropout re-activates the LoRA dropout (`lora_dropout=0.1`) at inference time by
calling `model.train()`. An additional `nn.Dropout` hook is inserted after the final
RMSNorm layer via `_add_mc_dropout_hook()`. Each of the `num_passes=20` greedy
forward passes produces a different stochastic prediction due to the active dropout.

Epistemic uncertainty is estimated from **disagreement across passes**: if all 20
passes give the same answer the model is highly certain; if each pass gives a
different answer the model has high epistemic uncertainty.

---

## Confidence measures

Three complementary confidence signals are computed per problem, forming a progression from naive to focused:

| Key in `results.json` | What it measures |
|---|---|
| `confidence` | Majority-vote fraction across passes — answer-level signal |
| `full_sequence_mean_confidence` | Unweighted geometric mean token prob — baseline token-level signal |
| `weighted_mean_confidence` | Weighted geometric mean token prob — focused token-level signal |

---

### 1. Majority-vote confidence (answer-level)

```
confidence = count(majority_answer) / num_passes
```

Fraction of the 20 passes that agreed on the most common answer.
- `confidence = 1.0` — all passes gave the same answer (certain)
- `confidence = 0.05` — each pass gave a different answer (maximally uncertain)

Captures epistemic uncertainty through *answer-level disagreement across passes*.
This is the primary UQ signal — it directly measures how consistently the model
commits to an answer under stochastic dropout.

---

### 2. Unweighted geometric mean token probability (token-level baseline)

```
full_sequence_mean_confidence = exp( (1/N) * Σ log(p_i) )
```

Geometric mean over all per-token probabilities in the full generated sequence,
averaged across all 20 MC Dropout passes. Treats every token equally regardless
of its role in the reasoning chain.

**Limitation**: dominated by high-frequency glue tokens ("the", "and", "therefore")
that the model assigns near-certainty probability to regardless of whether the
math is correct. This inflates confidence and weakens its correlation with
correctness — included as a baseline to demonstrate this effect.

---

### 3. Weighted geometric mean token probability (token-level, focused)

```
weighted_mean_confidence = exp( Σ w_i * log(p_i) / Σ w_i )
```

Same geometric mean but with per-token weights that focus the signal on the
tokens the model explicitly commits to as arithmetic facts. The weighting
exploits the `<<expr=result>>` annotation format:

| Token region | Weight |
|---|---|
| Tokens in `Final Answer: X` span | **25×** |
| Result tokens inside `<<expr=result>>` | **10×** |
| All other tokens | **1×** |

Because all prompts instruct the model to annotate arithmetic inline, result
tokens are identified precisely by regex rather than heuristic numeric detection.
This makes the weighting more principled than detecting "numeric tokens" by
character content.

Averaged across all 20 MC Dropout passes.

**Hypothesis**: by down-weighting glue tokens and up-weighting the tokens where
arithmetic errors actually occur, this measure should correlate better with
correctness than the unweighted baseline.

---

### 4. Answer entropy

```
H = -Σ p(a) * log₂(p(a))
```

Shannon entropy over the answer distribution across the 20 passes.
- `H = 0` — all passes agree
- `H = log₂(N)` — all N passes give different answers

Complements majority-vote confidence by capturing the full shape of the answer
distribution, not just the plurality fraction. A model that splits 10/10 between
two answers has the same `confidence = 0.5` as one that splits 10/10 between ten
answers, but very different entropy. Reported in `summary.json` as
`mean_answer_entropy`.
