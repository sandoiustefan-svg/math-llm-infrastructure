# Experiment Findings

This document records what has been run, what was found, and what is planned next.

---

## Phase 1 — Exploratory (GSM8K, three prompt variants)

### Setup

- **Models:** 1B LoRA (rank 16, seed 42), 8B LoRA (rank 64, seed 42)
- **Test set:** GSM8K, 500 problems
- **Method:** MC Dropout, 20 passes, greedy decoding, repetition_penalty=1.3
- **Prompts:** zero_shot, cot, cot_step_by_step (all with `<<expr=result>>` annotation instruction)
- **Results location:** `results/results_GM8SK_ONLY/`

### Accuracy

| Model | zero_shot | cot | cot_step_by_step |
|---|---|---|---|
| 1B | 40.6% | 34.4% | 34.8% |
| 8B | 57.6% | 58.0% | 59.0% |

### Calibration (ECE, majority-vote confidence)

| Model | zero_shot | cot | cot_step_by_step |
|---|---|---|---|
| 1B | 0.119 | 0.179 | 0.164 |
| 8B | 0.051 | 0.057 | 0.068 |

### Discrimination (AUROC, majority-vote confidence)

| Model | zero_shot | cot | cot_step_by_step |
|---|---|---|---|
| 1B | 0.837 | 0.836 | 0.795 |
| 8B | 0.861 | 0.845 | 0.888 |

### Overconfidence Rate (high-confidence predictions that are wrong)

| Model | zero_shot | cot | cot_step_by_step |
|---|---|---|---|
| 1B | 20.3% | 26.2% | 30.2% |
| 8B | 4.3% | 7.4% | 4.1% |

---

## Phase 1 — What We Learned

### Finding 1: MC Dropout uncertainty is informative
AUROC >0.83 in all 6 conditions. The model's disagreement across stochastic passes
reliably separates correct from incorrect answers regardless of model size or prompt.

### Finding 2: Model scale improves calibration by ~3×
8B ECE (0.051) vs 1B ECE (0.119) under zero-shot. The larger model's confidence
tracks its actual accuracy substantially better.

### Finding 3: Zero-shot is best for both accuracy and calibration
Zero-shot outperforms both CoT variants for the 1B model on all three metrics
(accuracy, ECE, overconfidence rate). For the 8B the differences are small but
zero-shot remains best or tied on ECE. This is explained by the format mismatch
below.

### Finding 4: The `<<expr=result>>` annotation format was never followed
The 1B model produced zero annotations across all 10,000 outputs. The 8B produced
them in fewer than 3% of passes. OpenMathInstruct-2 training data uses natural prose
and LaTeX — no `<<expr=result>>` format — so fine-tuning reinforced the opposite
habit. As a result:

- The `weighted_mean_confidence` measure (which relies on identifying these tokens)
  collapsed into a degraded answer-span signal.
- `weighted_mean_confidence` had worse ECE than majority-vote in every condition —
  not because weighting is a bad idea, but because the annotation it depends on
  does not exist in the model's outputs.

### Finding 5: CoT prompts are out-of-distribution for LoRA fine-tuned models
Fine-tuning used exclusively single-turn conversations (system + user + assistant).
CoT prompts add 3 user/assistant example turns, creating a 7-turn conversation the
model was never trained on. This explains the 1B accuracy drop (~6pp) and calibration
degradation under CoT. The 8B is less affected because its base model has stronger
residual multi-turn capability from pre-training.

### Finding 6: Token-level confidence predicts solution quality better than binary correctness
For the 8B model, full-sequence AUROC against ROUGE-L reaches 0.95 — substantially
higher than its AUROC against binary correctness (0.86). Token probability is a better
signal for *how good* an answer is than for *whether* it is right or wrong.

---

## Phase 1 — What Did Not Work

| Design element | Problem | Consequence |
|---|---|---|
| `<<expr=result>>` annotation instruction | Never produced by either model | `weighted_mean_confidence` is invalid as designed |
| CoT few-shot examples | Out-of-distribution conversation structure | Hurts 1B accuracy and calibration; minimal effect on 8B |
| `repetition_penalty=1.3` | Likely suppresses `<<` and `>>` tokens | Contributed to 0% annotation adherence |

---

## Phase 2 — Plan (Primary UQ Experiment)

### Changes from Phase 1

1. **Prompts:** `zero_shot_aligned` + `cot` (2 prompts, see rationale below).
   - `zero_shot_aligned`: matches fine-tuning format exactly — in-distribution anchor.
   - `cot`: 3 few-shot reasoning examples, different system prompt, multi-turn structure —
     intentionally out-of-distribution to test whether prompt engineering hurts calibration.
     `<<expr=result>>` annotations removed from `cot.py`; examples now use plain arithmetic.
2. **Test sets:** `zero_shot_aligned` runs on GSM8K + MATH; `cot` runs on GSM8K only
   (prompting effect is separate from domain shift — no need to conflate them).
3. **Confidence signals:** majority-vote, full-sequence, numeric-span (replaces weighted,
   which was invalid). `numeric_span_mean_confidence` focuses on numeric tokens within
   the Final Answer span — the most principled signal given what the model actually produces.
4. **Correctness signals:** binary exact-match, ROUGE-L, LLM judge rank (good/medium/bad).
   All three are used to rule out artefacts of any single correctness definition.
5. **Bug fixes applied before rerun:**
   - `math.isfinite()` guard on `normalize_math_answer` (OverflowError on "inf" answers)
   - Depth-aware `_extract_boxed()` replacing `[^}]+` regex for nested LaTeX braces
   - `data/test_sets/math.jsonl` must be deleted and rebuilt with the fix

### Experiment Design

```
2 models × 2 prompts × 3 confidence signals × 3 correctness metrics
```

`zero_shot_aligned` runs on both test sets; `cot` runs on GSM8K only.

| | GSM8K | MATH |
|---|---|---|
| **1B — zero_shot_aligned** | accuracy + ECE + AUROC (3 signals × 3 correctness) | same |
| **8B — zero_shot_aligned** | accuracy + ECE + AUROC (3 signals × 3 correctness) | same |
| **1B — cot** | accuracy + ECE + AUROC (3 signals × 3 correctness) | — |
| **8B — cot** | accuracy + ECE + AUROC (3 signals × 3 correctness) | — |

### Why Two Prompts

Both prompts share the **identical system prompt**. The only difference is conversation
structure:
- `zero_shot_aligned`: single-turn (system + user) — matches fine-tuning format exactly.
- `cot`: 7-turn (system + 3× user/assistant examples + user) — OOD multi-turn structure.

This isolates the effect of conversation structure alone, with system prompt controlled.

The thesis argument: *MC Dropout produces reliable uncertainty estimates only when the
inference prompt matches the training distribution. Multi-turn structure — even with the
same system prompt — degrades calibration, a silent failure mode for UQ in practice.*

The 1B model is expected to show both accuracy and calibration degradation under `cot`.
The 8B is expected to show calibration degradation even if accuracy holds — the more
interesting and subtle finding.

### Run Order

Run `zero_shot_aligned` first — its GSM8K cells serve double duty as the in-distribution
anchor for the prompting experiment and as Phase 2's primary result.

**Step 1 — zero_shot_aligned, both models, GSM8K + MATH**
```bash
python scripts/python/run_uq_eval.py \
    --cluster macross_1b_3090 \
    --method mc_dropout \
    --test-source all \
    --prompt zero_shot_aligned \
    --seed 42

python scripts/python/run_uq_eval.py \
    --cluster macross_8b_3090 \
    --method mc_dropout \
    --test-source all \
    --prompt zero_shot_aligned \
    --seed 42 \
    --base-model meta-llama/Llama-3.1-8B-Instruct
```

**Step 2 — cot, both models, GSM8K only**
```bash
python scripts/python/run_uq_eval.py \
    --cluster macross_1b_3090 \
    --method mc_dropout \
    --test-source gsm8k \
    --prompt cot \
    --seed 42

python scripts/python/run_uq_eval.py \
    --cluster macross_8b_3090 \
    --method mc_dropout \
    --test-source gsm8k \
    --prompt cot \
    --seed 42 \
    --base-model meta-llama/Llama-3.1-8B-Instruct
```

### Research Questions This Answers

1. Does MC Dropout produce reliable uncertainty estimates for LoRA fine-tuned LLMs? (AUROC)
2. Does model scale improve calibration? (1B vs 8B ECE)
3. Does distribution shift degrade calibration? (GSM8K vs MATH ECE/AUROC)
4. Which confidence signal best predicts correctness — majority-vote, full-sequence, or numeric-span?
5. Does prompt-induced distribution shift silently degrade UQ reliability? (zero_shot_aligned vs cot)

### Remaining Steps After Inference

- Run LLM judge on all results (`src/uq/llm_judge.py`) — adds judge_rank as third correctness signal
- Delete `data/test_sets/math.jsonl` before running MATH to rebuild with the boxed-brace fix
