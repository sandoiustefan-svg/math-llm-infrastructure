# math-llm-infrastructure

LLaMA LoRA fine-tuning infrastructure for OpenMathInstruct-2, with epistemic
Uncertainty Quantification (MC Dropout) evaluated across two model sizes, two
prompt variants, and two test sets.

The core research question: **does MC Dropout produce reliable epistemic uncertainty
estimates for LoRA fine-tuned LLMs on mathematical reasoning, and what factors affect
that reliability?**

Correctness is measured by three signals: binary answer match (+ sympy equivalence
for MATH), LLM-as-judge rank (`good / medium / bad`), and NLG baselines (ROUGE-L /
METEOR, retained to demonstrate their inadequacy for mathematical reasoning).
See `docs/metrics.md` and `docs/plots.md`.

---

## Environment Setup

```bash
bash scripts/bash/setup_env.sh
```

---

## Preprocessing Pipeline

Converts raw OpenMathInstruct-2 examples into packed `.npy` shards for training.

### Pipeline overview

```
HuggingFace (nvidia/OpenMathInstruct-2)
    │
    ▼  inspect_data.py
raw.jsonl          {problem, generated_solution, expected_answer, problem_source}
    │
    ▼  format_data.py          (Strategy 3 — chat messages + loss mask boundary)
formatted.jsonl    {messages, prompt_messages, completion_text}
    │
    ▼  tokenize_data.py        (apply_chat_template, dual-pass loss masking)
tokens.jsonl       {input_ids, attention_mask, loss_mask}
    │
    ▼  pack_data.py            (fixed-length rows, shard files)
shards/
  input_ids_XXXXX.npy        (S, seq_len)  int32
  attention_mask_XXXXX.npy   (S, seq_len)  int8
  loss_mask_XXXXX.npy        (S, seq_len)  int8
  manifest.json
```

**Strategy 3 formatting** — each example becomes three chat messages:

| Role | Content |
|---|---|
| `system` | "You are a careful mathematical reasoning assistant. Solve the problem step by step." |
| `user` | Problem text |
| `assistant` | Solution + `\n\nFinal Answer: {answer}` |

Loss is applied **only to completion tokens** (assistant response + EOS). Prompt tokens are masked out.

**Packing** — variable-length examples are chunked into fixed `(seq_len,)` rows. Each shard holds `shard_num_seqs` rows.

### Run preprocessing

```bash
# Debug (5k examples, fast)
bash scripts/bash/preprocess.sh --debug

# Full dataset
bash scripts/bash/preprocess.sh
```

### Inspect the dataset

```bash
python scripts/python/inspect_data.py --limit 3
```

---

## Training

### Models

| Model | Base | LoRA rank | Alpha | Trainable params | Steps | Precision |
|---|---|---|---|---|---|---|
| 1B | Llama-3.2-1B-Instruct | 16 | 32 | ~21 M | 408 000 | bf16 |
| 8B | Llama-3.1-8B-Instruct | 64 | 128 | ~168 M | 408 000 | bf16 |

Both use plain LoRA (no quantisation). LoRA adapts 7 modules per layer:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
`lora_dropout=0.05` is set on all adapter layers — this is the sole stochasticity
source for MC Dropout at inference. No additional hooks are added.

### Data split

```
total shards T  (13 646 for full OpenMathInstruct-2)
  train  = [0,        T − val − test)   ≈ 80 %
  val    = [T−val−test,  T − test)       ≈ 10 %
  test   = [T − test,    T)              ≈ 10 %   — held out
```

### Launch

```bash
# 1B model, seed 42
bash scripts/bash/train.sh configs/llama3_1b_lora.yaml 42

# 8B model, seed 42
bash scripts/bash/train.sh configs/llama3_8b_lora.yaml 42
```

---

## Uncertainty Quantification

### Method — MC Dropout

`lora_dropout=0.05` is active on all adapter layers during training and reactivated
at inference via `model.train()` — no additional hooks are added. Running
**N=20 greedy forward passes** per problem yields a distribution over answers;
disagreement and variance across passes estimate epistemic uncertainty over the
LoRA adapter weights.

```python
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, checkpoint_path)
model.train()   # re-activates LoRA dropout

answers = [model.generate(input_ids, do_sample=False) for _ in range(20)]
```

### Confidence scores

| Key | What it measures |
|---|---|
| `confidence` | Majority-vote fraction — fraction of 20 passes agreeing on the final answer |
| `consistency_rate` | Σ p(a)² — considers the full answer distribution, not just the top answer |

### Uncertainty measures

| Key | What it measures |
|---|---|
| `entropy` | Shannon entropy over answer distribution across 20 passes |
| `n_unique_answers` | Count of distinct answers across 20 passes |
| `std_log_prob` | Std of per-pass avg log-prob — token-level epistemic variance |
| `std_numeric_span_log_prob` | Same restricted to numeric tokens in the Final Answer span |

### Correctness signals

| Key | What it measures |
|---|---|
| `correct` | Binary string match; sympy symbolic equivalence fallback for MATH |
| `judge_rank` | `good / medium / bad` from LLM judge — produced by `src/uq/llm_judge.py` |
| `mean_rougeL`, `mean_meteor` | Mean NLG scores across 20 passes vs `reference_solution` — weak baselines |

### Aggregate metrics (summary.json)

| Metric | Meaning |
|---|---|
| `ece_confidence`, `ece_consistency` | ECE against binary correctness |
| `auroc_confidence`, `auroc_consistency` | AUROC — binary correct vs wrong |
| `auroc_entropy`, `auroc_n_unique`, `auroc_std_log_prob`, `auroc_std_numeric_span` | AUROC for uncertainty measures (negated) |
| `ece_judge_*`, `auroc_judge_*` | ECE / AUROC against LLM judge score |
| `ece_rougeL_*`, `auroc_rougeL_*` | ECE / AUROC against ROUGE-L (NLG baseline) |
| `overconf_rate` | Fraction of high-confidence (≥ 0.8) predictions that are wrong |

### Evaluation scripts

```bash
# Run inference + all metrics
python scripts/python/run_uq_eval.py \
    --cluster macross_1b_3090 \
    --method mc_dropout \
    --test-source all \
    --prompt zero_shot_aligned \
    --seed 42

# Available prompts: zero_shot_aligned | cot
# Available test sources: gsm8k | math | all

# Post-inference: LLM judge labelling
python -m src.uq.llm_judge \
    --results results/mc_dropout/1b/seed42/gsm8k/zero_shot_aligned/results.json \
    --provider anthropic --model claude-haiku-4-5-20251001
```

---

## Experiment Design

The main experiment is a **2×2 design** across model size and prompt variant:

| | zero_shot_aligned | cot |
|---|---|---|
| **1B LoRA rank 16** | GSM8K + MATH | GSM8K only |
| **8B LoRA rank 64** | GSM8K + MATH | GSM8K only |

`zero_shot_aligned` matches the fine-tuning format exactly (in-distribution anchor).
`cot` uses a multi-turn few-shot structure that is intentionally out-of-distribution,
isolating the effect of prompt-induced distribution shift on UQ reliability.

### Research questions

1. Does MC Dropout produce reliable epistemic uncertainty estimates for LoRA fine-tuned LLMs? (AUROC)
2. Does model scale improve calibration? (1B vs 8B ECE)
3. Does prompt-induced distribution shift silently degrade epistemic UQ? (zero_shot_aligned vs cot)
4. Does domain shift degrade calibration? (GSM8K vs MATH ECE/AUROC)
5. Which signal better reflects epistemic uncertainty — answer-level (majority-vote, entropy) or token-level (std_log_prob)?

See `docs/metrics.md` for metric definitions and `docs/plots.md` for plot interpretation.

---

## Experiment Registry

Every training run is automatically registered in `experiments/registry.json`.

```bash
python scripts/python/experiment.py list
python scripts/python/experiment.py show exp_001
```

---

## Output Structure

```
outputs/lora_1b_seed42/
  checkpoints/
    step_2000/
      adapter_config.json
      adapter_model.safetensors
      training_state.pt
    best/
    final/
  splits.json
  metrics.json
  training_metrics.png

results/mc_dropout/
  1b/seed42/gsm8k/zero_shot_aligned/
    results.json             ← per-problem: answers, confidence scores, uncertainty measures
    summary.json             ← accuracy, ECE, AUROC for all signals
    results_judged.json      ← results.json + judge_rank (after llm_judge.py)
    summary_judged.json      ← summary.json + ece_judge_*, auroc_judge_*
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
    judge_correctness/       ← produced by src/uq/llm_judge.py
      reliability_judge_confidence.png
      reliability_judge_consistency_rate.png
      roc_judge.png
  1b/seed42/gsm8k/cot/
    ...
  1b/seed42/math/zero_shot_aligned/
    ...
  8b/seed42/gsm8k/zero_shot_aligned/
    ...
```
