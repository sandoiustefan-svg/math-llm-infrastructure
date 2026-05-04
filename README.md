# math-llm-infrastructure

LLaMA LoRA fine-tuning infrastructure for OpenMathInstruct-2, with Uncertainty Quantification (MC Dropout) evaluated via zero-shot and chain-of-thought prompting.

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

**Tokenization** — two passes per example:
1. Full sequence → `input_ids`
2. Prompt only → defines the `loss_mask` boundary (0 = prompt, 1 = completion)

**Packing** — variable-length examples are chunked into fixed `(seq_len,)` rows. Short examples are padded; long examples are hard-chunked (no truncation, no dropped tokens). Each shard holds `shard_num_seqs` rows.

### Run preprocessing

```bash
# Debug (5k examples, fast)
bash scripts/bash/preprocess.sh --debug

# Full dataset
bash scripts/bash/preprocess.sh

# Full dataset — intermediates in /tmp, only shards saved
bash scripts/bash/preprocess.sh --shards-only
```

Each step is idempotent — safe to re-run after a failure.

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

Both use plain LoRA (no quantisation). The 8B model fits on 2× RTX 3090 (~20 GB/GPU) at bf16 with gradient checkpointing.

LoRA adapts 7 modules per layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Dropout `p=0.05` is set on all adapter layers — this is the sole stochasticity source for MC Dropout at inference.

### Data split

```
total shards T  (13 646 for full OpenMathInstruct-2)
  train  = [0,        T − val − test)   ≈ 80 %   (10 918 shards)
  val    = [T−val−test,  T − test)       ≈ 10 %   ( 1 364 shards)
  test   = [T − test,    T)              ≈ 10 %   ( 1 364 shards)  — held out
```

The split is written to `<output_dir>/splits.json` at training start.

### Launch

```bash
# 1B model
bash scripts/bash/train.sh configs/llama3_1b_lora.yaml 42

# 8B model
bash scripts/bash/train.sh configs/llama3_8b_lora.yaml 42
```

`train.sh` reads the nested YAML, writes a flat runtime config to `/tmp/train_config_seed42.yaml`, sets `CUDA_VISIBLE_DEVICES` and NCCL env vars, then launches:

```
torchrun --nproc_per_node=2 scripts/python/train.py --config <runtime_config>
```

Checkpoints are saved every 2 000 steps to `<output_dir>/checkpoints/step_XXXXX/`. Only the two most recent step checkpoints are kept. The `best/` checkpoint tracks the lowest validation loss.

To resume:
```yaml
# in the cluster config yaml:
logging:
  resume: true
```

### Cluster configs

| File | Model | Output dir |
|---|---|---|
| `configs/clusters/macross.yaml` | 1B | `outputs/lora_1b_seed42` |
| `configs/clusters/macross_8b.yaml` | 8B | `outputs/lora_8b_seed42` |

---

## Uncertainty Quantification

### Method — MC Dropout

LoRA dropout (`p=0.05`) is active on all 112 adapter layers during both training and inference. At inference time, `model.train()` keeps dropout active; running the same problem through the model **N=20 times** yields a distribution over answers.

No additional dropout hook is needed — the LoRA adapter dropout is the stochasticity source.

```python
# Inference recipe (handled automatically by run_uq_eval.py)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
model.train()   # activates LoRA dropout

answers = [model.generate(input_ids, ...) for _ in range(20)]
```

### UQ metrics

| Metric | Meaning |
|---|---|
| `confidence` | Fraction of passes agreeing with the majority answer |
| `entropy` | Spread of answer distribution (0 = unanimous) |
| `token_mean_confidence` | Geometric mean token probability across the generation |
| `token_perplexity` | Inverse of token confidence |
| `token_min_prob` | Probability of the least-certain token |
| `ece` | Expected Calibration Error — does confidence=0.8 mean 80% accuracy? |

### Evaluation scripts

```bash
# 1B model — MC Dropout
bash run_eval_1b.sh [method] [test_source] [checkpoint_step] [limit]

# 8B model — MC Dropout
bash run_eval_8b.sh [method] [test_source] [checkpoint_step] [limit]
```

| Argument | Options | Default |
|---|---|---|
| `method` | `mc_dropout`, `ensemble` | `mc_dropout` |
| `test_source` | `all`, `gsm8k`, `math`, `openmath_tail` | `all` |
| `checkpoint_step` | step number or empty (→ `final`) | final |
| `limit` | max problems per benchmark | 500 |

Examples:
```bash
bash run_eval_1b.sh mc_dropout gsm8k 408000 500
bash run_eval_1b.sh mc_dropout all 408000 50    # quick check
bash run_eval_8b.sh mc_dropout all              # final checkpoint
```

---

## Experiment Design — 2×2

The main experimental comparison is a **2×2 factorial design** across model size and prompt style:

|  | Zero-shot | Few-shot CoT |
|---|---|---|
| **1B LoRA rank 16** | zero-shot accuracy + UQ | CoT accuracy + UQ |
| **8B LoRA rank 64** | zero-shot accuracy + UQ | CoT accuracy + UQ |

**Zero-shot**: the model is given only the problem and asked to solve it directly.

**Few-shot CoT (chain-of-thought)**: the prompt includes 3–5 worked examples that demonstrate step-by-step reasoning before the target problem. The model is expected to follow the same pattern. This is *true* CoT — demonstrated reasoning chains, not just "think step by step".

**Research question**: does chain-of-thought prompting improve the *correlation between model confidence and correctness* (calibration), and does this hold across model sizes?

Key metrics per cell: accuracy, ECE, confidence–correctness correlation (AUROC), and coverage at confidence thresholds 0.8 / 0.9.

---

## Experiment Registry

Every training run is automatically registered in `experiments/registry.json`.

```bash
python scripts/python/experiment.py list
python scripts/python/experiment.py show exp_001
python scripts/python/experiment.py note exp_001 "used in Table 2"
```

---

## Output Structure

```
outputs/lora_8b_seed42/
  checkpoints/
    step_2000/
      adapter_config.json
      adapter_model.safetensors
      training_state.pt          ← optimizer state, step, metrics
    best/                        ← lowest val-loss adapter
    final/                       ← end-of-training adapter
  splits.json                    ← train/val/test shard indices
  metrics.json                   ← loss, lr, tokens/sec per step
  training_metrics.png           ← 2×2 plot (loss, log loss, LR, throughput)
  loss_curve.png                 ← live loss curve updated every log_every steps

results/
  uq_eval_1b_mc_dropout_gsm8k_YYYYMMDD_HHMMSS/
    results.json                 ← per-problem answers, confidence, correctness
    summary.json                 ← accuracy, ECE, coverage
    reliability_diagram.png      ← calibration plot
```
