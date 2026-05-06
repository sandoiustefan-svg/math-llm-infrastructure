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

LoRA dropout (`p=0.05`) is active on all adapter layers during both training and inference. At inference time, `model.train()` keeps dropout active; running the same problem through the model **N=20 times** yields a distribution over answers.

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
bash run_eval_1b.sh [method] [test_source] [prompt] [checkpoint_step] [limit]

# 8B model — MC Dropout
bash run_eval_8b.sh [method] [test_source] [prompt] [checkpoint_step] [limit]
```

| Argument | Options | Default |
|---|---|---|
| `method` | `mc_dropout`, `ensemble` | `mc_dropout` |
| `test_source` | `gsm8k`, `openmath_test` | `gsm8k` |
| `prompt` | `zero_shot`, `cot`, `rag` | `zero_shot` |
| `checkpoint_step` | step number or empty (→ `final`) | final |
| `limit` | max problems per benchmark | 500 |

Examples:
```bash
bash run_eval_1b.sh mc_dropout gsm8k zero_shot 408000 500
bash run_eval_1b.sh mc_dropout gsm8k cot 408000 500
bash run_eval_1b.sh mc_dropout gsm8k rag 408000 500
bash run_eval_8b.sh mc_dropout openmath_test zero_shot    # final checkpoint
```

---

## RAG Corpus

The RAG retrieval corpus is built from three elementary arithmetic word problem datasets that are independent of both OpenMathInstruct-2's construction and the evaluation sets:

| Dataset | Size | Source |
|---|---|---|
| SVAMP | ~1 000 | Patel et al., 2021 |
| ASDiv | ~2 300 | Miao et al., 2020 |
| MAWPS | ~3 000 | Koncel-Kedziorski et al., 2016 |

These datasets are at GSM8K difficulty level and were not used to construct OpenMathInstruct-2 (which was built exclusively from GSM8K and MATH training problems).

### Corpus pipeline

```
SVAMP + ASDiv + MAWPS
    ↓ format: Strategy 3 style  {system, user: problem, assistant: solution + Final Answer}
    ↓ embed with sentence-transformers (all-MiniLM-L6-v2 or similar)
    ↓ FAISS index (cosine similarity)
    ↓ saved to data/rag_corpus/

At inference (RAG prompt):
    ↓ embed test problem
    ↓ retrieve top-K most similar problems
    ↓ prepend as in-context examples before the target problem
```

### Build the RAG corpus

```bash
python scripts/python/build_rag_corpus.py \
    --out-dir data/rag_corpus \
    --top-k 3
```

---

## Experiment Design — 2×3

The main experimental comparison is a **2×3 factorial design** across model size and prompt style:

|  | Zero-shot | Few-shot CoT | RAG |
|---|---|---|---|
| **1B LoRA rank 16** | accuracy + UQ | accuracy + UQ | accuracy + UQ |
| **8B LoRA rank 64** | accuracy + UQ | accuracy + UQ | accuracy + UQ |

**Zero-shot**: the model is given only the problem and asked to solve it directly.

**Few-shot CoT (chain-of-thought)**: the prompt includes 3–5 hand-written worked examples demonstrating step-by-step reasoning. Fixed — the same examples are used for every test problem.

**RAG (retrieval-augmented generation)**: the top-K most similar problems from the RAG corpus are retrieved per test problem and prepended as in-context examples. Dynamic — different examples are retrieved for each problem.

All three prompt conditions use the same fine-tuned model weights and the same MC Dropout UQ method (N=20 stochastic passes with LoRA dropout active).

### Test sets

| Test set | Size | Notes |
|---|---|---|
| GSM8K test | 1 319 problems | Out-of-distribution — decontaminated from OpenMathInstruct-2 |
| OpenMathInstruct-2 test shards | ~1.4M rows (10% of shards) | In-distribution — held out from training and validation |

**Research questions**:
1. Does prompt style (zero-shot → CoT → RAG) improve the *correlation between model confidence and correctness* (calibration)?
2. Does this effect scale with model size (1B vs 8B)?
3. Does RAG (dynamic retrieval) improve calibration over CoT (fixed demonstrations)?

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

data/
  rag_corpus/
    corpus.jsonl                 ← formatted SVAMP + ASDiv + MAWPS problems
    faiss.index                  ← FAISS cosine similarity index
    embeddings.npy               ← sentence-transformer embeddings
    metadata.json                ← dataset sizes, embedding model, top-k default

results/
  mc_dropout/
    1b/
      gsm8k/
        zero_shot/
          results.json           ← per-problem answers, confidence, correctness
          summary.json           ← accuracy, ECE, coverage, AUROC
          reliability_diagram.png
        cot/
          ...
        rag/
          ...
      openmath_test/
        zero_shot/ cot/ rag/
    8b/
      gsm8k/
        zero_shot/ cot/ rag/
      openmath_test/
        zero_shot/ cot/ rag/
```
