# Training Pipeline

This document explains every component of the training pipeline — the base models, LoRA adaptation, data loading, loss computation, optimisation schedule, MC dropout, checkpointing, and how to launch a run.

---

## Overview

```
Preprocessed shards (input_ids / attention_mask / loss_mask .npy)
    │
    ▼  80 / 10 / 10 split
    ├─ train shards  ──→ NpyShardDataset + DistributedSampler
    ├─ val shards    ──→ NpyShardDataset (eval only, no grad)
    └─ test shards   ──→ held out → splits.json (MC-dropout inference later)
                                │
                                ▼
     Llama-3.2-1B-Instruct  /  Meta-Llama-3.1-8B-Instruct  (bf16, no quantisation)
                                │
                     Plain LoRA adapters  (rank 16 / rank 64)
                                │         lora_dropout=0.05 on all adapter layers
                   masked causal loss  (completion tokens only)
                                │
              AdamW + cosine LR decay + gradient accumulation
                                │
                        checkpoints/
                          best/          ← adapter weights only
                          step_XXXXX/    ← adapter weights + optimizer state
                          final/         ← adapter weights + optimizer state
                        splits.json      ← train/val/test shard index lists
                        metrics.json
                        training_metrics.png
```

---

## Base Models

Two models were fine-tuned independently using the same pipeline:

| Property | 1B model | 8B model |
|---|---|---|
| Model ID | `meta-llama/Llama-3.2-1B-Instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Parameters | ~1.24B | ~8.03B |
| Architecture | LLaMA-3.2 (GQA, SwiGLU, RoPE) | LLaMA-3.1 (GQA, SwiGLU, RoPE) |
| Context window | 128k (trained with seq_len=2048) | 128k (trained with seq_len=2048) |
| Starting point | Instruction-tuned by Meta | Instruction-tuned by Meta |
| Training steps | 1 100 000 | 408 000 |

Both models are loaded in **bf16** — no quantisation. Base weights are frozen; only the LoRA adapter weights are updated.

Using the instruct-tuned variants (rather than base pretrain) means the chat template, role tokens, and instruction-following behaviour are already in place before fine-tuning begins. The additional training sharpens the models on mathematical reasoning without re-teaching the dialogue format.

---

## LoRA Adaptation

Both models use plain LoRA (no quantisation). The adapter update rule for a weight matrix **W** is:

```
W_effective = W_frozen + (α/r) · B · A
```

where **A** (rank × d_in) and **B** (d_out × rank) are the trained adapter matrices.

### Adapter configuration

| Setting | 1B | 8B |
|---|---|---|
| Target modules | q/k/v/o_proj, gate/up/down_proj | q/k/v/o_proj, gate/up/down_proj |
| Rank `r` | 16 | 64 |
| Alpha `α` | 32 | 128 |
| Effective scale `α/r` | 2.0 | 2.0 |
| LoRA dropout | 0.05 | 0.05 |
| Bias | none | none |
| Trainable parameters | ~11.1M (≈0.9%) | ~167.8M (≈2.1%) |

`target_modules` covers all seven linear projections in each transformer block — attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`) — but not the embedding or LM head.

---

## MC Dropout

MC Dropout uses the `lora_dropout=0.05` that is already set on every LoRA adapter layer during training. No additional dropout layers or hooks are added — the same dropout that regularised training is the sole source of stochasticity at inference.

**During training** (`model.train()`): LoRA dropout is active — 5% of adapter activations are zeroed at each forward pass. The model learns to produce reliable predictions despite this perturbation.

**During validation** (`model.eval()`): dropout is automatically deactivated by PyTorch. Validation loss is measured without noise.

**At inference (MC Dropout UQ)**: `model.train()` is called to re-activate the LoRA dropout. Running 20 stochastic forward passes then yields a distribution over final answers whose disagreement and variance reflect epistemic uncertainty over the adapter weights.

```python
# Inference recipe
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
model.train()   # re-activates lora_dropout=0.05 on all adapter layers

samples = [model.generate(input_ids, ...) for _ in range(20)]
# disagreement across samples → epistemic uncertainty estimate
```

---

## Data Loading

### Shard Split

At the start of every training run, the total number of shards is counted and split into three non-overlapping index lists:

```
total_shards  T
  train  = shards [0,        T - val - test)   ≈ 80%
  val    = shards [T-val-test,  T - test)       ≈ 10%
  test   = shards [T - test,    T)              ≈ 10%
```

The split is persisted to `<output_dir>/splits.json` on rank 0 at startup:

```json
{
  "total_shards":        13646,
  "data_dir":            "/home/bsandoiu/math-llm-infrastructure/data/processed/openmathinstruct2/shards",
  "shard_num_seqs":      1024,
  "pad_token_id":        128004,
  "train_shard_indices": [0, 1, ..., 10917],
  "val_shard_indices":   [10918, ..., 12281],
  "test_shard_indices":  [12282, ..., 13645]
}
```

Test shards are never loaded during training. They are read by the MC-dropout inference pipeline using `NpyShardDataset(data_dir, shard_indices=test_shard_indices)`.

### Dataset and Sampler

Each shard is a pair of NumPy arrays of shape `(1024, 2048)`. `NpyShardDataset` memory-maps all shards and presents them as a flat indexed dataset. Each `__getitem__` returns one row:

```python
{
  "input_ids":      torch.Tensor  shape (2048,)  dtype int64
  "attention_mask": torch.Tensor  shape (2048,)  dtype int64
  "loss_mask":      torch.Tensor  shape (2048,)  dtype int64
}
```

In DDP mode, a `DistributedSampler` partitions the dataset across ranks so each GPU sees a disjoint subset of the training data. `set_epoch` is called on each DataLoader restart to re-seed the shuffle, preventing the same ordering across passes.

| DataLoader setting | Value |
|---|---|
| Batch size per GPU | 1 |
| Sampler (DDP) | `DistributedSampler(shuffle=True, drop_last=True)` |
| Sampler (single GPU) | random shuffle |
| `num_workers` | 0 in DDP, 2 otherwise |
| `pin_memory` | true (CUDA) |

---

## Loss Function

The loss is **masked causal cross-entropy** computed only over completion tokens:

```
loss = Σ [ CE(logit_t, token_{t+1}) · loss_mask_{t+1} ]
       ────────────────────────────────────────────────
              Σ loss_mask_{t+1}
```

The sequence is shifted by one: `x = input_ids[:, :-1]` feeds the model; `y = input_ids[:, 1:]` is the target. `loss_mask[:, 1:]` zeros out all prompt-token positions (system prompt, user message, assistant header) so they never contribute to the gradient.

This means the model only learns to predict the assistant's mathematical reasoning and the terminal `<|eot_id|>`.

---

## Optimisation

| Setting | 1B | 8B |
|---|---|---|
| Optimiser | AdamW | AdamW |
| Learning rate | 2 × 10⁻⁴ | 2 × 10⁻⁴ |
| LR schedule | Linear warmup → cosine decay | Linear warmup → cosine decay |
| Warmup steps | 1 000 | 3 000 |
| Gradient accumulation | 32 micro-steps | 32 micro-steps |
| Effective batch size | 1 × 32 × 2 GPUs = **64** | 1 × 32 × 2 GPUs = **64** |
| Gradient clipping | max norm 1.0 | max norm 1.0 |
| Precision | bf16 (AMP autocast) | bf16 (AMP autocast) |
| Training steps | 1 100 000 | 408 000 |

The LR schedule:

```
step < warmup_steps  :  lr = (step / warmup_steps) × base_lr
step ≥ warmup_steps  :  lr = 0.5 × (1 + cos(π × progress)) × base_lr
```

where `progress = (step − warmup) / (total − warmup)`.

Gradient accumulation is handled at the micro-step level: `optimizer.zero_grad` and `optimizer.step` are called once every 32 micro-steps. The DDP `no_sync()` context suppresses gradient all-reduce on accumulation steps, communicating only at the true optimizer step boundary.

---

## Hardware and Distributed Training

| Setting | Value |
|---|---|
| Cluster | macross |
| GPUs | 2 × RTX 3090 (24 GB each) |
| CUDA devices | `0,1` |
| Backend | NCCL |
| NCCL P2P | disabled (`NCCL_P2P_DISABLE=1`) — PCIe topology |
| NCCL InfiniBand | disabled (`NCCL_IB_DISABLE=1`) |
| Launch | `torchrun --nproc_per_node=2` |

The RTX 3090 supports bfloat16 natively. P2P and IB are disabled because the two cards are connected over PCIe (not NVLink), which makes peer-to-peer transfers unreliable under NCCL. All-reduce happens through the host CPU instead.

---

## Checkpointing

Checkpoints are saved to `<output_dir>/checkpoints/`:

```
checkpoints/
  step_2000/
    adapter_config.json       ← LoRA config
    adapter_model.safetensors ← trained adapter weights only
    training_state.pt         ← optimizer state, step, metrics, exp_id
  step_4000/
    ...
  best/
    adapter_config.json
    adapter_model.safetensors ← best val-loss checkpoint
  final/
    adapter_config.json
    adapter_model.safetensors
    training_state.pt
```

Only adapter weights are saved via `model.save_pretrained` — the base model is not written to disk (it is always re-loaded from HuggingFace at inference time). The `training_state.pt` stores the optimizer state dict, current step, all metrics, and experiment ID for resuming.

Old checkpoints are pruned: only the two most recent step checkpoints are kept on disk at any time. The `best/` directory is updated whenever a new lowest validation loss (or, if no validation was run, lowest average training loss) is achieved.

To resume a run:

```yaml
logging:
  resume: true
```

The trainer finds the latest `step_XXXXX/` checkpoint automatically, restores the optimizer state and step counter, and continues from there.

---

## Experiment Registry

Every training run is registered in `experiments/registry.json` with:

- Full `TrainConfig` (all hyperparameters, paths, seed)
- Model architecture summary
- Dataset metadata (tokenizer, data dir, seq len, total steps, batch size)
- Final loss, samples consumed, and checkpoint path on completion

The registry provides a searchable audit trail of all experiments without opening TensorBoard or reading log files.

---

## Running a Training Job

### Prerequisites

1. Preprocessing complete: `data/processed/openmathinstruct2/shards/manifest.json` exists.
2. HuggingFace token available: `huggingface-cli login` (gated model).

### Launch

```bash
# 1B model, seed 42
bash scripts/bash/train.sh configs/llama3_1b_lora.yaml 42

# 8B model, seed 42
bash scripts/bash/train.sh configs/llama3_8b_lora.yaml 42
```

`train.sh` will:
1. Parse the nested yaml and write a flat runtime config to `/tmp/train_config_seed42.yaml`
2. Set `CUDA_VISIBLE_DEVICES`, `HF_HOME`, NCCL env vars
3. Launch `torchrun --nproc_per_node=2 scripts/python/train.py --config <runtime_config>`

### Expected console output (startup)

```
============================================================
 TRAIN CONFIG
  config     : configs/llama3_1b_lora.yaml
  seed       : 42
  ...
============================================================
  Splits saved → outputs/lora_1b_seed42/splits.json
  Experiment registered → exp_XXXXXXXX
Data mode: DISK shards from .../shards
Total shards: 13646
Train shards: 10918 | Val shards: 1364 | Test shards: 1364 (held out)
Loading pretrained model: meta-llama/Llama-3.2-1B-Instruct
Training mode: LoRA (plain, bf16)
trainable params: 11,141,120 || all params: 1,235,814,400 || trainable%: 0.90
DDP: 2 GPUs
Effective batch size: 64
Step      0/1100000 | Loss 2.3412 | LR 6.67e-08 | Tok/s ... | ...
```

---

## Design Decisions

**Why plain LoRA instead of QLoRA?**
Both the 1B and 8B models fit comfortably in bf16 on 2×RTX 3090 (48 GB combined). The 1B uses ~2.5 GB at bf16; the 8B uses ~16 GB. Plain LoRA avoids the bitsandbytes quantisation overhead and the associated compute-dtype conversion on every forward pass, giving cleaner gradients and simpler reproducibility.

**Why different ranks for 1B and 8B?**
The 8B model has ~6.5× more parameters and deeper representations. Rank 64 gives the 8B adapter ~167M trainable parameters (~2.1% of total), keeping the trainable fraction comparable across the two models and giving the larger model sufficient capacity to adapt. Rank 16 for the 1B gives ~11M trainable parameters (~0.9%).

**Why bf16 and not fp16?**
Both precisions fit on the 3090. However, fp16 requires a `GradScaler` (loss scaling to prevent underflow), which introduces occasional scale-backoff events and can cause training instability. bf16 has a larger exponent range than fp16 and does not need loss scaling, making it more stable for long training runs.

**Why use LoRA dropout for MC Dropout rather than adding a separate hook?**
A separate post-hoc dropout layer applied only at inference would perturb a model that never saw that perturbation during training. The variance across passes would then reflect the model's fragility under an unfamiliar noise source rather than genuine epistemic uncertainty. Using `lora_dropout=0.05` — the same dropout that was active throughout training — is principled: the model learned its adapter weights in the presence of this stochasticity, giving a valid Bayesian approximation (Gal & Ghahramani, 2016) over the LoRA adapter weights specifically.

**Why 80 / 10 / 10 shards and not examples?**
Splitting at the shard level is simpler and deterministic — no shuffling of individual examples is needed, and the split is trivially reproducible from the shard count alone. Since shards are filled sequentially during preprocessing (examples arrive in dataset order), the split approximates a random 80/10/10 partition of the underlying examples.

**Why persist `splits.json` at training time?**
The test shard indices need to be known at inference time to load the correct held-out shards. Writing them once at the start of training, co-located with the checkpoints, ties the shard split unambiguously to the experiment and makes the inference pipeline self-contained.
