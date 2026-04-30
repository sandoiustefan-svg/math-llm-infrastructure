# Training Pipeline

This document explains every component of the training pipeline — the base model, QLoRA adaptation, data loading, loss computation, optimisation schedule, MC dropout, checkpointing, and how to launch a run.

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
              Meta-Llama-3.1-8B-Instruct  (4-bit NF4 quantisation)
                                │
                          QLoRA adapters  (rank 16, all-linear)
                                │
                     MC dropout hook  (rate 0.1, post-LayerNorm)
                                │
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

## Base Model

| Property | Value |
|---|---|
| Model ID | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| Parameters | 8B |
| Architecture | LLaMA-3.1 (GQA, SwiGLU, RoPE) |
| Vocab size | 128 256 tokens |
| Context window | 128 k (trained here with seq_len=2048) |
| Starting point | Instruction-tuned by Meta |

The model is loaded in **4-bit NF4** quantisation using bitsandbytes (`load_in_4bit=True`). Its base weights are frozen — no gradient flows through them. Only the LoRA adapter weights are updated.

Using the instruct-tuned variant (rather than the base pretrain) means the chat template, role tokens, and instruction-following behaviour are already in place before fine-tuning begins. The additional training sharpens the model on mathematical reasoning without re-teaching the dialogue format.

---

## QLoRA Adaptation

QLoRA (Quantised LoRA) combines 4-bit weight quantisation with low-rank adapter injection. This allows fine-tuning a model that would require ~16 GB at float16 using roughly ~5 GB of GPU memory.

### Quantisation

| Setting | Value |
|---|---|
| Quantisation | 4-bit NF4 (NormalFloat4) |
| Compute dtype | `bfloat16` |
| Double quantisation | enabled (quantises the quantisation constants, saves ~0.4 GB) |
| Library | `bitsandbytes` |

NF4 is information-theoretically optimal for normally-distributed weights. Double quantisation further reduces memory at negligible accuracy cost.

### LoRA Adapters

| Setting | Value |
|---|---|
| Target modules | `all-linear` (every `nn.Linear` in the model) |
| Rank `r` | 16 |
| Alpha `α` | 32 |
| Effective scale `α/r` | 2.0 |
| LoRA dropout | 0.05 |
| Bias | none |
| Trainable parameters | ~40 M (≈0.5% of total) |

`target_modules="all-linear"` applies adapters to every linear layer — attention projections (`q/k/v/o_proj`), MLP projections (`gate/up/down_proj`), and the LM head. This is the recommended setting for QLoRA as the quantisation error is distributed across all linear layers.

The adapter update rule for a weight matrix **W** is:

```
W_effective = W_frozen + (α/r) · B · A
```

where **A** (rank × d_in) and **B** (d_out × rank) are the trained adapter matrices, and **W_frozen** remains at 4-bit precision throughout.

Gradient checkpointing is enabled via `prepare_model_for_kbit_training` to keep activation memory bounded during backward passes.

---

## MC Dropout

A single `nn.Dropout(p=0.1)` layer is injected after the final LayerNorm of the transformer stack:

```
[transformer layers]  →  RMSNorm (final)  →  Dropout(0.1)  →  lm_head  →  logits
```

This is implemented as a PyTorch forward hook registered on `model.model.norm`. The hook intercepts the LayerNorm output and passes it through the dropout before it reaches the LM head.

**During training** (`model.train()`): dropout is active — 10 % of final hidden-state activations are zeroed at each forward pass. The model learns to produce reliable predictions despite this perturbation.

**During validation** (`model.eval()`): dropout is automatically deactivated by PyTorch. Validation loss is measured without noise.

**At inference (MC dropout UQ)**: the hook must be re-applied to the loaded checkpoint, and `model.train()` must be called to re-activate dropout. Running N stochastic forward passes then yields a distribution over outputs whose variance reflects epistemic uncertainty.

```python
# Inference recipe
model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_cfg)
model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
_add_mc_dropout_hook(model, rate=0.1)   # re-inject the same hook
model.train()                            # activate dropout

samples = [model.generate(input_ids, ...) for _ in range(20)]
# variance across samples → uncertainty estimate
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

| Setting | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 2 × 10⁻⁴ |
| LR schedule | Linear warmup → cosine decay |
| Warmup steps | 3 000 |
| Gradient accumulation | 32 micro-steps |
| Effective batch size | 1 × 32 × 2 GPUs = **64** |
| Gradient clipping | max norm 1.0 |
| Precision | bf16 (AMP autocast) |
| Epochs | 1 (over 80% of shards) |

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

Only adapter weights are saved via `model.save_pretrained` — the 4-bit base model is not written to disk (it is always re-loaded from HuggingFace at inference time). The `training_state.pt` stores the optimizer state dict, current step, all metrics, and experiment ID for resuming.

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
2. Shard counts computed and filled into `configs/llama3_8b_qlora.yaml`:
   ```bash
   T=$(ls $DATA_DIR/input_ids_*.npy | wc -l)
   # set val_shard_count = T/10, test_shard_count = T/10 in the yaml
   ```
3. HuggingFace token available: `huggingface-cli login` (gated model).

### Launch

```bash
# From the repo root on macross:
bash scripts/bash/train.sh configs/llama3_8b_qlora.yaml 42
#                                                        └── seed
```

`train.sh` will:
1. Parse the nested yaml and write a flat runtime config to `/tmp/train_config_seed42.yaml`
2. Set `CUDA_VISIBLE_DEVICES`, `HF_HOME`, NCCL env vars
3. Launch `torchrun --nproc_per_node=2 scripts/python/train.py --config <runtime_config>`

### Expected console output (startup)

```
============================================================
 TRAIN CONFIG
  config     : configs/llama3_8b_qlora.yaml
  seed       : 42
  ...
============================================================
  Splits saved → outputs/qlora_8b_seed42/splits.json
  Experiment registered → exp_XXXXXXXX
Data mode: DISK shards from .../shards
Total shards: 6840
Train shards: 10918 | Val shards: 1364 | Test shards: 1364 (held out)
Epochs mode: 1 epochs × ... steps/epoch = ... steps
Loading pretrained model: meta-llama/Meta-Llama-3.1-8B-Instruct
Training mode: QLoRA 4-bit
trainable params: 41,943,040 || all params: 8,072,884,224 || trainable%: 0.52
MC Dropout hook added with rate=0.1
Model parameters: 8,072,884,224 total | 41,943,040 trainable (41.94M)
DDP: 2 GPUs
Effective batch size: 64
Step      0/...... | Loss 2.3412 | LR 6.67e-08 | Tok/s ... | ...
```

---

## Design Decisions

**Why QLoRA instead of full fine-tuning?**
Full fine-tuning of an 8B model at bf16 requires ~16 GB per GPU just for parameters, plus optimiser states (~48 GB for AdamW). Two 3090s (48 GB combined) cannot hold this. QLoRA reduces the base model to ~5 GB and adds only ~160 MB of adapter weights, fitting comfortably with room for activations and gradient accumulation.

**Why `all-linear` target modules for QLoRA?**
When the base weights are quantised, quantisation error is present in every linear layer — not just the attention projections. Adapting only attention while leaving MLP layers unadapted means the MLP quantisation error is never corrected. `all-linear` addresses this across the full model.

**Why bf16 and not fp16?**
Both precisions fit on the 3090. However, QLoRA + fp16 requires a `GradScaler` (loss scaling to prevent underflow), which introduces occasional scale-backoff events and can cause training instability. bf16 has a larger exponent range than fp16 and does not need loss scaling, making it more stable for long training runs.

**Why train with MC dropout rather than adding it only at inference?**
A model trained without dropout produces representations that are not robust to activation zeroing. Post-hoc MC dropout on such a model gives miscalibrated uncertainty — the variance across passes reflects the model's fragility under perturbation rather than genuine epistemic uncertainty. Training with the same dropout rate that is used at inference ensures the learned representations are robust to it, yielding better-calibrated uncertainty estimates.

**Why 80 / 10 / 10 shards and not examples?**
Splitting at the shard level is simpler and deterministic — no shuffling of individual examples is needed, and the split is trivially reproducible from the shard count alone. Since shards are filled sequentially during preprocessing (examples arrive in dataset order), the split approximates a random 80/10/10 partition of the underlying examples.

**Why persist `splits.json` at training time?**
The test shard indices need to be known at inference time to load the correct held-out shards. Writing them once at the start of training, co-located with the checkpoints, ties the shard split unambiguously to the experiment and makes the inference pipeline self-contained.
