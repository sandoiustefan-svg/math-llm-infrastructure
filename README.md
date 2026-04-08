# math-llm-infrastructure

LLaMA-style training infrastructure for large-scale mathematical instruction datasets (OpenMathInstruct-2). Covers preprocessing, token packing, scratch training, fine-tuning pretrained models, and uncertainty quantification (MC Dropout + Deep Ensembles).

---

## Environment Setup

```bash
bash scripts/bash/setup_env.sh
```

---

## Inspecting the Dataset

```bash
python scripts/python/inspect_data.py --limit 3
```

| Argument | Description |
|---|---|
| `--split` | Dataset split to load (default: `train`) |
| `--limit` | Number of examples to display |
| `--skip` | Number of examples to skip before reading |

---

## Preprocessing Pipeline

Converts raw OpenMathInstruct-2 examples into packed training shards saved as `.npy` files.

**Strategy 3 formatting:**
- **Prompt:** `Problem`
- **Completion:** `Solution + Final Answer`
- Loss applied **only to completion tokens** (prompt tokens masked out)

### Steps

1. Stream dataset from HuggingFace
2. Format each example into `prompt_text` + `completion_text`
3. Tokenize separately, apply loss mask (`0` = prompt, `1` = completion)
4. Pack into fixed-length sequences
5. Save `.npy` shards + `manifest.json`

### Run Preprocessing

```bash
# Debug (5k examples, fast)
python scripts/python/preprocess_data.py \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --seq-len 1024 \
  --limit 5000 \
  --out-dir data/processed/openmathinstruct2_debug \
  --shard-num-seqs 256

# Full dataset
python scripts/python/preprocess_data.py \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --seq-len 2048 \
  --out-dir data/processed/openmathinstruct2
```

| Argument | Description |
|---|---|
| `--tokenizer` | HuggingFace tokenizer name or local path |
| `--seq-len` | Tokens per packed sequence (default: `2048`) |
| `--shard-num-seqs` | Sequences per `.npy` shard (default: `1024`) |
| `--no-loss-mask` | Train on all tokens (no prompt masking) |
| `--limit` | Max examples to preprocess (`0` = no limit) |
| `--skip` | Examples to skip before processing |
| `--split` | Dataset split (default: `train`) |

---

## Training From Scratch

Builds a LLaMA model from scratch and trains it on the preprocessed shards.

### Local (debug)

```bash
python scripts/python/train.py \
  --data-dir data/processed/openmathinstruct2_debug \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --n-layers 8 \
  --hidden-size 512 \
  --n-heads 8 \
  --batch-size 4 \
  --steps 500 \
  --output-dir outputs/debug_run
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 scripts/python/train.py \
  --data-dir data/processed/openmathinstruct2 \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --n-layers 16 \
  --hidden-size 1536 \
  --n-heads 12 \
  --batch-size 4 \
  --steps 1200000 \
  --fp16 \
  --resume \
  --output-dir outputs/scratch_16L
```

### Habrok (SLURM)

```bash
sbatch scripts/bash/habrok_train.sh
```

The job self-resubmits before training starts so it survives the 4-hour wall time limit. Checkpoints are saved every 2000 steps; only the last 2 are kept.

### Model size presets

| Preset | Layers | Hidden | ~Params |
|---|---|---|---|
| debug | 8 | 512 | ~30M |
| small | 12 | 768 | ~125M |
| medium | 16 | 1536 | ~500M |
| large | 24 | 2048 | ~2B |

### Training arguments

| Argument | Description |
|---|---|
| `--data-dir` | Path to preprocessed shards (use with disk mode) |
| `--online` | Stream from HuggingFace directly, no preprocessing step |
| `--tokenizer` | HuggingFace tokenizer ID or local path |
| `--pretrained-model` | HF model ID or checkpoint path — **skips scratch init, enables fine-tuning** |
| `--n-layers` | Number of transformer layers |
| `--hidden-size` | Hidden dimension |
| `--n-heads` | Number of attention heads |
| `--batch-size` | Per-GPU batch size |
| `--lr` | Learning rate (default: `3e-4`) |
| `--steps` | Total training steps |
| `--fp16` | Enable mixed precision (bfloat16) |
| `--seed` | Random seed — controls weight init and shard shuffling (default: `42`) |
| `--save-every` | Save checkpoint every N steps (default: `500`) |
| `--log-every` | Print log line every N steps (default: `50`) |
| `--resume` | Resume from the latest checkpoint in `--output-dir` |
| `--output-dir` | Where to write checkpoints, metrics, and plots |

---

## Fine-tuning a Pretrained Model

Use `--pretrained-model` to start from an existing HuggingFace model instead of training from scratch. The model architecture, vocab size, and sequence length are taken from the pretrained model's config — `--n-layers`, `--hidden-size`, `--n-heads` are ignored.

### Local (quick test)

```bash
python scripts/python/train.py \
  --pretrained-model mistralai/Mistral-7B-v0.1 \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --data-dir data/processed/openmathinstruct2_debug \
  --batch-size 1 \
  --lr 1e-5 \
  --steps 100 \
  --output-dir outputs/finetune_test
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 scripts/python/train.py \
  --pretrained-model mistralai/Mistral-7B-v0.1 \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --data-dir data/processed/openmathinstruct2 \
  --batch-size 2 \
  --lr 1e-5 \
  --steps 50000 \
  --fp16 \
  --resume \
  --output-dir outputs/finetune_seed42
```

### Habrok (SLURM)

```bash
# Single run
sbatch scripts/bash/finetune_habrok.sh

# Three independent seeds for Deep Ensemble
sbatch --export=SEED=42,OUTDIR=finetune_seed42 scripts/bash/finetune_habrok.sh
sbatch --export=SEED=43,OUTDIR=finetune_seed43 scripts/bash/finetune_habrok.sh
sbatch --export=SEED=44,OUTDIR=finetune_seed44 scripts/bash/finetune_habrok.sh
```

Each seed produces a different checkpoint. The three checkpoints are used together as a Deep Ensemble for uncertainty quantification.

### Fine-tuning a fine-tuned checkpoint (iterative)

You can pass any previously saved checkpoint as `--pretrained-model` — all checkpoints are saved in HuggingFace format via `model.save_pretrained()`:

```bash
python scripts/python/train.py \
  --pretrained-model outputs/finetune_seed42/checkpoints/final \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --data-dir data/processed/openmathinstruct2 \
  --lr 5e-6 \
  --steps 20000 \
  --output-dir outputs/finetune_seed42_v2
```

---

## Experiment Registry

Every training run (scratch or fine-tune) is automatically registered in `experiments/registry.json`. Use the CLI to inspect runs:

```bash
# List all runs
python scripts/python/experiment.py list

# Full detail for one run (config, metrics, checkpoint path)
python scripts/python/experiment.py show exp_001

# Add a note to a run
python scripts/python/experiment.py note exp_001 "used in Table 2"

# Manually register a run that predates auto-registration
python scripts/python/experiment.py register --type finetune --name "mistral-baseline"
```

---

## Uncertainty Quantification

UQ answers: **when the model gives an answer, how much should we trust it?**

Two methods are supported. Both compute the same output metrics, computed over the **full generated text** (reasoning chain + final answer):

| Metric | Meaning |
|---|---|
| `confidence` | Fraction of passes/members agreeing with the majority answer |
| `entropy` | Spread of the answer distribution (0 = unanimous, log₂K = max disagreement) |
| `token_mean_confidence` | Geometric mean token probability across the full generation |
| `token_perplexity` | Inverse of token confidence — lower = more certain word-by-word |
| `token_min_prob` | Probability of the least-certain token (weakest link in the chain) |
| `ece` | Expected Calibration Error — does confidence=0.8 mean 80% accuracy? |

### Method 1 — MC Dropout (single model)

Keeps dropout active at inference and runs the same problem through the model N times. Each pass produces a slightly different generation due to stochastic dropout masks. Measures disagreement across passes.

**Requires:** a model trained with dropout (the scratch-trained model has `attention_dropout=0.1`).

```bash
python scripts/python/evaluate_uq.py \
  --method mc_dropout \
  --model-path outputs/scratch_16L/checkpoints/final \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --problems-file data/problems.jsonl \
  --output-dir outputs/uq_mc_dropout \
  --num-passes 20
```

### Method 2 — Deep Ensemble (multiple models)

Runs K independently fine-tuned models (different seeds) on the same problem. Measures disagreement across models. More expensive but a stronger uncertainty signal.

**Requires:** K checkpoints trained with different `--seed` values.

```bash
python scripts/python/evaluate_uq.py \
  --method ensemble \
  --model-paths \
      outputs/finetune_seed42/checkpoints/final \
      outputs/finetune_seed43/checkpoints/final \
      outputs/finetune_seed44/checkpoints/final \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --problems-file data/problems.jsonl \
  --output-dir outputs/uq_ensemble
```

### On Habrok

```bash
# MC Dropout
sbatch --export=METHOD=mc_dropout,\
MODEL_PATH=/scratch/s5549329/outputs/scratch/checkpoints/final,\
PROBLEMS_FILE=/scratch/s5549329/data/problems.jsonl,\
OUTPUT_DIR=/scratch/s5549329/outputs/uq_mc_dropout \
scripts/bash/evaluate_habrok.sh

# Deep Ensemble
sbatch --export=METHOD=ensemble,\
"MODEL_PATHS=/scratch/s5549329/outputs/finetune_seed42/checkpoints/final /scratch/s5549329/outputs/finetune_seed43/checkpoints/final",\
PROBLEMS_FILE=/scratch/s5549329/data/problems.jsonl,\
OUTPUT_DIR=/scratch/s5549329/outputs/uq_ensemble \
scripts/bash/evaluate_habrok.sh
```

### Problems file format

Each line is a JSON object:

```jsonl
{"problem": "What is 2 + 2?", "expected_answer": "4"}
{"problem": "Solve x² - 5x + 6 = 0.", "expected_answer": "2, 3"}
```

`expected_answer` is optional — omit it for unlabeled problems (correctness metrics will be skipped, UQ metrics still computed).

### Output files

| File | Contents |
|---|---|
| `results.json` | Per-problem: answers, confidence, entropy, token metrics, correctness |
| `summary.json` | Aggregate: accuracy, mean confidence, ECE, coverage at 0.8/0.9 |
| `reliability_diagram.png` | Calibration plot — perfect model lies on the diagonal |
