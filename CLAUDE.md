# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research infrastructure for uncertainty quantification (UQ) on fine-tuned LLMs applied to mathematical reasoning. The pipeline covers: preprocessing OpenMathInstruct-2, LoRA/QLoRA fine-tuning of LLaMA models (1B and 8B), and evaluating epistemic uncertainty via MC Dropout and Deep Ensembles against GSM8K, MATH, and OpenMathInstruct-2 test sets.

## Common Commands

### Setup
```bash
bash scripts/bash/setup_env.sh
```

### Run Tests
```bash
pytest tests/data/
pytest tests/data/test_pack_shards.py   # single file
```

### Preprocess Data
```bash
bash scripts/bash/preprocess.sh          # full dataset
bash scripts/bash/preprocess.sh --debug  # 30 examples, seq_len=512
```

### Train (cluster config drives everything)
```bash
bash scripts/bash/train.sh configs/llama3_1b_lora.yaml 42    # seed=42
bash scripts/bash/train.sh configs/llama3_8b_lora.yaml 123
```

### UQ Evaluation
```bash
# MC Dropout (single model)
python scripts/python/run_uq_eval.py \
    --cluster macross \
    --method mc_dropout \
    --test-source all \
    --seed 42

# Deep Ensemble (multiple seeds)
python scripts/python/run_uq_eval.py \
    --cluster macross \
    --method ensemble \
    --test-source gsm8k \
    --ensemble-seeds 42 123 456
```

### Utilities
```bash
python scripts/python/replot_metrics.py outputs/lora_1b_seed42/metrics.json
python scripts/python/inspect_data.py --limit 3
python -m src.uq.test_sets --source all --out-dir data/test_sets --limit 500
```

## Architecture

### Data Flow
```
OpenMathInstruct-2 (HuggingFace)
  → preprocess.sh / preprocess_to_shards.py
      → iter_openmathinstruct2()          # streaming generator
      → format_openmathinstruct2_example() # Strategy 3: Problem → Solution → Final Answer
      → tokenize + pack → .npy shards    # input_ids, attention_mask, loss_mask
  → NpyShardDataset                      # manifest.json + .npy shard loader
  → DataLoader → trainer.py             # masked causal loss (only completion tokens)
```

### Key Design Decisions

**Strategy 3 formatting**: Each example is `Problem + Solution + Final Answer`. Loss masking ensures only completion tokens (solution + answer) contribute — prompts are masked out.

**LoRA fine-tuning**: Primary training mode. Both 1B and 8B use plain LoRA (no quantisation, both fit in bf16). `lora_dropout=0.05` is set on all adapter layers — this dropout is reactivated via `model.train()` at inference to enable MC Dropout.

**MC Dropout inference**: `model.train()` at inference time re-activates the LoRA dropout (`lora_dropout=0.05`) — this is the sole source of stochasticity. No additional hooks are added. Each of `num_passes` greedy forward passes produces a different stochastic prediction; answer disagreement estimates epistemic uncertainty.

**Deep Ensembles**: Multiple LoRA adapters trained from different seeds loaded sequentially (to avoid multiplying VRAM by K). Uncertainty from inter-model disagreement is complementary to MC Dropout.

**UQ metrics**: Six confidence measures are computed per problem — full-sequence, answer-span, numeric-only, numeric-span, position-weighted, and majority-vote fraction — each reported with mean confidence, perplexity, min token prob, and std token prob. ECE and reliability diagrams are produced for all six.

**Config split**: Training uses `configs/llama3_{1b,8b}_lora.yaml` (full hyperparams: LoRA, optimiser, scheduler, validation). Inference uses `configs/clusters/macross_{1b,8b}_3090.yaml` (hardware and paths only, plus `test_sets_dir`). The two configs share `output_dir` so checkpoints written during training are found by the evaluator.

**Experiment registry**: `experiments/registry.json` is a lightweight JSON log of all runs. `ExperimentRegistry` is called automatically from `trainer.py` at start and completion of each run.

### Module Map

| Module | Purpose |
|---|---|
| `src/data/read_openmathinstruct2.py` | HF dataset streaming |
| `src/data/format_openmathinstruct2.py` | Strategy 3 example formatting; `FormatConfig` |
| `src/data/tokenizer.py` | Tokenizer loading helpers |
| `src/data/pack_shards.py` | Tokenize, pack, write `.npy` shards |
| `src/data/load_shards.py` | `NpyShardDataset` — manifest-based shard loader for training |
| `src/model/llama_model.py` | From-scratch LLaMA builder with presets (debug/small/medium/large) |
| `src/training/trainer.py` | `TrainConfig`, `load_train_config()`, training loop, LoRA/QLoRA, MC Dropout hook |
| `src/uq/mc_dropout.py` | `MCDropoutEvaluator` — stochastic forward passes |
| `src/uq/ensemble.py` | `EnsembleEvaluator` — sequential per-member inference |
| `src/uq/metrics.py` | Token confidence measures, ECE, reliability diagrams, `summarise()` |
| `src/uq/test_sets.py` | Download/cache GSM8K, MATH, OpenMathInstruct-2 tail as JSONL |
| `src/experiments/registry.py` | `ExperimentRegistry` — JSON-backed experiment log |
| `scripts/python/run_uq_eval.py` | End-to-end UQ eval entry point |
| `scripts/python/replot_metrics.py` | Regenerate training plots from a saved `metrics.json` |
| `scripts/bash/train.sh` | Training entry point — reads cluster YAML, launches `torchrun` |
| `scripts/bash/preprocess.sh` | Preprocessing entry point — HF → packed `.npy` shards |

### Cluster Configs (`configs/clusters/`)

Inference-only configs: hardware (cuda_devices, n_gpus), paths (base_dir, data_dir, output_dir, hf_cache, test_sets_dir). Available: `macross_1b_3090.yaml` (1B, GPU 1), `macross_8b_3090.yaml` (8B, GPU 0). Training hyperparams live in `configs/llama3_{1b,8b}_lora.yaml`.

The `output_dir` in a cluster config is a base; `train.sh` appends `_seed{N}` to produce the per-seed output directory (e.g., `outputs/lora_1b_seed42/`).

### Checkpoint Layout
```
outputs/lora_1b_seed42/
  checkpoints/
    step_10000/   # LoRA adapter saved with save_pretrained()
    final/        # adapter saved at end of training
  metrics.json    # {steps, loss, lr, tokens_per_sec, val_steps, val_loss}
  training_metrics.png
```

### UQ Output Layout
```
results/<method>/<label>/<source>/
  results.json        # per-problem raw results
  summary.json        # aggregate metrics (accuracy, ECE, coverage, ...)
  reliability_*.png   # one diagram per confidence measure
```

### Model Presets (from-scratch LlamaModelConfig)
- **debug**: ~30M params, 8 layers, 512 hidden
- **small**: ~125M params
- **medium**: ~500M params
- **large**: ~2B params

Fine-tuning experiments use `pretrained_model` in TrainConfig / cluster YAML instead of these presets.
