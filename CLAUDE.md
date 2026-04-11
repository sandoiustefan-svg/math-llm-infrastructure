# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LLaMA-style training infrastructure for large-scale mathematical instruction datasets (OpenMathInstruct-2). The pipeline covers preprocessing, token packing, and training LLaMA models from scratch with support for distributed training (DDP).

## Common Commands

### Setup
```bash
bash scripts/bash/setup_env.sh
```

### Run Tests
```bash
pytest tests/data/
# Single test file:
pytest tests/data/test_tokenize_pack.py
```

### Preprocess Data
```bash
python scripts/python/preprocess_data.py \
  --tokenizer meta-llama/Llama-3.2-1B \
  --seq-len 1024 \
  --limit 5000 \
  --out-dir data/processed/openmathinstruct2_debug
```

### Train
```bash
python scripts/python/train.py \
  --data-dir data/processed/openmathinstruct2 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --n-layers 16 --hidden-size 1536 \
  --batch-size 4 --steps 10000
```

### Distributed Training
```bash
torchrun --nproc_per_node=2 scripts/python/train.py \
  --data-dir data/processed/openmathinstruct2 \
  --tokenizer meta-llama/Llama-3.2-1B \
  --batch-size 4 --fp16
```

### Full Pipeline (Debug Mode)
```bash
bash scripts/bash/run_pipeline.sh --debug    # 5k examples, small model, 100 steps
bash scripts/bash/run_pipeline.sh            # Full run
bash scripts/bash/run_pipeline.sh --online   # Stream from HF, no preprocessing
```

### Inspect Dataset
```bash
python scripts/python/inspect_data.py --limit 3
```

## Architecture

### Data Flow
```
OpenMathInstruct-2 (HuggingFace)
  → iter_openmathinstruct2()         # streaming generator
  → format_openmathinstruct2_example() # Strategy 3: Problem → Solution → Final Answer
  → tokenize_pack_and_save()         # tokenize + pack into fixed-length sequences + save .npy shards
  → PackedShardDataset / OnlinePackedDataset  # DDP-aware iterable dataset
  → DataLoader → train()             # masked causal loss (loss only on completion tokens)
```

### Key Design Decisions

**Strategy 3 formatting**: Each example is formatted as `Problem + Solution + Final Answer`. Loss masking is applied so only completion tokens (solution + answer) contribute to the loss — prompt tokens are masked out.

**Token packing**: Multiple tokenized examples are packed into fixed-length sequences to maximize GPU utilization. Shards are saved as `.npy` files (`input_ids_XXXXX.npy` + `loss_mask_XXXXX.npy`) with a `manifest.json` index.

**Two dataset modes**:
- *Disk-based* (`PackedShardDataset`): Load preprocessed shards; supports checkpointing, shard shuffling, DDP rank splitting.
- *Online* (`OnlinePackedDataset`): Streams → formats → tokenizes → packs on-the-fly; no preprocessing step needed.

**Configuration pattern**: All components use frozen dataclasses (`ReadConfig`, `FormatConfig`, `PackConfig`, `LlamaModelConfig`, `TrainConfig`). These are instantiated in the script entry points and passed into library functions.

### Module Map

| Module | Purpose |
|---|---|
| `src/data/read_openmathinstruct2.py` | HF dataset streaming |
| `src/data/format_openmathinstruct2.py` | Strategy 3 example formatting |
| `src/data/tokenize_pack.py` | Tokenize, pack, and save shards |
| `src/data/packed_dataset.py` | Disk-based DDP-aware IterableDataset |
| `src/data/online_dataset.py` | Streaming IterableDataset |
| `src/model/llama_model.py` | LLaMA model builder with presets (debug/small/medium/large) |
| `src/training/trainer.py` | Training loop, `masked_causal_loss`, checkpoint save/resume |
| `scripts/python/train.py` | Training entry point |
| `scripts/python/preprocess_data.py` | Preprocessing entry point |

### Model Presets (LlamaModelConfig)
- **debug**: ~30M params, 8 layers, 512 hidden
- **small**: ~125M params
- **medium**: ~500M params
- **large**: ~2B params

The model includes dropout support for MC Dropout uncertainty quantification.
