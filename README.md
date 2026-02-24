# math-llm-infrastructure

LLaMA-style training infrastructure for large-scale mathematical instruction datasets, including preprocessing, token packing, and distributed benchmarking.

---

## Environment Setup

Follow the steps below to set up the development environment.

### Run the following command that calls the env script

```bash
bash scripts/bash/setup_env.sh
```

## Inspecting the Dataset

Before preprocessing, you can inspect the dataset structure using the inspection script.

### Run the following command that calls the data inspection script 

```bash
python scripts/python/inspect_data.py --limit 3
```

It has available three arguments:

### Available arguments

| Argument  | Description |
|----------|-------------|
| `--split` | Dataset split to load (default: `train`) |
| `--limit` | Number of examples to display |
| `--skip`  | Number of examples to skip before reading |


## Preprocessing Pipeline

The preprocessing stage converts raw OpenMathInstruct-2 examples into
packed LLaMA-ready training shards.

This pipeline implements **Strategy 3 formatting**:

- **Prompt:** `Problem`
- **Completion:** `Solution + Final Answer`
- Loss is applied **only to completion tokens**
- Data is packed into fixed-length token sequences

---

### Pipeline Overview

The preprocessing script performs the following steps:

1. **Stream dataset** (RAM-safe using HuggingFace `datasets`)
2. **Format examples** into:
   - `prompt_text`
   - `completion_text`
3. **Tokenize separately**
4. **Apply loss mask**
   - `0` → prompt tokens
   - `1` → completion tokens
5. **Concatenate across examples**
6. **Pack into fixed-length sequences**
7. **Save `.npy` shards to disk**

---

## Running Preprocessing

### Laptop (recommended first)
```bash
python scripts/python/preprocess_data.py \
  --tokenizer mistralai/Mistral-7B-v0.1 \
  --seq-len 1024 \
  --limit 5000 \
  --out-dir data/processed/openmathinstruct2_debug \
  --shard-num-seqs 256
```

### Available arguments (Tokenization & Packing)

| Argument | Description |
|----------|------------|
| `--tokenizer` | HuggingFace tokenizer name or local path (LLaMA-compatible) |
| `--seq-len` | Number of tokens per packed sequence (default: `2048`) |
| `--shard-num-seqs` | Number of sequences per saved `.npy` shard (default: `1024`) |
| `--no-loss-mask` | Disable saving the loss mask (train on all tokens) |
| `--limit` | Maximum number of examples to preprocess (`0` = no limit) |
| `--skip` | Number of examples to skip before processing |
| `--split` | Dataset split to load (default: `train`) |