# Data Preprocessing Pipeline

This document explains every step performed when running `scripts/bash/preprocess.sh`, from the raw HuggingFace dataset to the packed `.npy` shards consumed by the training loop.

---

## Overview

```
HuggingFace (nvidia/OpenMathInstruct-2)
    │
    ▼  Step 1 — inspect_data.py
raw.jsonl          {problem, generated_solution, expected_answer, problem_source}
    │
    ▼  Step 2 — format_data.py
formatted.jsonl    {messages, prompt_messages, completion_text}
    │
    ▼  Step 3 — tokenize_data.py
tokens.jsonl       {input_ids, attention_mask, loss_mask}
    │
    ▼  Step 4 — pack_data.py
shards/
  input_ids_00000.npy        shape (shard_num_seqs, seq_len)  dtype int32
  attention_mask_00000.npy   shape (shard_num_seqs, seq_len)  dtype int8
  loss_mask_00000.npy        shape (shard_num_seqs, seq_len)  dtype int8
  ...
  manifest.json
```

---

## Step 1 — Stream Raw Data

**Script:** `scripts/python/inspect_data.py`  
**Source:** `src/data/read_openmathinstruct2.py`  
**Output:** `raw.jsonl`

The dataset is streamed directly from HuggingFace using the `datasets` library in **streaming mode** — examples are yielded one at a time without loading the full dataset into memory. This makes it RAM-safe even for the full ~14 million example dataset.

Each raw example looks like this:

```json
{
  "problem": "Ava is planning a camping trip ...",
  "generated_solution": "There will be a total of 5 people. ...",
  "expected_answer": "45",
  "problem_source": "augmented_gsm8k"
}
```

The `--limit` flag stops streaming after N examples. `--skip` skips the first K examples, which is useful for resuming an interrupted run or splitting the dataset across machines.

---

## Step 2 — Format into Chat Messages

**Script:** `scripts/python/format_data.py`  
**Source:** `src/data/format_openmathinstruct2.py`  
**Input:** `raw.jsonl`  
**Output:** `formatted.jsonl`

Each raw example is converted into the Llama chat message format — a list of role-tagged messages that the tokenizer's chat template can process. This is called **Strategy 3** formatting.

The three fields from the raw example are mapped as follows:

| Raw field            | Chat role   | Content                                      |
|----------------------|-------------|----------------------------------------------|
| *(system prompt)*    | `system`    | "You are a careful mathematical reasoning assistant. Solve the problem step by step." |
| `problem`            | `user`      | The math problem text                        |
| `generated_solution` + `expected_answer` | `assistant` | Solution + `\n\nFinal Answer: {answer}` |

The `Final Answer:` line is appended explicitly to give the model a clear, consistent termination target that it can learn to produce.

Each formatted example looks like this:

```json
{
  "messages": [
    {"role": "system",    "content": "You are a careful mathematical reasoning assistant. Solve the problem step by step."},
    {"role": "user",      "content": "Ava is planning a camping trip ..."},
    {"role": "assistant", "content": "There will be a total of 5 people. ...\n\nFinal Answer: 45"}
  ],
  "prompt_messages": [
    {"role": "system", "content": "..."},
    {"role": "user",   "content": "..."}
  ],
  "completion_text": "There will be a total of 5 people. ...\n\nFinal Answer: 45"
}
```

`prompt_messages` (system + user only) and `completion_text` are stored for convenience at inference time, but are not used during training.

Examples with an empty assistant response are silently skipped.

---

## Step 3 — Tokenize

**Script:** `scripts/python/tokenize_data.py`  
**Source:** `src/data/tokenizer.py`  
**Input:** `formatted.jsonl`  
**Output:** `tokens.jsonl`

Each formatted example is tokenized using the Llama 3.2 Instruct chat template via `AutoTokenizer.apply_chat_template`. The template wraps every message with Llama 3's special role tokens:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 29 Apr 2026

You are a careful mathematical reasoning assistant. Solve the problem step by step.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>

Ava is planning a camping trip ...
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

There will be a total of 5 people. ...

Final Answer: 45
<|eot_id|>
```

The tokenizer is called **twice** per example:

1. **Full sequence** (`messages` with `add_generation_prompt=False`) → produces `input_ids` for the entire conversation including the assistant response.
2. **Prompt only** (`prompt_messages` with `add_generation_prompt=True`) → produces the token ids up to and including `<|start_header_id|>assistant<|end_header_id|>\n\n`.

The length of the prompt tokenization defines the **loss mask boundary**:

```
Tokens:     [<BOS>] [system header] [system text] [<EOT>] [user header] [user text] [<EOT>] [assistant header] [response...] [<EOT>]
loss_mask:     0         0              0             0        0             0           0           0               1  ...  1     1
```

- `loss_mask = 0` — prompt tokens (system, user, assistant header). The model sees these but does not learn to predict them.
- `loss_mask = 1` — completion tokens (assistant response + final `<|eot_id|>`). Only these positions contribute to the training loss.

Each tokenized example in the output:

```json
{
  "input_ids":      [128000, 128006, 9125, 128007, 271, ...],
  "attention_mask": [1, 1, 1, 1, 1, ...],
  "loss_mask":      [0, 0, 0, 0, 0, ..., 1, 1, 1, 1, 1]
}
```

`attention_mask` is all 1s at this stage because there is no padding — every token is real, including the terminal `<|eot_id|>`. Although LLaMA sets `pad_token = eos_token`, the EOS here is a genuine end-of-sequence marker, not padding. It is deliberately kept as `attention_mask = 1` so the model learns to predict it. Padding (and the corresponding `attention_mask = 0`) is introduced only in step 4.

If `--max-length` is set (default 4096), sequences are **truncated** to that length. The truncation happens on the full sequence, so very long solutions may be cut off. This is the only case where content is lost.

---

## Step 4 — Pack into Shards

**Script:** `scripts/python/pack_data.py`  
**Source:** `src/data/pack_shards.py`  
**Input:** `tokens.jsonl`  
**Output:** `shards/`

This step converts the variable-length tokenized examples into fixed-shape NumPy arrays suitable for efficient batched training.

### Chunking and Padding

Each example is placed into one or more **rows** of shape `(seq_len,)`:

- **Short example** (`len < seq_len`): padded to `seq_len` with `pad_token_id`. Pad positions get `attention_mask = 0` and `loss_mask = 0`.
- **Exact fit** (`len == seq_len`): one row, no padding.
- **Long example** (`len > seq_len`): **hard-chunked** into `ceil(len / seq_len)` rows. The last chunk is padded. No overlap between chunks. No examples are dropped.

The `attention_mask` encodes exactly one distinction: **real tokens** (including EOS) vs **artificial padding**. It does not mask prompt tokens — that role belongs to `loss_mask`.

```
attention_mask: 1 = real token (including the terminal <|eot_id|>)
                0 = artificial pad token appended to fill seq_len

loss_mask:      1 = assistant/completion token (model learns to predict these)
                0 = prompt token or padding (excluded from loss)
```

```
Example length = 700, seq_len = 512:

  Row 0: [token_0 ... token_511]   mask: [0...0, 1...1]
  Row 1: [token_512 ... token_699, PAD, PAD, ..., PAD]   mask: [1...1, 0...0]
```

Split examples (those that produce more than one row) are counted in `manifest.json` under `split_examples`.

### Shard Files

Rows are buffered in memory. Once the buffer reaches `shard_num_seqs` rows, it is flushed to disk as three NumPy files:

```
input_ids_00000.npy        shape: (shard_num_seqs, seq_len)   dtype: int32
attention_mask_00000.npy   shape: (shard_num_seqs, seq_len)   dtype: int8
loss_mask_00000.npy        shape: (shard_num_seqs, seq_len)   dtype: int8
```

The final shard may have fewer than `shard_num_seqs` rows if the total number of rows is not a multiple of `shard_num_seqs`. Its actual size is recorded in `manifest.json` under `final_shard_rows`.

### Manifest

A `manifest.json` is written alongside the shards:

```json
{
  "seq_len":            2048,
  "shard_num_seqs":     1024,
  "dtype":              "int32",
  "num_shards":         137,
  "total_rows":         140250,
  "final_shard_rows":   378,
  "padded_tokens":      8273441,
  "split_examples":     3821,
  "pad_token_id":       128009,
  "has_attention_mask": true,
  "has_loss_mask":      true,
  "tokenizer":          "meta-llama/Meta-Llama-3.1-8B-Instruct"
}
```

The training dataset class (`NpyShardDataset`) reads this manifest to discover all shards and their shapes. It expects all three file families (`input_ids_*.npy`, `attention_mask_*.npy`, `loss_mask_*.npy`) to be present and equal in count. The trainer validates `has_attention_mask: true` at startup and raises if the flag is missing.

---

## File Summary

| File | Shape / Format | Description |
|---|---|---|
| `raw.jsonl` | variable | One raw HF example per line |
| `formatted.jsonl` | variable | Chat messages with roles |
| `tokens.jsonl` | variable | `input_ids`, `attention_mask`, `loss_mask` as integer lists |
| `input_ids_XXXXX.npy` | `(S, L)` int32 | Token IDs; S = shard_num_seqs, L = seq_len |
| `attention_mask_XXXXX.npy` | `(S, L)` int8 | 1 = real token (incl. EOS), 0 = artificial padding |
| `loss_mask_XXXXX.npy` | `(S, L)` int8 | 1 = compute loss (completion tokens), 0 = ignore |
| `manifest.json` | JSON | Shard index and metadata |

---

## Running the Pipeline

```bash
# Debug run — 5k examples, seq_len=512, all intermediate files saved
bash scripts/bash/preprocess.sh --debug

# Full dataset — all intermediate files saved (large but resumable)
bash scripts/bash/preprocess.sh

# Full dataset — only shards saved to disk (intermediates written to /tmp)
bash scripts/bash/preprocess.sh --shards-only

# Debug + shards only
bash scripts/bash/preprocess.sh --debug --shards-only
```

Each step checks whether its output already exists before running, so the script is safe to re-run after a failure — it will resume from the last completed step.

---

## Design Decisions

**Why loss masking on prompt tokens?**  
The problem statement is always provided as input at inference time. Training the model to predict it would waste capacity and dilute the learning signal. Masking prompt tokens means the model only learns to generate the mathematical reasoning and the final answer.

**Why append `Final Answer:`?**  
OpenMathInstruct-2 solutions end with a `\boxed{}` expression embedded in the solution text. The explicit `Final Answer: {answer}` line gives the model a second, unambiguous termination signal and makes answer extraction at evaluation time trivial.

**Why hard-chunk instead of truncate long examples?**  
Truncation silently discards the end of long solutions — often the part containing the final answer and `\boxed{}` expression — which are the most training-signal-rich tokens. Hard-chunking preserves all tokens at the cost of splitting context across rows.

**Why `loss_mask` (0/1) instead of `labels` (−100)?**  
The `loss_mask` convention is used consistently throughout the pipeline — from the tokenizer output through the packed shards to the training loss function. This avoids the conversion between the two conventions that would otherwise be required at pack time.
