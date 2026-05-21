# math-llm-infrastructure

LLaMA LoRA fine-tuning infrastructure for OpenMathInstruct-2, with Uncertainty Quantification (MC Dropout) evaluated via three prompt variants across two model sizes and two test sets.

The core research question: **does structured prompting improve the alignment between a model's confidence and the correctness of its outputs?**

Correctness is measured by three signals: binary answer match (+ sympy equivalence for MATH), LLM-as-judge rank (`good / medium / bad`), and NLG baselines (BLEU / ROUGE / METEOR, retained to demonstrate their inadequacy for mathematical reasoning). See `docs/metrics.md` and `docs/plots.md`.

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

LoRA dropout (`lora_dropout=0.1`) is active on all adapter layers during training and reactivated at inference time via `model.train()`. An additional `nn.Dropout` hook is inserted after the final RMSNorm layer (`_add_mc_dropout_hook()`). Running the same problem **N=20 times** with active dropout yields a distribution over answers — disagreement estimates epistemic uncertainty.

```python
# Inference recipe (handled automatically by run_uq_eval.py)
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, checkpoint_path, is_trainable=False)
model.train()   # re-activates LoRA dropout

answers = [model.generate(input_ids, ...) for _ in range(20)]
```

### UQ metrics

**Confidence signals (3 per problem)**

| Key | What it measures |
|---|---|
| `confidence` | Majority-vote fraction across 20 passes — answer-level signal |
| `full_sequence_mean_confidence` | Unweighted geometric mean of all token probs — token-level baseline, inflated by glue tokens |
| `weighted_mean_confidence` | Weighted geometric mean token prob — `<<expr=result>>` result tokens at 10×, Final Answer span at 25× |

**Correctness signals (3 per problem)**

| Key | What it measures |
|---|---|
| `correct` | Binary string match on extracted final answer; sympy symbolic equivalence fallback for MATH |
| `judge_rank` | `good / medium / bad` from LLM judge — produced by `src/uq/llm_judge.py` post-inference |
| `mean_bleu`, `mean_rougeL`, `mean_meteor` | Mean NLG scores across 20 passes vs full `reference_solution` — weak baselines |

**Aggregate metrics (per cell in summary.json)**

| Metric | Meaning |
|---|---|
| `ece_confidence`, `ece_full_sequence`, `ece_weighted` | ECE against binary correctness |
| `ece_judge_confidence`, `ece_judge_full_sequence`, `ece_judge_weighted` | ECE against judge score (good=1, medium=0.5, bad=0) |
| `ece_rougeL_confidence`, `ece_rougeL_full_sequence`, `ece_rougeL_weighted` | ECE against ROUGE-L (NLG baseline) |
| `auroc_confidence`, `auroc_full_sequence`, `auroc_weighted` | AUROC — binary correct vs wrong |
| `auroc_judge_confidence`, `auroc_judge_full_sequence`, `auroc_judge_weighted` | AUROC — judge `good` vs `medium+bad` |
| `auroc_rougeL_confidence`, `auroc_rougeL_full_sequence`, `auroc_rougeL_weighted` | AUROC — ROUGE-L above threshold vs below (NLG baseline) |
| `overconf_rate` | Fraction of high-confidence (≥ 0.8) predictions that are wrong |

### Evaluation scripts

```bash
# Run inference + all metrics for one prompt variant
python scripts/python/run_uq_eval.py \
    --cluster macross \
    --method mc_dropout \
    --test-source all \
    --prompt zero_shot \
    --seed 42

# Available prompts: zero_shot | cot | cot_step_by_step
# Available test sources: gsm8k | math | all

# Post-inference: LLM judge labelling (requires .env with API key)
python -m src.uq.llm_judge \
    --results results/mc_dropout/seed42/gsm8k/cot/results.json \
    --model gpt-4o-mini

# Preview rendered prompts before running inference
python scripts/python/show_prompts.py --source gsm8k --n 5 --print
```

### LLM-as-judge

After `results.json` is written, run `src/uq/llm_judge.py` to label each problem
`good / medium / bad` using an external LLM judge (default: `gpt-4o-mini`). Labels are
stored in `results_judged.json`; plots go to `judge_correctness/`.

Cost for the full experiment (6 000 majority-answer judgments): ~$0.90 with GPT-4o-mini.
Requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` (copy from `.env.example`).

---

## Experiment Design — 2×3

The main experimental comparison is a **2×3 factorial design** across model size and prompt variant:

|  | Zero-shot | Few-shot (CoT) | Few-shot + Step-by-step |
|---|---|---|---|
| **1B LoRA rank 16** | accuracy + UQ | accuracy + UQ | accuracy + UQ |
| **8B LoRA rank 64** | accuracy + UQ | accuracy + UQ | accuracy + UQ |

Each cell runs on two test sets: **GSM8K** (in-distribution) and **MATH** (out-of-distribution).
All three prompt variants instruct the model to annotate arithmetic as `<<expr=result>>`,
which is used to weight the `weighted_mean_confidence` signal. See `docs/prompts.md` for full
prompt definitions and `docs/uncertainty_methods.md` for the MC Dropout setup.

### Prompt variants

| Variant | Key | Description |
|---|---|---|
| Zero-shot | `zero_shot` | System prompt only; `<<expr=result>>` instruction in system message |
| Few-shot CoT | `cot` | System prompt + 3 hand-written examples in natural prose |
| Few-shot + Step-by-step | `cot_step_by_step` | Same examples with explicit `Step N:` labels |

### Correctness signals

All three signals are computed for every cell:

1. **Binary** — string match on extracted final answer (`answers_are_equal`); sympy equivalence fallback for MATH
2. **LLM judge rank** — `good / medium / bad` from external judge (post-inference step)
3. **NLG baselines** — BLEU, ROUGE-1/2/L, METEOR computed per pass vs full `reference_solution` (weak baselines)

### Test sets

| Test set | Size | Distribution |
|---|---|---|
| GSM8K | 500 problems (sampled) | In-distribution — models fine-tuned on OpenMathInstruct-2 which derives from GSM8K |
| MATH | 500 problems (sampled) | Out-of-distribution — competition mathematics |

### Research questions

1. How well does model confidence align with correctness (binary and judge) across prompt variants?
2. Does structured prompting (CoT → step-by-step) improve confidence–correctness alignment?
3. Does the weighted confidence measure outperform both majority-vote and unweighted token probability on ECE and AUROC?
4. Does the effect of prompting scale with model size (1B vs 8B)?
5. How does calibration degrade from in-distribution (GSM8K) to OOD (MATH)?
6. Do NLG baselines (BLEU/ROUGE/METEOR) carry meaningful calibration signal, or does the LLM judge provide substantially higher AUROC?

Key metrics per cell: accuracy, ECE, AUROC, overconfidence rate — computed against all correctness signals. See `docs/metrics.md` for definitions and `docs/plots.md` for plot interpretation.

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
  mc_dropout/
    seed42/
      gsm8k/
        zero_shot/
          results.json             ← per-problem: answers, confidence, correctness, nlg scores
          summary.json             ← accuracy, ECE, AUROC, overconf_rate for all signals
          results_judged.json      ← results.json + judge_rank/judge_score (after llm_judge.py)
          summary_judged.json      ← summary.json + ece_judge_*, auroc_judge_*
          confidence/
            selective_prediction.png
          binary_correctness/
            reliability_confidence.png
            reliability_weighted_mean_confidence.png
            roc_binary.png
          nlg_baselines/
            reliability_rougeL_confidence.png
            reliability_rougeL_weighted_mean_confidence.png
            roc_rougeL.png
          judge_correctness/       ← produced by src/uq/llm_judge.py
            reliability_judge_confidence.png
            reliability_judge_weighted_mean_confidence.png
            roc_judge.png
        cot/
          ...
        cot_step_by_step/
          ...
      math/
        zero_shot/ cot/ cot_step_by_step/
```
