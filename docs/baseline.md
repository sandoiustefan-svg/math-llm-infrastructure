# Raw-Model Baseline Evaluation

Baseline inference on the raw (un-fine-tuned) base models — Llama-3.2-1B-Instruct and
Llama-3.1-8B-Instruct with no LoRA adapter — on the same test sets used for MC Dropout.

The purpose is to give the fine-tuned model results a reference point: how well does the
pre-trained model already do, and how well does it know what it doesn't know?

---

## How it differs from MC Dropout

| | MC Dropout (fine-tuned) | Raw baseline |
|---|---|---|
| Model | Base + LoRA adapter | Base model only, no adapter |
| Mode | `model.train()` — dropout active | `model.eval()` — no stochasticity |
| Passes per problem | 20 stochastic greedy passes | 1 greedy pass |
| `confidence` field | Majority-vote fraction across 20 passes | Verbalized score parsed from output |
| Uncertainty measures | Entropy, std log-prob, consistency rate | Not computed (meaningless for 1 pass) |

Because there is only one pass there is no disagreement to measure, so ECE/AUROC and
calibration metrics are **not** computed for the baseline. The output is just the final
answer and the model's self-reported confidence.

---

## Prompts

One prompt is run by default:

**`verbalized_confidence`** — asks the model to state its confidence after the final answer:

```
You are a careful mathematical reasoning assistant.
Solve the problem step by step.
End your response with "Final Answer: {answer}" on one line,
then on the next line write "Confidence: {score}" where {score} is
a number between 0.0 and 1.0 representing how confident you are that
your final answer is correct.
```

Expected model output:
```
... reasoning steps ...

Final Answer: 42
Confidence: 0.85
```

---

## Output format

### `results.json`

One entry per problem. Simpler than the MC Dropout format — no `raws` list, no
multi-pass uncertainty fields.

```json
{
  "problem": "Janet's ducks lay 16 eggs per day...",
  "expected_answer": "9",
  "reference_solution": "Janet sells 16 - 3 - 4 = 9 eggs...",
  "raw": "Let me work through this step by step.\n\nJanet's ducks lay 16 eggs...\n\nFinal Answer: 9\nConfidence: 0.92",
  "answer": "9",
  "correct": true,
  "confidence": 0.92,
  "verbalized_confidence": 0.92,
  "token_full_seq_confidence": 0.031,
  "token_answer_span_confidence": 0.048,
  "token_numeric_confidence": 0.071,
  "token_perplexity": 32.2,
  "mean_rougeL": 0.41,
  "mean_meteor": 0.38
}
```

When the model fails to produce a `Confidence: X.XX` line, `confidence` and
`verbalized_confidence` are `NaN`.

### `summary.json`

Accuracy and verbalized-confidence coverage only — no ECE/AUROC.

```json
{
  "n_problems": 500,
  "n_labeled": 500,
  "accuracy": 0.402,
  "mean_confidence": 0.831,
  "verbalized_conf_coverage": 487,
  "verbalized_conf_coverage_rate": 0.974
}
```

---

## Output location

```
results/baseline/
  1b/
    gsm8k/
      zero_shot_aligned/
        results.json
        summary.json
      verbalized_confidence/
        results.json
        summary.json
    math/
      ...
  8b/
    gsm8k/
      ...
    math/
      ...
```

`base_dir` is read from the cluster config, so on macross the full path is
`/home/bsandoiu/math-llm-infrastructure/results/baseline/`.

---

## Running

Both models run in parallel, one per RTX 3090 (8B on GPU 0, 1B on GPU 1):

```bash
bash scripts/bash/run_baselines_parallel.sh
# optional overrides:
bash scripts/bash/run_baselines_parallel.sh --test-source gsm8k --limit 200
```

Or individually:

```bash
python scripts/python/run_baseline_eval.py --model-size 1b --cluster macross_1b_3090 --test-source all
python scripts/python/run_baseline_eval.py --model-size 8b --cluster macross_8b_3090 --test-source all
```
