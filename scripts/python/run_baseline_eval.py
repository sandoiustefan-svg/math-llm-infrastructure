#!/usr/bin/env python3
"""
Baseline UQ evaluation on raw (un-fine-tuned) base models.

Loads the raw Llama 1B or 8B with NO LoRA adapter and runs greedy
single-pass inference on the same test sets used for fine-tuned evaluation.
Two confidence signals are captured per problem:

  verbalized_confidence  — the model is prompted to state a score [0,1] after
                           its Final Answer; parsed from "Confidence: X.XX".
  token_confidence       — geometric mean of per-token softmax probabilities
                           over the generated response (always available).

The primary `confidence` field in results is the verbalized confidence so
that ECE / AUROC plots are directly comparable to the teacher's pedagogical
question.  Token-level confidence is stored as a supplementary field.

Usage:
    python scripts/python/run_baseline_eval.py \\
        --model-size 1b \\
        --test-source gsm8k \\
        --cluster macross_1b_3090

    python scripts/python/run_baseline_eval.py \\
        --model-size 8b \\
        --test-source all \\
        --prompt all \\
        --cluster macross_8b_3090
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

_BASE_MODELS: dict[str, str] = {
    "1b": "meta-llama/Llama-3.2-1B-Instruct",
    "8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

# Prompts to run when --prompt all is given.
_DEFAULT_PROMPTS = ["verbalized_confidence"]


# ---------------------------------------------------------------------------
# Cluster config helpers  (same logic as run_uq_eval.py)
# ---------------------------------------------------------------------------

def _load_cluster_cfg(cluster: str) -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "clusters" / f"{cluster}.yaml"
    if not cfg_path.exists():
        available = [p.stem for p in (PROJECT_ROOT / "configs" / "clusters").glob("*.yaml")]
        sys.exit(f"Unknown cluster '{cluster}'. Available: {', '.join(available)}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Verbalized-confidence parsing
# ---------------------------------------------------------------------------

_CONF_RE = re.compile(r"[Cc]onfidence\s*:\s*([0-9]*\.?[0-9]+)")


def _parse_verbalized_confidence(text: str) -> float:
    """Return float in [0, 1] parsed from 'Confidence: X.XX', or NaN."""
    m = _CONF_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
        except ValueError:
            pass
    return float("nan")


# ---------------------------------------------------------------------------
# Answer extraction  (mirrors MCDropoutEvaluator logic)
# ---------------------------------------------------------------------------

def _extract_final_answer(text: str) -> str:
    marker = "Final Answer:"
    idx = text.find(marker)
    if idx != -1:
        lines = text[idx + len(marker):].strip().splitlines()
        return lines[0].strip() if lines else ""
    from src.uq.metrics import _extract_boxed
    last, search = None, text
    while True:
        inner = _extract_boxed(search)
        if inner is None:
            break
        last = inner
        search = search[search.find(r"\boxed{") + 1:]
    return last.strip() if last else ""


# ---------------------------------------------------------------------------
# Baseline evaluator
# ---------------------------------------------------------------------------

class BaselineEvaluator:
    """Single-pass greedy evaluator for a raw (no LoRA) base model."""

    def __init__(self, model_id: str, hf_cache: str, device: str = "cuda"):
        os.environ["HF_HOME"] = hf_cache
        print(f"Loading tokenizer  : {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"Loading base model : {model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.model.to(device)
        self.model.eval()

        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None

        self.device = device
        self._eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    @torch.no_grad()
    def _generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> str:
        output_ids = self.model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id, self._eot_id],
            repetition_penalty=1.3,
        )
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def evaluate(
        self,
        problems: list[dict],
        prompt_fn,
        max_new_tokens: int = 512,
    ) -> list[dict]:
        from src.uq.metrics import (
            normalize_math_answer,
            answers_are_equal,
            token_probability_confidence,
            compute_nlg_scores,
        )

        results = []
        for i, item in enumerate(problems):
            messages = prompt_fn(item["problem"])
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

            raw = self._generate(input_ids, max_new_tokens)

            answer   = normalize_math_answer(_extract_final_answer(raw))
            verb_conf = _parse_verbalized_confidence(raw)

            expected = item.get("expected_answer")
            correct  = answers_are_equal(answer, str(expected)) if expected is not None else None

            tok_conf = token_probability_confidence(
                self.model, self.tokenizer, prompt, raw, self.device
            )

            reference = item.get("reference_solution") or ""
            nlg = compute_nlg_scores(raw, reference)

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(problems)}] correct={correct}  "
                      f"verb_conf={verb_conf:.2f}" if not math.isnan(verb_conf)
                      else f"  [{i+1}/{len(problems)}] correct={correct}  verb_conf=NaN")

            results.append({
                "problem":            item["problem"],
                "expected_answer":    expected,
                "reference_solution": item.get("reference_solution"),
                "raw":                raw,
                "answer":             answer,
                "majority_answer":    answer,   # alias for compatibility with summarise()
                "correct":            correct,
                # Primary confidence — verbalized (NaN if model omitted it)
                "confidence":         verb_conf,
                "verbalized_confidence": verb_conf,
                # summarise() also reads these; set to NaN — meaningless for single-pass
                "consistency_rate":   verb_conf,
                "entropy":            float("nan"),
                "n_unique_answers":   1,
                "std_log_prob":       float("nan"),
                "std_numeric_span_log_prob": float("nan"),
                # Token-level confidence from logits (always available)
                "token_full_seq_confidence":    tok_conf.get("full_sequence_mean_confidence", float("nan")),
                "token_answer_span_confidence": tok_conf.get("answer_span_mean_confidence",   float("nan")),
                "token_numeric_confidence":     tok_conf.get("numeric_mean_confidence",        float("nan")),
                "token_perplexity":             tok_conf.get("full_sequence_perplexity",       float("nan")),
                # NLG baselines
                "mean_rougeL":        nlg["rougeL"],
                "mean_meteor":        nlg["meteor"],
            })

        return results


# ---------------------------------------------------------------------------
# Result writing / plotting  (mirrors run_uq_eval.py)
# ---------------------------------------------------------------------------

def _write_results(results: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rp = out_dir / "results.json"
    sp = out_dir / "summary.json"
    with open(rp, "w") as f:
        json.dump(results, f, indent=2)
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results  → {rp}")
    print(f"Summary  → {sp}")


def _plot_results(results: list[dict], out_dir: Path) -> None:
    from src.uq.metrics import (
        plot_reliability_diagram,
        plot_reliability_diagram_rougeL,
        plot_roc_curve,
        plot_selective_prediction,
    )
    import math

    conf_dir = out_dir / "confidence"
    conf_dir.mkdir(parents=True, exist_ok=True)
    plot_selective_prediction(results, str(conf_dir / "selective_prediction.png"))

    binary_dir = out_dir / "binary_correctness"
    binary_dir.mkdir(parents=True, exist_ok=True)
    for key, title in [("confidence", "Verbalized Confidence"),
                       ("token_full_seq_confidence", "Token Full-Seq Confidence")]:
        valid = [r for r in results if not math.isnan(float(r.get(key, float("nan"))))]
        if not valid:
            continue
        plot_reliability_diagram(
            valid, str(binary_dir / f"reliability_{key}.png"),
            confidence_key=key, title=title,
        )
        print(f"Reliability  → {binary_dir / f'reliability_{key}.png'}")
    plot_roc_curve(
        results, str(binary_dir / "roc_binary.png"),
        label_fn=lambda r: r.get("correct"),
        title="ROC Curve — Binary Correctness (verbalized conf)",
    )

    nlg_dir = out_dir / "nlg_baselines"
    nlg_dir.mkdir(parents=True, exist_ok=True)
    valid_nlg = [r for r in results if not math.isnan(r.get("mean_rougeL", float("nan")))]
    if valid_nlg:
        plot_reliability_diagram_rougeL(
            valid_nlg, str(nlg_dir / "reliability_rougeL_confidence.png"),
            confidence_key="confidence", title="Verbalized Confidence vs ROUGE-L",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Baseline eval: raw (no LoRA) base model on math test sets."
    )
    ap.add_argument("--model-size", required=True, choices=["1b", "8b"],
                    help="Base model size (1b = Llama-3.2-1B-Instruct, 8b = Llama-3.1-8B-Instruct)")
    ap.add_argument("--cluster", default="macross_1b_3090",
                    choices=["macross_1b_3090", "macross_8b_3090"],
                    help="Cluster config for GPU + HF cache paths")
    ap.add_argument("--test-source", default="all",
                    choices=["gsm8k", "math", "all"])
    ap.add_argument("--prompt", default="all",
                    choices=["zero_shot", "zero_shot_aligned", "cot",
                             "verbalized_confidence", "all"],
                    help="Prompt style. 'all' runs zero_shot_aligned + verbalized_confidence")
    ap.add_argument("--limit", type=int, default=500,
                    help="Max problems per test set (0 = all)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--test-sets-dir", default="",
                    help="Override directory for cached JSONL test sets")
    args = ap.parse_args()

    cluster_cfg  = _load_cluster_cfg(args.cluster)
    hw           = cluster_cfg["hardware"]
    paths        = cluster_cfg["paths"]

    first_gpu = str(hw["cuda_devices"]).split(",")[0].strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
    os.environ["HF_HOME"] = paths["hf_cache"]
    print(f"Cluster  : {args.cluster}")
    print(f"GPU      : {first_gpu}")
    print(f"HF home  : {paths['hf_cache']}")

    model_id    = _BASE_MODELS[args.model_size]
    model_label = args.model_size
    base_dir    = Path(paths["base_dir"])
    test_sets_dir = (
        Path(args.test_sets_dir) if args.test_sets_dir
        else Path(paths.get("test_sets_dir", f"{paths['base_dir']}/data/test_sets"))
    )

    sources = ["gsm8k", "math"] if args.test_source == "all" else [args.test_source]
    prompts = _DEFAULT_PROMPTS if args.prompt == "all" else [args.prompt]

    print(f"Model    : {model_id}")
    print(f"Sources  : {sources}")
    print(f"Prompts  : {prompts}")

    evaluator = BaselineEvaluator(model_id, hf_cache=paths["hf_cache"])

    from src.uq.test_sets import load_or_build
    from src.uq.metrics import summarise
    from src.prompts import PROMPT_BUILDERS

    for source in sources:
        test_set_path = test_sets_dir / f"{source}.jsonl"
        problems = load_or_build(source, test_set_path, limit=args.limit)

        for prompt in prompts:
            print(f"\n{'='*60}")
            print(f"Source: {source}  |  Prompt: {prompt}  |  Model: {model_label} (raw)")
            print(f"{'='*60}")
            print(f"Problems: {len(problems)}")

            prompt_fn = PROMPT_BUILDERS[prompt]
            out_dir   = base_dir / "results" / "baseline" / model_label / source / prompt

            results = evaluator.evaluate(problems, prompt_fn, args.max_new_tokens)

            import math as _math
            n_verb = sum(1 for r in results if not _math.isnan(r["verbalized_confidence"]))
            print(f"Verbalized confidence coverage: {n_verb}/{len(results)}")

            # summarise() uses r["confidence"] directly; NaN values propagate into
            # mean_confidence but ECE/AUROC filter them, so the summary is still valid.
            summary = summarise(results)
            summary["verbalized_conf_coverage"] = n_verb
            summary["verbalized_conf_coverage_rate"] = round(n_verb / len(results), 4) if results else 0.0

            _write_results(results, summary, out_dir)
            _plot_results(results, out_dir)

            print(f"\n--- Summary: {source} / {prompt} ---")
            for k, v in summary.items():
                print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
