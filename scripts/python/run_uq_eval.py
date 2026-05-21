#!/usr/bin/env python3
"""
End-to-end UQ evaluation: download test set → format → inference → metrics.

Usage:
    python scripts/python/run_uq_eval.py \
        --cluster macross \
        --method mc_dropout \
        --test-source gsm8k \
        --prompt zero_shot \
        --seed 42

Cluster configs are read from configs/clusters/<cluster>.yaml — the same ones
used by run.sh. GPU selection, HF cache, and output paths are derived from the
YAML so the two scripts stay in sync.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import yaml


# ---------------------------------------------------------------------------
# Cluster config helpers (mirrors run.sh logic)
# ---------------------------------------------------------------------------

def _load_cluster_cfg(cluster: str) -> dict:
    cfg_path = PROJECT_ROOT / "configs" / "clusters" / f"{cluster}.yaml"
    if not cfg_path.exists():
        available = [p.stem for p in (PROJECT_ROOT / "configs" / "clusters").glob("*.yaml")]
        sys.exit(f"Unknown cluster '{cluster}'. Available: {', '.join(available)}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _resolve_paths(cluster_cfg: dict, seed: int) -> dict:
    paths    = cluster_cfg["paths"]
    gpus     = cluster_cfg["hardware"]
    base_dir = paths["base_dir"]
    # Derive a short model label from the output_dir basename:
    # "lora_1b" → "1b",  "lora_8b" → "8b"
    model_label = Path(paths["output_dir"]).name.replace("lora_", "")
    return {
        "base_dir":        base_dir,
        "data_dir":        paths["data_dir"],
        "base_output_dir": paths["output_dir"],
        "output_dir":      f"{paths['output_dir']}_seed{seed}",
        "hf_cache":        paths["hf_cache"],
        "cuda_devices":    str(gpus["cuda_devices"]),
        "test_sets_dir":   paths.get("test_sets_dir", f"{base_dir}/data/test_sets"),
        "model_label":     model_label,
    }


# ---------------------------------------------------------------------------
# Result writing helpers
# ---------------------------------------------------------------------------

def _write_results(results: list[dict], summary: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results  → {results_path}")
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary  → {summary_path}")


_CONF_KEYS = [
    ("confidence",                    "Majority Vote Confidence"),
    ("full_sequence_mean_confidence", "Unweighted Confidence"),
    ("weighted_mean_confidence",      "Weighted Mean Confidence"),
]


def _plot_binary_correctness(results: list[dict], out_dir: Path) -> None:
    from src.uq.metrics import plot_reliability_diagram, plot_roc_curve
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, title in _CONF_KEYS:
        p = str(out_dir / f"reliability_{key}.png")
        plot_reliability_diagram(results, p, confidence_key=key, title=title)
        print(f"Reliability       → {p}")
    p = str(out_dir / "roc_binary.png")
    plot_roc_curve(
        results, p,
        label_fn=lambda r: r.get("correct"),
        title="ROC Curve — Binary Correctness",
    )
    print(f"ROC (binary)      → {p}")


def _plot_nlg_baselines(results: list[dict], out_dir: Path) -> None:
    from src.uq.metrics import plot_reliability_diagram_rougeL, plot_roc_curve
    import math
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, title in _CONF_KEYS:
        p = str(out_dir / f"reliability_rougeL_{key}.png")
        plot_reliability_diagram_rougeL(results, p, confidence_key=key, title=title)
        print(f"Reliability-rougeL → {p}")
    p = str(out_dir / "roc_rougeL.png")
    plot_roc_curve(
        results, p,
        label_fn=lambda r: (
            r.get("mean_rougeL", float("nan")) >= 0.4
            if not math.isnan(r.get("mean_rougeL", float("nan")))
            else None
        ),
        title="ROC Curve — ROUGE-L (NLG baseline, threshold ≥ 0.4)",
    )
    print(f"ROC (rougeL)      → {p}")


# ---------------------------------------------------------------------------
# Evaluator builders
# ---------------------------------------------------------------------------

def _build_mc_dropout_evaluator(args, ckpt_path: str):
    from src.uq.mc_dropout import MCDropoutConfig, MCDropoutEvaluator
    cfg = MCDropoutConfig(
        model_path=ckpt_path,
        base_model=args.base_model,
        tokenizer_name=args.base_model,
        num_passes=args.num_passes,
        max_new_tokens=args.max_new_tokens,
        mc_dropout_rate=args.mc_dropout_rate,
        device="cuda",
    )
    print(f"Loading MC-Dropout evaluator (adapter: {ckpt_path}) ...")
    return MCDropoutEvaluator(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end UQ evaluation: download data → format → inference → metrics."
    )
    ap.add_argument("--cluster", default="macross_1b_3090",
                    choices=["macross_1b_3090", "macross_8b_3090"],
                    help="Cluster config — picks paths, GPU, HF cache from configs/clusters/*.yaml")
    ap.add_argument("--method", required=True, choices=["mc_dropout"])
    ap.add_argument("--test-source", default="all",
                    choices=["gsm8k", "math", "all"])
    ap.add_argument("--seed", type=int, default=42,
                    help="Training seed — used to locate output_dir_seed{N}/checkpoints/final")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct",
                    help="HF id of the base model (adapter is loaded on top of this)")
    ap.add_argument("--num-passes", type=int, default=20,
                    help="[mc_dropout] Number of stochastic forward passes per problem")
    ap.add_argument("--mc-dropout-rate", type=float, default=0.1,
                    help="[mc_dropout] Dropout rate for the post-RMSNorm hook")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=500,
                    help="Max problems per test set (0 = all)")
    ap.add_argument("--checkpoint-step", type=int, default=None,
                    help="Use step_N checkpoint instead of final (e.g. --checkpoint-step 408000)")
    ap.add_argument("--test-sets-dir", default="",
                    help="Directory for cached test-set JSONLs. Defaults to <data_dir>/../test_sets/")
    ap.add_argument("--prompt", default="all",
                    choices=["zero_shot", "cot", "cot_step_by_step", "all"],
                    help="Prompt style — use 'all' to run all three variants sequentially")
    args = ap.parse_args()

    # --- Resolve cluster config ---
    cluster_cfg = _load_cluster_cfg(args.cluster)
    paths = _resolve_paths(cluster_cfg, args.seed)

    os.environ["HF_HOME"] = paths["hf_cache"]
    first_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if first_gpu is None:
        first_gpu = str(paths["cuda_devices"]).split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
    print(f"Cluster : {args.cluster}")
    print(f"Model   : {paths['model_label']}")
    print(f"GPU     : {first_gpu} (CUDA_VISIBLE_DEVICES={first_gpu})")
    print(f"HF home : {paths['hf_cache']}")

    sources = ["gsm8k", "math"] if args.test_source == "all" else [args.test_source]

    if args.checkpoint_step is not None:
        ckpt_path = f"{paths['output_dir']}/checkpoints/step_{args.checkpoint_step}"
    else:
        ckpt_path = f"{paths['output_dir']}/checkpoints/final"
    if not Path(ckpt_path).exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    base_dir      = Path(paths["base_dir"])
    test_sets_dir = (
        Path(args.test_sets_dir) if args.test_sets_dir
        else Path(paths["test_sets_dir"])
    )

    from src.prompts import PROMPT_BUILDERS
    prompts = ["zero_shot", "cot", "cot_step_by_step"] if args.prompt == "all" else [args.prompt]
    print(f"Prompts : {prompts}")

    evaluator = _build_mc_dropout_evaluator(args, ckpt_path)

    from src.uq.test_sets import load_or_build
    from src.uq.metrics import summarise, plot_selective_prediction

    for source in sources:
        test_set_path = test_sets_dir / f"{source}.jsonl"
        problems = load_or_build(source, test_set_path, limit=args.limit)

        for prompt in prompts:
            print(f"\n{'='*60}")
            print(f"Test source: {source}  |  Prompt: {prompt}")
            print(f"{'='*60}")
            print(f"Problems: {len(problems)}")

            prompt_fn = PROMPT_BUILDERS[prompt]
            label   = f"seed{args.seed}"
            out_dir = base_dir / "results" / args.method / paths["model_label"] / label / source / prompt

            print(f"Running {args.method} ({args.num_passes} passes)...")
            results = evaluator.evaluate(problems, prompt_fn=prompt_fn)

            summary = summarise(results)
            _write_results(results, summary, out_dir)

            conf_dir = out_dir / "confidence"
            conf_dir.mkdir(parents=True, exist_ok=True)
            plot_selective_prediction(results, str(conf_dir / "selective_prediction.png"))
            print(f"Selective    → {conf_dir / 'selective_prediction.png'}")

            _plot_binary_correctness(results, out_dir / "binary_correctness")
            _plot_nlg_baselines(results, out_dir / "nlg_baselines")

            print(f"\n--- Summary: {source} / {prompt} ---")
            for k, v in summary.items():
                print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
