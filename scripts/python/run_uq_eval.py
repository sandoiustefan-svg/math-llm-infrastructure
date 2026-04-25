#!/usr/bin/env python3
"""
End-to-end UQ evaluation: download test set → format → inference → metrics.

Usage:
    python scripts/python/run_uq_eval.py \
        --cluster macross \
        --method mc_dropout \
        --test-source all \
        --seed 42

    python scripts/python/run_uq_eval.py \
        --cluster macross \
        --method ensemble \
        --test-source gsm8k \
        --ensemble-seeds 42 123 456

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
    paths = cluster_cfg["paths"]
    gpus  = cluster_cfg["gpus"]
    return {
        "base_dir":         paths["base_dir"],
        "data_dir":         paths["data_dir"],
        "base_output_dir":  paths["output_dir"],          # e.g. …/outputs/lora_8b
        "output_dir":       f"{paths['output_dir']}_seed{seed}",  # e.g. …/outputs/lora_8b_seed42
        "hf_cache":         paths["hf_cache"],
        "cuda_devices":     str(gpus["cuda_devices"]),
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


def _plot_all_reliability_diagrams(results: list[dict], out_dir: Path) -> None:
    from src.uq.metrics import plot_reliability_diagram

    conf_keys = [
        "confidence",
        "full_sequence_mean_confidence",
        "answer_span_mean_confidence",
        "numeric_mean_confidence",
        "numeric_span_mean_confidence",
        "weighted_mean_confidence",
    ]
    for key in conf_keys:
        out_path = str(out_dir / f"reliability_{key}.png")
        plot_reliability_diagram(results, out_path, confidence_key=key, title=key.replace("_", " ").title())
        print(f"Diagram  → {out_path}")


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


def _build_ensemble_evaluator(args, output_dir: str):
    from src.uq.ensemble import EnsembleConfig, EnsembleEvaluator

    paths = [f"{output_dir}_seed{s}/checkpoints/final" for s in args.ensemble_seeds]
    for p in paths:
        if not Path(p).exists():
            sys.exit(f"Ensemble checkpoint not found: {p}")

    cfg = EnsembleConfig(
        model_paths=paths,
        base_model=args.base_model,
        tokenizer_name=args.base_model,
        max_new_tokens=args.max_new_tokens,
        device="cuda",
    )
    print(f"Loading Ensemble evaluator ({len(paths)} members) ...")
    return EnsembleEvaluator(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end UQ evaluation: download data → format → inference → metrics."
    )
    ap.add_argument("--cluster", default="macross",
                    choices=["macross", "a100-1", "a100-2", "a100-3"],
                    help="Cluster config — picks paths, GPU, HF cache from configs/clusters/*.yaml")
    ap.add_argument("--method", required=True, choices=["mc_dropout", "ensemble"])
    ap.add_argument("--test-source", default="all",
                    choices=["openmath_tail", "gsm8k", "math", "all"])
    ap.add_argument("--seed", type=int, default=42,
                    help="Training seed — used to locate output_dir_seed{N}/checkpoints/final")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HF id of the base model (adapter is loaded on top of this)")
    ap.add_argument("--num-passes", type=int, default=20,
                    help="[mc_dropout] Number of stochastic forward passes per problem")
    ap.add_argument("--mc-dropout-rate", type=float, default=0.1,
                    help="[mc_dropout] Dropout rate for the post-RMSNorm hook")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=500,
                    help="Max problems per test set (0 = all; for openmath_tail default 500 applies)")
    ap.add_argument("--ensemble-seeds", nargs="+", type=int, default=[42, 123, 456],
                    help="[ensemble] Training seeds whose checkpoints form the ensemble")
    args = ap.parse_args()

    # --- Resolve cluster config ---
    cluster_cfg = _load_cluster_cfg(args.cluster)
    paths = _resolve_paths(cluster_cfg, args.seed)

    os.environ["HF_HOME"] = paths["hf_cache"]
    # Use first GPU from the cluster's device list for inference (single GPU is sufficient)
    first_gpu = str(paths["cuda_devices"]).split(",")[0].strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
    print(f"Cluster : {args.cluster}")
    print(f"GPU     : {first_gpu} (CUDA_VISIBLE_DEVICES={first_gpu})")
    print(f"HF home : {paths['hf_cache']}")

    # --- Test sources ---
    sources = ["openmath_tail", "gsm8k", "math"] if args.test_source == "all" else [args.test_source]

    # --- Checkpoint ---
    ckpt_path = f"{paths['output_dir']}/checkpoints/final"
    if args.method == "mc_dropout" and not Path(ckpt_path).exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    # Base dir for caching test sets and writing results
    base_dir = Path(paths["base_dir"])
    test_sets_dir = base_dir / "data" / "test_sets"

    # --- Build evaluator once (loaded outside the source loop) ---
    if args.method == "mc_dropout":
        evaluator = _build_mc_dropout_evaluator(args, ckpt_path)
    else:
        evaluator = _build_ensemble_evaluator(args, paths["base_output_dir"])

    # --- Per-source evaluation ---
    from src.uq.test_sets import load_or_build
    from src.uq.metrics import summarise

    for source in sources:
        print(f"\n{'='*60}")
        print(f"Test source: {source}")
        print(f"{'='*60}")

        test_set_path = test_sets_dir / f"{source}.jsonl"
        problems = load_or_build(source, test_set_path, limit=args.limit)
        print(f"Problems: {len(problems)}")

        label = f"seed{args.seed}" if args.method == "mc_dropout" else "ensemble"
        out_dir = base_dir / "results" / args.method / label / source

        print(f"Running {args.method} ({args.num_passes if args.method == 'mc_dropout' else len(args.ensemble_seeds)} passes/members)...")
        results = evaluator.evaluate(problems)

        summary = summarise(results)
        _write_results(results, summary, out_dir)
        _plot_all_reliability_diagrams(results, out_dir)

        print(f"\n--- Summary: {source} ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
