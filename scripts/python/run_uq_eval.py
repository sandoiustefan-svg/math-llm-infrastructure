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
    paths = cluster_cfg["paths"]
    gpus  = cluster_cfg["hardware"]
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="End-to-end UQ evaluation: download data → format → inference → metrics."
    )
    ap.add_argument("--cluster", default="macross",
                    choices=[
                        "macross",
                        "macross_8b",
                        "fse-4a100-2-1b",
                        "fse-4a100-2-8b",
                    ],
                    help="Cluster config — picks paths, GPU, HF cache from configs/clusters/*.yaml")
    ap.add_argument("--method", required=True, choices=["mc_dropout"])
    ap.add_argument("--test-source", default="all",
                    choices=["openmath_tail", "gsm8k", "math", "all"])
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
                    help="Max problems per test set (0 = all; for openmath_tail default 500 applies)")
    ap.add_argument("--checkpoint-step", type=int, default=None,
                    help="Use step_N checkpoint instead of final (e.g. --checkpoint-step 408000)")
    ap.add_argument("--test-sets-dir", default="",
                    help="Directory for cached test-set JSONLs. Defaults to <data_dir>/../test_sets/")
    ap.add_argument("--prompt", default="zero_shot", choices=["zero_shot", "cot", "rag"],
                    help="Prompt style: zero_shot | cot | rag")
    ap.add_argument("--rag-corpus-dir", default="data/rag_corpus",
                    help="[rag] Path to the built RAG corpus directory")
    ap.add_argument("--rag-top-k", type=int, default=3,
                    help="[rag] Number of examples to retrieve per problem")
    args = ap.parse_args()

    # --- Resolve cluster config ---
    cluster_cfg = _load_cluster_cfg(args.cluster)
    paths = _resolve_paths(cluster_cfg, args.seed)

    os.environ["HF_HOME"] = paths["hf_cache"]
    # Use first GPU from the cluster's device list for inference (single GPU is sufficient)
    first_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")

    if first_gpu is None:
        first_gpu = str(paths["cuda_devices"]).split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = first_gpu
    print(f"Cluster : {args.cluster}")
    print(f"GPU     : {first_gpu} (CUDA_VISIBLE_DEVICES={first_gpu})")
    print(f"HF home : {paths['hf_cache']}")

    # --- Test sources ---
    sources = ["openmath_tail", "gsm8k", "math"] if args.test_source == "all" else [args.test_source]

    # --- Checkpoint ---
    if args.checkpoint_step is not None:
        ckpt_path = f"{paths['output_dir']}/checkpoints/step_{args.checkpoint_step}"
    else:
        ckpt_path = f"{paths['output_dir']}/checkpoints/final"
    if not Path(ckpt_path).exists():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    # Base dir for writing results; test sets sit next to the training data dir
    base_dir = Path(paths["base_dir"])
    test_sets_dir = (
        Path(args.test_sets_dir)
        if args.test_sets_dir
        else Path(paths["data_dir"]).parent / "test_sets"
    )

    # --- Build prompt function ---
    if args.prompt == "zero_shot":
        from src.prompts.zero_shot import build_zero_shot_messages
        prompt_fn = build_zero_shot_messages
    elif args.prompt == "cot":
        from src.prompts.cot import build_cot_messages
        prompt_fn = build_cot_messages
    elif args.prompt == "rag":
        from src.rag.retriever import Retriever
        from src.prompts.rag import build_rag_messages
        retriever = Retriever(args.rag_corpus_dir)
        prompt_fn = lambda p: build_rag_messages(p, retriever.retrieve(p, top_k=args.rag_top_k))

    print(f"Prompt  : {args.prompt}")

    # --- Build evaluator once (loaded outside the source loop) ---
    evaluator = _build_mc_dropout_evaluator(args, ckpt_path)

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

        label = f"seed{args.seed}"
        out_dir = base_dir / "results" / args.method / label / source / args.prompt

        print(f"Running {args.method} ({args.num_passes} passes)...")
        results = evaluator.evaluate(problems, prompt_fn=prompt_fn)

        summary = summarise(results)
        _write_results(results, summary, out_dir)
        _plot_all_reliability_diagrams(results, out_dir)

        print(f"\n--- Summary: {source} ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
