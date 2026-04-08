from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.uq.metrics import plot_reliability_diagram, summarise


def load_problems(path: str) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def run_mc_dropout(args) -> list[dict]:
    from src.uq.mc_dropout import MCDropoutConfig, MCDropoutEvaluator
    cfg = MCDropoutConfig(
        model_path=args.model_path,
        tokenizer_name=args.tokenizer,
        num_passes=args.num_passes,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    evaluator = MCDropoutEvaluator(cfg)
    problems = load_problems(args.problems_file)
    print(f"Running MC Dropout ({args.num_passes} passes) on {len(problems)} problems...")
    return evaluator.evaluate(problems)


def run_ensemble(args) -> list[dict]:
    from src.uq.ensemble import EnsembleConfig, EnsembleEvaluator
    cfg = EnsembleConfig(
        model_paths=args.model_paths,
        tokenizer_name=args.tokenizer,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    evaluator = EnsembleEvaluator(cfg)
    problems = load_problems(args.problems_file)
    print(f"Running Ensemble ({len(args.model_paths)} members) on {len(problems)} problems...")
    return evaluator.evaluate(problems)


def main():
    ap = argparse.ArgumentParser(description="Evaluate UQ methods on math problems.")
    ap.add_argument("--method", required=True, choices=["mc_dropout", "ensemble"])
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer ID or local path")
    ap.add_argument("--problems-file", required=True, help="Path to .jsonl file with problems")
    ap.add_argument("--output-dir", required=True, help="Directory to write results.json and reliability_diagram.png")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda")

    # MC Dropout
    ap.add_argument("--model-path", default="", help="[mc_dropout] Path to scratch model checkpoint")
    ap.add_argument("--num-passes", type=int, default=20, help="[mc_dropout] Number of stochastic forward passes")

    # Ensemble
    ap.add_argument("--model-paths", nargs="+", default=[], help="[ensemble] Paths to ensemble member checkpoints")

    args = ap.parse_args()

    if args.method == "mc_dropout" and not args.model_path:
        ap.error("--model-path is required for mc_dropout")
    if args.method == "ensemble" and not args.model_paths:
        ap.error("--model-paths is required for ensemble")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.method == "mc_dropout":
        results = run_mc_dropout(args)
    else:
        results = run_ensemble(args)

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {results_path}")

    diagram_path = os.path.join(args.output_dir, "reliability_diagram.png")
    plot_reliability_diagram(results, diagram_path)
    print(f"Reliability diagram → {diagram_path}")

    summary = summarise(results)
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()
