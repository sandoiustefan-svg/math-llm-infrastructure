"""
Regenerate summary.json and all plots from an existing results.json.

Useful after updating metrics.py without rerunning inference.

Usage:
    # Single file
    python scripts/python/summarise_results.py \
        results/mc_dropout/1b/seed42/gsm8k/zero_shot_aligned/results.json

    # All results files under a directory
    python scripts/python/summarise_results.py --all results/mc_dropout/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_CONF_KEYS = [
    ("confidence",       "Majority Vote Confidence"),
    ("consistency_rate", "Consistency Rate"),
]


def _make_plots(results: list[dict], out_dir: Path) -> None:
    from src.uq.metrics import (
        plot_reliability_diagram,
        plot_reliability_diagram_rougeL,
        plot_roc_curve,
        plot_selective_prediction,
    )
    import math

    (out_dir / "binary_correctness").mkdir(parents=True, exist_ok=True)
    (out_dir / "nlg_baselines").mkdir(parents=True, exist_ok=True)
    (out_dir / "confidence").mkdir(parents=True, exist_ok=True)

    for key, title in _CONF_KEYS:
        p = str(out_dir / "binary_correctness" / f"reliability_{key}.png")
        plot_reliability_diagram(results, p, confidence_key=key, title=title)
        print(f"  Reliability (binary)  → {p}")

    p = str(out_dir / "binary_correctness" / "roc_binary.png")
    plot_roc_curve(results, p, label_fn=lambda r: r.get("correct"),
                   title="ROC Curve — Binary Correctness")
    print(f"  ROC (binary)          → {p}")

    for key, title in _CONF_KEYS:
        p = str(out_dir / "nlg_baselines" / f"reliability_rougeL_{key}.png")
        plot_reliability_diagram_rougeL(results, p, confidence_key=key, title=title)
        print(f"  Reliability (rougeL)  → {p}")

    p = str(out_dir / "nlg_baselines" / "roc_rougeL.png")
    plot_roc_curve(
        results, p,
        label_fn=lambda r: (
            r.get("mean_rougeL", float("nan")) >= 0.4
            if not math.isnan(r.get("mean_rougeL", float("nan")))
            else None
        ),
        title="ROC Curve — ROUGE-L (NLG baseline, threshold ≥ 0.4)",
    )
    print(f"  ROC (rougeL)          → {p}")

    p = str(out_dir / "confidence" / "selective_prediction.png")
    plot_selective_prediction(results, p)
    print(f"  Selective prediction  → {p}")


def process(results_path: Path, no_plots: bool) -> None:
    from src.uq.metrics import summarise

    if not results_path.exists():
        print(f"[skip] not found: {results_path}")
        return

    with open(results_path) as f:
        results = json.load(f)

    summary = summarise(results)
    out_dir = results_path.parent
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    if not no_plots:
        _make_plots(results, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Regenerate summary.json and plots from results.json."
    )
    ap.add_argument("results", nargs="?",
                    help="Path to a single results.json")
    ap.add_argument("--all", metavar="DIR",
                    help="Recursively find and process all results.json under DIR")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip plot generation")
    args = ap.parse_args()

    if args.all:
        paths = sorted(Path(args.all).rglob("results.json"))
        # Skip results_judged.json equivalents and judge subdirs
        paths = [p for p in paths if p.parent.name != "judge_correctness"]
        if not paths:
            sys.exit(f"No results.json found under {args.all}")
        print(f"Found {len(paths)} results.json files\n")
        for p in paths:
            print(f"--- {p}")
            process(p, args.no_plots)
            print()
    elif args.results:
        process(Path(args.results), args.no_plots)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
