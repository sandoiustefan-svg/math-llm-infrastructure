"""Regenerate loss_curve.png and training_metrics.png from a saved metrics.json."""

import argparse
import json
from pathlib import Path

from src.training.trainer import plot_loss_curve, plot_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Replot training metrics from metrics.json")
    parser.add_argument("metrics_json", type=str, help="Path to metrics.json")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for plots (default: same dir as metrics.json)",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    out_dir = Path(args.out_dir) if args.out_dir else metrics_path.parent

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    plot_metrics(metrics, str(out_dir))
    plot_loss_curve(metrics, str(out_dir / "loss_curve.png"))


if __name__ == "__main__":
    main()
