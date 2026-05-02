"""Regenerate training_metrics.png from a saved metrics.json."""

import argparse
import json
import os
from pathlib import Path


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

    if not metrics.get("steps"):
        print("No steps in metrics — nothing to plot.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss = metrics["loss"]
    steps = metrics["steps"]
    window = max(1, len(loss) // 50)

    smoothed = [
        sum(loss[max(0, i - window): i + 1]) / len(loss[max(0, i - window): i + 1])
        for i in range(len(loss))
    ]

    val_steps = metrics.get("val_steps", [])
    val_loss = metrics.get("val_loss", [])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

    for ax, yscale, title, ylabel in [
        (axes[0, 0], "linear", "Training Loss", "Loss"),
        (axes[0, 1], "log",    "Training Loss Log Scale", "Loss (log)"),
    ]:
        ax.plot(steps, loss, linewidth=0.8, alpha=0.3, label="train raw")
        ax.plot(steps, smoothed, linewidth=2, label="train smoothed")
        if val_steps:
            ax.plot(val_steps, val_loss, linewidth=2, marker="o", markersize=4, label="val")
        if yscale == "log":
            ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(metrics["steps"], metrics["lr"], linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if metrics.get("tokens_per_sec"):
        ax.plot(metrics["steps"], metrics["tokens_per_sec"], linewidth=0.8, alpha=0.4)
        if len(metrics["tokens_per_sec"]) > window:
            smoothed = [
                sum(metrics["tokens_per_sec"][max(0, i - window): i + 1])
                / len(metrics["tokens_per_sec"][max(0, i - window): i + 1])
                for i in range(len(metrics["tokens_per_sec"]))
            ]
            ax.plot(metrics["steps"], smoothed, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Throughput")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "training_metrics.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
