import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse
from src.training.trainer import TrainConfig, train


def main():
    ap = argparse.ArgumentParser(description="Train a LLaMA model from scratch.")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        tokenizer=args.tokenizer,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        num_workers=args.num_workers,
        fp16=args.fp16,
    )

    train(cfg)


if __name__ == "__main__":
    main()