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
    ap.add_argument("--output-dir", default="outputs/training")
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--hidden-size", type=int, default=512)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        n_layers=args.n_layers,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        num_workers=args.num_workers,
        fp16=args.fp16,
        save_every=args.save_every,
        log_every=args.log_every,
    )

    train(cfg)


if __name__ == "__main__":
    main()