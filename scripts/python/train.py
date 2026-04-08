import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse
from src.training.trainer import TrainConfig, train


def main():
    ap = argparse.ArgumentParser(description="Train a LLaMA model from scratch.")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output-dir", default="outputs/training")

    # Data source
    ap.add_argument("--data-dir", default="", help="Path to preprocessed shards (disk mode)")
    ap.add_argument("--online", action="store_true", help="Stream from HF directly (no disk)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit (online mode only)")

    # Model (ignored when --pretrained-model is set)
    ap.add_argument("--pretrained-model", default="", help="HF model ID or local path for fine-tuning (skips scratch init)")
    ap.add_argument("--n-layers", type=int, default=24)
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--n-heads", type=int, default=16)

    # Training
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (use different seeds for ensemble members)")

    args = ap.parse_args()

    if not args.online and not args.data_dir:
        ap.error("Either --data-dir or --online is required")

    cfg = TrainConfig(
        tokenizer=args.tokenizer,
        pretrained_model=args.pretrained_model,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        online=args.online,
        seq_len=args.seq_len,
        limit=args.limit,
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
        resume=args.resume,
        seed=args.seed,
    )

    train(cfg)


if __name__ == "__main__":
    main()