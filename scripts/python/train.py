import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse
from src.training.trainer import TrainConfig, load_train_config, train


def main():
    ap = argparse.ArgumentParser(description="Train/fine-tune a LLaMA model.")

    ap.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file, e.g. configs/llama3_8b_qlora.yaml",
    )

    # Optional overrides
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--pretrained-model", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--data-dir", default=None)

    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--use-qlora", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lora-dropout", type=float, default=None)

    ap.add_argument("--bnb-4bit-quant-type", default=None)
    ap.add_argument("--bnb-4bit-compute-dtype", default=None)
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true")


    ap.add_argument("--warmup-steps", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=None)

    ap.add_argument("--val-shard-count", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--val-batches", type=int, default=None)

    args = ap.parse_args()

    if args.config:
        cfg = load_train_config(args.config)
    else:
        if args.tokenizer is None:
            ap.error("--tokenizer is required when --config is not provided")
        if args.data_dir is None:
            ap.error("--data-dir is required when --config is not provided")

        cfg = TrainConfig(
            tokenizer=args.tokenizer,
            data_dir=args.data_dir,
        )

    overrides = {
        "tokenizer": args.tokenizer,
        "pretrained_model": args.pretrained_model,
        "output_dir": args.output_dir,
        "data_dir": args.data_dir,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "steps": args.steps,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,

        "warmup_steps": args.warmup_steps,
        "save_every": args.save_every,
        "log_every": args.log_every,
        "seed": args.seed,
        "val_shard_count": args.val_shard_count,
        "val_every": args.val_every,
        "val_batches": args.val_batches,
    }

    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    if args.fp16:
        cfg.fp16 = True
        cfg.bf16 = False

    if args.bf16:
        cfg.bf16 = True
        cfg.fp16 = False

    if args.use_lora:
        cfg.use_lora = True

    if args.use_qlora:
        cfg.use_qlora = True
        cfg.use_lora = True

    if args.bnb_4bit_use_double_quant:
        cfg.bnb_4bit_use_double_quant = True

    if args.resume:
        cfg.resume = True

    if not cfg.data_dir:
        ap.error("data_dir is required. Set it in YAML or pass --data-dir.")

    train(cfg)


if __name__ == "__main__":
    main()