from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.packed_dataset import PackedShardDataset
from src.model.llama_model import LlamaModelConfig, build_llama

import torch.distributed as dist

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


def load_manifest(data_dir: str) -> dict:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {data_dir}")
    with open(manifest_path) as f:
        return json.load(f)


def masked_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute masked causal language modeling loss.

    logits: (B, T, V)
    labels: (B, T)
    loss_mask: (B, T) or None
    """
    loss_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)

    if loss_mask is None:
        return loss_token.mean()

    loss_mask = loss_mask.to(loss_token.dtype)
    return (loss_token * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tokenizer", required=True)

    # Model size
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load manifest
    manifest = load_manifest(args.data_dir)
    seq_len = manifest["seq_len"]
    vocab_size = manifest["vocab_size"]
    use_loss_mask = manifest["save_loss_mask"]

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    assert (
        tokenizer.vocab_size == vocab_size
        or len(tokenizer) == vocab_size
    ), "Tokenizer vocab size does not match manifest."

    # Build model from scratch
    model_cfg = LlamaModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=seq_len,
        num_hidden_layers=args.n_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.n_heads,
    )

    model = build_llama(model_cfg)
    model.to(device)
    model.train()

    # Dataset

    # rank = dist.get_rank()
    # world_size = dist.get_world_size()

    dataset = PackedShardDataset(
        args.data_dir,
        shuffle=True,
        seed=42,
        rank=0,
        world_size=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.fp16 and device == "cuda"))

    data_iter = iter(loader)

    for step in range(args.steps):
        batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)

        loss_mask = None
        if use_loss_mask and "loss_mask" in batch:
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        # Shift for causal LM
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        if loss_mask is not None:
            loss_mask = loss_mask[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(args.fp16 and device == "cuda")):
            outputs = model(input_ids=x)
            logits = outputs.logits
            loss = masked_causal_loss(logits, y, loss_mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()