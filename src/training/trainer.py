from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.packed_dataset import PackedShardDataset
from src.model.llama_model import LlamaModelConfig, build_llama

@dataclass
class TrainConfig:
    data_dir: str
    tokenizer: str
    n_layers: int = 8
    hidden_size: int = 512
    n_heads: int = 8
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 1000
    num_workers: int = 2
    fp16: bool = False

def masked_causal_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor | None,
) -> torch.Tensor:
    loss_token = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)

    if loss_mask is None:
        return loss_token.mean()

    loss_mask = loss_mask.to(loss_token.dtype)
    return (loss_token * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)

def load_manifest(data_dir: str) -> dict:
    import json
    from pathlib import Path

    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {data_dir}")
    with open(manifest_path) as f:
        return json.load(f)
    
def train(cfg: TrainConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load manifest
    manifest = load_manifest(cfg.data_dir)
    seq_len = manifest["seq_len"]
    vocab_size = manifest["vocab_size"]
    use_loss_mask = manifest["save_loss_mask"]

    # Build model
    model_cfg = LlamaModelConfig(
        vocab_size=vocab_size,
        max_position_embeddings=seq_len,
        num_hidden_layers=cfg.n_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.n_heads,
    )

    model = build_llama(model_cfg)
    model.to(device)
    model.train()

    # Dataset
    dataset = PackedShardDataset(
        cfg.data_dir,
        shuffle=True,
        seed=42,
        rank=0,
        world_size=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.fp16 and device == "cuda"))

    data_iter = iter(loader)

    for step in range(cfg.steps):
        batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)

        loss_mask = None
        if use_loss_mask and "loss_mask" in batch:
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        if loss_mask is not None:
            loss_mask = loss_mask[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(cfg.fp16 and device == "cuda")):
            outputs = model(input_ids=x)
            logits = outputs.logits
            loss = masked_causal_loss(logits, y, loss_mask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            print(f"Step {step} | Loss {loss.item():.4f}")

    print("Training complete.")