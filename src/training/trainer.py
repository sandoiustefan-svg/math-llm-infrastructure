from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.packed_dataset import PackedShardDataset
from src.data.online_dataset import OnlinePackedDataset
from src.model.llama_model import LlamaModelConfig, build_llama


@dataclass
class TrainConfig:
    tokenizer: str
    output_dir: str = "outputs/training"

    # Data
    data_dir: str = ""
    online: bool = False
    seq_len: int = 1024
    limit: int = 0

    # Model
    n_layers: int = 24
    hidden_size: int = 2048
    n_heads: int = 16

    # Training
    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 1000
    num_workers: int = 2
    fp16: bool = False

    # Checkpointing
    save_every: int = 500
    log_every: int = 50

def _format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


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
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {data_dir}")
    with open(manifest_path) as f:
        return json.load(f)


def save_checkpoint(model, optimizer, step, metrics, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
    }, os.path.join(path, "training_state.pt"))
    print(f"  Checkpoint saved → {path}")


def plot_metrics(metrics: dict, output_dir: str):
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

        window = max(1, len(metrics["loss"]) // 50)

        # Loss curve
        ax = axes[0, 0]
        ax.plot(metrics["steps"], metrics["loss"], linewidth=0.8, alpha=0.4, label="raw")
        if len(metrics["loss"]) > window:
            smoothed = [
                sum(metrics["loss"][max(0, i - window):i + 1])
                / len(metrics["loss"][max(0, i - window):i + 1])
                for i in range(len(metrics["loss"]))
            ]
            ax.plot(metrics["steps"], smoothed, linewidth=2, color="red", label="smoothed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Log-scale loss
        ax = axes[0, 1]
        ax.plot(metrics["steps"], metrics["loss"], linewidth=0.8, alpha=0.4)
        if len(metrics["loss"]) > window:
            ax.plot(metrics["steps"], smoothed, linewidth=2, color="red")
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (log)")
        ax.set_title("Training Loss (Log Scale)")
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = axes[1, 0]
        ax.plot(metrics["steps"], metrics["lr"], linewidth=1.5, color="green")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

        # Throughput
        ax = axes[1, 1]
        if metrics.get("tokens_per_sec"):
            ax.plot(metrics["steps"], metrics["tokens_per_sec"], linewidth=0.8, alpha=0.4)
            if len(metrics["tokens_per_sec"]) > window:
                smoothed_tps = [
                    sum(metrics["tokens_per_sec"][max(0, i - window):i + 1])
                    / len(metrics["tokens_per_sec"][max(0, i - window):i + 1])
                    for i in range(len(metrics["tokens_per_sec"]))
                ]
                ax.plot(metrics["steps"], smoothed_tps, linewidth=2, color="orange")
            ax.set_xlabel("Step")
            ax.set_ylabel("Tokens/sec")
            ax.set_title("Training Throughput")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No throughput data", ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_metrics.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plots saved → {plot_path}")

    except ImportError:
        print("  matplotlib not available — skipping plots (metrics.json still saved)")


def train(cfg: TrainConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if cfg.online:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
        vocab_size = len(tokenizer)
        seq_len = cfg.seq_len
        use_loss_mask = True

        dataset = OnlinePackedDataset(
            tokenizer_name=cfg.tokenizer,
            seq_len=seq_len,
            split="train",
            limit=cfg.limit if cfg.limit > 0 else None,
            add_eos=True,
            save_loss_mask=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=0,
            pin_memory=(device == "cuda"),
        )
        print(f"Data mode: ONLINE (streaming from HF)")
    else:
        manifest = load_manifest(cfg.data_dir)
        seq_len = manifest["seq_len"]
        vocab_size = manifest["vocab_size"]
        use_loss_mask = manifest["save_loss_mask"]

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
        print(f"Data mode: DISK (shards from {cfg.data_dir})")

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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.fp16 and device == "cuda"))

    metrics = {
        "steps": [],
        "loss": [],
        "lr": [],
        "tokens_per_sec": [],
    }

    data_iter = iter(loader)
    train_start = time.time()
    step_start = time.time()

    for step in range(cfg.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart if dataset is exhausted (epoch boundary)
            data_iter = iter(loader)
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

        step_end = time.time()
        tokens_in_batch = x.numel()
        tokens_per_sec = tokens_in_batch / (step_end - step_start)

        metrics["steps"].append(step)
        metrics["loss"].append(loss.item())
        metrics["lr"].append(optimizer.param_groups[0]["lr"])
        metrics["tokens_per_sec"].append(tokens_per_sec)

        step_start = time.time()

        if step % cfg.log_every == 0:
            elapsed = time.time() - train_start
            steps_done = step + 1
            steps_left = cfg.steps - steps_done
            time_per_step = elapsed / steps_done
            eta = steps_left * time_per_step

            elapsed_str = _format_time(elapsed)
            eta_str = _format_time(eta)

            print(
                f"Step {step:>6d}/{cfg.steps} | "
                f"Loss {loss.item():.4f} | "
                f"LR {optimizer.param_groups[0]['lr']:.2e} | "
                f"Tok/s {tokens_per_sec:,.0f} | "
                f"Elapsed {elapsed_str} | "
                f"ETA {eta_str}"
            )

        if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            save_checkpoint(
                model, optimizer, step, metrics,
                os.path.join(checkpoint_dir, f"step_{step}"),
            )

    print("")
    print("Saving final model...")
    save_checkpoint(
        model, optimizer, cfg.steps, metrics,
        os.path.join(checkpoint_dir, "final"),
    )

    print("Generating training plots...")
    plot_metrics(metrics, cfg.output_dir)

    print("")
    print("Training complete.")
    print(f"  Total steps   : {cfg.steps}")
    print(f"  Final loss    : {metrics['loss'][-1]:.4f}")
    print(f"  Checkpoints   : {checkpoint_dir}")
    print(f"  Metrics       : {cfg.output_dir}")