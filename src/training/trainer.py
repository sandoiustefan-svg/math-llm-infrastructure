from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

from src.data.load_shards import NpyShardDataset, load_manifest
from src.model.llama_model import LlamaModelConfig, build_llama
from src.experiments.registry import ExperimentRegistry


@dataclass
class TrainConfig:
    tokenizer: str
    output_dir: str = "outputs/training"

    data_dir: str = ""
    seq_len: int = 1024
    limit: int = 0

    n_layers: int = 24
    hidden_size: int = 2048
    n_heads: int = 16

    batch_size: int = 8
    lr: float = 3e-4
    steps: int = 1000
    epochs: float = 0.0
    num_workers: int = 2
    fp16: bool = False
    bf16: bool = False
    warmup_steps: int = 2000
    grad_accum_steps: int = 1

    pretrained_model: str = ""
    trainable_layers: int = 0

    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    use_qlora: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True

    mc_dropout_rate: float = 0.0

    seed: int = 42
    save_every: int = 500
    log_every: int = 50
    resume: bool = False

    val_shard_count: int = 0
    test_shard_count: int = 0
    val_every: int = 1000
    val_batches: int = 50


def load_train_config(config_path: str | Path) -> TrainConfig:
    path = Path(config_path)

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError(f"Empty config file: {path}")

    valid_fields = {f.name for f in fields(TrainConfig)}
    unknown = set(raw.keys()) - valid_fields

    if unknown:
        raise ValueError(f"Unknown config fields in {path}: {sorted(unknown)}")

    return TrainConfig(**raw)


def _dtype_from_string(dtype: str) -> torch.dtype:
    dtype = dtype.lower()

    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32"}:
        return torch.float32

    raise ValueError(f"Unsupported dtype: {dtype}")


def _build_bnb_config(cfg: TrainConfig) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_dtype_from_string(cfg.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def _apply_lora(model, cfg: TrainConfig):
    if not _PEFT_AVAILABLE:
        raise ImportError("peft not installed. Run: pip install peft")

    if cfg.use_qlora:
        target_modules: Any = "all-linear"
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    return get_peft_model(model, lora_cfg)


def _add_mc_dropout_hook(model, rate: float) -> None:
    dropout = nn.Dropout(p=rate)

    m = model
    if hasattr(m, "base_model"):
        m = m.base_model.model

    inner = m.model
    dropout.to(next(inner.parameters()).device)

    inner.norm.add_module("_mc_dropout", dropout)
    inner.norm.register_forward_hook(lambda _m, _i, o: inner.norm._mc_dropout(o))


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


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


@torch.no_grad()
def _run_validation(
    model,
    val_loader,
    device,
    use_fp16: bool,
    use_bf16: bool,
    max_batches: int,
    world_size: int,
) -> float:
    was_training = model.training
    model.eval()

    total_loss = torch.zeros(1, device=device)
    total_count = torch.zeros(1, device=device)

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_enabled = (use_fp16 or use_bf16) and "cuda" in str(device)

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        attention_mask = attention_mask[:, :-1]
        loss_mask = loss_mask[:, 1:]

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(input_ids=x, attention_mask=attention_mask)
            loss = masked_causal_loss(outputs.logits, y, loss_mask)

        total_loss += loss.detach()
        total_count += 1

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)

    if was_training:
        model.train()

    return (total_loss / total_count.clamp_min(1.0)).item()


def save_checkpoint(
    model,
    optimizer,
    step,
    metrics,
    samples_consumed,
    path,
    exp_id=None,
    use_lora=False,
    use_qlora=False,
    base_model_id="",
):
    os.makedirs(path, exist_ok=True)

    model.save_pretrained(path)

    state = {
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "samples_consumed": samples_consumed,
        "exp_id": exp_id,
        "use_lora": use_lora,
        "use_qlora": use_qlora,
        "base_model_id": base_model_id,
    }

    buf = io.BytesIO()
    torch.save(state, buf)
    data = buf.getvalue()
    del buf

    state_path = os.path.join(path, "training_state.pt")
    chunk = 256 * 1024 * 1024

    with open(state_path, "wb") as f:
        for i in range(0, len(data), chunk):
            f.write(data[i : i + chunk])

    del data
    print(f"  Checkpoint saved → {path}")


def plot_loss_curve(metrics: dict, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = metrics["steps"]
        loss = metrics["loss"]

        if not steps:
            return

        window = max(1, len(loss) // 50)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, loss, linewidth=0.8, alpha=0.4, label="train raw")

        if len(loss) > window:
            smoothed = [
                sum(loss[max(0, i - window) : i + 1])
                / len(loss[max(0, i - window) : i + 1])
                for i in range(len(loss))
            ]
            ax.plot(steps, smoothed, linewidth=2, label="train smoothed")

        if metrics.get("val_steps"):
            ax.plot(
                metrics["val_steps"],
                metrics["val_loss"],
                linewidth=2,
                marker="o",
                markersize=4,
                label="val",
            )

        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss step {steps[-1]}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Loss curve saved → {path}")

    except ImportError:
        pass


def plot_metrics(metrics: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "metrics.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Metrics saved → {metrics_path}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not metrics["steps"]:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

        window = max(1, len(metrics["loss"]) // 50)

        ax = axes[0, 0]
        ax.plot(metrics["steps"], metrics["loss"], linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(metrics["steps"], metrics["loss"], linewidth=0.8, alpha=0.4)
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss log")
        ax.set_title("Training Loss Log Scale")
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
                smoothed_tps = [
                    sum(metrics["tokens_per_sec"][max(0, i - window) : i + 1])
                    / len(metrics["tokens_per_sec"][max(0, i - window) : i + 1])
                    for i in range(len(metrics["tokens_per_sec"]))
                ]
                ax.plot(metrics["steps"], smoothed_tps, linewidth=2)

        ax.set_xlabel("Step")
        ax.set_ylabel("Tokens/sec")
        ax.set_title("Throughput")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = os.path.join(output_dir, "training_metrics.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Plots saved → {plot_path}")

    except ImportError:
        print("  matplotlib not available — skipping plots")


def _find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    try:
        ckpt_dirs = sorted(
            [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")],
            key=lambda x: int(x.split("_")[1]),
        )
    except FileNotFoundError:
        return None

    if not ckpt_dirs:
        return None

    latest = os.path.join(checkpoint_dir, ckpt_dirs[-1])
    state_path = os.path.join(latest, "training_state.pt")

    if not os.path.exists(state_path):
        return None

    return latest


def _load_pretrained_model(
    cfg: TrainConfig,
    local_rank: int,
    resume_state_path: str | None,
):
    if cfg.use_qlora:
        if not cfg.use_lora:
            raise ValueError("QLoRA requires use_lora=True")

        if not _PEFT_AVAILABLE:
            raise ImportError("QLoRA requires peft. Run: pip install peft")

        quantization_config = _build_bnb_config(cfg)

        model = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model,
            quantization_config=quantization_config,
            device_map={"": local_rank},
            low_cpu_mem_usage=True,
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

        if resume_state_path is not None:
            model = PeftModel.from_pretrained(
                model,
                resume_state_path,
                is_trainable=True,
            )
        else:
            model = _apply_lora(model, cfg)

        return model

    torch_dtype = torch.bfloat16 if cfg.bf16 else torch.float16 if cfg.fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )

    if cfg.use_lora:
        if resume_state_path is not None:
            model = PeftModel.from_pretrained(
                model,
                resume_state_path,
                is_trainable=True,
            )
        else:
            model = _apply_lora(model, cfg)

    model.gradient_checkpointing_enable()

    if cfg.use_lora:
        model.enable_input_require_grads()

    return model


def train(cfg: TrainConfig) -> None:
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.set_device(local_rank)

    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")

    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    metrics = {
        "steps": [],
        "loss": [],
        "lr": [],
        "tokens_per_sec": [],
        "val_steps": [],
        "val_loss": [],
    }

    start_step = 0
    skip_samples = 0
    exp_id = None
    resume_state_path = None

    if cfg.resume:
        resume_state_path = _find_latest_checkpoint(checkpoint_dir)

        if resume_state_path is not None:
            state = torch.load(
                os.path.join(resume_state_path, "training_state.pt"),
                map_location="cpu",
            )
            start_step = state["step"]
            metrics = state.get("metrics", metrics)
            metrics.setdefault("val_steps", [])
            metrics.setdefault("val_loss", [])
            skip_samples = state.get("samples_consumed", 0)
            exp_id = state.get("exp_id", None)

            if rank == 0:
                print(f"Will resume from {resume_state_path} at step {start_step}")

        elif rank == 0:
            print("No checkpoint found — starting from scratch.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    manifest = load_manifest(cfg.data_dir)
    seq_len = int(manifest.get("seq_len", cfg.seq_len))

    total_shards = len(sorted(Path(cfg.data_dir).glob("input_ids_*.npy")))

    if total_shards == 0:
        raise FileNotFoundError(f"No input_ids_*.npy shards found in {cfg.data_dir}")

    if not bool(manifest.get("has_attention_mask", False)):
        raise ValueError(
            "Your shards do not contain attention_mask_*.npy files. "
            "Rerun preprocessing with the updated pack_shards.py."
        )

    total_holdout = cfg.val_shard_count + cfg.test_shard_count

    if total_holdout > 0:
        if total_holdout >= total_shards:
            raise ValueError(
                f"val + test shards ({total_holdout}) must be smaller than total_shards ({total_shards})"
            )

        test_start = total_shards - cfg.test_shard_count
        val_start = test_start - cfg.val_shard_count

        train_shard_indices = list(range(0, val_start))
        val_shard_indices = (
            list(range(val_start, test_start)) if cfg.val_shard_count > 0 else None
        )
        test_shard_indices = (
            list(range(test_start, total_shards)) if cfg.test_shard_count > 0 else None
        )

        all_indices = list(train_shard_indices)
        if val_shard_indices is not None:
            all_indices.extend(val_shard_indices)
        if test_shard_indices is not None:
            all_indices.extend(test_shard_indices)
        assert len(set(all_indices)) == len(all_indices), "shard splits overlap"
        assert sorted(all_indices) == list(range(total_shards)), "shard splits do not cover dataset"
    else:
        train_shard_indices = None
        val_shard_indices = None
        test_shard_indices = None

    if rank == 0:
        os.makedirs(cfg.output_dir, exist_ok=True)
        splits_path = os.path.join(cfg.output_dir, "splits.json")
        splits = {
            "total_shards": total_shards,
            "data_dir": str(cfg.data_dir),
            "shard_num_seqs": int(manifest.get("shard_num_seqs", 0)),
            "pad_token_id": manifest.get("pad_token_id"),
            "train_shard_indices": train_shard_indices,
            "val_shard_indices": val_shard_indices,
            "test_shard_indices": test_shard_indices,
        }
        with open(splits_path, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        print(f"  Splits saved → {splits_path}")

    if cfg.epochs > 0:
        train_shards = len(train_shard_indices) if train_shard_indices is not None else total_shards
        train_seqs = train_shards * int(manifest["shard_num_seqs"])
        steps_per_epoch = math.ceil(train_seqs / world_size / cfg.batch_size)
        cfg.steps = math.ceil(cfg.epochs * steps_per_epoch)

        if rank == 0:
            print(f"Epochs mode: {cfg.epochs} epochs × {steps_per_epoch} steps/epoch = {cfg.steps} steps")

    safe_num_workers = 0 if world_size > 1 else cfg.num_workers

    train_dataset = NpyShardDataset(
        cfg.data_dir,
        shard_indices=train_shard_indices,
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=cfg.seed,
        )
        if world_size > 1
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=safe_num_workers,
        pin_memory=("cuda" in str(device)),
        drop_last=True,
    )

    val_loader = None
    val_sampler = None

    if val_shard_indices is not None:
        val_dataset = NpyShardDataset(
            cfg.data_dir,
            shard_indices=val_shard_indices,
        )

        val_sampler = (
            DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
            if world_size > 1
            else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=safe_num_workers,
            pin_memory=("cuda" in str(device)),
            drop_last=False,
        )

    if rank == 0:
        print(f"Data mode: DISK shards from {cfg.data_dir}")
        print(f"Total shards: {total_shards}")
        train_count = len(train_shard_indices) if train_shard_indices is not None else total_shards
        val_count = len(val_shard_indices) if val_shard_indices is not None else 0
        test_count = len(test_shard_indices) if test_shard_indices is not None else 0
        print(
            f"Train shards: {train_count} | Val shards: {val_count} | "
            f"Test shards: {test_count} (held out)"
        )

    if cfg.pretrained_model:
        model = _load_pretrained_model(cfg, local_rank, resume_state_path)

        if rank == 0:
            print(f"Loaded pretrained model: {cfg.pretrained_model}")
            if cfg.use_qlora:
                print("Training mode: QLoRA 4-bit")
            elif cfg.use_lora:
                print("Training mode: LoRA")
            else:
                print("Training mode: full/partial fine-tuning")

        if hasattr(model, "print_trainable_parameters") and rank == 0:
            model.print_trainable_parameters()

    else:
        model_cfg = LlamaModelConfig(
            vocab_size=len(tokenizer),
            max_position_embeddings=seq_len,
            num_hidden_layers=cfg.n_layers,
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.n_heads,
        )
        model = build_llama(model_cfg)

    if not cfg.use_qlora:
        model.to(device)

    if cfg.pretrained_model and cfg.trainable_layers > 0 and not cfg.use_lora and not cfg.use_qlora:
        for p in model.parameters():
            p.requires_grad_(False)

        inner = model.model

        for layer in inner.layers[-cfg.trainable_layers :]:
            for p in layer.parameters():
                p.requires_grad_(True)

        for p in inner.norm.parameters():
            p.requires_grad_(True)

        for p in model.lm_head.parameters():
            p.requires_grad_(True)

        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"Partial fine-tune: last {cfg.trainable_layers} layers trainable "
                f"({trainable:,} / {total:,}, {100 * trainable / total:.2f}%)"
            )

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    if cfg.mc_dropout_rate > 0 and cfg.pretrained_model:
        target = model.module if world_size > 1 else model
        _add_mc_dropout_hook(target, cfg.mc_dropout_rate)

        if rank == 0:
            print(f"MC Dropout hook added with rate={cfg.mc_dropout_rate}")

    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    use_fp16 = cfg.fp16 and not cfg.bf16 and "cuda" in str(device)
    use_bf16 = cfg.bf16 and "cuda" in str(device)

    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    total_opt_steps = max(1, cfg.steps // cfg.grad_accum_steps)
    warmup_opt_steps = max(1, cfg.warmup_steps)
    start_opt_step = start_step // cfg.grad_accum_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(
            step + start_opt_step,
            warmup_opt_steps,
            total_opt_steps,
        ),
    )

    if resume_state_path is not None:
        resume_state = torch.load(
            os.path.join(resume_state_path, "training_state.pt"),
            map_location=device,
        )

        try:
            optimizer.load_state_dict(resume_state["optimizer"])
        except ValueError as exc:
            if rank == 0:
                print(f"Could not load optimizer state, continuing with fresh optimizer: {exc}")

        del resume_state

        if rank == 0:
            print(f"Resumed training state from step {start_step}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(
            f"Model parameters: {total_params:,} total | "
            f"{n_params:,} trainable ({n_params / 1e6:.2f}M)"
        )
        if world_size > 1:
            print(f"DDP: {world_size} GPUs")
        print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps * world_size}")

    if rank == 0:
        try:
            registry = ExperimentRegistry()

            if exp_id is not None:
                registry.update(exp_id, status="running")
                print(f"  Experiment resumed → {exp_id}")
            else:
                exp_id = registry.start_training(cfg, n_params)
                print(f"  Experiment registered → {exp_id}")

        except Exception as e:
            print(f"  [registry] Warning: could not register experiment: {e}")

    best_loss = float("inf")
    samples_consumed = skip_samples
    epoch_idx = 0
    if train_sampler is not None:
        train_sampler.set_epoch(epoch_idx)
    data_iter = iter(train_loader)

    train_start = time.time()
    step_start = time.time()

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_enabled = (use_fp16 or use_bf16) and "cuda" in str(device)

    for step in range(start_step, cfg.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch_idx += 1
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_idx)
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        attention_mask = attention_mask[:, :-1]
        loss_mask = loss_mask[:, 1:]

        is_window_start = step % cfg.grad_accum_steps == 0
        is_step_boundary = (step + 1) % cfg.grad_accum_steps == 0

        if is_window_start:
            optimizer.zero_grad(set_to_none=True)

        sync_ctx = (
            model.no_sync()
            if world_size > 1 and not is_step_boundary
            else contextlib.nullcontext()
        )

        with sync_ctx:
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    input_ids=x,
                    attention_mask=attention_mask,
                )
                loss = masked_causal_loss(outputs.logits, y, loss_mask)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

        if is_step_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        samples_consumed += cfg.batch_size * world_size

        step_end = time.time()
        tokens_in_batch = x.numel() * world_size
        tokens_per_sec = tokens_in_batch / max(1e-8, step_end - step_start)

        raw_loss = (loss.detach() * cfg.grad_accum_steps).item()

        metrics["steps"].append(step)
        metrics["loss"].append(raw_loss)
        metrics["lr"].append(scheduler.get_last_lr()[0])
        metrics["tokens_per_sec"].append(tokens_per_sec)

        step_start = time.time()

        if rank == 0 and step % cfg.log_every == 0:
            elapsed = time.time() - train_start
            steps_done = step - start_step + 1
            steps_left = cfg.steps - step - 1
            time_per_step = elapsed / max(1, steps_done)
            eta = steps_left * time_per_step

            print(
                f"Step {step:>6d}/{cfg.steps} | "
                f"Loss {raw_loss:.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e} | "
                f"Tok/s {tokens_per_sec:,.0f} | "
                f"Elapsed {_format_time(elapsed)} | "
                f"ETA {_format_time(eta)}"
            )

            plot_loss_curve(metrics, os.path.join(cfg.output_dir, "loss_curve.png"))

        if (
            val_loader is not None
            and cfg.val_every > 0
            and step > 0
            and step % cfg.val_every == 0
        ):
            val_loss = _run_validation(
                model=model,
                val_loader=val_loader,
                device=device,
                use_fp16=use_fp16,
                use_bf16=use_bf16,
                max_batches=cfg.val_batches,
                world_size=world_size,
            )

            if rank == 0:
                metrics["val_steps"].append(step)
                metrics["val_loss"].append(val_loss)
                print(f"  Val loss @ step {step}: {val_loss:.4f}")

        if rank == 0 and cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            save_model = model.module if world_size > 1 else model
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step}")

            save_checkpoint(
                save_model,
                optimizer,
                step,
                metrics,
                samples_consumed,
                ckpt_path,
                exp_id=exp_id,
                use_lora=cfg.use_lora,
                use_qlora=cfg.use_qlora,
                base_model_id=cfg.pretrained_model,
            )

            plot_metrics(metrics, cfg.output_dir)
            plot_loss_curve(metrics, os.path.join(cfg.output_dir, "loss_curve.png"))

            if metrics.get("val_loss"):
                recent_loss = metrics["val_loss"][-1]
                metric_name = "val loss"
            else:
                window = min(cfg.save_every, len(metrics["loss"]))
                recent_loss = sum(metrics["loss"][-window:]) / window
                metric_name = "avg train loss"

            if recent_loss < best_loss:
                best_loss = recent_loss
                best_path = os.path.join(checkpoint_dir, "best")
                save_model.save_pretrained(best_path)
                print(f"  Best checkpoint updated → {best_path} ({metric_name} {best_loss:.4f})")

            prev2_step = step - 2 * cfg.save_every
            prev2_path = os.path.join(checkpoint_dir, f"step_{prev2_step}")

            if os.path.exists(prev2_path):
                shutil.rmtree(prev2_path)
                print(f"  Deleted old checkpoint → {prev2_path}")

    if rank == 0:
        print("")
        print("Saving final model...")

        save_model = model.module if world_size > 1 else model

        save_checkpoint(
            save_model,
            optimizer,
            cfg.steps,
            metrics,
            samples_consumed,
            os.path.join(checkpoint_dir, "final"),
            exp_id=exp_id,
            use_lora=cfg.use_lora,
            use_qlora=cfg.use_qlora,
            base_model_id=cfg.pretrained_model,
        )

        print("Generating training plots...")
        plot_metrics(metrics, cfg.output_dir)
        plot_loss_curve(metrics, os.path.join(cfg.output_dir, "loss_curve.png"))

        final_loss = metrics["loss"][-1] if metrics["loss"] else float("nan")

        print("")
        print("Training complete.")
        print(f"  Total steps : {cfg.steps}")
        print(f"  Final loss  : {final_loss:.4f}")
        print(f"  Checkpoints : {checkpoint_dir}")
        print(f"  Metrics     : {cfg.output_dir}")

        if exp_id is not None:
            try:
                registry = ExperimentRegistry()
                registry.complete_training(
                    exp_id,
                    final_loss=final_loss,
                    steps=cfg.steps,
                    samples_consumed=samples_consumed,
                    checkpoint_path=os.path.join(checkpoint_dir, "final"),
                    output_dir=cfg.output_dir,
                )
                print(f"  Experiment updated → {exp_id}")

            except Exception as e:
                print(f"  [registry] Warning: could not update experiment: {e}")

    if world_size > 1:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file, e.g. configs/llama3_8b_qlora.yaml",
    )
    args = parser.parse_args()

    cfg = load_train_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()