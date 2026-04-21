from __future__ import annotations

import io
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False

from src.data.packed_dataset import PackedShardDataset
from src.data.online_dataset import OnlinePackedDataset
from src.data.doc_mask import build_doc_mask_and_positions
from src.model.llama_model import LlamaModelConfig, build_llama
from src.experiments.registry import ExperimentRegistry


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
    fp16: bool = False   # float16 + GradScaler
    bf16: bool = False   # bfloat16, no GradScaler (preferred on A100)
    warmup_steps: int = 2000
    grad_accum_steps: int = 1

    # Fine-tuning
    pretrained_model: str = ""  # HF model ID or local path; empty = train from scratch
    trainable_layers: int = 0   # 0 = all layers; N > 0 = freeze all except last N transformer layers + norm + lm_head

    # LoRA
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # MC Dropout (UQ at inference)
    mc_dropout_rate: float = 0.1  # dropout applied after final norm, before lm_head

    # Reproducibility
    seed: int = 42

    # Checkpointing
    save_every: int = 500
    log_every: int = 50
    resume: bool = False


def _apply_lora(model, cfg: TrainConfig):
    if not _PEFT_AVAILABLE:
        raise ImportError("peft not installed — run: pip install peft")
    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_cfg)


def _add_mc_dropout_hook(model, rate: float) -> None:
    """Add dropout after the final RMSNorm (before lm_head) for MC Dropout UQ.

    Works with bare LlamaForCausalLM and PEFT-wrapped models.
    Call model.train() at inference time to enable stochastic sampling.
    """
    dropout = nn.Dropout(p=rate)
    # Navigate: (PeftModel ->) LlamaForCausalLM -> LlamaModel -> norm
    m = model
    if hasattr(m, 'base_model'):       # PEFT LoraModel
        m = m.base_model.model
    inner = m.model                    # LlamaModel
    dropout.to(next(inner.parameters()).device)
    inner.norm.register_forward_hook(lambda _m, _i, o: dropout(o))


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    import math
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


def load_manifest(data_dir: str) -> dict:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {data_dir}")
    with open(manifest_path) as f:
        return json.load(f)


def save_checkpoint(model, optimizer, step, metrics, samples_consumed, path, exp_id=None, use_lora=False, base_model_id=""):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)  # saves adapters only if PEFT, full model otherwise
    state = {
        "optimizer": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
        "samples_consumed": samples_consumed,
        "exp_id": exp_id,
        "use_lora": use_lora,
        "base_model_id": base_model_id,
    }
    # Serialize to memory first, then flush in 256 MB chunks.
    # Avoids Lustre large-write EINVAL from PyTorch's C++ _write_file syscall.
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
    """Save a simple loss-only PNG, overwriting on each call."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = metrics["steps"]
        loss  = metrics["loss"]
        window = max(1, len(loss) // 50)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, loss, linewidth=0.8, alpha=0.4, label="raw")
        if len(loss) > window:
            smoothed = [
                sum(loss[max(0, i - window):i + 1]) / len(loss[max(0, i - window):i + 1])
                for i in range(len(loss))
            ]
            ax.plot(steps, smoothed, linewidth=2, color="red", label="smoothed")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss (step {steps[-1] if steps else 0})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Loss curve saved → {path}")
    except ImportError:
        pass


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

        ax = axes[0, 1]
        ax.plot(metrics["steps"], metrics["loss"], linewidth=0.8, alpha=0.4)
        if len(metrics["loss"]) > window:
            ax.plot(metrics["steps"], smoothed, linewidth=2, color="red")
        ax.set_yscale("log")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss (log)")
        ax.set_title("Training Loss (Log Scale)")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(metrics["steps"], metrics["lr"], linewidth=1.5, color="green")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

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
    torch.cuda.manual_seed_all(cfg.seed)

    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoints")
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=True)

    metrics = {
        "steps": [],
        "loss": [],
        "lr": [],
        "tokens_per_sec": [],
    }

    start_step = 0
    skip_samples = 0
    resume_state_path = None
    exp_id = None

    if cfg.resume:
        try:
            ckpt_dirs = sorted(
                [d for d in os.listdir(checkpoint_dir) if d.startswith("step_")],
                key=lambda x: int(x.split("_")[1])
            )
        except FileNotFoundError:
            ckpt_dirs = []

        if ckpt_dirs:
            latest = os.path.join(checkpoint_dir, ckpt_dirs[-1])
            state_path = os.path.join(latest, "training_state.pt")
            if os.path.exists(state_path):
                resume_state_path = latest
                state = torch.load(state_path, map_location="cpu")
                start_step = state["step"]
                metrics = state.get("metrics", metrics)
                skip_samples = state.get("samples_consumed", 0)
                exp_id = state.get("exp_id", None)

                if rank == 0:
                    print(f"Will resume from {latest} at step {start_step}, skipping {skip_samples} samples")
        else:
            if rank == 0:
                print("No checkpoint found — starting from scratch.")

    if cfg.online:
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
            pin_memory=("cuda" in str(device)),
        )
        if rank == 0:
            print(f"Data mode: ONLINE (streaming from HF)")
    else:
        manifest = load_manifest(cfg.data_dir)
        seq_len = manifest["seq_len"]
        vocab_size = manifest["vocab_size"]
        use_loss_mask = manifest["save_loss_mask"]

        dataset = PackedShardDataset(
            cfg.data_dir,
            shuffle=True,
            seed=cfg.seed,
            rank=rank,
            world_size=world_size,
            skip_samples=skip_samples,
        )
        # num_workers > 0 causes SIGSEGV under DDP because DataLoader forks
        # after CUDA is initialised — safe to use workers only in single-GPU mode.
        safe_num_workers = 0 if world_size > 1 else cfg.num_workers
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=safe_num_workers,
            pin_memory=("cuda" in str(device)),
        )
        if rank == 0:
            print(f"Data mode: DISK (shards from {cfg.data_dir})")

    if cfg.pretrained_model:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_model,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        seq_len = model.config.max_position_embeddings
        vocab_size = model.config.vocab_size
        if rank == 0:
            print(f"Loaded pretrained model: {cfg.pretrained_model}")
        if cfg.use_lora:
            model = _apply_lora(model, cfg)
            if rank == 0:
                model.print_trainable_parameters()
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # required for PEFT + gradient checkpointing
    else:
        model_cfg = LlamaModelConfig(
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            num_hidden_layers=cfg.n_layers,
            hidden_size=cfg.hidden_size,
            num_attention_heads=cfg.n_heads,
        )
        model = build_llama(model_cfg)

    model.to(device)

    if cfg.pretrained_model and cfg.trainable_layers > 0:
        # Freeze all parameters, then unfreeze the last N transformer layers + norm + lm_head
        for p in model.parameters():
            p.requires_grad_(False)
        inner = model.model  # LlamaModel inside LlamaForCausalLM
        for layer in inner.layers[-cfg.trainable_layers:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for p in inner.norm.parameters():
            p.requires_grad_(True)
        for p in model.lm_head.parameters():
            p.requires_grad_(True)
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Partial fine-tune: last {cfg.trainable_layers} layers trainable "
                  f"({trainable:,} / {total:,} params, {100*trainable/total:.1f}%)")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=cfg.use_lora)

    if cfg.mc_dropout_rate > 0 and cfg.pretrained_model:
        target = model.module if world_size > 1 else model
        _add_mc_dropout_hook(target, cfg.mc_dropout_rate)
        if rank == 0:
            print(f"MC Dropout hook added (rate={cfg.mc_dropout_rate}) after final norm")

    model.train()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,} ({total_params / 1e6:.1f}M), "
              f"trainable: {n_params:,} ({n_params / 1e6:.1f}M)")
        if world_size > 1:
            print(f"DDP: {world_size} GPUs, effective batch size = {cfg.batch_size * world_size}")

    if rank == 0:
        try:
            registry = ExperimentRegistry()
            if exp_id is not None:
                # Resuming — reuse existing experiment entry, just mark it running again
                registry.update(exp_id, status="running")
                print(f"  Experiment resumed → {exp_id}")
            else:
                exp_id = registry.start_training(cfg, n_params)
                print(f"  Experiment registered → {exp_id}")
        except Exception as e:
            print(f"  [registry] Warning: could not register experiment: {e}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)
    # GradScaler only for fp16 (bf16 has wide dynamic range and doesn't need it)
    use_fp16 = cfg.fp16 and not cfg.bf16 and "cuda" in str(device)
    use_bf16 = cfg.bf16 and "cuda" in str(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Scheduler counter advances only at optimizer boundaries (every grad_accum_steps).
    # Convert cfg.steps (micro-batch iterations) and start_step to optimizer-step units
    # so warmup/cosine decay line up with actual optimizer updates.
    total_opt_steps = max(1, cfg.steps // cfg.grad_accum_steps)
    warmup_opt_steps = max(1, cfg.warmup_steps)
    start_opt_step = start_step // cfg.grad_accum_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(
            step + start_opt_step, warmup_opt_steps, total_opt_steps
        ),
    )

    if resume_state_path is not None:
        raw_model = model.module if world_size > 1 else model
        resume_state = torch.load(
            os.path.join(resume_state_path, "training_state.pt"),
            map_location=device,
        )
        if resume_state.get("use_lora") and _PEFT_AVAILABLE:
            base_id = resume_state.get("base_model_id") or cfg.pretrained_model
            base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
            loaded = PeftModel.from_pretrained(base, resume_state_path)
            raw_model.load_state_dict(loaded.state_dict())
            del base, loaded
        else:
            loaded = AutoModelForCausalLM.from_pretrained(
                resume_state_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            )
            raw_model.load_state_dict(loaded.state_dict())
            del loaded
        optimizer.load_state_dict(resume_state["optimizer"])
        del resume_state
        if rank == 0:
            print(f"Resumed model and optimizer from step {start_step}")

    best_loss = float("inf")
    samples_consumed = skip_samples
    data_iter = iter(loader)

    bos_tokenizer = tokenizer if cfg.online else AutoTokenizer.from_pretrained(cfg.tokenizer)
    bos_id = bos_tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    if bos_id is None or bos_id < 0:
        # Fallback: LLaMA 3.x hardcoded BOS id
        bos_id = 128000
    if rank == 0:
        print(f"Document-boundary BOS id: {bos_id}")

    train_start = time.time()
    step_start = time.time()

    for step in range(start_step, cfg.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            if rank == 0:
                print("Data exhausted — ending training.")
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)

        loss_mask = None
        if use_loss_mask and "loss_mask" in batch:
            loss_mask = batch["loss_mask"].to(device, non_blocking=True)

        x = input_ids[:, :-1]
        y = input_ids[:, 1:]

        if loss_mask is not None:
            loss_mask = loss_mask[:, 1:]

        is_window_start = (step % cfg.grad_accum_steps == 0)
        is_step_boundary = ((step + 1) % cfg.grad_accum_steps == 0)

        if is_window_start:
            optimizer.zero_grad(set_to_none=True)

        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        amp_enabled = use_fp16 or use_bf16
        attn_mask_4d, pos_ids = build_doc_mask_and_positions(x, bos_id)
        if amp_enabled:
            attn_mask_4d = attn_mask_4d.to(amp_dtype)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(
                input_ids=x,
                attention_mask=attn_mask_4d,
                position_ids=pos_ids,
            )
            logits = outputs.logits
            loss = masked_causal_loss(logits, y, loss_mask)
            loss = loss / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        if is_step_boundary:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        samples_consumed += cfg.batch_size

        step_end = time.time()
        tokens_in_batch = x.numel()
        tokens_per_sec = tokens_in_batch / (step_end - step_start)

        metrics["steps"].append(step)
        metrics["loss"].append((loss * cfg.grad_accum_steps).item())
        metrics["lr"].append(scheduler.get_last_lr()[0])
        metrics["tokens_per_sec"].append(tokens_per_sec)

        step_start = time.time()

        if rank == 0 and step % cfg.log_every == 0:
            elapsed = time.time() - train_start
            steps_done = step - start_step + 1
            steps_left = cfg.steps - step - 1
            time_per_step = elapsed / steps_done
            eta = steps_left * time_per_step

            elapsed_str = _format_time(elapsed)
            eta_str = _format_time(eta)

            print(
                f"Step {step:>6d}/{cfg.steps} | "
                f"Loss {(loss * cfg.grad_accum_steps).item():.4f} | "
                f"LR {scheduler.get_last_lr()[0]:.2e} | "
                f"Tok/s {tokens_per_sec:,.0f} | "
                f"Elapsed {elapsed_str} | "
                f"ETA {eta_str}"
            )
            plot_loss_curve(metrics, os.path.join(cfg.output_dir, "loss_curve.png"))

        if rank == 0 and cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
            save_model = model.module if world_size > 1 else model
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step}")
            save_checkpoint(
                save_model, optimizer, step, metrics, samples_consumed, ckpt_path,
                exp_id=exp_id, use_lora=cfg.use_lora, base_model_id=cfg.pretrained_model,
            )
            plot_metrics(metrics, cfg.output_dir)
            plot_loss_curve(metrics, os.path.join(cfg.output_dir, "loss_curve.png"))

            # Save best checkpoint based on average loss over the last save_every steps
            window = min(cfg.save_every, len(metrics["loss"]))
            recent_loss = sum(metrics["loss"][-window:]) / window
            if recent_loss < best_loss:
                best_loss = recent_loss
                best_path = os.path.join(checkpoint_dir, "best")
                save_model.save_pretrained(best_path)
                print(f"  Best checkpoint updated → {best_path} (avg loss {best_loss:.4f})")

            # Keep only the last 2 checkpoints (safety net for hard job kills)
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
            save_model, optimizer, cfg.steps, metrics, samples_consumed,
            os.path.join(checkpoint_dir, "final"),
            exp_id=exp_id, use_lora=cfg.use_lora, base_model_id=cfg.pretrained_model,
        )

        print("Generating training plots...")
        plot_metrics(metrics, cfg.output_dir)
        plot_loss_curve(metrics, os.path.join(cfg.output_dir, "loss_curve.png"))

        print("")
        print("Training complete.")
        print(f"  Total steps   : {cfg.steps}")
        print(f"  Final loss    : {metrics['loss'][-1]:.4f}")
        print(f"  Checkpoints   : {checkpoint_dir}")
        print(f"  Metrics       : {cfg.output_dir}")

        if exp_id is not None:
            try:
                registry = ExperimentRegistry()
                registry.complete_training(
                    exp_id,
                    final_loss=metrics["loss"][-1],
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