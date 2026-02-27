from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
import json
from datetime import datetime

import numpy as np
from transformers import AutoTokenizer

from src.utils.logging_utils import setup_logger


@dataclass(frozen=True)
class PackConfig:
    """
    Configuration container for tokenization and sequence packing.

    Attributes:
        tokenizer_name_or_path (str):
            HuggingFace tokenizer identifier or local path.

        seq_len (int):
            Number of tokens per packed training sequence.
            Each saved sequence will have shape (seq_len,).

        out_dir (str):
            Directory where processed .npy shard files will be saved.

        shard_num_seqs (int):
            Number of packed sequences stored per shard file.
            Each shard will have shape (shard_num_seqs, seq_len).

        dtype (str):
            Integer precision used to store token IDs ("int32" or "int64").

        add_eos (bool):
            Whether to append the tokenizer's EOS token to each completion.

        save_loss_mask (bool):
            Whether to save a loss mask indicating which tokens should
            contribute to the training loss (0 = prompt, 1 = completion).
    """
    tokenizer_name_or_path: str
    seq_len: int = 2048
    out_dir: str = "data/processed/openmathinstruct2"
    shard_num_seqs: int = 1024
    dtype: str = "int32"
    add_eos: bool = True
    save_loss_mask: bool = True


def _ensure_out_dir(out_dir: str) -> Path:
    """Create the output directory if it does not already exist."""
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_shard(
    out_dir: Path,
    shard_idx: int,
    input_ids_arr: np.ndarray,
    loss_mask_arr: Optional[np.ndarray],
    logger,
) -> None:
    """Save a shard of packed sequences (and optional loss masks) to disk."""
    ids_path = out_dir / f"input_ids_{shard_idx:05d}.npy"
    np.save(ids_path, input_ids_arr)
    logger.info("Saved %s shape=%s", str(ids_path), input_ids_arr.shape)

    if loss_mask_arr is not None:
        mask_path = out_dir / f"loss_mask_{shard_idx:05d}.npy"
        np.save(mask_path, loss_mask_arr)
        logger.info("Saved %s shape=%s", str(mask_path), loss_mask_arr.shape)


def tokenzie_pack_and_save(
    examples: Iterable[Dict[str, Any]],
    pack_cfg: PackConfig,
    logger_name: str = "preprocess_openmath",
) -> None:
    """
    Tokenize, pack, and save instruction-style training data into fixed-length shards.

    Produces:
        - input_ids_{idx}.npy: (shard_num_seqs, seq_len)
        - loss_mask_{idx}.npy: (shard_num_seqs, seq_len) if enabled
        - manifest.json: metadata needed for training reproducibility
    """
    logger = setup_logger(logger_name)
    out_dir = _ensure_out_dir(pack_cfg.out_dir)

    logger.info("Loading tokenizer: %s", pack_cfg.tokenizer_name_or_path)
    tok = AutoTokenizer.from_pretrained(pack_cfg.tokenizer_name_or_path, use_fast=True)

    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Provide a LLaMA-compatible tokenizer.")

    vocab_size = tok.vocab_size if tok.vocab_size is not None else len(tok)

    seq_len = pack_cfg.seq_len
    dtype = np.int32 if pack_cfg.dtype == "int32" else np.int64

    token_buf: List[int] = []
    loss_buf: List[int] = []  # 0 for prompt, 1 for completion

    shard_input_ids: List[np.ndarray] = []
    shard_loss_mask: List[np.ndarray] = []

    shard_idx = 0
    saved_shards = 0
    total_seqs = 0
    total_examples = 0

    for ex in examples:
        prompt_text = ex["prompt_text"]
        completion_text = ex["completion_text"]

        # Tokenize separately (no special tokens)
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        completion_ids = tok.encode(completion_text, add_special_tokens=False)

        if pack_cfg.add_eos:
            completion_ids = completion_ids + [tok.eos_token_id]

        # Append to stream buffers
        token_buf.extend(prompt_ids)
        loss_buf.extend([0] * len(prompt_ids))

        token_buf.extend(completion_ids)
        loss_buf.extend([1] * len(completion_ids))

        total_examples += 1

        # Pack into fixed-length sequences
        while len(token_buf) >= seq_len:
            seq_ids = np.array(token_buf[:seq_len], dtype=dtype)
            token_buf = token_buf[seq_len:]

            if pack_cfg.save_loss_mask:
                seq_mask = np.array(loss_buf[:seq_len], dtype=dtype)
                loss_buf = loss_buf[seq_len:]
            else:
                seq_mask = None
                loss_buf = loss_buf[seq_len:]

            shard_input_ids.append(seq_ids)
            if pack_cfg.save_loss_mask and seq_mask is not None:
                shard_loss_mask.append(seq_mask)

            total_seqs += 1

            # Save shard if enough sequences
            if len(shard_input_ids) >= pack_cfg.shard_num_seqs:
                input_ids_arr = np.stack(shard_input_ids, axis=0)
                loss_mask_arr = np.stack(shard_loss_mask, axis=0) if pack_cfg.save_loss_mask else None
                _save_shard(out_dir, shard_idx, input_ids_arr, loss_mask_arr, logger)

                shard_idx += 1
                saved_shards += 1
                shard_input_ids.clear()
                shard_loss_mask.clear()

        if total_examples % 10_000 == 0:
            logger.info(
                "Processed examples=%d | packed_seqs=%d | buffer_tokens=%d",
                total_examples, total_seqs, len(token_buf)
            )

    # Save remainder shard (partial shard of sequences, but each sequence is full length)
    if shard_input_ids:
        input_ids_arr = np.stack(shard_input_ids, axis=0)
        loss_mask_arr = np.stack(shard_loss_mask, axis=0) if pack_cfg.save_loss_mask else None
        _save_shard(out_dir, shard_idx, input_ids_arr, loss_mask_arr, logger)
        saved_shards += 1

    # Manifest (DATA artifact metadata)
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "tokenizer": pack_cfg.tokenizer_name_or_path,
        "tokenizer_class": tok.__class__.__name__,
        "vocab_size": int(vocab_size),
        "eos_token_id": int(tok.eos_token_id) if tok.eos_token_id is not None else None,
        "pad_token_id": int(tok.pad_token_id) if tok.pad_token_id is not None else None,
        "seq_len": int(pack_cfg.seq_len),
        "shard_num_seqs": int(pack_cfg.shard_num_seqs),
        "dtype": pack_cfg.dtype,
        "add_eos": bool(pack_cfg.add_eos),
        "save_loss_mask": bool(pack_cfg.save_loss_mask),
        "num_shards": int(saved_shards),
        "packed_seqs": int(total_seqs),
        "packed_tokens": int(total_seqs) * int(pack_cfg.seq_len),
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Saved manifest %s", str(manifest_path))
    logger.info("DONE: examples=%d | packed_seqs=%d | out_dir=%s",
                total_examples, total_seqs, str(out_dir))