from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple, Optional

import numpy as np
from transformers import AutoTokenizer

from src.utils.logging_utils import setup_logger

@dataclass(frozen=True)
class PackConfig:
    tokenizer_name_or_path: str
    seq_len: int = 2048
    out_dir: str = "data/processed/openmathinstruct2"
    shard_num_seqs: int = 1024  #sequence per saved .npy
    dtype: str = "int32"
    add_eos: bool = True
    save_loss_mask: bool = True

def _ensure_out_dir(out_dir: str) -> Path:
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
    logger = setup_logger(logger_name)
    out_dir = _ensure_out_dir(pack_cfg.out_dir)

    logger.info("Loading tokenizer: %s", pack_cfg.tokenizer_name_or_path)
    tok = AutoTokenizer.from_pretrained(pack_cfg.tokenizer_name_or_path, use_fast=True)

    # LLaMA tokenizers have no pad token; we don't need padding here.
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Provide a LLaMA-compatible tokenizer.")

    seq_len = pack_cfg.seq_len
    dtype = np.int32 if pack_cfg.dtype == "int32" else np.int64

    # Buffers for streaming pack
    token_buf: List[int] = []
    loss_buf: List[int] = [] # 0 for prompt, 1 for completion

    shard_input_ids: List[np.ndarray] = []
    shard_loss_mask: List[np.ndarray] = []

    shard_idx = 0
    total_seqs = 0
    total_examples = 0

    for ex in examples:
        prompt_text = ex["prompt_text"]
        completion_text = ex["completion_text"]

        # we tokenize them separatedly
        prompts_ids = tok.encode(prompt_text, add_special_tokens=False)
        completion_ids = tok.encode(prompt_text, add_special_tokens=False)

        if pack_cfg.add_eos:
            completion_ids = completion_ids + [tok.eos_token_id]

        # append to stream buff
        token_buf.extend(prompts_ids)
        loss_buf.extend([0] * len(prompts_ids))

        token_buf.extend(completion_ids)
        loss_buf.extend([1] * len(completion_ids))

        total_examples += 1

        # here we pack into fixed-lentgh sequences
        while len(token_buf) > seq_len:
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
                shard_input_ids.clear()
                shard_loss_mask.clear()

        if total_examples % 10_000 == 0:
            logger.info("Processed examples=%d | packed_seqs=%d | buffer_tokens=%d",
                        total_examples, total_seqs, len(token_buf))

    # Save remainder
    if shard_input_ids:
        input_ids_arr = np.stack(shard_input_ids, axis=0)
        loss_mask_arr = np.stack(shard_loss_mask, axis=0) if pack_cfg.save_loss_mask else None
        _save_shard(out_dir, shard_idx, input_ids_arr, loss_mask_arr, logger)

    logger.info("DONE: examples=%d | packed_seqs=%d | out_dir=%s",
                total_examples, total_seqs, str(out_dir))





