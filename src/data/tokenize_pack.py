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
    seq_len: int = 2048     # we have 2048 tokens per sequence 
    out_dir: str = "data/processed/openmathinstruct2"
    shard_num_seqs: int = 1024  # we have 1024 sequences per shard(.npy) => (1024, 2048)
    dtype: str = "int32"
    add_eos: bool = True
    save_loss_mask: bool = True

def _ensure_out_dir(out_dir: str) -> Path:
    """
    Create the output directory if it does not already exist.

    Args:
        out_dir (str):
            Path to the directory where processed data will be saved.

    Returns:
        Path:
            A pathlib.Path object pointing to the created/existing directory.
    """
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
    """
    Save a shard of packed sequences to disk.

    Each shard contains multiple fixed-length token sequences
    (and optionally corresponding loss masks) stored as .npy files.

    Args:
        out_dir (Path):
            Directory where shard files will be written.

        shard_idx (int):
            Index of the shard, used for naming the file.

        input_ids_arr (np.ndarray):
            Array of token IDs with shape (num_sequences, seq_len).

        loss_mask_arr (Optional[np.ndarray]):
            Array of loss masks with shape (num_sequences, seq_len),
            or None if loss masking is disabled.

        logger:
            Logger instance used to record saving information.
    """
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

    This function:
        1. Loads a HuggingFace tokenizer.
        2. Streams over input examples containing prompt and completion text.
        3. Tokenizes prompt and completion separately.
        4. Concatenates tokens into a continuous stream buffer.
        5. Splits the stream into fixed-length sequences of size `seq_len`.
        6. Groups sequences into shards of size `shard_num_seqs`.
        7. Saves shards as NumPy arrays to disk.

    Optionally, a loss mask is generated where:
        - 0 indicates prompt tokens (excluded from loss computation),
        - 1 indicates completion tokens (included in loss computation).

    Output:
        Saves multiple .npy shard files to `pack_cfg.out_dir`.
        Each shard contains:
            - input_ids_{idx}.npy  → shape (shard_num_seqs, seq_len)
            - loss_mask_{idx}.npy  → same shape (if enabled)

    Notes:
        - Remaining tokens that do not fill a complete sequence
          are discarded.
        - No padding is applied; sequences are strictly packed.
        - Designed for LLaMA-style causal language model training.
    """
    logger = setup_logger(logger_name)
    out_dir = _ensure_out_dir(pack_cfg.out_dir)

    # use the tokenizer from the argument
    logger.info("Loading tokenizer: %s", pack_cfg.tokenizer_name_or_path)
    tok = AutoTokenizer.from_pretrained(pack_cfg.tokenizer_name_or_path, use_fast=True)

    # LLaMA tokenizers have no pad token; we don't need padding here.
    if tok.eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id. Provide a LLaMA-compatible tokenizer.")

    seq_len = pack_cfg.seq_len
    dtype = np.int32 if pack_cfg.dtype == "int32" else np.int64

    # Buffers for streaming pack
    token_buf: List[int] = []   # this buffer represents a filled sequence with tokens (2048, )
    loss_buf: List[int] = []    # this buffer represented a filled sequence with 1 or 0 (2048, )

    shard_input_ids: List[np.ndarray] = []  # this shard has 1024 sequences of tokens (1024, 2048)
    shard_loss_mask: List[np.ndarray] = []  # this shard has 1024 sequences of 1s or 0s (1024, 2048)

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
