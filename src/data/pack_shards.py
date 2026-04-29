from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

_NUMPY_DTYPES = {"int32": np.int32, "int64": np.int64}
_MASK_DTYPE = np.int8


@dataclass(frozen=True)
class PackShardsConfig:
    seq_len: int
    shard_num_seqs: int
    pad_token_id: int
    out_dir: Path
    dtype: str = "int32"


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _chunk_example(
    input_ids: List[int],
    attention_mask: List[int],
    loss_mask: List[int],
    seq_len: int,
    pad_id: int,
) -> List[Tuple[List[int], List[int], List[int], int]]:
    """
    Split one example into fixed-length rows.

    Returns:
        (input_ids_row, attention_mask_row, loss_mask_row, pad_len)

    attention_mask:
        1 = real token, including EOS
        0 = artificial padding only

    loss_mask:
        1 = assistant/completion token
        0 = prompt token or padding
    """
    if not (len(input_ids) == len(attention_mask) == len(loss_mask)):
        raise ValueError(
            "input_ids, attention_mask, and loss_mask must have the same length "
            f"but got {len(input_ids)}, {len(attention_mask)}, {len(loss_mask)}"
        )

    n = len(input_ids)
    num_chunks = max(1, math.ceil(n / seq_len))

    rows = []

    for i in range(num_chunks):
        start = i * seq_len

        chunk_ids = input_ids[start : start + seq_len]
        chunk_attention = attention_mask[start : start + seq_len]
        chunk_loss = loss_mask[start : start + seq_len]

        pad_len = seq_len - len(chunk_ids)

        rows.append(
            (
                chunk_ids + [pad_id] * pad_len,
                chunk_attention + [0] * pad_len,
                chunk_loss + [0] * pad_len,
                pad_len,
            )
        )

    return rows


def _flush_shard(
    id_rows: List[List[int]],
    attention_rows: List[List[int]],
    loss_rows: List[List[int]],
    shard_idx: int,
    cfg: PackShardsConfig,
) -> None:
    out = Path(cfg.out_dir)

    np.save(
        out / f"input_ids_{shard_idx:05d}.npy",
        np.array(id_rows, dtype=_NUMPY_DTYPES[cfg.dtype]),
    )

    np.save(
        out / f"attention_mask_{shard_idx:05d}.npy",
        np.array(attention_rows, dtype=_MASK_DTYPE),
    )

    np.save(
        out / f"loss_mask_{shard_idx:05d}.npy",
        np.array(loss_rows, dtype=_MASK_DTYPE),
    )


def pack_jsonl_to_shards(
    input_jsonl: Path,
    cfg: PackShardsConfig,
    tokenizer_name: str = "",
) -> dict:
    """
    Stream tokenized JSONL into fixed-shape .npy shards.

    Each JSONL record must contain:
        input_ids
        attention_mask
        loss_mask

    Output:
        input_ids_00000.npy
        attention_mask_00000.npy
        loss_mask_00000.npy
        manifest.json
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    id_buf: List[List[int]] = []
    attention_buf: List[List[int]] = []
    loss_buf: List[List[int]] = []

    shard_idx = 0
    total_rows = 0
    padded_tokens = 0
    split_examples = 0

    for record in _iter_jsonl(input_jsonl):
        attention_mask = record.get("attention_mask")

        if attention_mask is None:
            attention_mask = [1] * len(record["input_ids"])

        rows = _chunk_example(
            input_ids=record["input_ids"],
            attention_mask=attention_mask,
            loss_mask=record["loss_mask"],
            seq_len=cfg.seq_len,
            pad_id=cfg.pad_token_id,
        )

        if len(rows) > 1:
            split_examples += 1

        for ids_row, attention_row, loss_row, pad_len in rows:
            padded_tokens += pad_len

            id_buf.append(ids_row)
            attention_buf.append(attention_row)
            loss_buf.append(loss_row)

            if len(id_buf) == cfg.shard_num_seqs:
                _flush_shard(
                    id_rows=id_buf,
                    attention_rows=attention_buf,
                    loss_rows=loss_buf,
                    shard_idx=shard_idx,
                    cfg=cfg,
                )

                total_rows += cfg.shard_num_seqs
                shard_idx += 1

                id_buf = []
                attention_buf = []
                loss_buf = []

    final_shard_rows = cfg.shard_num_seqs

    if id_buf:
        _flush_shard(
            id_rows=id_buf,
            attention_rows=attention_buf,
            loss_rows=loss_buf,
            shard_idx=shard_idx,
            cfg=cfg,
        )

        final_shard_rows = len(id_buf)
        total_rows += final_shard_rows
        shard_idx += 1

    manifest = {
        "seq_len": cfg.seq_len,
        "shard_num_seqs": cfg.shard_num_seqs,
        "dtype": cfg.dtype,
        "num_shards": shard_idx,
        "total_rows": total_rows,
        "final_shard_rows": final_shard_rows,
        "padded_tokens": padded_tokens,
        "split_examples": split_examples,
        "pad_token_id": cfg.pad_token_id,
        "has_attention_mask": True,
        "has_loss_mask": True,
        "tokenizer": tokenizer_name,
    }

    with (Path(cfg.out_dir) / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest