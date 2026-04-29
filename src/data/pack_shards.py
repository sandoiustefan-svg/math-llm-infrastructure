from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np

_NUMPY_DTYPES = {"int32": np.int32, "int64": np.int64}
_MASK_DTYPE = np.int8  # loss_mask is binary 0/1, no need for int32


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
    loss_mask: List[int],
    seq_len: int,
    pad_id: int,
) -> List[Tuple[List[int], List[int]]]:
    """Split one example into fixed-length (seq_len,) rows, padding the last.

    Returns list of (input_ids_row, loss_mask_row). Pad positions get mask=0.
    """
    n = len(input_ids)
    num_chunks = max(1, math.ceil(n / seq_len))
    rows = []
    for i in range(num_chunks):
        start = i * seq_len
        chunk_ids = input_ids[start : start + seq_len]
        chunk_mask = loss_mask[start : start + seq_len]
        pad_len = seq_len - len(chunk_ids)
        rows.append((
            chunk_ids + [pad_id] * pad_len,
            chunk_mask + [0] * pad_len,
        ))
    return rows


def _flush_shard(
    id_rows: List[List[int]],
    mask_rows: List[List[int]],
    shard_idx: int,
    cfg: PackShardsConfig,
) -> None:
    out = Path(cfg.out_dir)
    np.save(out / f"input_ids_{shard_idx:05d}.npy", np.array(id_rows, dtype=_NUMPY_DTYPES[cfg.dtype]))
    np.save(out / f"loss_mask_{shard_idx:05d}.npy", np.array(mask_rows, dtype=_MASK_DTYPE))


def pack_jsonl_to_shards(
    input_jsonl: Path,
    cfg: PackShardsConfig,
    tokenizer_name: str = "",
) -> dict:
    """
    Stream tokenized JSONL, pack into fixed-shape .npy shards, write manifest.

    Each record must have 'input_ids' and 'loss_mask' lists of equal length.
    loss_mask is 1 for tokens that contribute to loss, 0 for prompt/pad tokens.
    Examples shorter than seq_len are padded; longer ones are hard-chunked
    into multiple rows. Returns the manifest dict.
    """
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    id_buf: List[List[int]] = []
    mask_buf: List[List[int]] = []

    shard_idx = 0
    total_rows = 0
    padded_tokens = 0
    split_examples = 0

    for record in _iter_jsonl(input_jsonl):
        rows = _chunk_example(
            record["input_ids"], record["loss_mask"], cfg.seq_len, cfg.pad_token_id
        )

        if len(rows) > 1:
            split_examples += 1

        for ids_row, mask_row in rows:
            padded_tokens += ids_row.count(cfg.pad_token_id)
            id_buf.append(ids_row)
            mask_buf.append(mask_row)

            if len(id_buf) == cfg.shard_num_seqs:
                _flush_shard(id_buf, mask_buf, shard_idx, cfg)
                total_rows += cfg.shard_num_seqs
                shard_idx += 1
                id_buf = []
                mask_buf = []

    final_shard_rows = cfg.shard_num_seqs
    if id_buf:
        _flush_shard(id_buf, mask_buf, shard_idx, cfg)
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
        "tokenizer": tokenizer_name,
    }

    with (Path(cfg.out_dir) / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest
