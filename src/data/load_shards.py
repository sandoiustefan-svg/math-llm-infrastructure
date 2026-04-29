from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyShardDataset(Dataset):
    """
    PyTorch Dataset for loading .npy shards created by pack_shards.py.

    Expected files:
        manifest.json
        input_ids_00000.npy
        attention_mask_00000.npy
        loss_mask_00000.npy
        ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        shard_indices: Optional[list[int]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)

        self.manifest_path = self.data_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {self.data_dir}")

        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.input_files = sorted(self.data_dir.glob("input_ids_*.npy"))
        self.attention_files = sorted(self.data_dir.glob("attention_mask_*.npy"))
        self.loss_files = sorted(self.data_dir.glob("loss_mask_*.npy"))

        if not self.input_files:
            raise FileNotFoundError(f"No input_ids_*.npy files found in {self.data_dir}")

        if len(self.input_files) != len(self.attention_files):
            raise ValueError(
                f"Found {len(self.input_files)} input shards but "
                f"{len(self.attention_files)} attention-mask shards."
            )

        if len(self.input_files) != len(self.loss_files):
            raise ValueError(
                f"Found {len(self.input_files)} input shards but "
                f"{len(self.loss_files)} loss-mask shards."
            )

        if shard_indices is not None:
            self.input_files = [self.input_files[i] for i in shard_indices]
            self.attention_files = [self.attention_files[i] for i in shard_indices]
            self.loss_files = [self.loss_files[i] for i in shard_indices]

        self.shard_sizes: list[int] = []

        for input_file in self.input_files:
            arr = np.load(input_file, mmap_mode="r")
            self.shard_sizes.append(arr.shape[0])

        self.cumulative_sizes = np.cumsum(self.shard_sizes).tolist()

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index = len(self) + index

        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")

        shard_idx = int(np.searchsorted(self.cumulative_sizes, index, side="right"))
        previous_size = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        row_idx = index - previous_size

        return shard_idx, row_idx

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_idx, row_idx = self._locate(index)

        input_ids = np.load(self.input_files[shard_idx], mmap_mode="r")[row_idx]
        attention_mask = np.load(self.attention_files[shard_idx], mmap_mode="r")[row_idx]
        loss_mask = np.load(self.loss_files[shard_idx], mmap_mode="r")[row_idx]

        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.as_tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.as_tensor(loss_mask, dtype=torch.long),
        }


def load_manifest(data_dir: str | Path) -> dict:
    manifest_path = Path(data_dir) / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {data_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)