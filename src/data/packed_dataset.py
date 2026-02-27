from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Dict, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset


class PackedShardDataset(IterableDataset):
    """
    Iterable dataset that streams pre-tokenized and packed LLM training data
    from disk.

    This dataset reads `.npy` shard files produced by the preprocessing
    pipeline. Each shard contains multiple fixed-length token sequences
    (and optionally corresponding loss masks).

    Data is memory-mapped using NumPy (`mmap_mode="r"`) to avoid loading
    entire shards into RAM, making it suitable for very large datasets.

    Expected directory structure:
        data_dir/
            manifest.json
            input_ids_00000.npy
            loss_mask_00000.npy (optional)
            input_ids_00001.npy
            ...

    Each yielded sample is a dictionary:
        {
            "input_ids": Tensor(seq_len),
            "loss_mask": Tensor(seq_len)  # optional
        }
    """

    def __init__(self, data_dir: str):
        """
        Initialize the dataset from a directory containing packed shards.

        Args:
            data_dir (str):
                Path to the directory containing:
                    - manifest.json
                    - input_ids_*.npy
                    - optional loss_mask_*.npy
        """
        super().__init__()

        self.data_dir = Path(data_dir)

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {self.data_dir}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.seq_len = manifest["seq_len"]
        self.save_loss_mask = manifest["save_loss_mask"]

        self.input_shards = sorted(self.data_dir.glob("input_ids_*.npy"))

        if not self.input_shards:
            raise RuntimeError("No input shards files found.")

        if self.save_loss_mask:
            self.mask_shards = sorted(self.data_dir.glob("loss_mask_*.npy"))
            if len(self.mask_shards) != len(self.input_shards):
                raise RuntimeError("Mismatch between input shards and mask shards.")
        else:
            self.mask_shards = None

    def _iter_shard(
        self,
        shard_path: Path,
        mask_path: Optional[Path],
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over a single shard file and yield individual samples.

        Args:
            shard_path (Path):
                Path to the input_ids shard (.npy file).

            mask_path (Optional[Path]):
                Path to the corresponding loss_mask shard, or None
                if loss masks are disabled.

        Yields:
            Dict[str, torch.Tensor]:
                A dictionary containing:
                    - "input_ids": LongTensor of shape (seq_len,)
                    - "loss_mask": LongTensor of shape (seq_len,) (optional)
        """
        input_arr = np.load(shard_path, mmap_mode="r")

        if mask_path is not None:
            mask_arr = np.load(mask_path, mmap_mode="r")
        else:
            mask_arr = None

        for i in range(input_arr.shape[0]):
            sample = {
                "input_ids": torch.from_numpy(input_arr[i]).long()
            }

            if mask_arr is not None:
                sample["loss_mask"] = torch.from_numpy(mask_arr[i]).long()

            yield sample

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over all shard files sequentially.

        This method streams shards one by one and yields individual
        token sequences, making it suitable for large-scale training
        without loading the full dataset into memory.

        Yields:
            Dict[str, torch.Tensor]:
                A single training sample of length `seq_len`.
        """
        for idx, shard_path in enumerate(self.input_shards):
            mask_path = self.mask_shards[idx] if self.mask_shards is not None else None
            yield from self._iter_shard(shard_path, mask_path)