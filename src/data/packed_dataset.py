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

    Supports:
        - Memory-mapped shard loading (mmap_mode="r")
        - Multi-worker DataLoader (shards split across workers, no duplicates)
        - DDP (shards split across ranks, no duplicates)
        - Shard-level shuffling with reproducible seeds
        - Epoch-aware shuffling (different order each epoch)

    Expected directory structure:
        data_dir/
            manifest.json
            input_ids_00000.npy
            loss_mask_00000.npy  (optional)
            input_ids_00001.npy
            ...

    Each yielded sample:
        {
            "input_ids": LongTensor(seq_len,),
            "loss_mask": LongTensor(seq_len,)  # optional
        }
    """

    def __init__(self, 
                 data_dir: str,
                 shuffle: bool = False,
                 seed: int = 42,
                 rank: int = 0,
                 world_size: int = 1):
        """
        Args:
            data_dir (str):
                Path to directory containing manifest.json and .npy shards.

            shuffle (bool):
                Whether to shuffle shard order each epoch.

            seed (int):
                Base random seed for shuffling. Combined with epoch for
                reproducible per-epoch ordering.

            rank (int):
                Current process rank for DDP. 0 for single-GPU.

            world_size (int):
                Total number of processes for DDP. 1 for single-GPU.
        """
        super().__init__()

        self.data_dir = Path(data_dir)
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        manifest_path = self.data_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {self.data_dir}")

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.seq_len = manifest["seq_len"]
        self.save_loss_mask = manifest["save_loss_mask"]
        self._packed_seq = manifest["packed_seqs"]
        

        self.input_shards = sorted(self.data_dir.glob("input_ids_*.npy"))
        if not self.input_shards:
            raise RuntimeError("No input shards files found.")

        if self.save_loss_mask:
            self.mask_shards = sorted(self.data_dir.glob("loss_mask_*.npy"))
            if len(self.mask_shards) != len(self.input_shards):
                raise RuntimeError(
                    f"Mismatch between input shards ({len(self.input_shards)}) "
                    f"and mask shards ({len(self.mask_shards)})."
                )
        else:
            self.mask_shards = None

    def set_epoch(self, epoch: int) -> None:
        """
        Set the current epoch for shuffle reproducibility.

        Call this at the start of each epoch in your training loop:
            dataset.set_epoch(epoch)

        This ensures each epoch sees a different shard order while
        remaining fully reproducible given the same seed.
        """
        self.epoch = epoch

    def _get_shard_indicies(self) -> list[int]:
        """
        Compute which shard indices this worker+rank should process.

        Shards are first divided across DDP ranks, then subdivided
        across DataLoader workers within each rank.
        """
        num_shards = len(self.input_shards)

        # Shuffle shard order (same order across all workers/ranks for correct splitting)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)
            indices = rng.permutation(num_shards).tolist()
        else:
            indices = list(range(num_shards))

        # Split across DDP ranks
        indices = indices[self.rank::self.world_size]

        # Split across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices = indices[worker_info.id::worker_info.num_workers]

        return indices
    
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
        for idx in self._get_shard_indices():
            shard_path = self.input_shards[idx]
            mask_path = self.mask_shards[idx] if self.mask_shards is not None else None
            yield from self._iter_shard(shard_path, mask_path)