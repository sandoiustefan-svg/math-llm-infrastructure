from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Dict

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.read_openmathinstruct2 import ReadConfig, iter_openmathinstruct2
from src.data.format_openmathinstruct2 import FormatConfig, format_openmathinstruct2_exmaple


class OnlinePackedDataset(IterableDataset):
    """
    Streams examples from HuggingFace, tokenizes, packs into
    fixed-length sequences, and yields training samples directly.
    No intermediate files are written to disk.

    Packing: multiple examples are concatenated into a single
    sequence of length `seq_len`. When an example doesn't fit
    the current buffer, the buffer is flushed as a sample and
    a new buffer starts.

    Each yielded sample:
        {
            "input_ids": LongTensor(seq_len,),
            "loss_mask": LongTensor(seq_len,)  # optional
        }
    """

    def __init__(
        self,
        tokenizer_name: str,
        seq_len: int = 1024,
        split: str = "train",
        limit: int | None = None,
        skip: int = 0,
        add_eos: bool = True,
        save_loss_mask: bool = True,
    ):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.seq_len = seq_len
        self.split = split
        self.limit = limit
        self.skip = skip
        self.add_eos = add_eos
        self.save_loss_mask = save_loss_mask

    def _pack_and_yield(
        self,
        token_iter: Iterator,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Pack tokenized examples into fixed-length sequences.
        Yields one sample each time the buffer reaches seq_len.
        """
        id_buf: list[int] = []
        mask_buf: list[int] = []

        for input_ids, loss_mask in token_iter:
            id_buf.extend(input_ids)
            mask_buf.extend(loss_mask)

            while len(id_buf) >= self.seq_len:
                sample = {
                    "input_ids": torch.tensor(
                        id_buf[: self.seq_len], dtype=torch.long
                    ),
                }
                if self.save_loss_mask:
                    sample["loss_mask"] = torch.tensor(
                        mask_buf[: self.seq_len], dtype=torch.long
                    )

                id_buf = id_buf[self.seq_len :]
                mask_buf = mask_buf[self.seq_len :]

                yield sample

    def _tokenize_examples(self, tokenizer):
        """
        Stream examples from HF, format, tokenize, and yield
        (input_ids, loss_mask) pairs.
        """
        read_cfg = ReadConfig(
            split=self.split,
            streaming=True,
            limit=self.limit,
            skip=self.skip,
        )
        fmt_cfg = FormatConfig(include_final_answer=True)

        for ex in iter_openmathinstruct2(read_cfg):
            formatted = format_openmathinstruct2_exmaple(ex, fmt_cfg)

            prompt_ids = tokenizer.encode(
                formatted["prompt_text"], add_special_tokens=False
            )
            completion_ids = tokenizer.encode(
                formatted["completion_text"], add_special_tokens=False
            )

            if self.add_eos and tokenizer.eos_token_id is not None:
                completion_ids.append(tokenizer.eos_token_id)

            all_ids = prompt_ids + completion_ids

            # Loss mask: 0 for prompt, 1 for completion
            loss_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

            yield all_ids, loss_mask

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        token_iter = self._tokenize_examples(tokenizer)
        yield from self._pack_and_yield(token_iter)