from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Any

from datasets import load_dataset

@dataclass
class ReadConfig:
    dataset_name: str = "nvidia/OpenMathInstruct-2"
    split: str = "train"
    streaming: bool = True
    limit: Optional[int] = None
    skip: int = 0

def iter_openmathinstruct2(cfg: ReadConfig) -> Iterator[Dict[str, Any]]:
    """
    Stream OpenMathInstruct-2 examples. RAM-safe by default.

    - streaming=True: does not load dataset into memory
    - limit: stops after N examples (useful for laptop)
    - skip: skips first K examples (useful for resume/debug)
    """

    # returns only training data, it is not downloaded into memory because we use streaming
    data = load_dataset(cfg.dataset_name, split=cfg.split, streaming=cfg.streaming)

    if cfg.skip:
        data = data.skip(cfg.skip)

    n = 0
    for ex in data:
        # yield reacts like return, but return one number than pause the function and so on, instead giving all the numbers
        yield ex
        n += 1
        if cfg.limit is not None and n >= cfg.limit:
            break