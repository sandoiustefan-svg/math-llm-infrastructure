"""
Inspect OpenMathInstruct-2 dataset examples via streaming.

This script provides a simple CLI utility to:
- Load a specified dataset split (default: train)
- Optionally skip the first N examples
- Optionally limit the number of examples returned
- Print formatted JSON previews of the examples

The dataset is streamed using HuggingFace `datasets` to avoid
loading the full dataset into memory. This makes it suitable
for large-scale datasets and quick inspection on both local
machines and HPC environments.

Typical usage:

    python scripts/python/inspect_data.py --limit 3
    python scripts/python/inspect_data.py --skip 1000 --limit 5
    python scripts/python/inspect_data.py --split validation

This script is intended for debugging and understanding
the dataset structure before preprocessing and tokenization.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse
import json

from src.utils.logging_utils import setup_logger
from src.data.read_openmathinstruct2 import ReadConfig, iter_openmathinstruct2

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--skip", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logger("inspect_data")
    logger.info("Starting inspection (split=%s, skip=%d, limit=%d)", args.split, args.skip, args.limit)

    cfg = ReadConfig(split=args.split, limit=args.limit, skip=args.skip)

    for i, ex in enumerate(iter_openmathinstruct2(cfg), start=1):
        logger.info("------ Example %d ------", i)
        logger.info(json.dumps(ex, ensure_ascii=False, indent=2)[:4000])

if __name__ == "__main__":
    main()