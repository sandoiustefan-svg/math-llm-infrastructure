"""
Download / inspect OpenMathInstruct-2 and save raw streamed data.
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
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None,
                    help="Output path. Defaults to data/raw/inspect/raw/openmathinstruct2_{split}.jsonl")

    args = ap.parse_args()

    logger = setup_logger("inspect_data")

    cfg = ReadConfig(
        split=args.split,
        limit=args.limit,
        skip=args.skip
    )

    if args.output is not None:
        output_file = args.output
    else:
        raw_dir = PROJECT_ROOT / "data" / "raw" / "inspect" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_file = raw_dir / f"openmathinstruct2_{args.split}.jsonl"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving data to %s", output_file)

    count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for ex in iter_openmathinstruct2(cfg):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1

            if count % 100 == 0:
                logger.info("Saved %d examples", count)

    logger.info("Finished. Total saved: %d", count)


if __name__ == "__main__":
    main()