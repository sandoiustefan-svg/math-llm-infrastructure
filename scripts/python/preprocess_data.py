"""
Preprocess OpenMathInstruct-2 into packed LLaMA token shards.

Pipeline:
  1) Stream dataset examples (RAM-safe)
  2) Format as Strategy 3:
       Problem + Solution + Final Answer
  3) Tokenize with LLaMA tokenizer
  4) Pack into fixed-length sequences
  5) Save .npy shards (and optional loss masks)

Example:
  python scripts/python/preprocess_openmathinstruct2.py \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --seq-len 2048 \
    --limit 50000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import argparse

from src.data.read_openmathinstruct2 import ReadConfig, iter_openmathinstruct2
from src.data.format_openmathinstruct2 import FormatConfig, format_openmathinstruct2_exmaple
from src.data.tokenize_pack import PackConfig, tokenzie_pack_and_save


def main():
    ap = argparse.ArgumentParser(description="Preprocess OpenMathInstruct-2 into packed token shards.")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--tokenizer", required=True, help="HF tokenizer name or local path (LLaMA tokenizer).")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--shard-num-seqs", type=int, default=1024)
    ap.add_argument("--no-loss-mask", action="store_true")
    args = ap.parse_args()

    read_cfg = ReadConfig(
        split=args.split,
        streaming=True,
        limit=None if args.limit == 0 else args.limit,
        skip=args.skip,
    )

    fmt_cfg = FormatConfig(include_final_answer=True, add_eos=True)

    pack_cfg = PackConfig(
        tokenizer_name_or_path=args.tokenizer,
        seq_len=args.seq_len,
        out_dir=args.out_dir,
        shard_num_seqs=args.shard_num_seqs,
        save_loss_mask=(not args.no_loss_mask),
        add_eos=True,
    )

    # Generator of formatted examples
    def formatted_iter():
        for ex in iter_openmathinstruct2(read_cfg):
            yield format_openmathinstruct2_exmaple(ex, fmt_cfg)

    tokenzie_pack_and_save(formatted_iter(), pack_cfg, logger_name="preprocess_openmath")


if __name__ == "__main__":
    main()