"""
End-to-end preprocessing: HuggingFace → format → tokenize → .npy shards.

Each example is fetched from HF, run through the full pipeline, and written
directly to shards. No intermediate JSONL files are created.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.data.read_openmathinstruct2 import ReadConfig, iter_openmathinstruct2
from src.data.format_openmathinstruct2 import FormatConfig, format_openmathinstruct2_example
from src.data.tokenizer import tokenize_example
from src.data.pack_shards import PackShardsConfig, stream_to_shards
from src.utils.logging_utils import setup_logger

_LLAMA3_PAD_TOKEN = "<|finetune_right_pad_id|>"
_LOG_EVERY = 10_000


def _resolve_pad_token_id(tok: PreTrainedTokenizerBase, logger) -> int:
    llama_pad = tok.convert_tokens_to_ids(_LLAMA3_PAD_TOKEN)
    if llama_pad != tok.unk_token_id:
        logger.info("pad_token_id: %d (%s)", llama_pad, _LLAMA3_PAD_TOKEN)
        return llama_pad

    if tok.pad_token_id is not None and tok.pad_token_id != tok.eos_token_id:
        logger.info("pad_token_id: %d (from tokenizer)", tok.pad_token_id)
        return tok.pad_token_id

    logger.warning(
        "pad_token_id falls back to eos_token_id (%d). "
        "Pad positions and real EOS tokens will share the same ID.",
        tok.eos_token_id,
    )
    return tok.eos_token_id


def _pipeline(
    hf_records: Iterator[Dict[str, Any]],
    format_cfg: FormatConfig,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    logger,
) -> Iterator[Dict[str, Any]]:
    processed = 0
    skipped = 0

    for ex in hf_records:
        try:
            formatted = format_openmathinstruct2_example(ex, format_cfg)
            tokenized = tokenize_example(formatted, tokenizer, max_length)
            yield tokenized
            processed += 1

            if processed % _LOG_EVERY == 0:
                logger.info("Processed %d examples  (skipped %d)", processed, skipped)

        except Exception as exc:
            skipped += 1
            if skipped <= 10:
                logger.warning("Skipping example: %s", exc)

    logger.info("Pipeline done. processed=%d  skipped=%d", processed, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-pass preprocessing: HF dataset → format → tokenize → .npy shards"
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N examples (0 = full dataset)")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first K examples")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--max-length", type=int, default=4096,
                        help="Truncate sequences longer than this many tokens")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Fixed sequence length of each shard row (pad/chunk to this)")
    parser.add_argument("--shard-num-seqs", type=int, default=1024,
                        help="Number of sequences per shard file")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory where shards and manifest.json are written")
    args = parser.parse_args()

    logger = setup_logger("preprocess_to_shards")

    logger.info("Loading tokenizer: %s", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    pad_token_id = _resolve_pad_token_id(tok, logger)

    read_cfg = ReadConfig(
        split=args.split,
        limit=args.limit if args.limit > 0 else None,
        skip=args.skip,
        streaming=True,
    )

    pack_cfg = PackShardsConfig(
        seq_len=args.seq_len,
        shard_num_seqs=args.shard_num_seqs,
        pad_token_id=pad_token_id,
        out_dir=args.out_dir,
    )

    logger.info(
        "Streaming nvidia/OpenMathInstruct-2 [%s]  limit=%s  skip=%d",
        args.split,
        args.limit if args.limit > 0 else "all",
        args.skip,
    )
    logger.info(
        "seq_len=%d  shard_num_seqs=%d  out_dir=%s",
        args.seq_len, args.shard_num_seqs, args.out_dir,
    )

    hf_records = iter_openmathinstruct2(read_cfg)
    records = _pipeline(hf_records, FormatConfig(), tok, args.max_length, logger)
    manifest = stream_to_shards(records, pack_cfg, tokenizer_name=args.tokenizer)

    logger.info(
        "Done. shards=%d  total_rows=%d  split_examples=%d  padded_tokens=%d",
        manifest["num_shards"],
        manifest["total_rows"],
        manifest["split_examples"],
        manifest["padded_tokens"],
    )


if __name__ == "__main__":
    main()
