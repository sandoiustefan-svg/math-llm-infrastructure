from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from transformers import AutoTokenizer

from src.data.pack_shards import PackShardsConfig, pack_jsonl_to_shards
from src.utils.logging_utils import setup_logger

_LLAMA3_PAD_TOKEN = "<|finetune_right_pad_id|>"


def _resolve_pad_token_id(tok, logger) -> int:
    """
    Pick a pad token that is distinct from EOS so pad positions and real
    end-of-sequence tokens can be told apart by ID.

    Priority:
      1. Llama 3's dedicated pad token <|finetune_right_pad_id|> (128004)
      2. tokenizer.pad_token_id if set and != eos_token_id
      3. eos_token_id as last resort (with a warning)
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack tokenized JSONL into fixed-shape .npy shards."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Tokenized JSONL produced by tokenize_data.py --mode tokens.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where shards and manifest.json are written.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--shard-num-seqs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer name, used only to look up pad_token_id.",
    )
    parser.add_argument(
        "--pad-token-id",
        type=int,
        default=None,
        help="Override pad token ID (skips loading tokenizer).",
    )
    args = parser.parse_args()

    logger = setup_logger("pack_data")

    if args.pad_token_id is not None:
        pad_token_id = args.pad_token_id
        logger.info("Using provided pad_token_id: %d", pad_token_id)
    else:
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        pad_token_id = _resolve_pad_token_id(tok, logger)

    cfg = PackShardsConfig(
        seq_len=args.seq_len,
        shard_num_seqs=args.shard_num_seqs,
        pad_token_id=pad_token_id,
        out_dir=args.out_dir,
    )

    logger.info(
        "Packing %s → %s (seq_len=%d, shard_num_seqs=%d)",
        args.input, args.out_dir, args.seq_len, args.shard_num_seqs,
    )

    manifest = pack_jsonl_to_shards(args.input, cfg, tokenizer_name=args.tokenizer)

    logger.info(
        "Done. shards=%d  total_rows=%d  split_examples=%d  padded_tokens=%d",
        manifest["num_shards"],
        manifest["total_rows"],
        manifest["split_examples"],
        manifest["padded_tokens"],
    )


if __name__ == "__main__":
    main()
