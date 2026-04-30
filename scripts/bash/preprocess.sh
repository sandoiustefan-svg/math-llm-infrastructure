#!/bin/bash
# End-to-end preprocessing: HuggingFace → format → tokenize → .npy shards.
#
# Each example is fetched from HF, run through the full pipeline in one pass,
# and written directly to shards. No intermediate JSONL files are created.
#
# Usage:
#   bash scripts/bash/preprocess.sh                 # full dataset
#   bash scripts/bash/preprocess.sh --debug         # 30 examples, seq_len=512
#
# Skips preprocessing if shards/manifest.json already exists (safe to re-run).

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
TOKENIZER="meta-llama/Meta-Llama-3.1-8B-Instruct"
SPLIT="train"
LIMIT=0          # 0 = no limit (full dataset)
SKIP=0
SEQ_LEN=2048
SHARD_NUM_SEQS=1024
OUT_DIR="data/processed/openmathinstruct2"
# ─────────────────────────────────────────────────────────────────────────────

for arg in "$@"; do
  case $arg in
    --debug)
      LIMIT=30
      SEQ_LEN=512
      SHARD_NUM_SEQS=64
      OUT_DIR="data/processed/openmathinstruct2_debug"
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

SHARDS_DIR="$OUT_DIR/shards"
mkdir -p "$OUT_DIR"

echo "============================================================"
echo " PREPROCESS CONFIG"
echo "  tokenizer      : $TOKENIZER"
echo "  split          : $SPLIT"
echo "  limit          : $LIMIT (0 = all)"
echo "  seq_len        : $SEQ_LEN"
echo "  shard_num_seqs : $SHARD_NUM_SEQS"
echo "  out_dir        : $OUT_DIR"
echo "============================================================"

LIMIT_FLAG=""
if [ "$LIMIT" -gt 0 ]; then
  LIMIT_FLAG="--limit $LIMIT"
fi

if [ -f "$SHARDS_DIR/manifest.json" ]; then
  echo "Shards already exist — skipping. ($SHARDS_DIR)"
else
  echo "Preprocessing..."

  python scripts/python/preprocess_to_shards.py \
    --split          "$SPLIT" \
    --skip           "$SKIP" \
    --tokenizer      "$TOKENIZER" \
    --max-length     "$SEQ_LEN" \
    --seq-len        "$SEQ_LEN" \
    --shard-num-seqs "$SHARD_NUM_SEQS" \
    --out-dir        "$SHARDS_DIR" \
    $LIMIT_FLAG

  echo "Done → $SHARDS_DIR"
fi

echo "============================================================"
echo " Preprocessing complete."
echo "  shards: $SHARDS_DIR"
echo "  $(python -c "import json; m=json.load(open('$SHARDS_DIR/manifest.json')); print(f\"{m['num_shards']} shards  |  {m['total_rows']} rows  |  {m['split_examples']} split examples\")")"
echo "============================================================"
