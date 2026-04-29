#!/bin/bash
# Preprocess OpenMathInstruct-2 into packed .npy shards ready for training.
#
# Pipeline:
#   1. Stream raw data from HuggingFace → raw.jsonl
#   2. Format into chat messages        → formatted.jsonl
#   3. Tokenize with loss_mask          → tokens.jsonl
#   4. Pack into fixed-shape shards     → shards/
#
# Usage:
#   bash scripts/bash/preprocess.sh                        # full dataset, save all
#   bash scripts/bash/preprocess.sh --debug                # 5k examples, save all
#   bash scripts/bash/preprocess.sh --shards-only          # full dataset, shards only
#   bash scripts/bash/preprocess.sh --debug --shards-only  # debug, shards only
#
# --shards-only: intermediate JSONL files are written to /tmp and deleted after
#   each step. Only the final shards/ directory is kept on disk. Useful when
#   disk space is limited (full dataset JSONL files can be 50–100 GB).
#
# Each step is skipped if its output already exists (safe to re-run).

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
TOKENIZER="meta-llama/Llama-3.2-1B-Instruct"
SPLIT="train"
LIMIT=0          # 0 = no limit (full dataset)
SKIP=0
SEQ_LEN=2048
SHARD_NUM_SEQS=1024
OUT_DIR="data/processed/openmathinstruct2"
SHARDS_ONLY=false
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
    --shards-only)
      SHARDS_ONLY=true
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

# ── Decide where intermediate files live ─────────────────────────────────────
if [ "$SHARDS_ONLY" = true ]; then
  RAW_FILE=$(mktemp /tmp/omi2_raw_XXXXXX.jsonl)
  FORMATTED_FILE=$(mktemp /tmp/omi2_formatted_XXXXXX.jsonl)
  TOKENS_FILE=$(mktemp /tmp/omi2_tokens_XXXXXX.jsonl)
  # Clean up temp files on exit (success or failure)
  trap 'rm -f "$RAW_FILE" "$FORMATTED_FILE" "$TOKENS_FILE"' EXIT
else
  RAW_FILE="$OUT_DIR/raw.jsonl"
  FORMATTED_FILE="$OUT_DIR/formatted.jsonl"
  TOKENS_FILE="$OUT_DIR/tokens.jsonl"
fi

echo "============================================================"
echo " PREPROCESS CONFIG"
echo "  tokenizer      : $TOKENIZER"
echo "  split          : $SPLIT"
echo "  limit          : $LIMIT (0 = all)"
echo "  seq_len        : $SEQ_LEN"
echo "  shard_num_seqs : $SHARD_NUM_SEQS"
echo "  out_dir        : $OUT_DIR"
echo "  shards_only    : $SHARDS_ONLY"
echo "============================================================"

LIMIT_FLAG=""
if [ "$LIMIT" -gt 0 ]; then
  LIMIT_FLAG="--limit $LIMIT"
fi

# ── Step 1: Stream raw data ───────────────────────────────────────────────────
if [ "$SHARDS_ONLY" = false ] && [ -f "$RAW_FILE" ]; then
  echo "[1/4] Raw data found — skipping. ($RAW_FILE)"
else
  echo "[1/4] Streaming raw data from HuggingFace..."

  python scripts/python/inspect_data.py \
    --split  "$SPLIT" \
    --skip   "$SKIP" \
    --output "$RAW_FILE" \
    $LIMIT_FLAG

  echo "[1/4] Done → $RAW_FILE"
fi

# ── Step 2: Format ────────────────────────────────────────────────────────────
if [ "$SHARDS_ONLY" = false ] && [ -f "$FORMATTED_FILE" ]; then
  echo "[2/4] Formatted data found — skipping. ($FORMATTED_FILE)"
else
  echo "[2/4] Formatting..."

  python scripts/python/format_data.py \
    --input  "$RAW_FILE" \
    --output "$FORMATTED_FILE"

  echo "[2/4] Done → $FORMATTED_FILE"

  # In shards-only mode, raw file is no longer needed
  if [ "$SHARDS_ONLY" = true ]; then
    rm -f "$RAW_FILE"
  fi
fi

# ── Step 3: Tokenize ──────────────────────────────────────────────────────────
if [ "$SHARDS_ONLY" = false ] && [ -f "$TOKENS_FILE" ]; then
  echo "[3/4] Tokenized data found — skipping. ($TOKENS_FILE)"
else
  echo "[3/4] Tokenizing..."

  python scripts/python/tokenize_data.py \
    --input              "$FORMATTED_FILE" \
    --output             "$TOKENS_FILE" \
    --mode               tokens \
    --model-name-or-path "$TOKENIZER" \
    --max-length         "$SEQ_LEN"

  echo "[3/4] Done → $TOKENS_FILE"

  # In shards-only mode, formatted file is no longer needed
  if [ "$SHARDS_ONLY" = true ]; then
    rm -f "$FORMATTED_FILE"
  fi
fi

# ── Step 4: Pack into shards ──────────────────────────────────────────────────
if [ -f "$SHARDS_DIR/manifest.json" ]; then
  echo "[4/4] Shards found — skipping. ($SHARDS_DIR)"
else
  echo "[4/4] Packing into shards..."

  python scripts/python/pack_data.py \
    --input          "$TOKENS_FILE" \
    --out-dir        "$SHARDS_DIR" \
    --seq-len        "$SEQ_LEN" \
    --shard-num-seqs "$SHARD_NUM_SEQS" \
    --tokenizer      "$TOKENIZER"

  echo "[4/4] Done → $SHARDS_DIR"
fi

echo "============================================================"
echo " Preprocessing complete."
echo "  shards: $SHARDS_DIR"
echo "  $(python -c "import json; m=json.load(open('$SHARDS_DIR/manifest.json')); print(f\"{m['num_shards']} shards  |  {m['total_rows']} rows  |  {m['split_examples']} split examples\")")"
echo "============================================================"
