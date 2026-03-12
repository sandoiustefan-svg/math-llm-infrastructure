#!/bin/bash
# Full pipeline: preprocess OpenMathInstruct-2 then train from scratch.
#
# Usage:
#   bash scripts/bash/run_pipeline.sh
#   bash scripts/bash/run_pipeline.sh --debug   # small run for testing
set -euo pipefail

# Defaults
TOKENIZER="mistralai/Mistral-7B-v0.1"
OUT_DIR="data/processed/openmathinstruct2"
SPLIT="train"
LIMIT=0
SKIP=0
SEQ_LEN=1024
SHARD_NUM_SEQS=256

BATCH_SIZE=8
LR=3e-4
STEPS=1000
NUM_WORKERS=2
N_LAYERS=8
HIDDEN_SIZE=512
N_HEADS=8

NO_LOSS_MASK=false
DEBUG=false

for arg in "$@"; do
  case $arg in
    --debug)
      DEBUG=true
      LIMIT=5000
      SHARD_NUM_SEQS=64
      OUT_DIR="data/processed/openmathinstruct2_debug"
      STEPS=100
      shift
      ;;
    --no-loss-mask)
      NO_LOSS_MASK=true
      shift
      ;;
    *)
      echo "Unknown argument: $arg"
      exit 1
      ;;
  esac
done

NO_LOSS_MASK_FLAG=""
if [ "$NO_LOSS_MASK" = true ]; then
  NO_LOSS_MASK_FLAG="--no-loss-mask"
fi

echo "============================================================"
echo "PIPELINE CONFIG"
echo "  tokenizer      : $TOKENIZER"
echo "  out_dir        : $OUT_DIR"
echo "  limit          : $LIMIT (0 = no limit)"
echo "  seq_len        : $SEQ_LEN"
echo "  shard_num_seqs : $SHARD_NUM_SEQS"
echo "  steps          : $STEPS"
echo "  debug          : $DEBUG"
echo "============================================================"

# Step 1: Preprocess (skip if already done)
if [ -f "$OUT_DIR/manifest.json" ]; then
  echo "[1/2] Preprocessed data found — skipping."
else
  echo "[1/2] Preprocessing..."
  python scripts/python/preprocess_data.py \
    --tokenizer "$TOKENIZER" \
    --split "$SPLIT" \
    --limit "$LIMIT" \
    --skip "$SKIP" \
    --seq-len "$SEQ_LEN" \
    --shard-num-seqs "$SHARD_NUM_SEQS" \
    --out-dir "$OUT_DIR" \
    $NO_LOSS_MASK_FLAG
  echo "[1/2] Preprocessing complete."
fi

# Step 2: Train
echo "[2/2] Starting training..."
python scripts/python/train.py \
  --data-dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --n-layers "$N_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --n-heads "$N_HEADS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --steps "$STEPS" \
  --num-workers "$NUM_WORKERS"

echo "[2/2] Training complete."
echo "============================================================"