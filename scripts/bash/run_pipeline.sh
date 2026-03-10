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
LIMIT=0                  # 0 = no limit
SKIP=0
SEQ_LEN=1024
SHARD_NUM_SEQS=256

# Training
BATCH_SIZE=8
LR=3e-4
STEPS=1000
NUM_WORKERS=2
N_LAYERS=8
HIDDEN_SIZE=512
N_HEADS=8

# Flags
NO_LOSS_MASK=false
DEBUG=false

# Argument parsing
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
      echo "Usage: bash run_pipeline.sh [--debug] [--no-loss-mask]"
      exit 1
      ;;
  esac
done

# Derived flags
NO_LOSS_MASK_FLAG=""
if [ "$NO_LOSS_MASK" = true ]; then
  NO_LOSS_MASK_FLAG="--no-loss-mask"
fi

# Print config
echo "============================================================"
echo "PIPELINE CONFIG"
echo "  tokenizer      : $TOKENIZER"
echo "  out_dir        : $OUT_DIR"
echo "  split          : $SPLIT"
echo "  limit          : $LIMIT (0 = no limit)"
echo "  seq_len        : $SEQ_LEN"
echo "  shard_num_seqs : $SHARD_NUM_SEQS"
echo "  batch_size     : $BATCH_SIZE"
echo "  lr             : $LR"
echo "  steps          : $STEPS"
echo "  n_layers       : $N_LAYERS"
echo "  hidden_size    : $HIDDEN_SIZE"
echo "  n_heads        : $N_HEADS"
echo "  debug          : $DEBUG"
echo "============================================================"

# Step 1: Preprocess
echo ""
echo "[1/2] Preprocessing OpenMathInstruct-2..."
echo ""

python scripts/python/preprocess_data.py \
  --tokenizer "$TOKENIZER" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --skip "$SKIP" \
  --seq-len "$SEQ_LEN" \
  --shard-num-seqs "$SHARD_NUM_SEQS" \
  --out-dir "$OUT_DIR" \
  $NO_LOSS_MASK_FLAG

echo ""
echo "[1/2] Preprocessing complete."
echo ""

# Step 2: Train
echo "[2/2] Starting training..."
echo ""

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

echo ""
echo "[2/2] Training complete."
echo "============================================================"