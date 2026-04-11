#!/bin/bash
# Full pipeline: preprocess OpenMathInstruct-2 then train from scratch.
#
# Usage:
#   bash scripts/bash/run_pipeline.sh                    # full run, disk mode
#   bash scripts/bash/run_pipeline.sh --debug            # small debug run
#   bash scripts/bash/run_pipeline.sh --online           # stream from HF, no disk
#   bash scripts/bash/run_pipeline.sh --debug --online   # debug + online

set -euo pipefail

# Defaults (full run — ~500M params)
TOKENIZER="meta-llama/Llama-3.2-1B"
OUT_DIR="data/processed/openmathinstruct2"
OUTPUT_DIR="outputs/training"
SPLIT="train"
LIMIT=0
SKIP=0
SEQ_LEN=1024
SHARD_NUM_SEQS=256

BATCH_SIZE=4
LR=3e-4
STEPS=150000
NUM_WORKERS=0
N_LAYERS=16
HIDDEN_SIZE=1536
N_HEADS=12
SAVE_EVERY=500
LOG_EVERY=50

NO_LOSS_MASK=false
DEBUG=false
FP16=false
ONLINE=false

for arg in "$@"; do
  case $arg in
    --debug)
      DEBUG=true
      LIMIT=5000
      SHARD_NUM_SEQS=64
      OUT_DIR="data/processed/openmathinstruct2_debug"
      OUTPUT_DIR="outputs/training_debug"
      STEPS=100
      SAVE_EVERY=50
      LOG_EVERY=10
      N_LAYERS=8
      HIDDEN_SIZE=512
      N_HEADS=8
      BATCH_SIZE=8
      NUM_WORKERS=2
      shift
      ;;
    --no-loss-mask)
      NO_LOSS_MASK=true
      shift
      ;;
    --fp16)
      FP16=true
      shift
      ;;
    --online)
      ONLINE=true
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

FP16_FLAG=""
if [ "$FP16" = true ]; then
  FP16_FLAG="--fp16"
fi

echo "============================================================"
echo "PIPELINE CONFIG"
echo "  tokenizer      : $TOKENIZER"
echo "  out_dir        : $OUT_DIR"
echo "  output_dir     : $OUTPUT_DIR"
echo "  limit          : $LIMIT (0 = no limit)"
echo "  seq_len        : $SEQ_LEN"
echo "  shard_num_seqs : $SHARD_NUM_SEQS"
echo "  n_layers       : $N_LAYERS"
echo "  hidden_size    : $HIDDEN_SIZE"
echo "  n_heads        : $N_HEADS"
echo "  batch_size     : $BATCH_SIZE"
echo "  lr             : $LR"
echo "  steps          : $STEPS"
echo "  save_every     : $SAVE_EVERY"
echo "  log_every      : $LOG_EVERY"
echo "  fp16           : $FP16"
echo "  online         : $ONLINE"
echo "  debug          : $DEBUG"
echo "============================================================"

if [ "$ONLINE" = true ]; then
  echo "[1/1] Training (online streaming)..."

  LIMIT_FLAG=""
  if [ "$LIMIT" -gt 0 ]; then
    LIMIT_FLAG="--limit $LIMIT"
  fi

  python scripts/python/train.py \
    --tokenizer "$TOKENIZER" \
    --output-dir "$OUTPUT_DIR" \
    --n-layers "$N_LAYERS" \
    --hidden-size "$HIDDEN_SIZE" \
    --n-heads "$N_HEADS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --steps "$STEPS" \
    --num-workers 0 \
    --save-every "$SAVE_EVERY" \
    --log-every "$LOG_EVERY" \
    --online \
    --seq-len "$SEQ_LEN" \
    $LIMIT_FLAG \
    $FP16_FLAG

else
  # ── Disk mode: preprocess then train ──

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
    --output-dir "$OUTPUT_DIR" \
    --n-layers "$N_LAYERS" \
    --hidden-size "$HIDDEN_SIZE" \
    --n-heads "$N_HEADS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --steps "$STEPS" \
    --num-workers "$NUM_WORKERS" \
    --save-every "$SAVE_EVERY" \
    --log-every "$LOG_EVERY" \
    $FP16_FLAG
fi

echo "============================================================"
echo "Training complete."
echo "============================================================"