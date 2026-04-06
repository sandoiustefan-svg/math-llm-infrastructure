#!/bin/bash
set -euo pipefail

cd /home/bsandoiu/math-llm-infrastructure
source .venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES=0,1
export HF_TOKEN="hf_..."

TOKENIZER="mistralai/Mistral-7B-v0.1"
OUT_DIR="data/processed/openmathinstruct2"
OUTPUT_DIR="outputs/training"

# Preprocess 
if [ -f "$OUT_DIR/manifest.json" ]; then
    echo "Preprocessed data found — skipping."
else
    echo "Preprocessing..."
    python3 scripts/python/preprocess_data.py \
        --tokenizer "$TOKENIZER" \
        --split train \
        --limit 0 \
        --skip 0 \
        --seq-len 1024 \
        --shard-num-seqs 256 \
        --out-dir "$OUT_DIR"
    echo "Preprocessing complete."
fi

# Train
torchrun --nproc_per_node=2 scripts/python/train.py \
    --data-dir "$OUT_DIR" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$OUTPUT_DIR" \
    --n-layers 16 \
    --hidden-size 1536 \
    --n-heads 12 \
    --batch-size 4 \
    --lr 3e-4 \
    --steps 1200000 \
    --num-workers 4 \
    --save-every 10000 \
    --log-every 100 \
    --fp16 \
    --resume