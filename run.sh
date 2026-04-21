#!/bin/bash
# Usage: bash run.sh [cluster] [seed]
#   cluster : macross (default) | a100-1 | a100-2 | a100-3 | habrok
#   seed    : random seed (default: 42)
# Examples:
#   bash run.sh macross        # MC Dropout run, seed 42
#   bash run.sh a100-1 42      # Ensemble member 1
#   bash run.sh a100-2 123     # Ensemble member 2
#   bash run.sh a100-3 456     # Ensemble member 3
set -euo pipefail

CLUSTER=${1:-macross}
SEED=${2:-42}
CONFIG="configs/clusters/${CLUSTER}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Unknown cluster '${CLUSTER}'. Config not found: $CONFIG"
    echo "Available: $(ls configs/clusters/*.yaml | xargs -n1 basename | sed 's/.yaml//')"
    exit 1
fi

eval "$(python3 - <<EOF
import yaml
with open("$CONFIG") as f:
    c = yaml.safe_load(f)
print(f"BASE_DIR={c['paths']['base_dir']}")
print(f"DATA_DIR={c['paths']['data_dir']}")
print(f"OUTPUT_DIR={c['paths']['output_dir']}_seed${SEED}")
print(f"HF_CACHE={c['paths']['hf_cache']}")
print(f"CUDA_DEVICES={c['gpus']['cuda_devices']}")
print(f"N_GPUS={c['gpus']['n_gpus']}")
EOF
)"

echo "Cluster  : $CLUSTER"
echo "Seed     : $SEED"
echo "Base dir : $BASE_DIR"
echo "Data dir : $DATA_DIR"
echo "Output   : $OUTPUT_DIR"
echo "GPUs     : $CUDA_DEVICES ($N_GPUS devices)"

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p logs

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="$HF_CACHE"

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
TOKENIZER="$BASE_MODEL"

# A100s have 80GB — use larger batches for faster training
# macross 3090s have 24GB — keep small batches
if [[ "$CLUSTER" == a100* ]]; then
    BATCH_SIZE=8
    GRAD_ACCUM=4
else
    BATCH_SIZE=1
    GRAD_ACCUM=16
fi

# MC Dropout on macross and a100-1 (fse-2a100-1), ensemble members have no dropout
if [[ "$CLUSTER" == "macross" || "$CLUSTER" == "a100-1" ]]; then
    MC_DROPOUT="--mc-dropout-rate 0.1"
else
    MC_DROPOUT="--mc-dropout-rate 0.0"
fi

# Preprocess
if [ -f "$DATA_DIR/manifest.json" ]; then
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
        --out-dir "$DATA_DIR"
    echo "Preprocessing complete."
fi

# Train
torchrun --nproc_per_node="$N_GPUS" scripts/python/train.py \
    --pretrained-model "$BASE_MODEL" \
    --tokenizer "$TOKENIZER" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    $MC_DROPOUT \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --lr 2e-4 \
    --steps 50000 \
    --warmup-steps 5000 \
    --num-workers 4 \
    --save-every 10000 \
    --log-every 100 \
    --bf16 \
    --resume
