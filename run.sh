#!/bin/bash
# Usage: bash run.sh [seed]
#   seed : random seed (default: 42)
# Examples:
#   bash run.sh        # MC Dropout run, seed 42
#   bash run.sh 7      # MC Dropout run, seed 7
set -euo pipefail

SEED=${1:-42}
CONFIG="configs/clusters/macross.yaml"

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
t = c.get('train', {})
print(f"STEPS={t.get('steps', 50000)}")
print(f"EPOCHS={t.get('epochs', 0)}")
EOF
)"

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
export NCCL_P2P_DISABLE=1   # RTX 3090s are PCIe-only; P2P causes large GPU memory allocation

BASE_MODEL="meta-llama/Llama-3.2-3B-Instruct"
TOKENIZER="$BASE_MODEL"

BATCH_SIZE=1
GRAD_ACCUM=16
MC_DROPOUT="--mc-dropout-rate 0.1"

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
    --steps "$STEPS" \
    --epochs "$EPOCHS" \
    --warmup-steps 500 \
    --num-workers 4 \
    --save-every 2500 \
    --log-every 100 \
    --val-shard-count 8 \
    --val-every 500 \
    --val-batches 32 \
    --bf16 \
    --resume
