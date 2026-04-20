#!/bin/bash
# Usage: bash run.sh [cluster]
#   cluster: macross (default) | habrok
# Example: bash run.sh habrok
set -euo pipefail

CLUSTER=${1:-macross}
CONFIG="configs/clusters/${CLUSTER}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Unknown cluster '${CLUSTER}'. Config not found: $CONFIG"
    echo "Available: $(ls configs/clusters/*.yaml | xargs -n1 basename | sed 's/.yaml//')"
    exit 1
fi

# Parse cluster YAML using Python (always available in the venv)
eval "$(python3 - <<EOF
import yaml
with open("$CONFIG") as f:
    c = yaml.safe_load(f)
print(f"BASE_DIR={c['paths']['base_dir']}")
print(f"DATA_DIR={c['paths']['data_dir']}")
print(f"OUTPUT_DIR={c['paths']['output_dir']}")
print(f"HF_CACHE={c['paths']['hf_cache']}")
print(f"CUDA_DEVICES={c['gpus']['cuda_devices']}")
print(f"N_GPUS={c['gpus']['n_gpus']}")
EOF
)"

echo "Cluster  : $CLUSTER"
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

# Train — LoRA fine-tuning of 8B Instruct with MC Dropout for UQ
torchrun --nproc_per_node="$N_GPUS" scripts/python/train.py \
    --pretrained-model "$BASE_MODEL" \
    --tokenizer "$TOKENIZER" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --use-lora \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --mc-dropout-rate 0.1 \
    --batch-size 2 \
    --grad-accum-steps 8 \
    --lr 2e-4 \
    --steps 1200000 \
    --warmup-steps 5000 \
    --num-workers 4 \
    --save-every 10000 \
    --log-every 100 \
    --bf16 \
    --resume
