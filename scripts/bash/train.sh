#!/bin/bash
# Usage:
#   bash scripts/bash/train.sh
#   bash scripts/bash/train.sh configs/clusters/macross.yaml
#   bash scripts/bash/train.sh configs/clusters/macross.yaml 7

set -euo pipefail

CONFIG=${1:-configs/clusters/macross.yaml}
SEED=${2:-42}

eval "$(python3 - <<EOF
import yaml

with open("$CONFIG", "r") as f:
    c = yaml.safe_load(f)

print(f"BASE_DIR={c['paths']['base_dir']}")
print(f"DATA_DIR={c['paths']['data_dir']}")
print(f"OUTPUT_DIR={c['paths']['output_dir']}_seed${SEED}")
print(f"HF_CACHE={c['paths']['hf_cache']}")

print(f"CUDA_DEVICES={c['hardware']['cuda_devices']}")
print(f"N_GPUS={c['hardware']['n_gpus']}")

print(f"TOKENIZER={c['model']['tokenizer']}")
print(f"PRETRAINED_MODEL={c['model']['pretrained_model']}")
print(f"SEQ_LEN={c['model'].get('seq_len', 2048)}")

print(f"BATCH_SIZE={c['training'].get('batch_size', 1)}")
print(f"GRAD_ACCUM_STEPS={c['training'].get('grad_accum_steps', 16)}")
print(f"LR={c['training'].get('lr', 2e-4)}")
print(f"STEPS={c['training'].get('steps', 10000)}")
print(f"EPOCHS={c['training'].get('epochs', 0)}")
print(f"NUM_WORKERS={c['training'].get('num_workers', 2)}")

print(f"FP16={str(c['precision'].get('fp16', True)).lower()}")
print(f"BF16={str(c['precision'].get('bf16', False)).lower()}")

print(f"USE_LORA={str(c['lora'].get('use_lora', True)).lower()}")
print(f"USE_QLORA={str(c['lora'].get('use_qlora', True)).lower()}")
print(f"LORA_RANK={c['lora'].get('rank', 16)}")
print(f"LORA_ALPHA={c['lora'].get('alpha', 32)}")
print(f"LORA_DROPOUT={c['lora'].get('dropout', 0.05)}")

print(f"BNB_4BIT_QUANT_TYPE={c.get('qlora', {}).get('quant_type', 'nf4')}")
print(f"BNB_4BIT_COMPUTE_DTYPE={c.get('qlora', {}).get('compute_dtype', 'bfloat16')}")
print(f"BNB_4BIT_USE_DOUBLE_QUANT={str(c.get('qlora', {}).get('double_quant', True)).lower()}")

print(f"MC_DROPOUT_RATE={c.get('uncertainty', {}).get('mc_dropout_rate', 0.0)}")

print(f"WARMUP_STEPS={c['logging'].get('warmup_steps', 100)}")
print(f"SAVE_EVERY={c['logging'].get('save_every', 500)}")
print(f"LOG_EVERY={c['logging'].get('log_every', 20)}")
print(f"RESUME={str(c['logging'].get('resume', False)).lower()}")

print(f"VAL_SHARD_COUNT={c['validation'].get('val_shard_count', 1)}")
print(f"TEST_SHARD_COUNT={c['validation'].get('test_shard_count', 0)}")
print(f"VAL_EVERY={c['validation'].get('val_every', 500)}")
print(f"VAL_BATCHES={c['validation'].get('val_batches', 50)}")
EOF
)"

echo "============================================================"
echo " TRAIN CONFIG"
echo "  config     : $CONFIG"
echo "  seed       : $SEED"
echo "  base dir   : $BASE_DIR"
echo "  data dir   : $DATA_DIR"
echo "  output dir : $OUTPUT_DIR"
echo "  model      : $PRETRAINED_MODEL"
echo "  tokenizer  : $TOKENIZER"
echo "  GPUs       : $CUDA_DEVICES ($N_GPUS devices)"
echo "============================================================"

cd "$BASE_DIR"

source .venv/bin/activate

mkdir -p logs

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE/datasets"

# Good for 2x RTX 3090 PCIe systems.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

RUN_CONFIG="/tmp/train_config_seed${SEED}.yaml"

python3 - <<EOF
import yaml

cfg = {
    "tokenizer": "$TOKENIZER",
    "pretrained_model": "$PRETRAINED_MODEL",
    "data_dir": "$DATA_DIR",
    "output_dir": "$OUTPUT_DIR",

    "seq_len": int("$SEQ_LEN"),

    "batch_size": int("$BATCH_SIZE"),
    "grad_accum_steps": int("$GRAD_ACCUM_STEPS"),
    "lr": float("$LR"),
    "steps": int("$STEPS"),
    "epochs": float("$EPOCHS"),
    "num_workers": int("$NUM_WORKERS"),

    "fp16": "$FP16" == "true",
    "bf16": "$BF16" == "true",

    "warmup_steps": int("$WARMUP_STEPS"),
    "save_every": int("$SAVE_EVERY"),
    "log_every": int("$LOG_EVERY"),
    "resume": "$RESUME" == "true",
    "seed": int("$SEED"),

    "use_lora": "$USE_LORA" == "true",
    "use_qlora": "$USE_QLORA" == "true",
    "lora_rank": int("$LORA_RANK"),
    "lora_alpha": int("$LORA_ALPHA"),
    "lora_dropout": float("$LORA_DROPOUT"),

    "bnb_4bit_quant_type": "$BNB_4BIT_QUANT_TYPE",
    "bnb_4bit_compute_dtype": "$BNB_4BIT_COMPUTE_DTYPE",
    "bnb_4bit_use_double_quant": "$BNB_4BIT_USE_DOUBLE_QUANT" == "true",

    "mc_dropout_rate": float("$MC_DROPOUT_RATE"),

    "val_shard_count": int("$VAL_SHARD_COUNT"),
    "test_shard_count": int("$TEST_SHARD_COUNT"),
    "val_every": int("$VAL_EVERY"),
    "val_batches": int("$VAL_BATCHES"),
}

with open("$RUN_CONFIG", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote runtime config → $RUN_CONFIG")
EOF

if [ ! -f "$DATA_DIR/manifest.json" ]; then
    echo "ERROR: Missing manifest.json in $DATA_DIR"
    echo "Run preprocessing first and make sure data_dir points to the shards folder."
    exit 1
fi

echo "Starting training..."

torchrun \
    --nproc_per_node="$N_GPUS" \
    scripts/python/train.py \
    --config "$RUN_CONFIG"

echo "Training finished."