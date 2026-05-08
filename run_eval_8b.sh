#!/bin/bash
# Eval script for Llama-3.1-8B LoRA (rank 64, seed 42)
#
# Usage: bash run_eval_8b.sh [cluster] [method] [test_source] [prompt] [checkpoint_step] [limit]
#   cluster         : fse-4a100-2-8b (default) | fse-4a100-2-8b-cot | macross_8b
#   method          : mc_dropout (default)
#   test_source     : all (default) | gsm8k | math
#   prompt          : zero_shot (default) | cot
#   checkpoint_step : step number (default: final); e.g. 408000
#   limit           : max problems per test set (default: 500; use 50 for a quick check)
#
# Examples:
#   bash run_eval_8b.sh fse-4a100-2-8b mc_dropout all zero_shot
#   bash run_eval_8b.sh fse-4a100-2-8b-cot mc_dropout all cot
set -euo pipefail

CLUSTER=${1:-fse-4a100-2-8b}
METHOD=${2:-mc_dropout}
TEST_SOURCE=${3:-all}
PROMPT=${4:-zero_shot}
CHECKPOINT_STEP=${5:-}
LIMIT=${6:-500}
SEED=42
BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
CONFIG="configs/clusters/${CLUSTER}.yaml"

eval "$(python3 - <<EOF
import yaml
with open("$CONFIG") as f:
    c = yaml.safe_load(f)
print(f"BASE_DIR={c['paths']['base_dir']}")
print(f"HF_CACHE={c['paths']['hf_cache']}")
print(f"CUDA_DEVICES={c['hardware']['cuda_devices']}")
EOF
)"

echo "Model       : 8B  ($BASE_MODEL)"
echo "Method      : $METHOD"
echo "Test source : $TEST_SOURCE"
echo "GPU devices : $CUDA_DEVICES"

cd "$BASE_DIR"
source .venv/bin/activate
mkdir -p logs results

export HF_HOME="$HF_CACHE"

CKPT_STEP_ARG=""
if [ -n "$CHECKPOINT_STEP" ]; then
    CKPT_STEP_ARG="--checkpoint-step $CHECKPOINT_STEP"
fi

python3 scripts/python/run_uq_eval.py \
    --cluster "$CLUSTER" \
    --base-model "$BASE_MODEL" \
    --method "$METHOD" \
    --test-source "$TEST_SOURCE" \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --num-passes 20 \
    --mc-dropout-rate 0.1 \
    --max-new-tokens 512 \
    --limit "$LIMIT" \
    $CKPT_STEP_ARG \
    2>&1 | tee "logs/uq_eval_8b_${METHOD}_${TEST_SOURCE}_$(date +%Y%m%d_%H%M%S).log"
