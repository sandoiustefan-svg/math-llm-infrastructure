#!/bin/bash
# Eval script for Llama-3.1-8B LoRA (rank 64, seed 42)
#
# Usage: bash run_eval_8b.sh [method] [test_source] [checkpoint_step] [limit]
#   method          : mc_dropout (default) | ensemble
#   test_source     : all (default) | gsm8k | math | openmath_tail
#   checkpoint_step : step number (default: final); e.g. 100000
#   limit           : max problems per test set (default: 500; use 50 for a quick check)
#
# Examples:
#   bash run_eval_8b.sh
#   bash run_eval_8b.sh mc_dropout gsm8k
#   bash run_eval_8b.sh mc_dropout gsm8k final 50
set -euo pipefail

METHOD=${1:-mc_dropout}
TEST_SOURCE=${2:-all}
CHECKPOINT_STEP=${3:-}
LIMIT=${4:-500}

CLUSTER="macross_8b"
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
    --seed "$SEED" \
    --num-passes 20 \
    --mc-dropout-rate 0.0 \
    --max-new-tokens 512 \
    --limit "$LIMIT" \
    $CKPT_STEP_ARG \
    2>&1 | tee "logs/uq_eval_8b_${METHOD}_${TEST_SOURCE}_$(date +%Y%m%d_%H%M%S).log"
