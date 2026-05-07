#!/bin/bash
# Eval script for Llama-3.2-1B LoRA (rank 16, seed 42)
#
# Usage: bash run_eval_1b.sh [cluster] [method] [test_source] [prompt] [checkpoint_step] [limit]
#   cluster         : macross (default) | fse-4a100-2-1b
#   method          : mc_dropout
#   test_source     : all (default) | gsm8k | math | openmath_tail
#   prompt          : zero_shot (default) | cot | rag
#   checkpoint_step : step number (default: final); e.g. 408000
#   limit           : max problems per test set (default: 500; use 50 for a quick check)
#
# Examples:
#   bash run_eval_1b.sh
#   bash run_eval_1b.sh macross mc_dropout gsm8k zero_shot
#   bash run_eval_1b.sh macross mc_dropout gsm8k cot 408000 50
#   bash run_eval_1b.sh fse-4a100-2-1b mc_dropout gsm8k rag "" 500
set -euo pipefail
CLUSTER=${1:-macross}
METHOD=${2:-mc_dropout}
TEST_SOURCE=${3:-all}
PROMPT=${4:-zero_shot}
CHECKPOINT_STEP=${5:-}
LIMIT=${6:-500}

SEED=42
BASE_MODEL="meta-llama/Llama-3.2-1B-Instruct"
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

echo "Model       : 1B  ($BASE_MODEL)"
echo "Method      : $METHOD"
echo "Test source : $TEST_SOURCE"
echo "Prompt      : $PROMPT"
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
    2>&1 | tee "logs/uq_eval_1b_${METHOD}_${TEST_SOURCE}_${PROMPT}_$(date +%Y%m%d_%H%M%S).log"
