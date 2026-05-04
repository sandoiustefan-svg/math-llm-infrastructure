#!/bin/bash
# Usage: bash run_eval.sh [cluster] [seed] [method] [test_source] [checkpoint_step] [limit]
#   cluster          : macross (default) | a100-1 | a100-2 | a100-3
#   seed             : training seed used to locate the checkpoint (default: 42)
#   method           : mc_dropout (default) | ensemble
#   test_source      : all (default) | gsm8k | math | openmath_tail
#   checkpoint_step  : step number to eval (default: final); e.g. 408000
#   limit            : max problems per test set (default: 500; use 50 for a quick check)
#
# Examples:
#   bash run_eval.sh                                    # MC Dropout, seed 42, all test sets, final ckpt
#   bash run_eval.sh macross 42 mc_dropout gsm8k
#   bash run_eval.sh macross 42 mc_dropout all 408000   # eval mid-training checkpoint
#   bash run_eval.sh macross 42 mc_dropout gsm8k 408000 50  # quick 50-problem smoke test
#   bash run_eval.sh macross 42 ensemble all            # needs seed-42/123/456 checkpoints on disk
set -euo pipefail

CLUSTER=${1:-macross}
SEED=${2:-42}
METHOD=${3:-mc_dropout}
TEST_SOURCE=${4:-all}
CHECKPOINT_STEP=${5:-}
LIMIT=${6:-500}
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
print(f"HF_CACHE={c['paths']['hf_cache']}")
print(f"CUDA_DEVICES={c['hardware']['cuda_devices']}")
EOF
)"

echo "Cluster     : $CLUSTER"
echo "Seed        : $SEED"
echo "Method      : $METHOD"
echo "Test source : $TEST_SOURCE"
echo "Base dir    : $BASE_DIR"
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
    --method "$METHOD" \
    --test-source "$TEST_SOURCE" \
    --seed "$SEED" \
    --num-passes 20 \
    --mc-dropout-rate 0.0 \
    --max-new-tokens 512 \
    --limit "$LIMIT" \
    $CKPT_STEP_ARG \
    2>&1 | tee "logs/uq_eval_${METHOD}_seed${SEED}_${TEST_SOURCE}_$(date +%Y%m%d_%H%M%S).log"
