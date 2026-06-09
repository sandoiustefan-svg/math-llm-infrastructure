#!/bin/bash
# Run raw-model baseline eval for 1B and 8B simultaneously, one per RTX 3090.
#
# Usage:
#   bash scripts/bash/run_baselines_parallel.sh [--test-source gsm8k|math|all] [--limit N] [--prompt STYLE]
#
# Defaults: test-source=all, limit=500, prompt=all (zero_shot_aligned + verbalized_confidence)
# Logs are written to logs/baseline_{1b,8b}_<timestamp>.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

TEST_SOURCE="all"
LIMIT=500
PROMPT="verbalized_confidence"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --test-source) TEST_SOURCE="$2"; shift 2 ;;
        --limit)       LIMIT="$2";       shift 2 ;;
        --prompt)      PROMPT="$2";      shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG_1B="logs/baseline_1b_${TS}.log"
LOG_8B="logs/baseline_8b_${TS}.log"

echo "Launching 1B baseline  (GPU 1) → ${LOG_1B}"
python scripts/python/run_baseline_eval.py \
    --model-size 1b \
    --cluster macross_1b_3090 \
    --test-source "$TEST_SOURCE" \
    --limit "$LIMIT" \
    --prompt "$PROMPT" \
    >"$LOG_1B" 2>&1 &
PID_1B=$!

echo "Launching 8B baseline  (GPU 0) → ${LOG_8B}"
python scripts/python/run_baseline_eval.py \
    --model-size 8b \
    --cluster macross_8b_3090 \
    --test-source "$TEST_SOURCE" \
    --limit "$LIMIT" \
    --prompt "$PROMPT" \
    >"$LOG_8B" 2>&1 &
PID_8B=$!

echo "Both processes running.  PIDs: 1B=${PID_1B}  8B=${PID_8B}"
echo "Tail logs with:"
echo "  tail -f ${LOG_1B}"
echo "  tail -f ${LOG_8B}"
echo ""

wait $PID_1B
STATUS_1B=$?
wait $PID_8B
STATUS_8B=$?

echo ""
if [[ $STATUS_1B -eq 0 ]]; then
    echo "1B baseline: DONE"
else
    echo "1B baseline: FAILED (exit $STATUS_1B) — see ${LOG_1B}"
fi

if [[ $STATUS_8B -eq 0 ]]; then
    echo "8B baseline: DONE"
else
    echo "8B baseline: FAILED (exit $STATUS_8B) — see ${LOG_8B}"
fi

if [[ $STATUS_1B -ne 0 ]] || [[ $STATUS_8B -ne 0 ]]; then
    exit 1
fi
echo "All baselines complete."
