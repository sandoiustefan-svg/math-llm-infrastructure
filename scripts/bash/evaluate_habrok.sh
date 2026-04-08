#!/bin/bash
#SBATCH --job-name=math-llm-uq-eval
#SBATCH --time=02:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

set -euo pipefail

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0

export CUDA_HOME="${EBROOTCUDA}"
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"

cd /scratch/s5549329/math-llm-infrastructure || exit 1
mkdir -p logs

source .venv/bin/activate

export HF_HOME=/scratch/s5549329/.cache/huggingface

# ---------------------------------------------------------------------------
# Configuration — override with environment variables when submitting:
#
#   MC Dropout (single model):
#     sbatch --export=METHOD=mc_dropout,\
#                     MODEL_PATH=/scratch/.../checkpoints/final,\
#                     TOKENIZER=mistralai/Mistral-7B-v0.1,\
#                     PROBLEMS_FILE=/scratch/.../problems.jsonl,\
#                     OUTPUT_DIR=/scratch/.../uq_mc_dropout \
#            scripts/bash/evaluate_habrok.sh
#
#   Deep Ensemble (multiple model paths, space-separated):
#     sbatch --export=METHOD=ensemble,\
#                     MODEL_PATHS="/scratch/.../seed42/final /scratch/.../seed43/final /scratch/.../seed44/final",\
#                     TOKENIZER=mistralai/Mistral-7B-v0.1,\
#                     PROBLEMS_FILE=/scratch/.../problems.jsonl,\
#                     OUTPUT_DIR=/scratch/.../uq_ensemble \
#            scripts/bash/evaluate_habrok.sh
# ---------------------------------------------------------------------------

METHOD=${METHOD:-mc_dropout}
TOKENIZER=${TOKENIZER:-mistralai/Mistral-7B-v0.1}
PROBLEMS_FILE=${PROBLEMS_FILE:-/scratch/s5549329/data/problems.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-/scratch/s5549329/outputs/uq_eval}
NUM_PASSES=${NUM_PASSES:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}

mkdir -p "${OUTPUT_DIR}"

if [ "${METHOD}" = "mc_dropout" ]; then
    MODEL_PATH=${MODEL_PATH:-/scratch/s5549329/outputs/training/checkpoints/final}
    python scripts/python/evaluate_uq.py \
        --method mc_dropout \
        --model-path "${MODEL_PATH}" \
        --tokenizer "${TOKENIZER}" \
        --problems-file "${PROBLEMS_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --num-passes "${NUM_PASSES}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --device cuda

elif [ "${METHOD}" = "ensemble" ]; then
    # MODEL_PATHS is a space-separated list of checkpoint directories
    MODEL_PATHS=${MODEL_PATHS:-/scratch/s5549329/outputs/finetune_seed42/checkpoints/final}
    # shellcheck disable=SC2086
    python scripts/python/evaluate_uq.py \
        --method ensemble \
        --model-paths ${MODEL_PATHS} \
        --tokenizer "${TOKENIZER}" \
        --problems-file "${PROBLEMS_FILE}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --device cuda

else
    echo "Unknown METHOD=${METHOD}. Use mc_dropout or ensemble." >&2
    exit 1
fi
