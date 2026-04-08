#!/bin/bash
#SBATCH --job-name=math-llm-train
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/12.2.0
module load cuDNN/8.9.2.26-CUDA-12.2.0

export CUDA_HOME="${EBROOTCUDA}"
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"

# Self-resubmit before training starts so the next job is always queued,
# even if this one is hard-killed near the end of the 4-hour window.
sbatch "$0"

cd /scratch/s5549329/math-llm-infrastructure || exit 1
mkdir -p logs

source .venv/bin/activate

# export HF_TOKEN="hf_..."
export HF_HOME=/scratch/s5549329/.cache/huggingface

torchrun --nproc_per_node=2 scripts/python/train.py \
    --data-dir /scratch/s5549329/data/openmathinstruct2 \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --output-dir /scratch/s5549329/outputs/training \
    --n-layers 16 \
    --hidden-size 1536 \
    --n-heads 12 \
    --batch-size 4 \
    --lr 3e-4 \
    --steps 1200000 \
    --num-workers 4 \
    --save-every 2000 \
    --log-every 100 \
    --fp16 \
    --resume
