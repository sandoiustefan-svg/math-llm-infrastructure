#!/bin/bash
#SBATCH --job-name=math-llm-train
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64000
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

module purge
module load CUDA/12.6.0
module load cuDNN/9.5.1.17-CUDA-12.6.0
module load NCCL/2.26.2-GCCcore-13.3.0-CUDA-12.6.0

export CUDA_HOME="${EBROOTCUDA}"
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"

# Self-resubmit before training starts so the next job is always queued,
# even if this one is hard-killed near the end of the 4-hour window.
sbatch "$0"

cd /home2/s5549329/math-llm-infrastructure || exit 1
mkdir -p logs

source .venv/bin/activate

export HF_HOME=/scratch/s5549329/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Architecture: ~1B params (16L / 2048H / 16heads)
# Previous run (exp_001): 16L/1536H/12heads ~650M, fp16+bf16 conflict, no warmup → loss stuck at 5.3
# This run: same data, fixed precision (bf16), warmup + cosine decay, grad accumulation

torchrun --nproc_per_node=4 scripts/python/train.py \
    --data-dir /scratch/s5549329/data/openmathinstruct2 \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --output-dir /scratch/s5549329/outputs/training_1b \
    --n-layers 16 \
    --hidden-size 2048 \
    --n-heads 16 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --lr 3e-4 \
    --warmup-steps 2000 \
    --steps 1200000 \
    --num-workers 4 \
    --save-every 2000 \
    --log-every 100 \
    --bf16 \
    --resume
