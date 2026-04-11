#!/bin/bash
#SBATCH --job-name=math-llm-finetune
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64000
#SBATCH --output=/home2/s5549329/math-llm-infrastructure/logs/finetune_%j.out
#SBATCH --error=/home2/s5549329/math-llm-infrastructure/logs/finetune_%j.err

set -euo pipefail

cd /home2/s5549329/math-llm-infrastructure || exit 1
mkdir -p logs

module purge
module load CUDA/12.6.0
module load cuDNN/9.5.1.17-CUDA-12.6.0
module load NCCL/2.26.2-GCCcore-13.3.0-CUDA-12.6.0

export CUDA_HOME="${EBROOTCUDA}"
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"

source .venv/bin/activate

export HF_HOME=/scratch/s5549329/.cache/huggingface
# Read HF token from the standard location so gated repos (Llama-3.2-1B) are accessible.
# Store your token with: echo "hf_..." > ~/.cache/huggingface/token && chmod 600 ~/.cache/huggingface/token
if [[ -f "${HOME}/.cache/huggingface/token" ]]; then
    export HF_TOKEN=$(cat "${HOME}/.cache/huggingface/token")
fi

# Configurable via --export when submitting:
#   SEED      — random seed (default 42); use different seeds for ensemble members
#   OUTDIR    — output subdirectory under /scratch/.../outputs/
#   PRETRAINED — HF model ID or local path
#   STEPS     — total training steps (default 5000 for an initial loss check;
#               increase and resubmit with --resume to continue)
#
# Initial probe (check loss curve, then resume with more steps):
#   sbatch --export=SEED=42,OUTDIR=finetune_seed42,STEPS=5000 scripts/bash/finetune_habrok.sh
#
# Resume with more steps after inspecting the plot:
#   sbatch --export=SEED=42,OUTDIR=finetune_seed42,STEPS=50000 scripts/bash/finetune_habrok.sh
#
# For deep ensembles, launch 3 seeds in parallel:
#   sbatch --export=SEED=42,OUTDIR=finetune_seed42,STEPS=5000 scripts/bash/finetune_habrok.sh
#   sbatch --export=SEED=43,OUTDIR=finetune_seed43,STEPS=5000 scripts/bash/finetune_habrok.sh
#   sbatch --export=SEED=44,OUTDIR=finetune_seed44,STEPS=5000 scripts/bash/finetune_habrok.sh
SEED=${SEED:-42}
OUTDIR=${OUTDIR:-finetune_seed42}
PRETRAINED=${PRETRAINED:-meta-llama/Llama-3.2-1B}
STEPS=${STEPS:-5000}

# Self-resubmit only when running open-ended (STEPS >= 50000) so short probe
# jobs don't keep requeueing indefinitely.
if [[ ${STEPS} -ge 50000 ]]; then
    sbatch --export=SEED=${SEED},OUTDIR=${OUTDIR},PRETRAINED=${PRETRAINED},STEPS=${STEPS} "$0"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable NVLink peer-to-peer and InfiniBand — Habrok A100s are PCIe-connected and
# NCCL's P2P probe segfaults when the topology doesn't support it.
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=4 scripts/python/train.py \
    --data-dir /scratch/s5549329/data/openmathinstruct2 \
    --pretrained-model ${PRETRAINED} \
    --tokenizer meta-llama/Llama-3.2-1B \
    --output-dir /scratch/s5549329/outputs/${OUTDIR} \
    --batch-size 1 \
    --lr 1e-5 \
    --steps ${STEPS} \
    --num-workers 4 \
    --save-every 1000 \
    --log-every 100 \
    --seed ${SEED} \
    --bf16 \
    --resume
