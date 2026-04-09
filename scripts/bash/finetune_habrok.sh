#!/bin/bash
#SBATCH --job-name=math-llm-finetune
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:a100:2
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

# For deep ensembles, launch 3 independent jobs with different --seed and --output-dir:
#   sbatch --export=SEED=42,OUTDIR=finetune_seed42 scripts/bash/finetune_habrok.sh
#   sbatch --export=SEED=43,OUTDIR=finetune_seed43 scripts/bash/finetune_habrok.sh
#   sbatch --export=SEED=44,OUTDIR=finetune_seed44 scripts/bash/finetune_habrok.sh
#
# To resume a previous run, resubmit the same command — --resume picks up from
# the latest checkpoint and continues from where the data left off.
SEED=${SEED:-42}
OUTDIR=${OUTDIR:-finetune_seed42}
PRETRAINED=${PRETRAINED:-meta-llama/Llama-3.2-1B}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=2 scripts/python/train.py \
    --data-dir /scratch/s5549329/data/openmathinstruct2 \
    --pretrained-model ${PRETRAINED} \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --output-dir /scratch/s5549329/outputs/${OUTDIR} \
    --batch-size 1 \
    --lr 1e-5 \
    --steps 9999999 \
    --num-workers 4 \
    --save-every 1000 \
    --log-every 100 \
    --seed ${SEED} \
    --fp16 \
    --resume
