#!/bin/bash
#SBATCH --job-name=math-llm-finetune
#SBATCH --time=04:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:2
#SBATCH --mem=64000
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err

set -euo pipefail

cd /home2/s5549329/math-llm-infrastructure || exit 1
mkdir -p logs

module purge
module load CUDA/12.6.0
module load cuDNN/9.5.1.17-CUDA-12.6.0

export CUDA_HOME="${EBROOTCUDA}"
export LD_LIBRARY_PATH="${EBROOTCUDA}/lib64:${EBROOTCUDNN}/lib64:${LD_LIBRARY_PATH:-}"

# Fine-tune a pretrained model on OpenMathInstruct-2.
# For deep ensembles, launch 3 independent jobs with different --seed and --output-dir:
#   sbatch --export=SEED=42,OUTDIR=finetune_seed42 finetune_habrok.sh
#   sbatch --export=SEED=43,OUTDIR=finetune_seed43 finetune_habrok.sh
#   sbatch --export=SEED=44,OUTDIR=finetune_seed44 finetune_habrok.sh
SEED=${SEED:-42}
OUTDIR=${OUTDIR:-finetune_seed42}
PRETRAINED=${PRETRAINED:-mistralai/Mistral-7B-v0.1}

FINAL_CKPT="/scratch/s5549329/outputs/${OUTDIR}/checkpoints/final"

# Stop resubmitting once training has fully completed.
if [ -d "${FINAL_CKPT}" ]; then
    echo "Training already complete (${FINAL_CKPT} exists). Exiting."
    exit 0
fi

# Resubmit now so the next job is queued in case this one hits the wall time.
# Capture the job ID so we can cancel it if training fails.
NEXT_JOB=$(sbatch --parsable \
    --output=/home2/s5549329/math-llm-infrastructure/logs/finetune_%j.out \
    --error=/home2/s5549329/math-llm-infrastructure/logs/finetune_%j.err \
    --export=ALL,SEED=${SEED},OUTDIR=${OUTDIR},PRETRAINED=${PRETRAINED} \
    "$0")
echo "Queued next job: ${NEXT_JOB}"

source .venv/bin/activate

# export HF_TOKEN="hf_..."
export HF_HOME=/scratch/s5549329/.cache/huggingface

# Use a very high step count — training stops naturally when data is exhausted.
if torchrun --nproc_per_node=2 scripts/python/train.py \
    --data-dir /scratch/s5549329/data/openmathinstruct2 \
    --pretrained-model ${PRETRAINED} \
    --tokenizer mistralai/Mistral-7B-v0.1 \
    --output-dir /scratch/s5549329/outputs/${OUTDIR} \
    --batch-size 2 \
    --lr 1e-5 \
    --steps 9999999 \
    --num-workers 4 \
    --save-every 1000 \
    --log-every 100 \
    --seed ${SEED} \
    --fp16 \
    --resume; then
    # Training finished cleanly — cancel the pending resubmission.
    scancel "${NEXT_JOB}" 2>/dev/null && echo "Training complete. Cancelled pending job ${NEXT_JOB}."
else
    # Training errored — cancel the pending resubmission so we don't loop forever.
    scancel "${NEXT_JOB}" 2>/dev/null && echo "Training failed. Cancelled pending job ${NEXT_JOB}. Fix the error and resubmit manually."
    exit 1
fi
