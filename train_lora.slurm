#!/bin/bash
#SBATCH --job-name=visionocr-lora-v2
#SBATCH --partition=main-gpu
#SBATCH --account=students
#SBATCH --qos=students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/lora-v2-%j.out
#SBATCH --error=logs/lora-v2-%j.err

set -euo pipefail

PROJECT_ROOT=/home/s3002152/LeeHoang_/vlm_invoice/visionocr
PYTHON=/home/s3002152/LeeHoang_/vlm_invoice/thanhocr/bin/python

cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=/tmp/visionocr-hf-${SLURM_JOB_ID}

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $($PYTHON --version)"
nvidia-smi

exec "$PYTHON" vlm/finetune/train.py
