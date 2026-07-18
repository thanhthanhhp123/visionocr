#!/bin/bash
#SBATCH --job-name=visionocr-awq-v2
#SBATCH --partition=main-gpu
#SBATCH --account=students
#SBATCH --qos=students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/awq-v2-%j.out
#SBATCH --error=logs/awq-v2-%j.err

set -euo pipefail

PROJECT_ROOT=/home/s3002152/LeeHoang_/vlm_invoice/visionocr
PYTHON=/home/s3002152/LeeHoang_/vlm_invoice/thanhocr/bin/python
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
exec "$PYTHON" vlm/finetune/merge_and_quantize.py --quantize-only
