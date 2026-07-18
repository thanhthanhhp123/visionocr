#!/bin/bash
#SBATCH --job-name=visionocr-merge-v2
#SBATCH --partition=main-gpu
#SBATCH --account=students
#SBATCH --qos=students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge-v2-%j.out
#SBATCH --error=logs/merge-v2-%j.err

set -euo pipefail

PROJECT_ROOT=/home/s3002152/LeeHoang_/vlm_invoice/visionocr
PYTHON=/home/s3002152/LeeHoang_/vlm_invoice/thanhocr/bin/python
cd "$PROJECT_ROOT"

export PYTHONUNBUFFERED=1
exec "$PYTHON" vlm/finetune/merge_and_quantize.py --merge-only
