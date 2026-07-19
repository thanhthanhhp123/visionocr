#!/bin/bash
#SBATCH --job-name=visionocr-serve-v2
#SBATCH --partition=main-gpu
#SBATCH --account=students
#SBATCH --qos=students
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/serve-v2-%j.out
#SBATCH --error=logs/serve-v2-%j.err

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
nvidia-smi

echo "GPU node reserved for 24h (job ${SLURM_JOB_ID} on $(hostname))."
echo "Attach to it with: srun --jobid=${SLURM_JOB_ID} --pty bash"
echo "Then run the web app with:"
echo "  cd /home/s3002152/LeeHoang_/vlm_invoice/visionocr"
echo "  /home/s3002152/LeeHoang_/vlm_invoice/thanhocr/bin/python -m streamlit run frontend/app_hpc.py --server.port 8501 --server.address 0.0.0.0"
echo "From your laptop:"
echo "  ssh -L 8501:$(hostname):8501 -J s3002152@hpc-head1 s3002152@$(hostname)"
echo "  open http://localhost:8501"

sleep infinity
