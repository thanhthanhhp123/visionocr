#!/bin/bash
# One-shot setup for a fresh vast.ai instance: git clone this repo, then run
# this script. It generates .env (with a random Postgres password), checks
# that the fine-tuned LoRA adapter weights are in place (they aren't in git —
# see the printed instructions below if missing), and brings up the full
# docker-compose stack (Postgres, Redis, API, Celery worker, Streamlit,
# Caddy). The base Qwen2.5-VL-3B model itself downloads automatically from
# the Hugging Face Hub on first worker start — no manual transfer needed.
#
# Usage:
#   git clone <repo> && cd visionocr
#   ./scripts/setup_vastai.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found. Most vast.ai templates ship with it — pick a" >&2
    echo "'PyTorch' or 'CUDA' template, or install docker + nvidia-container-toolkit manually." >&2
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "'docker compose' plugin not found (this repo's compose file needs the v2 CLI, not standalone docker-compose)." >&2
    exit 1
fi

if [ ! -f .env ]; then
    echo "No .env found — creating one from .env.example with a random POSTGRES_PASSWORD."
    cp .env.example .env
    RANDOM_PW=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
    # portable in-place sed for both GNU and BSD sed
    sed -i.bak "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=${RANDOM_PW}|" .env && rm -f .env.bak
else
    echo ".env already exists, leaving it as-is."
fi

ADAPTER_DIR="models/qwen-lora-invoice-adapter-v2"
MERGED_DIR="models/qwen-merged-invoice-v2"
if [ ! -d "$ADAPTER_DIR" ] && [ ! -d "$MERGED_DIR" ]; then
    echo
    echo "Missing fine-tuned weights — neither $ADAPTER_DIR nor $MERGED_DIR exists." >&2
    echo "These are the trained model artifacts and are NOT in git (see .gitignore)." >&2
    echo "Copy the adapter (~158MB, smallest option) from wherever it's trained, e.g.:" >&2
    echo >&2
    echo "  scp -r <user>@hpc-head1.ewi.utwente.nl:/home/s3002152/LeeHoang_/vlm_invoice/visionocr/models/qwen-lora-invoice-adapter-v2 \\" >&2
    echo "      $(pwd)/models/" >&2
    echo >&2
    echo "Then re-run this script." >&2
    exit 1
fi

echo
echo "Building and starting the stack (this can take a while on first run — worker image installs torch/transformers/paddleocr)..."
docker compose up -d --build

echo
echo "Done. Services:"
docker compose ps
echo
SITE_ADDRESS=$(grep '^SITE_ADDRESS=' .env | cut -d= -f2-)
if [ "$SITE_ADDRESS" = ":80" ] || [ -z "$SITE_ADDRESS" ]; then
    echo "Public entrypoint (Caddy, plain HTTP): http://<this-machine-public-ip>"
else
    echo "Public entrypoint (Caddy, auto-HTTPS once DNS points here): https://${SITE_ADDRESS}"
fi
echo
echo "Tail worker logs (model loading / inference) with:"
echo "  docker compose logs -f worker"
