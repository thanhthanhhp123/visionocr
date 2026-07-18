.PHONY: help install install-train install-dev check validate prepare train evaluate api worker frontend docker-up docker-down mlflow lint lint-fix test migrate

help:
	@echo "VisionOCR — Available commands:"
	@echo ""
	@echo "  Setup"
	@echo "    make install          Install inference/API dependencies"
	@echo "    make install-train    Install fine-tuning dependencies"
	@echo "    make install-dev      Install test and lint dependencies"
	@echo ""
	@echo "  Data (Phase 1)"
	@echo "    make check            Check dataset integrity"
	@echo "    make validate         Pydantic validate + clean labels"
	@echo "    make prepare          Build train/val/test JSONL splits"
	@echo ""
	@echo "  Training (Phase 2)"
	@echo "    make train            Run LoRA fine-tuning"
	@echo "    make evaluate         Evaluate on test set"
	@echo "    make mlflow           Open MLflow UI"
	@echo ""
	@echo "  Services (Phase 3+)"
	@echo "    make api              Start FastAPI dev server"
	@echo "    make worker           Start Celery worker"
	@echo "    make frontend         Start Streamlit dashboard"
	@echo "    make docker-up        Start all services with Docker Compose"
	@echo "    make docker-down      Stop all services"
	@echo ""
	@echo "  Dev"
	@echo "    make lint             Check code with ruff"
	@echo "    make lint-fix         Apply safe ruff fixes"
	@echo "    make test             Run unit/API tests"
	@echo "    make migrate          Apply Alembic migrations"

install:
	pip install -r requirements.txt

install-train:
	pip install -r requirements-train.txt

install-dev:
	pip install -r requirements-dev.txt

# ── Data pipeline ──────────────────────────────────────────
check:
	python scripts/check_data.py

validate:
	python scripts/validate_labels.py

prepare:
	python scripts/prepare_finetune_data.py

# ── Training ───────────────────────────────────────────────
train:
	python vlm/finetune/train.py

evaluate:
	python scripts/evaluate.py

mlflow:
	mlflow ui --port 5000

migrate:
	alembic upgrade head

# ── Services ───────────────────────────────────────────────
api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

worker:
	celery -A worker.tasks worker --loglevel=info --concurrency=2

frontend:
	streamlit run frontend/app.py --server.port 8501

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

# ── Dev ────────────────────────────────────────────────────
lint:
	ruff check .

lint-fix:
	ruff check . --fix

test:
	pytest -q
