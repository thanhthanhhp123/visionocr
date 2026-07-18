# VisionOCR — Invoice Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Model-Qwen2.5--VL--3B-purple?style=flat-square"/>
  <img src="https://img.shields.io/badge/Framework-FastAPI-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-gray?style=flat-square"/>
</p>

> End-to-end AI system for extracting structured information from Vietnamese retail invoices.
> Built as a portfolio project targeting **AI/ML Engineer Fresher** positions.

---

## Demo

<!-- TODO: Add demo GIF after Phase 6 -->

```
Input:  Image of Vietnamese supermarket invoice (VinMart, Co.opmart, etc.)
Output: {"store_name": "VinCommerce", "date": "2024-01-15", "total": 81302,
         "discount": 8800, "items": [{"name": "Bơ đặc biệt", "unit_price": 52700,
         "quantity": 0.408, "total_price": 21502}]}
```

---

## Architecture

```
Invoice Image
      │
      ▼
PaddleOCR ──── Raw text + bounding boxes
      │
      ▼
Qwen2.5-VL-3B (LoRA fine-tuned)
  - Sees the image directly (spatial understanding)
  - Uses OCR text as hint
  - Outputs structured JSON
      │
      ▼
FastAPI + Celery + Redis
  - Async processing queue
  - PostgreSQL storage
  - vLLM inference engine (3–5x faster)
      │
      ▼
Streamlit Dashboard
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| **ML / AI** | PyTorch, Transformers, LoRA (PEFT), PaddleOCR, Qwen2.5-VL |
| **Inference** | Hugging Face + LoRA v2 (active); vLLM/AWQ INT4 (pending) |
| **MLOps** | MLflow (experiment tracking, model registry) |
| **Backend** | FastAPI, Celery, Redis, PostgreSQL, SQLAlchemy, Alembic |
| **Frontend** | Streamlit |
| **DevOps** | Docker, Docker Compose |

---

## Project Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Data pipeline & labeling | ✅ Done |
| 2 | Fine-tune Qwen2.5-VL + MLflow | 🟡 LoRA v2 trained, evaluated and merged; AWQ/vLLM pending |
| 3 | FastAPI + Celery + PostgreSQL | ✅ Implemented; end-to-end runtime verification pending |
| 4 | Frontend Streamlit | ✅ Implemented; end-to-end runtime verification pending |
| 5 | Docker + Docker Compose | 🟡 Configured; runtime verification pending (Docker unavailable locally) |
| 6 | Polish, README, demo video | 🟡 Docs/tests done; demo recording pending |

### Remaining work

- Produce an AWQ artifact from the merged LoRA v2 model, validate vLLM visual
  inference, and benchmark accuracy/latency against the HF path.
- Run the full Docker stack on a Docker-enabled GPU host using the local model
  artifacts or an authenticated model source.
- Record the actual upload-to-result demo GIF/video after the live stack check.

---

## Evaluation Results

Results on the same 87-sample held-out test set. Augmented variants from the
same source invoice are kept in a single split to prevent train/test leakage.

| Metric | Zero-shot | LoRA v2 | LoRA change |
|---|---:|---:|---:|
| Store Name Accuracy | 79.3% | 88.5% | +9.2 pp |
| Date Accuracy | 87.4% | 90.8% | +3.4 pp |
| Total Amount Accuracy | 23.0% | 90.8% | +67.8 pp |
| Item Name F1 | 0.931 | 0.923 | -0.008 |
| JSON Parse Rate | 98.9% | 98.9% | 0.0 pp |
| Avg Latency (HF, 4-bit) | 16.77s | 36.59s | +19.82s |

LoRA v2 substantially improves structured fields, especially total amount, but
does not improve item-name F1 and is slower in the current Hugging Face 4-bit
evaluation path. AWQ + vLLM latency remains pending.

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/thanhthanhhp123/visionocr.git
cd visionocr
pip install -r requirements.txt
```

### 2. Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The worker requires an NVIDIA-enabled Docker host. Mount or configure access to
the Qwen base model and the local `models/` artifacts before running offline.

### 3. API

```bash
curl -X POST http://localhost:8000/api/v1/extract \
  -F "file=@invoice.jpg"
```

---

## Project Structure

```
visionocr/
├── api/                  FastAPI routes and schemas
├── worker/               Celery async tasks
├── ocr/                  PaddleOCR engine wrapper
├── vlm/
│   ├── inference.py      vLLM inference engine
│   └── finetune/         LoRA training scripts
├── db/                   SQLAlchemy models, session and persistence helpers
├── alembic/              Database migration scripts
├── mlops/                MLflow tracking utilities
├── frontend/             Streamlit dashboard
├── scripts/              Data preparation and evaluation
├── datasets/
│   ├── images/           Invoice images (not committed)
│   ├── labels/           JSON labels (not committed)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── models/               Local model weights (not committed)
├── docker/               Dockerfiles per service
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Invoice Schema

```json
{
  "store_name": "VinCommerce",
  "date": "2024-01-15",
  "total": 81302,
  "discount": 8800,
  "items": [
    {
      "name": "Bơ đặc biệt",
      "unit_price": 52700,
      "quantity": 0.408,
      "total_price": 21502
    }
  ]
}
```
