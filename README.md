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
| **Inference** | vLLM, AWQ INT4 quantization |
| **MLOps** | MLflow (experiment tracking, model registry) |
| **Backend** | FastAPI, Celery, Redis, PostgreSQL, SQLAlchemy, Alembic |
| **Frontend** | Streamlit |
| **DevOps** | Docker, Docker Compose |

---

## Project Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Data pipeline & labeling | ✅ Done |
| 2 | Fine-tune Qwen2.5-VL + MLflow | 🔄 In progress |
| 3 | FastAPI + Celery + PostgreSQL | ⬜ Planned |
| 4 | Frontend Streamlit | ⬜ Planned |
| 5 | Docker + Docker Compose | ⬜ Planned |
| 6 | Polish, README, demo video | ⬜ Planned |

---

## Evaluation Results

Results on 26-invoice held-out test set (Vietnamese retail invoices):

| Metric | Score |
|---|---|
| Store Name Accuracy | — |
| Date Accuracy | — |
| Total Amount Accuracy | — |
| Item Name F1 | — |
| JSON Parse Rate | — |
| Avg Latency (AWQ + vLLM) | — |

*Results will be updated after Phase 2.*

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
docker-compose up --build
```

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
├── db/                   SQLAlchemy models, CRUD, Alembic migrations
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
