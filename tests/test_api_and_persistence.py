import os
import asyncio
from io import BytesIO
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/visionocr-tests.db")
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")

import httpx
from PIL import Image

from api.main import app
from api.routes import invoice as invoice_routes
from api.schemas.invoice import InvoiceSchema
from db.repository import create_invoice, get_invoice
from db.session import SessionLocal, init_db
from vlm.inference import _parse_json


def _image_bytes() -> bytes:
    image = Image.new("RGB", (8, 8), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _request(method: str, url: str, **kwargs) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(send_request())


def test_extract_rejects_non_image():
    response = _request("POST", "/api/v1/extract", files={"file": ("note.txt", b"x", "text/plain")})
    assert response.status_code == 400


def test_extract_queues_base64_image(monkeypatch):
    monkeypatch.setattr(invoice_routes.extract_invoice_task, "delay", lambda *_: SimpleNamespace(id="task-123"))
    response = _request(
        "POST",
        "/api/v1/extract",
        files={"file": ("invoice.png", _image_bytes(), "image/png")},
    )
    assert response.status_code == 202
    assert response.json() == {"task_id": "task-123", "status": "processing", "result": None, "error": None, "latency_ms": None, "invoice_id": None}


def test_task_poll_returns_worker_payload(monkeypatch):
    payload = {
        "result": {
            "store_name": "Store",
            "date": "2026-01-01",
            "total": 100,
            "discount": 0,
            "items": [{"name": "Item", "unit_price": 100, "quantity": 1, "total_price": 100}],
        },
        "latency_ms": 12.5,
        "invoice_id": "0e4e4b7b-3faa-4df6-9ed3-2ae3261aa549",
    }
    monkeypatch.setattr(invoice_routes.celery_app, "AsyncResult", lambda _: SimpleNamespace(state="SUCCESS", result=payload))
    response = _request("GET", "/api/v1/tasks/task-123")
    assert response.status_code == 200
    assert response.json()["result"]["total"] == 100.0
    assert response.json()["latency_ms"] == 12.5


def test_invoice_persistence_round_trip():
    init_db()
    invoice = InvoiceSchema(
        store_name="Store",
        date="2026-01-01",
        total=100,
        discount=0,
        items=[{"name": "Item", "unit_price": 100, "quantity": 1, "total_price": 100}],
    )
    with SessionLocal() as db:
        record = create_invoice(db, invoice, "invoice.png", 12.5)
        loaded = get_invoice(db, record.id)
    assert loaded is not None
    assert loaded.raw_json["total"] == 100.0
    assert len(loaded.items) == 1


def test_json_parser_handles_fenced_and_embedded_json():
    assert _parse_json('```json\n{"total": 1}\n```') == {"total": 1}
    assert _parse_json('Answer: {"total": 2}') == {"total": 2}
    assert _parse_json("not json") == {}
