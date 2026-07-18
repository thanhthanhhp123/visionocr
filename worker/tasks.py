import os
import time
import tempfile
import base64
from celery import Celery

BROKER_URL = os.getenv(
    "CELERY_BROKER_URL", os.getenv("REDIS_URL", "redis://localhost:6379/0")
)
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL.rsplit("/", 1)[0] + "/1")

celery_app = Celery(
    "visionocr",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    task_track_started=True,
)


@celery_app.task(bind=True, name="extract_invoice")
def extract_invoice_task(self, image_base64: str, filename: str = "invoice.jpg"):
    """
    Async Celery task:
    1. Decode and save image to a temp file
    2. Run PaddleOCR for text extraction
    3. Run Qwen2.5-VL (via vLLM) for structured extraction
    4. Return JSON result
    """
    start = time.time()
    self.update_state(state="STARTED", meta={"filename": filename})

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)

        # Write to temp file
        suffix = ".jpg" if "jpg" in (filename or "").lower() else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name

        # OCR
        from ocr.paddle_engine import extract_text

        ocr_text = extract_text(tmp_path)

        # VLM inference
        from vlm.inference import extract_invoice

        result = extract_invoice(tmp_path, ocr_text)

        from api.schemas.invoice import InvoiceSchema

        invoice = InvoiceSchema.model_validate(result)

        latency = round((time.time() - start) * 1000, 1)
        from db.repository import create_invoice
        from db.session import SessionLocal, init_db

        init_db()
        with SessionLocal() as db:
            record = create_invoice(db, invoice, filename, latency)

        return {
            "result": invoice.model_dump(),
            "latency_ms": latency,
            "invoice_id": str(record.id),
        }

    except Exception as e:
        raise self.retry(exc=e, countdown=5, max_retries=2)

    finally:
        import os as _os

        tmp_path = locals().get("tmp_path")
        if tmp_path:
            try:
                _os.unlink(tmp_path)
            except OSError:
                pass
