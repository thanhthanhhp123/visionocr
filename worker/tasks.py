import os
import time
import tempfile
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "visionocr",
    broker=REDIS_URL,
    backend=REDIS_URL.replace("/0", "/1"),
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    task_track_started=True,
)


@celery_app.task(bind=True, name="extract_invoice")
def extract_invoice_task(self, image_bytes: bytes, filename: str = "invoice.jpg"):
    """
    Async Celery task:
    1. Save image to temp file
    2. Run PaddleOCR for text extraction
    3. Run Qwen2.5-VL (via vLLM) for structured extraction
    4. Return JSON result
    """
    start = time.time()
    self.update_state(state="STARTED", meta={"filename": filename})

    try:
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

        latency = round((time.time() - start) * 1000, 1)
        return {**result, "_latency_ms": latency}

    except Exception as e:
        raise self.retry(exc=e, countdown=5, max_retries=2)

    finally:
        import os as _os
        try: _os.unlink(tmp_path)
        except: pass
