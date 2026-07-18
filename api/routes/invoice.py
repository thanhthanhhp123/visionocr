import base64
from io import BytesIO
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.schemas.invoice import ExtractResponse, InvoiceRecordResponse
from db.repository import as_schema, get_invoice, list_invoices
from db.session import get_db
from worker.tasks import celery_app, extract_invoice_task

router = APIRouter(prefix="/api/v1", tags=["invoice"])
MAX_FILE_SIZE = 10 * 1024 * 1024
SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def _record_response(record) -> InvoiceRecordResponse:
    return InvoiceRecordResponse(
        id=record.id,
        filename=record.filename,
        invoice=as_schema(record),
        latency_ms=record.latency_ms,
        created_at=record.created_at,
    )


@router.post("/extract", response_model=ExtractResponse, status_code=202)
async def extract_invoice(file: UploadFile = File(...)):
    """Queue a validated Vietnamese invoice image for asynchronous extraction."""
    if file.content_type not in SUPPORTED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Image is empty")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Invalid image data") from None

    payload = base64.b64encode(image_bytes).decode("ascii")
    task = extract_invoice_task.delay(payload, file.filename or "invoice.jpg")
    return ExtractResponse(task_id=task.id, status="processing")


@router.get("/tasks/{task_id}", response_model=ExtractResponse)
async def get_task_result(task_id: str):
    """Poll an extraction task and return its result when it is complete."""
    result = celery_app.AsyncResult(task_id)
    if result.state == "PENDING":
        return ExtractResponse(task_id=task_id, status="pending")
    if result.state == "SUCCESS":
        payload = result.result
        return ExtractResponse(task_id=task_id, status="success", **payload)
    if result.state == "FAILURE":
        return ExtractResponse(task_id=task_id, status="failed", error=str(result.info))
    return ExtractResponse(task_id=task_id, status=result.state.lower())


@router.get("/invoices", response_model=list[InvoiceRecordResponse])
async def get_invoices(limit: int = 20, db: Session = Depends(get_db)):
    return [_record_response(record) for record in list_invoices(db, min(max(limit, 1), 100))]


@router.get("/invoices/{invoice_id}", response_model=InvoiceRecordResponse)
async def get_invoice_record(invoice_id: UUID, db: Session = Depends(get_db)):
    record = get_invoice(db, invoice_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Invoice not found")
    return _record_response(record)


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    return {"status": "ok", "service": "visionocr-api", "database": "ok"}
