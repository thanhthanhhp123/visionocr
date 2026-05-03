import time
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from api.schemas.invoice import ExtractResponse
from worker.tasks import extract_invoice_task

router = APIRouter(prefix="/api/v1", tags=["invoice"])


@router.post("/extract", response_model=ExtractResponse)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Upload a Vietnamese invoice image and extract structured information.
    Returns a task_id for async polling, or the result directly for small files.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")

    # Dispatch to Celery worker
    task = extract_invoice_task.delay(image_bytes, file.filename)

    return ExtractResponse(task_id=task.id, status="processing")


@router.get("/tasks/{task_id}", response_model=ExtractResponse)
async def get_task_result(task_id: str):
    """Poll for async task result."""
    from worker.tasks import extract_invoice_task
    from celery.result import AsyncResult

    result = AsyncResult(task_id)

    if result.state == "PENDING":
        return ExtractResponse(task_id=task_id, status="pending")
    elif result.state == "SUCCESS":
        return ExtractResponse(task_id=task_id, status="success", result=result.get())
    elif result.state == "FAILURE":
        return ExtractResponse(task_id=task_id, status="failed", error=str(result.info))
    else:
        return ExtractResponse(task_id=task_id, status=result.state.lower())


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "visionocr-api"}
