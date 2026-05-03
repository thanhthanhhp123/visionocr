from pydantic import BaseModel, field_validator
from typing import List, Optional
from datetime import datetime


class InvoiceItem(BaseModel):
    name: str
    unit_price: float
    quantity: float
    total_price: float


class InvoiceSchema(BaseModel):
    store_name: str
    date: str
    total: float
    discount: float = 0.0
    items: List[InvoiceItem]

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        datetime.strptime(v, "%Y-%m-%d")
        return v


class ExtractRequest(BaseModel):
    """For JSON body requests (base64 image)"""
    image_base64: str
    filename: Optional[str] = None


class ExtractResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[InvoiceSchema] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
