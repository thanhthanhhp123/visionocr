"""Persistence helpers for extracted invoices."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session, selectinload

from api.schemas.invoice import InvoiceSchema
from db.models import InvoiceItem, InvoiceRecord


def create_invoice(
    db: Session, invoice: InvoiceSchema, filename: str | None, latency_ms: float | None
) -> InvoiceRecord:
    record = InvoiceRecord(
        filename=filename,
        store_name=invoice.store_name,
        date=invoice.date,
        total=invoice.total,
        discount=invoice.discount,
        raw_json=invoice.model_dump(),
        latency_ms=latency_ms,
        items=[
            InvoiceItem(
                name=item.name,
                unit_price=item.unit_price,
                quantity=item.quantity,
                total_price=item.total_price,
            )
            for item in invoice.items
        ],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_invoice(db: Session, invoice_id: UUID) -> InvoiceRecord | None:
    return (
        db.query(InvoiceRecord)
        .options(selectinload(InvoiceRecord.items))
        .filter(InvoiceRecord.id == invoice_id)
        .one_or_none()
    )


def list_invoices(db: Session, limit: int = 20) -> list[InvoiceRecord]:
    return (
        db.query(InvoiceRecord)
        .options(selectinload(InvoiceRecord.items))
        .order_by(InvoiceRecord.created_at.desc())
        .limit(limit)
        .all()
    )


def as_schema(record: InvoiceRecord) -> InvoiceSchema:
    if record.raw_json:
        return InvoiceSchema.model_validate(record.raw_json)
    return InvoiceSchema(
        store_name=record.store_name or "",
        date=record.date or "1970-01-01",
        total=record.total or 0,
        discount=record.discount or 0,
        items=[
            {
                "name": item.name,
                "unit_price": item.unit_price or 0,
                "quantity": item.quantity or 0,
                "total_price": item.total_price or 0,
            }
            for item in record.items
        ],
    )
