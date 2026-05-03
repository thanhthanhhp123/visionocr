import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class InvoiceRecord(Base):
    __tablename__ = "invoices"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename   = Column(String, nullable=True)
    store_name = Column(String, nullable=True)
    date       = Column(String, nullable=True)
    total      = Column(Float, nullable=True)
    discount   = Column(Float, default=0.0)
    raw_json   = Column(JSON, nullable=True)
    latency_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    items = relationship("InvoiceItem", back_populates="invoice", cascade="all, delete-orphan")


class InvoiceItem(Base):
    __tablename__ = "invoice_items"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    invoice_id  = Column(UUID(as_uuid=True), ForeignKey("invoices.id"), nullable=False)
    name        = Column(String, nullable=False)
    unit_price  = Column(Float, nullable=True)
    quantity    = Column(Float, nullable=True)
    total_price = Column(Float, nullable=True)

    invoice = relationship("InvoiceRecord", back_populates="items")
