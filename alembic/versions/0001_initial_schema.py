"""Create invoice persistence tables.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-07-17
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "invoices",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("filename", sa.String(), nullable=True),
        sa.Column("store_name", sa.String(), nullable=True),
        sa.Column("date", sa.String(), nullable=True),
        sa.Column("total", sa.Float(), nullable=True),
        sa.Column("discount", sa.Float(), nullable=True),
        sa.Column("raw_json", sa.JSON(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "invoice_items",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column(
            "invoice_id", sa.Uuid(), sa.ForeignKey("invoices.id"), nullable=False
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("unit_price", sa.Float(), nullable=True),
        sa.Column("quantity", sa.Float(), nullable=True),
        sa.Column("total_price", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("invoice_items")
    op.drop_table("invoices")
