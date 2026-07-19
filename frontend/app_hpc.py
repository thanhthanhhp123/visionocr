"""
frontend/app_hpc.py — Standalone VisionOCR web app for running directly on
an HPC GPU node (no FastAPI/Celery/Redis/Postgres — those aren't available
on a shared login node without root). OCR + VLM inference run in-process;
results persist to a local SQLite file via db/repository.py.

Run on the GPU node (inside a SLURM allocation):
    streamlit run frontend/app_hpc.py --server.port 8501 --server.address 0.0.0.0

SSH port-forwarding is blocked on this cluster, so the above can't be reached
from another machine. For demos, run scripts/prepare_demo.py on the GPU node
to pre-populate visionocr.db + datasets/demo_images, scp both to a laptop,
then run this app there with VISIONOCR_DISPLAY_ONLY=1 (no GPU/model needed —
just browses the pre-computed results):
    VISIONOCR_DISPLAY_ONLY=1 streamlit run frontend/app_hpc.py
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DISPLAY_ONLY = os.getenv("VISIONOCR_DISPLAY_ONLY", "0") == "1"
DEMO_IMAGES_DIR = PROJECT_ROOT / "datasets" / "demo_images"

st.set_page_config(
    page_title="VisionOCR — Invoice Intelligence (HPC)",
    page_icon="🧾",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading OCR + VLM models onto GPU (first run only)...")
def _load_pipeline():
    from ocr.paddle_engine import extract_text
    from vlm.inference import backend_name, extract_invoice

    return extract_text, extract_invoice, backend_name()


def _render_extract_tab() -> None:
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("Upload invoice image", type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded invoice", use_container_width=True)

    with col2:
        if not uploaded:
            st.info("Upload an invoice image to get started")
            return
        if not st.button("🔍 Extract", type="primary", use_container_width=True):
            return

        extract_text, extract_invoice, backend = _load_pipeline()
        st.caption(f"Backend: `{backend}`")

        tmp_path = None
        try:
            with st.spinner("Running OCR + VLM extraction..."):
                start = time.time()

                suffix = ".jpg" if "jpg" in uploaded.type else ".png"
                tmp_path = str(PROJECT_ROOT / f".tmp_upload{suffix}")
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getvalue())

                ocr_text = extract_text(tmp_path)
                result = extract_invoice(tmp_path, ocr_text)

                from api.schemas.invoice import InvoiceSchema

                invoice = InvoiceSchema.model_validate(result)
                elapsed = round((time.time() - start) * 1000)

                from db.repository import create_invoice
                from db.session import SessionLocal, init_db

                init_db()
                with SessionLocal() as db:
                    create_invoice(db, invoice, uploaded.name, elapsed)

            st.success(f"Extracted in {elapsed}ms")
            _render_invoice_details(invoice.store_name, invoice.date, invoice.total, invoice.discount, invoice.items)
            with st.expander("Raw JSON"):
                st.json(invoice.model_dump())

        except Exception as e:
            st.error(f"Extraction failed: {e}")
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)


def _render_invoice_details(store_name, date, total, discount, items) -> None:
    st.subheader("📋 Invoice Details")
    st.metric("Store", store_name or "—")
    c1, c2, c3 = st.columns(3)
    c1.metric("Date", date or "—")
    c2.metric("Total", f"{total:,.0f}đ")
    c3.metric("Discount", f"{discount:,.0f}đ")

    st.subheader("🛒 Items")
    if items:
        st.dataframe(
            [
                {
                    "Name": i.name,
                    "Unit Price": f"{i.unit_price:,.0f}đ",
                    "Quantity": i.quantity,
                    "Total": f"{i.total_price:,.0f}đ",
                }
                for i in items
            ],
            use_container_width=True,
        )


def _render_history_tab() -> None:
    from db.repository import list_invoices
    from db.session import SessionLocal, init_db

    init_db()
    with SessionLocal() as db:
        records = list_invoices(db, limit=50)

    if not records:
        st.info("No invoices extracted yet.")
        return

    st.dataframe(
        [
            {
                "Created": r.created_at,
                "File": r.filename,
                "Store": r.store_name,
                "Date": r.date,
                "Total": r.total,
                "Latency (ms)": r.latency_ms,
            }
            for r in records
        ],
        use_container_width=True,
    )

    st.divider()
    labels = [f"{r.created_at} — {r.store_name or r.filename}" for r in records]
    choice = st.selectbox("View details", options=range(len(records)), format_func=lambda i: labels[i])
    record = records[choice]

    col1, col2 = st.columns([1, 1])
    with col1:
        image_path = DEMO_IMAGES_DIR / (record.filename or "")
        if record.filename and image_path.exists():
            st.image(str(image_path), caption=record.filename, use_container_width=True)
        else:
            st.caption("(source image not bundled with this DB copy)")

    with col2:
        from db.repository import as_schema

        invoice = as_schema(record)
        _render_invoice_details(invoice.store_name, invoice.date, invoice.total, invoice.discount, invoice.items)
        with st.expander("Raw JSON"):
            st.json(invoice.model_dump())


st.title("🧾 VisionOCR — Invoice Intelligence Platform")

if DISPLAY_ONLY:
    st.caption("Display-only mode: browsing results pre-computed on the HPC GPU node.")
    (tab_history,) = st.tabs(["History"])
    with tab_history:
        _render_history_tab()
else:
    st.caption("Standalone mode: OCR + VLM run in this process, on this GPU node.")
    tab_extract, tab_history = st.tabs(["Extract", "History"])
    with tab_extract:
        _render_extract_tab()
    with tab_history:
        _render_history_tab()

st.divider()
st.caption("VisionOCR · Qwen2.5-VL-3B + LoRA · Running standalone on HPC GPU node")
