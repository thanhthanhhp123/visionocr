"""
frontend/app.py — Streamlit dashboard for VisionOCR
Run: streamlit run frontend/app.py
"""

import streamlit as st
import httpx
import time
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1").rstrip("/")

st.set_page_config(
    page_title="VisionOCR — Invoice Intelligence",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 VisionOCR — Invoice Intelligence Platform")
st.caption("Upload a Vietnamese retail invoice to extract structured information")

# ── Upload ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload invoice image", type=["jpg", "jpeg", "png"])
    if uploaded:
        st.image(uploaded, caption="Uploaded invoice", use_container_width=True)

with col2:
    if uploaded:
        if st.button("🔍 Extract", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                # Send to API
                start = time.time()
                try:
                    resp = httpx.post(
                        f"{API_BASE}/extract",
                        files={
                            "file": (uploaded.name, uploaded.getvalue(), uploaded.type)
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    task_id = data["task_id"]

                    # Poll for result
                    result = {"status": "pending"}
                    for _ in range(90):
                        time.sleep(2)
                        poll = httpx.get(f"{API_BASE}/tasks/{task_id}", timeout=10)
                        poll.raise_for_status()
                        result = poll.json()
                        if result["status"] in ("success", "failed"):
                            break

                    elapsed = round((time.time() - start) * 1000)

                    if result["status"] == "success" and result.get("result"):
                        inv = result["result"]
                        st.success(f"Extracted in {elapsed}ms")

                        # Show result
                        st.subheader("📋 Invoice Details")
                        st.metric("Store", inv.get("store_name", "—"))
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Date", inv.get("date", "—"))
                        c2.metric("Total", f"{inv.get('total', 0):,.0f}đ")
                        c3.metric("Discount", f"{inv.get('discount', 0):,.0f}đ")

                        st.subheader("🛒 Items")
                        items = inv.get("items", [])
                        if items:
                            st.dataframe(
                                [
                                    {
                                        "Name": i["name"],
                                        "Unit Price": f"{i['unit_price']:,.0f}đ",
                                        "Quantity": i["quantity"],
                                        "Total": f"{i['total_price']:,.0f}đ",
                                    }
                                    for i in items
                                ],
                                use_container_width=True,
                            )

                        with st.expander("Raw JSON"):
                            st.json(inv)
                    elif result["status"] == "pending":
                        st.warning(
                            "Task is still processing. Please try again in a moment."
                        )
                    else:
                        st.error(
                            f"Extraction failed: {result.get('error', 'Unknown error')}"
                        )

                except Exception as e:
                    st.error(f"API error: {e}")
    else:
        st.info("Upload an invoice image to get started")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "VisionOCR · Qwen2.5-VL-3B + LoRA + AWQ + vLLM · Built for AI/ML Engineer portfolio"
)
