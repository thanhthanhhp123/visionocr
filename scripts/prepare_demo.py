"""
scripts/prepare_demo.py — Pre-run OCR+VLM extraction on a few held-out test
images and persist the results to visionocr.db, so the Streamlit UI can be
demoed from a laptop (no GPU) by just copying the DB + demo images over.

Run this on the HPC GPU node, from the repo root, with the thanhocr venv:
    /home/s3002152/LeeHoang_/vlm_invoice/thanhocr/bin/python scripts/prepare_demo.py --count 6

Then from your laptop:
    scp s3002152@hpc-head1.ewi.utwente.nl:/home/s3002152/LeeHoang_/vlm_invoice/visionocr/visionocr.db .
    scp -r s3002152@hpc-head1.ewi.utwente.nl:/home/s3002152/LeeHoang_/vlm_invoice/visionocr/datasets/demo_images .
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEMO_IMAGES_DIR = PROJECT_ROOT / "datasets" / "demo_images"


def _iter_test_images(count: int) -> list[str]:
    test_jsonl = PROJECT_ROOT / "datasets" / "test.jsonl"
    paths = []
    with open(test_jsonl, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            image_path = record["messages"][0]["content"][0]["image"]
            paths.append(image_path)
    return paths[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6)
    args = parser.parse_args()

    from ocr.paddle_engine import extract_text
    from vlm.inference import backend_name, extract_invoice

    from api.schemas.invoice import InvoiceSchema
    from db.repository import create_invoice
    from db.session import SessionLocal, init_db

    init_db()
    DEMO_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {backend_name()}")
    images = _iter_test_images(args.count)
    print(f"Extracting {len(images)} demo invoices...")

    ok, failed = 0, 0
    with SessionLocal() as db:
        for i, image_path in enumerate(images, 1):
            filename = Path(image_path).name
            print(f"[{i}/{len(images)}] {filename} ...", end=" ", flush=True)
            try:
                start = time.time()
                ocr_text = extract_text(image_path)
                result = extract_invoice(image_path, ocr_text)
                invoice = InvoiceSchema.model_validate(result)
                elapsed = round((time.time() - start) * 1000, 1)

                create_invoice(db, invoice, filename, elapsed)
                shutil.copy(image_path, DEMO_IMAGES_DIR / filename)

                print(f"ok ({elapsed}ms)")
                ok += 1
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1

    print(f"\nDone: {ok} ok, {failed} failed.")
    print(f"DB: {PROJECT_ROOT / 'visionocr.db'}")
    print(f"Images: {DEMO_IMAGES_DIR}")


if __name__ == "__main__":
    main()
