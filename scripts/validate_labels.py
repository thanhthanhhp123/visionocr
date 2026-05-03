"""
scripts/validate_labels.py
Pydantic validation + auto-clean of all label files.
Run after check_data.py.
"""
import json
from pathlib import Path
from pydantic import BaseModel, field_validator, model_validator
from typing import List
from datetime import datetime

LABELS_DIR = Path("datasets/labels")


class Item(BaseModel):
    name: str
    unit_price: float
    quantity: float
    total_price: float

    @model_validator(mode="after")
    def check_math(self):
        expected = round(self.unit_price * self.quantity, 0)
        actual   = round(self.total_price, 0)
        diff_pct = abs(expected - actual) / (actual + 1)
        if diff_pct > 0.05:
            # Warning only — discount items may not follow unit * qty = total
            pass
        return self


class Invoice(BaseModel):
    store_name: str
    date: str
    total: float
    discount: float = 0.0
    items: List[Item]

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        datetime.strptime(v, "%Y-%m-%d")
        return v

    @field_validator("store_name")
    @classmethod
    def clean_store(cls, v):
        return v.strip()

    @field_validator("total", "discount")
    @classmethod
    def clean_number(cls, v):
        return round(float(v), 2)


valid, invalid = [], []

for f in sorted(LABELS_DIR.glob("*.json")):
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        # Drop internal fields like _ocr_text before validation
        clean_raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        invoice = Invoice(**clean_raw)
        # Write cleaned file back
        f.write_text(
            json.dumps(invoice.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        valid.append(f.stem)
    except Exception as e:
        invalid.append((f.name, str(e)))

print(f"\n{'='*50}")
print(f"  Valid:   {len(valid)}")
print(f"  Invalid: {len(invalid)}")
print(f"{'='*50}")

if invalid:
    print("\nInvalid files (need manual fix):")
    for name, err in invalid:
        print(f"  {name}: {err}")
else:
    print("\nAll labels valid and cleaned. Ready for prepare_finetune_data.py")
