# scripts/validate_labels.py — version mới, thay toàn bộ file cũ

import json
from pathlib import Path
from datetime import datetime

LABELS_DIR = Path("datasets/labels")

def try_fix_date(v: str) -> str | None:
    for fmt in ["%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d/%m/%y"]:
        try:
            return datetime.strptime(v.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None

def clean_number(v) -> float:
    if v is None:
        return 0.0
    try:
        return round(float(str(v).replace(",", ".")), 2)
    except (TypeError, ValueError):
        return 0.0

def clean_invoice(raw: dict) -> dict | None:
    """Auto-fix tất cả lỗi null, format sai. Return None nếu không cứu được."""
    
    # store_name
    store = raw.get("store_name")
    if not store or not isinstance(store, str):
        store = "UNKNOWN"
    raw["store_name"] = store.strip()

    # date
    date = raw.get("date", "")
    if date:
        try:
            datetime.strptime(date, "%Y-%m-%d")  # đã đúng
        except ValueError:
            fixed = try_fix_date(str(date))
            if fixed:
                raw["date"] = fixed
            else:
                return None  # date không fix được → bỏ
    else:
        return None  # không có date → bỏ

    # total
    raw["total"] = clean_number(raw.get("total"))
    if raw["total"] <= 0:
        return None  # total = 0 → bỏ

    # discount: null → 0.0
    raw["discount"] = clean_number(raw.get("discount"))

    # items
    items = raw.get("items")
    if not isinstance(items, list):
        items = []

    clean_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        if not name:
            continue
        clean_items.append({
            "name": str(name).strip(),
            "unit_price":  clean_number(item.get("unit_price")),
            "quantity":    clean_number(item.get("quantity")) or 1.0,  # null → 1
            "total_price": clean_number(item.get("total_price")),
        })

    if not clean_items:
        return None  # không có items → bỏ

    raw["items"] = clean_items
    return raw


valid, fixed_and_saved, dropped = [], [], []

for f in sorted(LABELS_DIR.glob("*.json")):
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
        # Bỏ internal fields
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        
        cleaned = clean_invoice(raw)
        
        if cleaned is None:
            dropped.append(f.name)
            f.unlink()  # xoá file không cứu được
        else:
            f.write_text(
                json.dumps(cleaned, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            valid.append(f.name)

    except Exception as e:
        dropped.append(f"{f.name} (exception: {e})")
        f.unlink()

print(f"\n{'='*50}")
print(f"  Valid + saved : {len(valid)}")
print(f"  Dropped       : {len(dropped)}")
print(f"{'='*50}")

if dropped:
    print("\nDropped files:")
    for d in dropped:
        print(f"  {d}")

print("\nDone. Run: make prepare")
