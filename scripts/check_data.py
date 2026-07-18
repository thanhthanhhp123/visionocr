"""
scripts/check_data.py
Run FIRST before any other data script.
Checks all label files have corresponding images and valid structure.
"""

import json
from pathlib import Path

LABELS_DIR = Path("datasets/labels")
IMAGES_DIR = Path("datasets/images")


def find_image(stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png"]:
        p = IMAGES_DIR / (stem + ext)
        if p.exists():
            return p
    return None


ok, errors = [], []

for f in sorted(LABELS_DIR.glob("*.json")):
    try:
        data = json.loads(f.read_text(encoding="utf-8"))

        img = find_image(f.stem)
        if not img:
            raise FileNotFoundError(f"No image found for {f.stem}")

        assert "store_name" in data, "Missing store_name"
        assert "date" in data, "Missing date"
        assert len(data["date"]) == 10, f"Bad date format: {data['date']}"
        assert "total" in data, "Missing total"
        assert float(data["total"]) > 0, "Total must be > 0"
        assert isinstance(data.get("items"), list), "items must be a list"
        assert len(data["items"]) > 0, "items must not be empty"

        for i, item in enumerate(data["items"]):
            assert "name" in item, f"Item {i}: missing name"
            assert "unit_price" in item, f"Item {i}: missing unit_price"
            assert "quantity" in item, f"Item {i}: missing quantity"
            assert "total_price" in item, f"Item {i}: missing total_price"

        ok.append(f.stem)

    except Exception as e:
        errors.append((f.name, str(e)))

print(f"\n{'=' * 50}")
print(f"  OK:    {len(ok)}")
print(f"  Error: {len(errors)}")
print(f"  Total: {len(ok) + len(errors)}")
print(f"{'=' * 50}")

if errors:
    print("\nFiles with errors:")
    for name, err in errors:
        print(f"  {name}: {err}")
else:
    print("\nAll files valid. Ready for validate_labels.py")
