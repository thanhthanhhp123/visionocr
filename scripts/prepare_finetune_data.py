"""
scripts/prepare_finetune_data.py
Build train/val/test JSONL splits from labeled data.
Run after validate_labels.py.

Output:
    datasets/train.jsonl   (~80%)
    datasets/val.jsonl     (~10%)
    datasets/test.jsonl    (~10%)
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas.invoice import InvoiceSchema  # noqa: E402

LABELS_DIR = Path("datasets/labels")
IMAGES_DIR = Path("datasets/images")
OUTPUT_DIR = Path("datasets")
RANDOM_SEED = 42

PROMPT_TEMPLATE = (
    "Trích xuất thông tin hóa đơn Việt Nam từ ảnh.\n"
    "{ocr_hint}"
    "Trả về JSON với format:\n"
    '{{"store_name":"","date":"YYYY-MM-DD","total":0,"discount":0,'
    '"items":[{{"name":"","unit_price":0,"quantity":0,"total_price":0}}]}}'
)


def find_image(stem: str) -> Path | None:
    for ext in [".jpg", ".jpeg", ".png"]:
        p = IMAGES_DIR / (stem + ext)
        if p.exists():
            return p
    return None


def build_sample(image_path: Path, label: dict) -> dict:
    ocr_text = label.pop("_ocr_text", "")
    ocr_hint = f"OCR text tham khảo:\n{ocr_text}\n\n" if ocr_text else ""

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.resolve())},
                    {"type": "text", "text": PROMPT_TEMPLATE.format(ocr_hint=ocr_hint)},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps(label, ensure_ascii=False)},
                ],
            },
        ]
    }


# Load samples
samples = []
skipped = []
invalid = []

for f in sorted(LABELS_DIR.glob("*.json")):
    label = json.loads(f.read_text(encoding="utf-8"))
    img = find_image(f.stem)
    if not img:
        skipped.append(f.stem)
        continue

    try:
        validated = InvoiceSchema.model_validate(label)
    except ValidationError:
        invalid.append(f.stem)
        continue

    # Giữ các biến thể augmentation của cùng một hóa đơn trong cùng
    # split để tránh leakage giữa train, validation và test.
    group_id = re.sub(r"_(?:aug\d+|orig)$", "", f.stem)
    samples.append((group_id, build_sample(img, validated.model_dump())))

print(
    f"Loaded valid: {len(samples)} | Invalid schema: {len(invalid)} | "
    f"Skipped (no image): {len(skipped)}"
)

# Shuffle & split by source invoice group
groups = defaultdict(list)
for group_id, sample in samples:
    groups[group_id].append(sample)

random.seed(RANDOM_SEED)
group_ids = list(groups)
random.shuffle(group_ids)

n_groups = len(group_ids)
n_val_groups = max(1, int(n_groups * 0.10))
n_test_groups = max(1, int(n_groups * 0.10))

train_group_ids = group_ids[: n_groups - n_val_groups - n_test_groups]
val_group_ids = group_ids[
    n_groups - n_val_groups - n_test_groups : n_groups - n_test_groups
]
test_group_ids = group_ids[n_groups - n_test_groups :]

train = [sample for group_id in train_group_ids for sample in groups[group_id]]
val = [sample for group_id in val_group_ids for sample in groups[group_id]]
test = [sample for group_id in test_group_ids for sample in groups[group_id]]

print(
    f"Groups — Train: {len(train_group_ids)} | Val: {len(val_group_ids)} | "
    f"Test: {len(test_group_ids)}"
)
print(f"Samples — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# Write JSONL
OUTPUT_DIR.mkdir(exist_ok=True)
for split_name, data in [("train", train), ("val", val), ("test", test)]:
    out = OUTPUT_DIR / f"{split_name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote  {out}")

print("\nDone. Ready for: make train")
