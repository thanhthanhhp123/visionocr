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
from pathlib import Path

LABELS_DIR  = Path("datasets/labels")
IMAGES_DIR  = Path("datasets/images")
OUTPUT_DIR  = Path("datasets")
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
                    {"type": "text",  "text": PROMPT_TEMPLATE.format(ocr_hint=ocr_hint)},
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

for f in sorted(LABELS_DIR.glob("*.json")):
    label = json.loads(f.read_text(encoding="utf-8"))
    img   = find_image(f.stem)
    if img:
        samples.append(build_sample(img, label))
    else:
        skipped.append(f.stem)

print(f"Loaded: {len(samples)} | Skipped (no image): {len(skipped)}")

# Shuffle & split
random.seed(RANDOM_SEED)
random.shuffle(samples)

n     = len(samples)
n_val = max(1, int(n * 0.10))
n_tst = max(1, int(n * 0.10))

train = samples[:n - n_val - n_tst]
val   = samples[n - n_val - n_tst : n - n_tst]
test  = samples[n - n_tst:]

print(f"Split  — Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# Write JSONL
OUTPUT_DIR.mkdir(exist_ok=True)
for split_name, data in [("train", train), ("val", val), ("test", test)]:
    out = OUTPUT_DIR / f"{split_name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote  {out}")

print("\nDone. Ready for: make train")
