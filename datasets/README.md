# datasets/

This directory contains training data for VisionOCR.

## Structure

```
datasets/
├── images/        Invoice images (*.jpg, *.png) — NOT committed to git
├── labels/        JSON label files — NOT committed to git
├── train.jsonl    Training split (~80%) — NOT committed to git
├── val.jsonl      Validation split (~10%) — NOT committed to git
└── test.jsonl     Test split (~10%) — NOT committed to git
```

## Invoice label format

```json
{
  "store_name": "VinCommerce",
  "date": "2024-01-15",
  "total": 81302,
  "discount": 8800,
  "items": [
    {
      "name": "Bơ đặc biệt",
      "unit_price": 52700,
      "quantity": 0.408,
      "total_price": 21502
    }
  ]
}
```

## Data pipeline

```bash
make check     # verify all labels have corresponding images
make validate  # Pydantic validation + auto-clean
make prepare   # build JSONL splits
```

## Stats

- Total labeled: 261 invoices
- Train: ~208 | Val: ~26 | Test: ~26
- Sources: Vietnamese retail (VinMart, Co.opmart, Bách Hoá Xanh...)
