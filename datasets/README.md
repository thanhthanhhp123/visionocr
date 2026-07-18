# datasets/

This directory contains training data for VisionOCR.

## Structure

```
datasets/
├── images/        Invoice images (*.jpg, *.png) — NOT committed to git
├── labels/        JSON label files — NOT committed to git
├── sources/       Bản gốc tải từ Google Drive — NOT committed to git
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
make prepare   # build JSONL splits
```

> Không chạy `make validate` nếu chưa sao lưu dữ liệu và chưa có sự
> cho phép rõ ràng. Script hiện tại ghi đè labels và xóa các file không
> thể tự động làm sạch.

## Stats

- Dataset chính: 1.757 cặp ảnh–nhãn (cập nhật 2026-07-15).
- Nguồn `data_son.zip`: 1.051 labels, trong đó 614 ảnh được đóng gói
  lại; 437 ảnh còn lại đã có sẵn trong dataset cục bộ.
- Nguồn thư mục Google Drive `tung-2 (1)`: 706 cặp ảnh–nhãn bổ sung.
- Hai nhóm tên file không trùng nhau; sau khi gộp có 1.757 ảnh và
  1.757 labels khớp basename.
- Các file `train.jsonl`, `val.jsonl`, `test.jsonl` chưa được tạo lại từ
  dataset 1.757 cặp này.
