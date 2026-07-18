# Báo cáo dự án VisionOCR

> Hệ thống trích xuất thông tin có cấu trúc từ hóa đơn bán lẻ Việt Nam bằng
> Qwen2.5-VL, LoRA và OCR.

**Tác giả:** [Điền họ tên]  
**MSSV:** [Điền MSSV]  
**Môn học / Đơn vị:** [Điền thông tin]  
**Ngày hoàn thành:** [Điền ngày]

---

## Tóm tắt

VisionOCR là hệ thống nhận ảnh hóa đơn bán lẻ Việt Nam và trả về dữ liệu JSON
có cấu trúc gồm tên cửa hàng, ngày hóa đơn, tổng tiền, giảm giá và các mặt hàng.
Hệ thống kết hợp PaddleOCR để cung cấp văn bản tham khảo với Qwen2.5-VL-3B được
fine-tune bằng LoRA. Phiên bản LoRA v2 được đánh giá trên tập kiểm thử 87 mẫu
theo cách chia không rò rỉ dữ liệu giữa các biến thể augmentation của cùng hóa
đơn gốc.

**Từ khóa:** OCR, Vision Language Model, Qwen2.5-VL, LoRA, invoice extraction,
Vietnamese receipts.

---

## 1. Giới thiệu

### 1.1. Bối cảnh

Nhập liệu hóa đơn giấy vào hệ thống kế toán/quản lý bán hàng hiện vẫn chủ yếu
làm thủ công tại các cửa hàng vừa và nhỏ ở Việt Nam, tốn thời gian và dễ sai
sót. Hóa đơn bán lẻ Việt Nam gây khó khăn đặc thù cho OCR truyền thống:

- Layout không thống nhất giữa các chuỗi cửa hàng (siêu thị, tiệm hoa, quán
  ăn, tiệm giặt ủi...), không có template cố định.
- Chất lượng ảnh chụp bằng điện thoại không đồng đều: nghiêng, mờ, thiếu
  sáng, nhàu giấy.
- Tiếng Việt có dấu, nhiều hóa đơn dùng chữ in hoa toàn bộ cho tên cửa hàng,
  dễ khiến OCR nhận sai dấu.
- Ký hiệu tiền tệ (đ, VNĐ), dấu phân cách hàng nghìn và danh sách mặt hàng dài
  với số lượng thập phân (ví dụ hàng cân theo kg) làm tăng độ khó khi trích
  xuất số liệu có cấu trúc.

Một pipeline kết hợp OCR truyền thống làm gợi ý văn bản và một Vision Language
Model (VLM) có khả năng đọc trực tiếp bố cục ảnh là hướng tiếp cận phù hợp để
xử lý các khó khăn này mà không cần xây template riêng cho từng chuỗi cửa
hàng.

### 1.2. Bài toán

Đầu vào là một ảnh hóa đơn. Đầu ra là dữ liệu có cấu trúc theo schema sau:

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

### 1.3. Mục tiêu

- Xây dựng pipeline trích xuất hóa đơn từ ảnh sang JSON.
- Fine-tune Vision Language Model cho hóa đơn tiếng Việt.
- So sánh baseline zero-shot với model LoRA.
- Triển khai kiến trúc API bất đồng bộ, database và giao diện người dùng.

---

## 2. Dữ liệu

### 2.1. Nguồn dữ liệu

- Hóa đơn bán lẻ Việt Nam từ các chuỗi như VinMart, Co.opmart, Bách Hóa Xanh,
  [bổ sung các nguồn khác nếu có].
- Mỗi mẫu bao gồm ảnh hóa đơn và một file JSON nhãn có cùng basename.

### 2.2. Quy mô và chia tập

| Hạng mục | Số lượng |
|---|---:|
| Ảnh và labels dataset chính | 1.757 cặp |
| Mẫu đạt schema dùng cho LoRA v2 | 1.017 |
| Train | 820 |
| Validation | 110 |
| Test | 87 |

Việc chia train/validation/test được thực hiện theo hóa đơn gốc. Các bản
augmentation của cùng một hóa đơn luôn thuộc cùng một split để tránh data
leakage.

### 2.3. Tiền xử lý và kiểm soát chất lượng

Pipeline dữ liệu gồm ba bước tuần tự (`scripts/check_data.py` →
`scripts/validate_labels.py` → `scripts/prepare_finetune_data.py`):

1. **`check_data.py`** (read-only) đối chiếu từng file nhãn với ảnh cùng
   basename và kiểm tra các trường bắt buộc (`store_name`, `date` đúng 10 ký
   tự, `total` > 0, `items` không rỗng và đủ 4 trường mỗi mặt hàng). Trên
   1.757 cặp ảnh–nhãn hiện có: 1.436 nhãn đạt kiểm tra cơ bản, 321 lỗi (chủ
   yếu thiếu `store_name` hoặc `date` rỗng).
2. **`validate_labels.py`** chuẩn hóa định dạng ngày/số tiền và làm sạch các
   nhãn có thể tự sửa được; đây là bước duy nhất ghi đè/xóa dữ liệu nên chỉ
   chạy khi có sao lưu và được xác nhận rõ ràng.
3. **`prepare_finetune_data.py`** lọc lại bằng `InvoiceSchema` (Pydantic) —
   chặt hơn `check_data.py` vì kiểm tra kiểu dữ liệu từng trường và định dạng
   ngày `strptime` — rồi build JSONL theo định dạng hội thoại (`messages`)
   dùng cho fine-tuning. Từ 1.757 cặp, 1.017 mẫu đạt schema; 740 mẫu không
   đạt bị loại khỏi tập huấn luyện.

Để tránh rò rỉ dữ liệu, việc chia train/val/test được thực hiện theo **nhóm
hóa đơn gốc** thay vì theo từng mẫu: hậu tố `_aug<N>`/`_orig` bị loại bỏ khỏi
tên file bằng regex để xác định `group_id`, sau đó toàn bộ mẫu (ảnh gốc và
các bản augmentation) của cùng một hóa đơn luôn nằm trong cùng một split.
Việc chia nhóm dùng `random.seed(42)` theo tỉ lệ ~80/10/10, cho kết quả
107/13/13 nhóm hóa đơn tương ứng 820/110/87 mẫu (đã xác minh lại việc rebuild
JSONL trên toàn bộ 1.757 cặp cho kết quả giống hệt tập đang dùng để đánh giá).

> Lưu ý: Phase dữ liệu được xem là hoàn tất theo phạm vi dự án. Trong quá trình
> train LoRA v2, chỉ các nhãn khớp schema được đưa vào tập 1.017 mẫu.

---

## 3. Phương pháp đề xuất

### 3.1. Kiến trúc tổng thể

```text
Invoice image
    │
    ├── PaddleOCR ──► OCR text hint
    │
    └── Qwen2.5-VL-3B + LoRA v2
                      │
                      ▼
               Structured invoice JSON
                      │
                      ▼
         FastAPI ─ Celery/Redis ─ PostgreSQL
                      │
                      ▼
             Streamlit dashboard
```

### 3.2. OCR và Vision Language Model

PaddleOCR trích xuất văn bản tham khảo từ ảnh. Qwen2.5-VL nhận đồng thời ảnh
và prompt tiếng Việt, sau đó sinh JSON theo schema đầu ra. Mô hình trực tiếp
quan sát bố cục ảnh nên có thể kết hợp vị trí không gian và văn bản OCR.

### 3.3. LoRA fine-tuning

| Siêu tham số | Giá trị |
|---|---:|
| Base model | Qwen2.5-VL-3B-Instruct |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0,05 |
| Learning rate | 5e-5 |
| Epochs | 5 |
| Batch size / GPU | 1 |
| Gradient accumulation | 8 |
| Quantization khi train | NF4 4-bit |

LoRA được áp dụng lên các projection layers `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj` và `down_proj`.

### 3.4. Hậu xử lý JSON

Kết quả sinh được parse theo ba chiến lược: parse JSON trực tiếp, trích xuất
object JSON nhúng trong văn bản và loại bỏ Markdown code fence. API tiếp tục
validate kết quả bằng Pydantic trước khi lưu database.

---

## 4. Thiết kế và triển khai hệ thống

### 4.1. Backend

- FastAPI cung cấp endpoint upload ảnh, polling task và tra cứu hóa đơn.
- Celery/Redis xử lý inference bất đồng bộ.
- Ảnh được Base64-encode trước khi đưa vào JSON task payload.
- PostgreSQL lưu hóa đơn, line items, JSON gốc và latency.
- Alembic quản lý migration database.

### 4.2. Frontend

Streamlit cho phép upload ảnh, theo dõi trạng thái tác vụ, hiển thị chỉ số chính,
bảng sản phẩm và JSON thô.

### 4.3. Triển khai

Docker Compose định nghĩa các service: PostgreSQL, Redis, API, Celery worker,
MLflow và Streamlit. [Bổ sung ảnh chụp Docker dashboard hoặc kiến trúc triển
khai nếu có.]

---

## 5. Thiết lập thí nghiệm

### 5.1. Baseline

Qwen2.5-VL-3B-Instruct được chạy zero-shot với cùng prompt và cùng tập test
87 mẫu.

### 5.2. Model đề xuất

Qwen2.5-VL-3B-Instruct được fine-tune bằng LoRA v2 trên 820 mẫu train; 110 mẫu
validation dùng để theo dõi loss; 87 mẫu test dùng để đánh giá cuối cùng.

### 5.3. Metrics

- Store Name Accuracy: tỉ lệ tên cửa hàng khớp chính xác.
- Date Accuracy: tỉ lệ ngày hóa đơn khớp sau khi chuẩn hóa định dạng.
- Total Amount Accuracy: tỉ lệ tổng tiền sai số không quá 1%.
- Item Name F1: F1 trung bình của tên mặt hàng với token-sort similarity.
- JSON Parse Rate: tỉ lệ output parse được thành JSON.
- Average Latency: thời gian inference trung bình trên một ảnh.

---

## 6. Kết quả và thảo luận

### 6.1. So sánh định lượng

| Metric | Zero-shot | LoRA v2 | Thay đổi |
|---|---:|---:|---:|
| Store Name Accuracy | 79,3% | 88,5% | +9,2 điểm % |
| Date Accuracy | 87,4% | 90,8% | +3,4 điểm % |
| Total Amount Accuracy | 23,0% | 90,8% | +67,8 điểm % |
| Item Name F1 | 0,931 | 0,923 | -0,008 |
| JSON Parse Rate | 98,9% | 98,9% | 0,0 điểm % |
| Latency trung bình | 16,77 giây | 36,59 giây | +19,82 giây |

### 6.2. Nhận xét

- LoRA v2 cải thiện mạnh khả năng nhận diện tổng tiền, từ 23,0% lên 90,8%.
- Accuracy cho tên cửa hàng và ngày hóa đơn cũng tăng.
- Item Name F1 không tăng; cần phân tích thêm tính đa dạng của tên hàng, chất
  lượng nhãn và chiến lược matching.
- LoRA làm inference chậm hơn baseline trong Hugging Face 4-bit path. Model đã
  được merge; AWQ/vLLM là hướng tối ưu latency tiếp theo.

### 6.3. Phân tích lỗi

Phân tích trực tiếp trên `results_finetuned_v2.json` (87 mẫu, so khớp từng
trường giữa `gold` và `pred`):

**JSON parse thất bại (1/87, mẫu #34 — "TIỆM GIẶT ỦI SẠCH & THƠM").** Đây là
hóa đơn dịch vụ (giặt ủi) với danh sách 5+ mặt hàng có tên dài; model không
đóng đúng cấu trúc JSON. Tỷ lệ 1/87 (98.9%) là chấp nhận được nhưng cho thấy
output cần luôn được bọc bởi validate + retry ở tầng API thay vì tin tưởng
tuyệt đối, đúng như thiết kế hiện tại của `worker/tasks.py`
(`InvoiceSchema.model_validate` chặn trước khi lưu DB).

**Sai tên cửa hàng (9/87).** Gần như toàn bộ là lỗi chính tả có dấu khi tên
cửa hàng được viết IN HOA toàn bộ trên hóa đơn gốc, ví dụ:

| Gold | Pred |
|---|---|
| `TIỆM BÁNH & CÀ PHÊ NẮNG THÁNG MƯỜI` | `Tiệm Bánh & Cà Phê Nắng Thắng Mười` |
| `TIỆM BÁNH & CÀ PHÊ NẮNG THÁNG MƯỜI` | `Tiệm Bánh & Cà Phê Nắng Thang Mười` |
| `Lá em flower & decor` | `Lê em flower & decor` |

Model có xu hướng tự "chuẩn hóa" chữ hoa toàn bộ thành viết hoa chữ cái đầu
(Title Case) và trong quá trình đó đoán sai dấu tiếng Việt (`Tháng` → `Thắng`/
`Thang`). Đây là cùng một cửa hàng lặp lại ở nhiều mẫu augmentation liên tiếp
(#39–44) với cùng kiểu lỗi — cho thấy lỗi mang tính hệ thống theo từng hóa đơn
gốc chứ không ngẫu nhiên theo từng ảnh.

**Sai ngày (7/87).** Chủ yếu nhầm giữa các chữ số có hình dạng gần giống nhau
khi viết tay/in mờ: `2023` ↔ `2025`, ngày `25` ↔ `26`, `18` ↔ `12`. Toàn bộ
7 ca đều rơi vào các mẫu augmentation của một số ít hóa đơn gốc, củng cố giả
thuyết lỗi theo nhóm hóa đơn (chất lượng ảnh gốc kém) hơn là lỗi ngẫu nhiên
của model.

**Sai tổng tiền (7/87).** Đáng chú ý là mẫu #33 và #35–38 (cùng một hóa đơn
gốc, các bản augmentation khác nhau) đều lệch đúng 20.000đ theo cùng một
hướng (`236.000` → `216.000`) — khớp với phần `discount` bị đọc nhầm/bỏ sót
trên hóa đơn đó, chứ không phải lỗi tính toán ngẫu nhiên. Điều này gợi ý cải
thiện bằng cách kiểm tra ràng buộc `total = Σ(item.total_price) − discount`
ở tầng hậu xử lý thay vì chỉ tin vào giá trị `total` model sinh ra trực tiếp.

**Kết luận phân tích lỗi:** phần lớn lỗi còn lại không rải rác ngẫu nhiên mà
tập trung theo từng hóa đơn gốc khó (chữ viết tay, chữ in hoa toàn bộ, ảnh
mờ) và lặp lại trên toàn bộ các bản augmentation của hóa đơn đó — đây cũng là
lý do Item Name F1 không cải thiện dù các trường số liệu cải thiện mạnh. Hai
hướng cải thiện rõ ràng nhất: (1) rà soát/gắn nhãn lại các hóa đơn gốc gây
lỗi hệ thống thay vì tăng số lượng augmentation của chúng, và (2) thêm ràng
buộc hậu xử lý đối chiếu `total` với tổng `items` + `discount`.

---

## 7. Hạn chế và hướng phát triển

- Hoàn thiện AWQ quantization và benchmark vLLM. Bốn lần chạy job SLURM đầu
  tiên thất bại do AutoAWQ (đã deprecated, không còn được bảo trì) giả định
  cấu trúc `model.model.layers` của một phiên bản `transformers` cũ hơn,
  trong khi bản `transformers` hiện cài đặt đã chuyển decoder layers sang
  `model.model.language_model.layers`; đã vá lỗi này trong
  `vlm/finetune/merge_and_quantize.py` và job đã được submit lại.
- Kiểm tra end-to-end Docker trên GPU host có NVIDIA Container Runtime.
- Bổ sung demo GIF/video từ luồng upload thực tế.
- Rà soát 321/1.757 nhãn còn lỗi (thiếu `store_name`/`date`) và 740/1.757
  nhãn không đạt `InvoiceSchema` để tăng số mẫu train khả dụng, thay vì tiếp
  tục augment các hóa đơn đã có nhãn tốt.
- Thêm ràng buộc hậu xử lý đối chiếu `total` với tổng `items.total_price` và
  `discount` (xem phân tích lỗi §6.3) để giảm các sai số tổng tiền mang tính
  hệ thống.
- Active learning trên các hóa đơn gốc gây lỗi lặp lại theo nhóm augmentation
  (§6.3), thay vì lấy mẫu ngẫu nhiên cho vòng gán nhãn tiếp theo.
- Mở rộng độ đa dạng nguồn hóa đơn (thêm chuỗi bán lẻ, hóa đơn viết tay) và
  cân nhắc confidence score / human-in-the-loop cho các trường số liệu quan
  trọng (`total`, `date`) trước khi lưu vào hệ thống kế toán thực tế.

---

## 8. Kết luận

VisionOCR xây dựng thành công một pipeline đầu-cuối trích xuất dữ liệu có cấu
trúc từ ảnh hóa đơn bán lẻ Việt Nam, kết hợp PaddleOCR làm gợi ý văn bản với
Qwen2.5-VL-3B fine-tune bằng LoRA để tận dụng khả năng đọc trực tiếp bố cục
ảnh của VLM. Trên tập kiểm thử 87 mẫu không rò rỉ dữ liệu, LoRA v2 cải thiện
rõ rệt các trường có cấu trúc chặt so với zero-shot — đặc biệt tổng tiền
(23.0% → 90.8%) và tên cửa hàng (79.3% → 88.5%) — trong khi tỷ lệ parse JSON
giữ ở mức cao (98.9%). Phân tích lỗi cho thấy phần lớn sai số còn lại không
ngẫu nhiên mà tập trung ở một số hóa đơn gốc khó (chữ viết tay, ảnh mờ, tên
cửa hàng viết hoa toàn bộ), gợi ý hướng cải thiện tiếp theo nên ưu tiên rà
soát dữ liệu và ràng buộc hậu xử lý hơn là tăng quy mô augmentation.

Về mặt hệ thống, dự án triển khai đầy đủ kiến trúc bất đồng bộ thực tế
(FastAPI + Celery/Redis + PostgreSQL + Alembic + Streamlit, có test tự động
cho tầng API/persistence) thay vì chỉ dừng ở một script inference đơn lẻ.
Giới hạn còn lại là latency của đường suy luận HF + LoRA 4-bit (36.6s/ảnh)
chưa được benchmark với đường AWQ + vLLM do vướng lỗi tương thích thư viện
(nay đã vá) và việc xác minh runtime Docker/GPU đầy đủ chưa thực hiện được
trên môi trường phát triển hiện tại. Trong phạm vi một dự án portfolio nhắm
tới vị trí AI/ML Engineer Fresher, hệ thống chứng minh được năng lực triển
khai một pipeline VLM fine-tuning + MLOps + backend bất đồng bộ hoàn chỉnh,
có đo lường định lượng rõ ràng thay vì demo minh họa đơn thuần.

---

## Tài liệu tham khảo

1. Qwen Team. *Qwen2.5-VL Technical Report*. [Bổ sung URL/DOI chính xác].
2. Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. 2021.
   https://arxiv.org/abs/2106.09685
3. PaddleOCR documentation. https://github.com/PaddlePaddle/PaddleOCR
4. FastAPI documentation. https://fastapi.tiangolo.com/
5. Celery documentation. https://docs.celeryq.dev/
6. PostgreSQL documentation. https://www.postgresql.org/docs/
7. MLflow documentation. https://mlflow.org/docs/latest/index.html
