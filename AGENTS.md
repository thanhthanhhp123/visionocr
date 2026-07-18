# Hướng dẫn làm việc với dự án VisionOCR

## Quy tắc bắt buộc khi bắt đầu session

- Mỗi agent/session **phải đọc toàn bộ file `AGENTS.md` này trước khi phân tích, chạy lệnh hoặc sửa mã nguồn**.
- Sau khi đọc, kiểm tra `git status --short --branch` để biết trạng thái hiện tại của repository.
- Đọc `README.md` và các file liên quan trực tiếp đến yêu cầu trước khi hành động.
- Không giả định roadmap trong README phản ánh chính xác mức độ hoàn thiện; phải đối chiếu với mã nguồn và kết quả kiểm thử thực tế.
- Tôn trọng thay đổi đang có của người dùng. Không hoàn tác, ghi đè hoặc định dạng lại phần không thuộc phạm vi công việc.

## Tổng quan dự án

VisionOCR là hệ thống trích xuất dữ liệu có cấu trúc từ ảnh hóa đơn bán lẻ Việt Nam.

Luồng dự kiến:

1. PaddleOCR trích xuất văn bản từ ảnh.
2. Qwen2.5-VL-3B đã fine-tune bằng LoRA nhận ảnh và OCR text.
3. Model trả về JSON hóa đơn.
4. FastAPI tiếp nhận upload và chuyển tác vụ sang Celery/Redis.
5. PostgreSQL lưu kết quả; Streamlit hiển thị dữ liệu.

Các khu vực chính:

- `datasets/`: dữ liệu, nhãn và các tập JSONL.
- `scripts/`: kiểm tra, làm sạch, chuẩn bị và đánh giá dữ liệu.
- `vlm/`: fine-tuning và inference VLM.
- `ocr/`: PaddleOCR wrapper.
- `api/`: FastAPI routes và schemas.
- `worker/`: Celery tasks.
- `db/`: SQLAlchemy models.
- `frontend/`: giao diện Streamlit.
- `docker/` và `docker-compose.yml`: môi trường container.

## Trạng thái gần nhất

- Phase 1 được chủ dự án xác nhận hoàn tất.
- Phase 2: LoRA v2 đã train, evaluate và merge; AWQ/vLLM vẫn chưa tạo artifact.
- Phase 3–4: API/Celery/database persistence và frontend đã được triển khai, có unit/API tests; chưa xác minh full GPU runtime.
- Phase 5: Docker Compose và Dockerfiles đã cấu hình; Docker runtime chưa có trên môi trường hiện tại để chạy stack.
- Kết quả LoRA v2 gần nhất trên 87 mẫu test leakage-safe:
  - Store name accuracy: 88.5%.
  - Date accuracy: 90.8%.
  - Total amount accuracy: 90.8%.
  - Item name F1: 0.923.
  - JSON parse rate: 98.9%.
  - Độ trễ trung bình HF + LoRA: 36.59 giây.

## An toàn dữ liệu

- Khi người dùng chỉ yêu cầu kiểm tra, chẩn đoán hoặc báo cáo, chỉ được chạy các lệnh **read-only**.
- Phải đọc nội dung script trước khi chạy. Tên như `check`, `validate` hoặc `lint` không đảm bảo script không sửa dữ liệu.
- **Không chạy `scripts/validate_labels.py` khi chưa được người dùng cho phép rõ ràng.** Script này ghi đè label hợp lệ và xóa label không sửa được.
- Không chạy `scripts/fix_errors.py`, `scripts/prepare_finetune_data.py`, lệnh lint có `--fix`, hoặc bất kỳ script nào ghi/xóa dữ liệu nếu yêu cầu chỉ là kiểm tra.
- Không xóa hoặc thay đổi dữ liệu trong `datasets/images`, `datasets/labels`, `models` hay `vlm/mlruns` nếu chưa có sự đồng ý rõ ràng.
- Trước thao tác dữ liệu hàng loạt, phải xác định phương án backup/khôi phục và báo cho người dùng.
- Không dùng các lệnh phá hủy như `rm`, `git reset --hard`, hoặc `git checkout --` nếu người dùng chưa yêu cầu rõ ràng.

## Quy trình thay đổi mã nguồn

1. Xác định chính xác phạm vi yêu cầu.
2. Kiểm tra các thay đổi hiện có bằng Git.
3. Đọc mã nguồn và cấu hình liên quan.
4. Thực hiện thay đổi nhỏ nhất đáp ứng yêu cầu.
5. Chạy kiểm tra phù hợp với mức độ rủi ro.
6. Kiểm tra lại `git diff` và `git status`.
7. Báo cáo file đã thay đổi, kiểm tra đã chạy và phần chưa thể xác minh.

## Quy ước kiểm thử

- Có thể dùng `python -m compileall` để kiểm tra cú pháp vì thao tác này không sửa mã nguồn; lưu ý nó có thể tạo `__pycache__` đã được ignore.
- Ưu tiên test nhỏ, cô lập và không cần tải model trước.
- Không tự động tải model, cài dependency, khởi động Docker stack hoặc chạy GPU workload lớn nếu chưa cần thiết.
- Không tuyên bố hệ thống hoạt động end-to-end nếu chưa kiểm tra API, Redis, worker, OCR, VLM và frontend trong cùng một luồng.

## Các vấn đề kỹ thuật đã biết cần lưu ý

- AWQ/vLLM chưa có artifact cục bộ. Runtime tự fallback sang model merged v2 (hoặc LoRA v2).
- Docker CLI/runtime không có trên môi trường hiện tại; cần chạy Docker Compose trên Docker-enabled GPU host.
- Chưa có demo GIF/video được ghi lại từ luồng upload thực tế.

## Ngôn ngữ và báo cáo

- Trao đổi và viết báo cáo cho người dùng bằng tiếng Việt, trừ khi người dùng yêu cầu ngôn ngữ khác.
- Báo cáo dựa trên bằng chứng từ mã nguồn, Git và kết quả lệnh thực tế.
- Nếu vô tình làm thay đổi dữ liệu hoặc gặp rủi ro mất dữ liệu, phải thông báo ngay, không che giấu.
