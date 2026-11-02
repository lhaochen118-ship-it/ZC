# ZC — Zalo AI Challenge 2026, track DataMate (v3)

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng bẩn (CSV, Excel, SQLite) của hộ kinh doanh nhỏ, biết trả
`unanswerable` hoặc từ chối (`abstain`). Điểm: Penalized Accuracy (+1 đúng, 0 abstain, −1 sai). Ràng buộc: offline,
≤ 15 B tham số, ≤ 6 giờ trên 2 × RTX 5090, nộp Docker trước 16/11/2026 23:59 GMT+7.

- **Branch làm việc: `claude/datamate-v3-0dny90`** (khung chat 3). Tạo từ `claude/peaceful-carson-0dny90` theo yêu cầu
  của user, sau khi quyết định **bỏ** phương án v1 (lý do + số liệu: `docs/04_DECISION_V3.md`).
- Phương án cũ còn nguyên ở branch `claude/peaceful-carson-0dny90` (v1: LLM tự viết chương trình pandas cho từng câu) và
  `claude/funny-euler-ceut3o` (v2: toolkit nghiệp vụ, chưa chạy với LLM).

## Đọc theo thứ tự

| # | File | Nội dung |
|---|---|---|
| 1 | `docs/ZAIC_REQUIREMENTS.md` (gốc: `ZAIC.docx`) | Yêu cầu của user: mục tiêu, quy trình nghiên cứu, dùng Kaggle |
| 2 | `docs/00_TASK.md` | Đề bài đầy đủ (portal + tab Data), luật chấm, luật kỹ thuật, lịch |
| 3 | `docs/04_DECISION_V3.md` | Đánh giá v1 / v2, quyết định, kiến trúc v3, kế hoạch thí nghiệm |
| 4 | `docs/03_WORKFLOW.md` | Quy trình làm việc + nguyên tắc user đã chốt |
| 5 | `docs/02_KAGGLE_OPS.md` | Vận hành Kaggle (GPU L4, dataset, lệnh) |
| — | `docs/01_PLAN.md` | Kế hoạch gốc + tổng quan paper |
| — | `docs/history/` | Kết quả, bài học kỹ thuật, nhật ký thí nghiệm của v1 |
| — | `EXPERIMENT_LOG.md` | Nhật ký thí nghiệm v3 |

## Cấu trúc

| Path | Nội dung |
|---|---|
| `docs/` | Đề bài, yêu cầu, quyết định, quy trình, vận hành |
| `automation/` | Chạy Kaggle: `fanout.py`, `make_kernel.py`, `run_kernel.py`, `watch_done.py`, `pull_kernels.py`, … |
| `src/`, `tests/` | Pipeline v3 (thêm dần, mỗi thành phần kèm test) |
| `data/` | `data/raw/` (không commit) = Kaggle `nhatlong1103/zc-2026` |
