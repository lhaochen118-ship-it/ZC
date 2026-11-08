# ZC — Zalo AI Challenge 2026, track DataMate (v3)

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng bẩn (CSV, Excel, SQLite) của hộ kinh doanh nhỏ, biết trả
`unanswerable` hoặc từ chối (`abstain`). Điểm: Penalized Accuracy (+1 đúng, 0 abstain, −1 sai). Ràng buộc: offline,
≤ 15 B tham số, ≤ 6 giờ trên 2 × RTX 5090, nộp Docker trước 16/11/2026 23:59 GMT+7.

- **Branch làm việc: `claude/datamate-v3-0dny90`** (khung chat 3). Tạo từ `claude/peaceful-carson-0dny90` theo yêu cầu
  của user, sau khi quyết định **bỏ** phương án v1 (lý do + số liệu: `docs/04_DECISION_V3.md`).
- Phương án cũ còn nguyên ở branch `claude/peaceful-carson-0dny90` (v1: LLM tự viết chương trình pandas cho từng câu) và
  `claude/funny-euler-ceut3o` (v2: toolkit nghiệp vụ, chưa chạy với LLM).

## Trạng thái 08/11/2026

- Hai đường đã xây: **đường LLM** (`src/agent.py`: Qwen3.5-9B ánh xạ dữ liệu theo DN + sinh chương trình trên toolkit
  `dm` / `industry`, bỏ phiếu, đồng thuận ánh xạ; v28 public 0.921) và **đường luật** (`src/rules_first.py`; public
  1000 / 1000 trong mẫu, yếu trên cách hỏi mới). Luồng nộp hiện tại `src/predict_flow.py` là rule-first.
- **User chốt 08/11: hybrid framework** — Agent sinh code cho mọi câu; luật tổng quát chỉ hỗ trợ (ý định, schema,
  toolkit, kiểm tra), không sinh thẳng đáp án. Thiết kế: `docs/06_HYBRID_FRAMEWORK.md`. Việc tiếp theo:
  `handoff/03_HANDOFF.pdf`; trạng thái sống: `handoff/STATUS.md`; cách đọc repo: `READING_GUIDE.md`.

## Đọc theo thứ tự

| # | File | Nội dung |
|---|---|---|
| 1 | `docs/ZAIC_REQUIREMENTS.md` (gốc: `ZAIC.docx`) | Yêu cầu của user: mục tiêu, quy trình nghiên cứu, dùng Kaggle |
| 2 | `docs/00_TASK.md` | Đề bài đầy đủ (portal + tab Data), luật chấm, luật kỹ thuật, lịch |
| 3 | `docs/04_DECISION_V3.md` | Đánh giá v1 / v2, quyết định, kiến trúc v3, kế hoạch thí nghiệm |
| 4 | `docs/03_WORKFLOW.md` | Quy trình làm việc + nguyên tắc user đã chốt |
| 5 | `docs/02_KAGGLE_OPS.md` | Vận hành Kaggle (GPU L4, dataset, lệnh) |
| 6 | `docs/05_GPU_OPTIMIZATION.md` (tóm tắt trong `CLAUDE.md`) | Playbook tốc độ GPU: đo, data parallel, prefix cache, pipeline, work stealing |
| — | `EXPERIMENT_LOG.md` | Nhật ký thí nghiệm (entry 0–27) |

## Cấu trúc

| Path | Nội dung |
|---|---|
| `docs/` | Đề bài, yêu cầu, quyết định, quy trình, vận hành |
| `automation/` | Chạy Kaggle: `fanout.py`, `make_kernel.py`, `run_kernel.py`, `watch_done.py`, `pull_kernels.py`, … |
| `src/`, `tests/` | Mã nguồn + test CPU (bản đồ: `READING_GUIDE.md` §3) |
| `handoff/` | Bàn giao: `STATUS.md`, PDF 1–3, `environment_summary.pdf` |
| `submissions/` | `v28/` (đường LLM, 0.921), `v30/` (luật, 1000 / 1000) |
| `data/` | `data/raw/` (không commit) = Kaggle `nhatlong1103/zc-2026` |
