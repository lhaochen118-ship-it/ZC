# READING_GUIDE — cách đọc repository và việc tiếp theo (cập nhật 11/11/2026)

Cho Agent nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm.

## 0. Năm phút đầu

1. Tạo routine tự tiếp tục mỗi giờ (`create_trigger`, lời nhắc trong `CLAUDE.md`) và **xin user token Kaggle** (24 acc).
2. `handoff/STATUS.md` → bản nộp hiện tại, kernel đang chạy, việc tiếp theo theo thứ tự.
3. Dựng lại môi trường: `handoff/ENVIRONMENT_SUMMARY.md` §1 + `Process/README.md` (dữ liệu, cache, log nén, dữ liệu huấn luyện).

## 1. Bắt buộc

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | quy định user (hybrid, chạy tự động, quota < 6 h, phiên ≤ 12 h, `final_submission`, cổng kiểm tra trước khi chạy, playbook GPU) |
| 2 | `handoff/STATUS.md` | trạng thái sống + bước tiếp theo |
| 3 | `handoff/03_HANDOFF.pdf` | mục tiêu, quyết định, vị trí code / dữ liệu / kết quả |
| 4 | `docs/13_TASK_AFTER_PLAN1.md` | nhiệm vụ hiện tại (nguyên văn user): lặp đến public 100 %, held-out không giảm |
| 5 | `docs/12_PLAN2.md` | kế hoạch 2: hướng R / M, gói việc, **quy tắc giữ / bỏ §4**, kết quả từng vòng |

## 2. Khi cần hiểu sâu

| File | Khi nào đọc |
|---|---|
| `handoff/01_SCIENTIFIC_FINDINGS.pdf` | trước khi đề xuất hướng mới: cái gì hiệu quả / thất bại, bằng chứng |
| `handoff/02_ENGINEERING_OPS_LESSONS.pdf` | trước khi sửa code / chạy Kaggle: lỗi đã gặp, công cụ mới (quota, 12 h, checkpoint) |
| `handoff/environment_summary.pdf` | máy, acc, quota, Docker |
| `docs/03_WORKFLOW.md` (PDF `docs/RESEARCH_WORKFLOW.pdf`) | quy trình: vòng lặp, sàng lọc → xác nhận, git, bàn giao |
| `docs/10_ERROR_ANALYSIS.md`, `docs/11_LITERATURE.md` | taxonomy lỗi 7 nhóm, tài liệu ngoài và gói việc tương ứng |
| `docs/06_HYBRID_FRAMEWORK.md`, `docs/08_MASTER_PLAN.md`, `docs/09_OPTIONS_DEEP_DIVE.md` | thiết kế luồng hybrid và kế hoạch 1 (đã xong) |
| `docs/00_TASK.md`, `docs/ZAIC_REQUIREMENTS.md` (gốc `ZAIC.docx`) | đề bài, luật chấm |
| `docs/02_KAGGLE_OPS.md`, `docs/05_GPU_OPTIMIZATION.md`, `docs/07_DATASETS.md` | vận hành Kaggle, tối ưu GPU, dataset ngoài |
| `EXPERIMENT_LOG.md` | số liệu kế hoạch 1 (entry 0–47); kế hoạch 2: `Process/history/`, `Process/scores.csv` |
| `tests/README.md` | bộ test CPU, lệnh đánh giá |

Các `handoff/*.md`, `docs/03_WORKFLOW.md` là nguồn của PDF cùng tên: sửa `.md` rồi
`python3 automation/md2pdf.py <in.md> <out.pdf>` (cần `pip install markdown xhtml2pdf`).

## 3. Bản đồ mã nguồn

- **Luồng nộp**: `src/predict_flow.py` (lối vào Docker) → `src/agent.py --flow hybrid` (`build_parser` = mọi cờ; cấu hình
  `configs/flow_final.json`) → `src/hybrid.py` (gợi ý → prompt → mẫu → precheck → sandbox → cờ → vòng 2 → phiếu) trên
  `src/pipeline.py` (vLLM từng request, work stealing, `--hard-deadline`).
- **Ngữ cảnh / luật tổng quát**: `src/hints.py` (kỳ, thực thể, kiểu đáp án, từ rulebook), `src/qnorm.py` (làm sạch câu),
  `src/qparse.py` (kỳ thời gian), `src/retrieval.py` (ví dụ đã kiểm), `src/prompting.py`, `src/shopctx.py`, `src/automap.py` / `src/ingest.py` (schema).
- **Tính toán / kiểm tra**: `src/dm.py`, `src/industry.py` (toolkit nghiệp vụ), `src/precheck.py` (lỗi → lượt sửa: size / unit / len_mask),
  `src/tools.py` (cờ sau khi chạy), `src/canon.py` (chạy chương trình an toàn), `src/sqlesc.py` (nhánh SQL / DuckDB).
- **Mô hình phụ**: `src/selector.py` (bộ chọn học được; `src/refs/selector/`).
- **Huấn luyện** (`training/README.md`): `training/data/*` (nhà máy dữ liệu D1–D4T, `rft_from_logs.py`, `dpo_from_logs.py`),
  `sft_unsloth.py` / `sft.py` / `dpo.py` (checkpoint + resume: `deadline.py`), `arms_job.py` (4 nhánh / phiên, `--base-adapters`),
  `full_job.py` (FSDP), `soup_job.py` (gộp + đánh giá, `--runner predict` = đường container), `merge.py`.
- **Vận hành**: `automation/launch_many.py` (plan → kernel, kiểm quota / 12 h), `fanout.py`, `make_kernel.py` (runner + hạn chót),
  `run_kernel.py`, `watch_done.py` (watcher nền), `resume.py` (gắn lại sau khởi động lại), `quota.py`,
  `update_final_submission.py`, `set_final_model.py`; `automation/jobs/` (job), `docker/` (bài nộp).
- **Đánh giá / phân tích**: `src/eval_flow.py` (tập `tests/splits/`), `src/score.py`, `tests/list_wrong.py`, `tests/pool_gens.py`,
  `tests/replay_vote.py`, `tests/train_selector.py`, `tests/hint_audit.py`; test CPU: `bash tests/run_all.sh`.

## 4. Việc tiếp theo

Theo `handoff/STATUS.md` §4: (1) kết quả WG5–8 → sàng lọc → xác nhận → đổi mô hình nộp nếu đạt; (2) đợt quota mới 14/11;
(3) chốt bản nộp + đường container trước 15/11, báo user build Docker; sau mỗi bước cập nhật STATUS, commit + push.
