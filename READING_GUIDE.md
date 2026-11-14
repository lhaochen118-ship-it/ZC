# READING_GUIDE — cách đọc repository (cập nhật 14/11/2026)

Cho Agent nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm.

## 0. Mười phút đầu

1. **Xin user token Kaggle** (container mới không còn `~/.kaggle/tok_*`); lưu `~/.kaggle/tok_<username>` (chmod 600), không ghi vào repo.
2. `pip install kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf`; dữ liệu / cache theo `handoff/ENVIRONMENT_SUMMARY.md` §1 và `Process/README.md`.
3. Đọc `handoff/STATUS.md` (trạng thái + việc tiếp theo). Dự án **đang dừng** (user 14/11): chờ user đưa chiến lược mới hướng tới public = 1.

## 1. Bắt buộc

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | quy định user (quyết định mới nhất 14/11: chỉ tối ưu public; không luật viết tay; GPU / quota / 12 h; cổng kiểm tra) |
| 2 | `handoff/STATUS.md` | trạng thái sống, hệ đang nộp, việc tiếp theo |
| 3 | `handoff/01_SCIENTIFIC_FINDINGS.pdf` (nguồn `.md`) | cái gì hiệu quả / thất bại, bằng chứng, giới hạn — đọc trước khi đề xuất hướng mới |
| 4 | `handoff/02_ENGINEERING_OPS_LESSONS.pdf` (nguồn `.md`) | lập trình, Kaggle, GPU, lỗi đã gặp và cách sửa — đọc trước khi sửa code / chạy Kaggle |

## 2. Khi cần

| File | Khi nào |
|---|---|
| `docs/17_PLAN4_SWV.md` | kế hoạch 4 S–W–V (đã xong) và nhật ký chi tiết §7 (mọi lần chạy 13–14/11) |
| `docs/00_TASK.md`, `docs/ZAIC_REQUIREMENTS.md` | đề bài, luật chấm, giới hạn (≤ 15B, 6 h, 2×RTX 5090, offline) |
| `docs/02_KAGGLE_OPS.md`, `docs/05_GPU_OPTIMIZATION.md` | vận hành Kaggle, tối ưu GPU |
| `docs/07_DATASETS.md`, `docs/11_LITERATURE.md` | dataset ngoài dùng được, tài liệu tham khảo |
| `handoff/03_HANDOFF.md`, `handoff/ENVIRONMENT_SUMMARY.md` | tóm tắt bàn giao, môi trường / máy / Docker |
| `Process/README.md` | những gì đã lưu (log, dữ liệu, kernel, công cụ) và cách khôi phục |
| `Process/scores.csv`, `Process/scores_p3.csv` | điểm mọi lần chạy (kế hoạch 1–2 / 3–4) |

Tài liệu kế hoạch cũ (docs/06, 08–10, 12–16, `EXPERIMENT_LOG.md`, `Process/history/`) đã xoá 14/11 — nội dung chính ở PDF 1 / 2; bản gốc
trong lịch sử git (commit trước `14/11 05:00`). PDF dựng lại: `python3 automation/md2pdf.py <in.md> <out.pdf>`.

## 3. Bản đồ mã nguồn

- **Luồng nộp** (`configs/flow_final.json` = S–W, không luật viết tay): `src/predict_flow.py` (W: `prepare_units` → `src/lrl_gen.py --mode unit`,
  `src/units.py`; thẻ luật: `prepare_lrl`, `src/lrl.py`) → `src/agent.py --flow hybrid --pure 1` → `src/hybrid.py` (prompt → mẫu → precheck
  chung → sandbox → sửa → phiếu / bộ chọn `src/selector.py`) trên `src/pipeline.py`. Hệ cũ P (luật viết tay): `configs/flow_p_final.json`.
- **Tính toán**: `src/dm.py`, `src/industry.py` (toolkit), `src/automap.py` / `src/ingest.py` (schema), `src/precheck.py`, `src/tools.py`.
- **Huấn luyện**: `training/full_job.py` (FSDP 4 GPU, `ZC_SESSION_S` + `--resume-from auto`), `training/sft.py`, `training/soup_job.py`
  (gộp adapter + đánh giá; `--parallel-gpus N` = mỗi GPU một biến thể; `--runner predict` = đường container); dữ liệu `training/data/*`.
- **Vận hành**: `automation/launch_many.py` (plan → kernel, kiểm quota, trần 11 h), `chain_next.py` (nối checkpoint qua acc), `watch_done.py`
  (watcher nền), `publish_many.py` (đăng adapter), `set_final_model.py` (đổi mô hình nộp), `quota.py`, `archive_runs.py`, `resume.py`.
- **Đánh giá**: `src/eval_flow.py` (tập `tests/splits/`; rulebook theo DN), `src/score.py`, `Process/tools/score_p3.py`; test CPU `bash tests/run_all.sh`.
