# ZC — Zalo AI Challenge 2026, track DataMate

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bảng bẩn (CSV, Excel, SQLite) của hộ kinh doanh nhỏ, biết trả
`unanswerable` hoặc từ chối (`abstain`). Điểm: Penalized Accuracy (+1 đúng, 0 abstain, −1 sai). Ràng buộc: offline,
≤ 15 B tham số, ≤ 6 giờ trên 2 × RTX 5090, nộp Docker trước 16/11.

**Bắt đầu đọc: `handoff/00_READ_ME_FIRST.md`.** Kết quả hiện tại: PA train dev +0.47 / holdout +0.44 (`handoff/01_SCIENTIFIC_FINDINGS.md`).

- Branch làm việc: **`claude/peaceful-carson-0dny90`** (tiếp nối `datamate`).
- Branch này là bản sao của `hmm-navigation-2` (repo `hmm-develop`), giữ nguyên lịch sử; nội dung nghiên cứu HMM/VPR đã được gỡ.
- Khôi phục file cũ: `git checkout de4360e -- <path>`.

## Cấu trúc

| Path | Nội dung |
|---|---|
| `handoff/` | Bàn giao: `00_READ_ME_FIRST.md` (cách đọc + việc tiếp theo), `01_SCIENTIFIC_FINDINGS`, `02_ENGINEERING_OPS_NOTES`, `03_HANDOFF` (bản md + pdf), `RUNNING_KERNELS.md` |
| `docs/` | `00_TASK.md` (đề + dữ liệu), `01_PLAN.md` (kế hoạch, tổng quan paper), `02_KAGGLE_OPS.md` (acc, assets, lệnh), `03_WORKFLOW.md` (quy trình + nguyên tắc của user), `RESEARCH_WORKFLOW_original.pdf` (prompt quy trình gốc) |
| `src/` | Chạy trên Kaggle / Docker. `ingest.py` (làm sạch tổng quát), `semantic.py` (luật ghi chép của rulebook: `build_sales`), `vnnorm.py` (chuẩn hoá câu chat), `agent.py` (pipeline LLM), `compare.py` (so run dev/holdout), `build_shots.py` + `shots/` (thư viện few-shot), `score.py` (bộ chấm đúng luật), `analyze.py` (phân tích log), `run_kaggle.py` (job Kaggle), `harvest.py` + `train_sft.py` (fine-tune) |
| `tests/` | `test_ingest_robust.py` (12 biến thể dữ liệu bẩn tổng hợp), `test_agent_local.py` (pipeline đầu-cuối không GPU) |
| `automation/` | `fanout.py`, `make_kernel.py`, `run_kernel.py` (chạy Kaggle), `watch_done.py` (báo khi kernel xong), `pull_kernels.py`, `resume.py`, `reattach.py`, `publish_dataset.py`, `md2pdf.py` |
| `docker/` | Dockerfile, `predict.sh` (/data → /result), build / chạy thử / export, `SETUP_DOCKER.md` (hướng dẫn user), `README_SOLUTION.md` |
| `results/chat2_runs/` | Log nén của mọi run khung chat 2 (dùng lại offline) |
| `data/` | `data/raw/` (không commit) = Kaggle `nhatlong1103/zc-2026` |
| `EXPERIMENT_LOG.md` | Nhật ký thí nghiệm |
