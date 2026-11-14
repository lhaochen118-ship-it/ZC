# READING_GUIDE — cách đọc repository và việc tiếp theo (cập nhật 14/11/2026 15:45 UTC)

Cho Agent nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm.

## 0. Mười phút đầu

1. **Xin user token Kaggle** (container mới không còn `~/.kaggle/tok_*`); lưu `~/.kaggle/tok_<username>` (chmod 600), không ghi vào repo.
2. `pip install kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf pandas openpyxl`
3. Dữ liệu: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_nhatlong1103) kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip`;
   đáp án public (chỉ để chấm): `git fetch --unshallow origin; python3 automation/rebuild_public_gt.py`.
4. Đọc `handoff/STATUS.md`. Dự án **đang dừng** (14/11 15:30): kế hoạch 5 xong; user sẽ gửi kế hoạch mới (docx) để chạy tiếp.

## 1. Bắt buộc

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | quy định user (quyết định mới nhất 14/11; GPU / quota / 12 h; cổng kiểm tra) |
| 2 | `handoff/STATUS.md` | trạng thái sống, hệ đang nộp, việc tiếp theo |
| 3 | `handoff/03_HANDOFF.pdf` (nguồn `.md`) | mục tiêu, quyết định, tiến độ, vị trí, việc tiếp theo |
| 4 | `handoff/01_SCIENTIFIC_FINDINGS.pdf` — **§8 kế hoạch 5** | cái gì hiệu quả / thất bại, bằng chứng |
| 5 | `handoff/02_ENGINEERING_OPS_LESSONS.pdf` — **§9 lỗi đã phạm + quy tắc tránh, §9.2 kỹ thuật dùng lại** | trước khi sửa code / đẩy Kaggle |

## 2. Khi cần

| File | Khi nào |
|---|---|
| `docs/18_PLAN5_TWO_LORA.md` | kế hoạch 5 (thiết kế, lịch, nhật ký §7 mọi lần chạy 14/11) |
| `docs/03_WORKFLOW.md` §9 | quy trình nghiên cứu dùng cho kế hoạch mới |
| `docs/00_TASK.md`, `docs/ZAIC_REQUIREMENTS.md` | đề bài, luật chấm, giới hạn |
| `docs/02_KAGGLE_OPS.md`, `docs/05_GPU_OPTIMIZATION.md`, `handoff/environment_summary.pdf` | vận hành Kaggle, GPU, môi trường |
| `Process/README.md` | mọi thứ đã lưu (dữ liệu W, log S, phiếu phán xử, chỉ số huấn luyện, công cụ) và cách khôi phục |
| `Process/scores_p5.csv`, `scores_p3.csv`, `scores.csv` | điểm mọi lần chạy |
| `docs/17_PLAN4_SWV.md` | kế hoạch 4 (W unit, bộ chọn V) |

## 3. Bản đồ mã nguồn (kế hoạch 5)

- **Luồng nộp** `src/predict_flow.py` + `configs/flow_p5_final.json`: W1 `prepare_lrl` (`lrl_gen.py --mode bind`) → `prepare_units` (`lrl_gen.py --mode unit`)
  → W2 `prepare_link` (`src/link.py`, adapter WL1 gộp theo pha `w_model`) → S `src/agent.py --flow hybrid` (`src/hybrid.py`, `--links`) → W3
  `judge_phase` (`src/judge.py`, adapter WJR1, tổ hợp list / point / point+think) → `merge_sub`.
- **Đánh giá trên Kaggle**: `training/soup_job.py` (`--runner eval` tập `tests/splits`, `--runner predict` đường container, `--runner judge` phán xử
  log đã lưu, `--runner script` chạy script `src/` trên mô hình gộp).
- **Huấn luyện W**: `training/full_job.py --base-adapters V3A --base-weights 0.70` + dữ liệu `training/data/w_judge.py`, `w_link.py`, `p_rows.py`.
- **Vận hành**: `automation/launch_many.py` (plan → kernel, kiểm quota), `watch_done.py` (+ `Process/tools/plan5/watch_loop.sh`), `resume.py`.
- **Offline**: `tests/replay_judge.py`, `Process/tools/plan5/{pol,ens2,err,pool}.py`.

## 4. Việc tiếp theo

1. Nhận kế hoạch docx của user → lập lịch theo `docs/03_WORKFLOW.md` §9 và quy tắc PDF 2 §9.
2. Nếu kế hoạch không thay: **D1 Docker** cho `configs/flow_p5_final.json` trước **16/11 04:00 UTC** (chi tiết `handoff/03_HANDOFF.md` §5).
