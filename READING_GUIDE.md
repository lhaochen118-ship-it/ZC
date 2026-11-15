# READING_GUIDE — cách đọc repository và việc tiếp theo (cập nhật 15/11/2026 04:30 UTC)

Cho Agent nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm. **Đóng băng container 16/11 04:00 UTC.**

## 0. Mười lăm phút đầu

1. **Xin user token Kaggle** (container mới mất `~/.kaggle/tok_*`): 19 acc (danh sách `handoff/environment_summary.pdf` §6); lưu
   `~/.kaggle/tok_<username>` (chmod 600), không ghi vào repo.
2. `pip install kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf pandas openpyxl`
3. Dữ liệu: `KAGGLE_API_TOKEN=$(cat ~/.kaggle/tok_nhatlong1103) kaggle datasets download nhatlong1103/zc-2026 -p data/raw --unzip`;
   đáp án public (chỉ để chấm): `cp "score 1.jsonl" data/raw/public_gt.jsonl` (hoặc `git fetch --unshallow origin; python3 automation/rebuild_public_gt.py`).
4. `git lfs pull` nếu cần adapter W cục bộ (`Process/adapters/WL1`, `WJR1`, ~640 MB; Docker tải từ Kaggle nên thường không cần).
5. Đọc `handoff/STATUS.md`. Trạng thái: **tạm dừng theo user** (15/11 04:00) — chờ user cho chạy tiếp ở khung mới. Có 7 kernel được để chạy xong
   lúc bàn giao: thu kết quả theo PDF 3 §5 bước 2 (không đẩy trùng).

## 1. Bắt buộc

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | quy định user (mới nhất: 15/11 04:00 tạm dừng; 15/11 01:15 chỉ public; 14/11 17:00 chế độ H, không heuristic thủ công); GPU / quota / 12 h; cổng kiểm tra |
| 2 | `handoff/STATUS.md` | trạng thái sống, cấu hình nộp, việc số 1 |
| 3 | `handoff/03_HANDOFF.pdf` (nguồn `.md`) | mục tiêu, quyết định, tiến độ, vị trí, **việc tiếp theo §5** |
| 4 | `handoff/01_SCIENTIFIC_FINDINGS.pdf` **§9** | kết quả v2 chế độ H: cái gì không giúp, vì sao (hàm WT sai nghĩa, thêm mẫu không giúp) |
| 5 | `handoff/02_ENGINEERING_OPS_LESSONS.pdf` **§10** (và §9) | lỗi đã phạm + quy tắc tránh, kỹ thuật dùng lại — trước khi sửa code / đẩy Kaggle |

## 2. Khi cần

| File | Khi nào |
|---|---|
| `handoff/06_PUBLIC_GRID.md` | lưới public v2: yếu tố, biến thể, acc, **nhật ký kết quả §5**, giai đoạn B′ §6 |
| `handoff/07_DATA_AUG_PLAN.md` | data augmentation S″ (UQ, kiểm ngược uqchk, S″-A / A′) — §8 là bản mới nhất |
| `handoff/04_PROPOSED_PLAN.md` / `.pdf`, `05_RUN_PLAN.md` / `.pdf` | kế hoạch v2 user đã duyệt; `05` §6 = nhật ký sự cố 14–15/11 |
| `handoff/research_workflow.pdf` (nguồn `docs/03_WORKFLOW.md`, §10 = v2) | quy trình nghiên cứu / chuyển khung chat |
| `handoff/environment_summary.pdf` (nguồn `ENVIRONMENT_SUMMARY.md`, §6 = 15/11) | môi trường, acc, dataset, thời gian chạy, Docker |
| `Process/README.md` | mọi thứ đã lưu (log, dữ liệu, công cụ, sổ kernel) và cách khôi phục |
| `Process/scores_v2.csv` (v2), `scores_p5.csv`, `scores_p3.csv`, `scores.csv` | điểm mọi lần chạy |
| `docs/00_TASK.md`, `docs/ZAIC_REQUIREMENTS.md` | đề bài, luật chấm |
| `docs/02_KAGGLE_OPS.md`, `docs/05_GPU_OPTIMIZATION.md` | vận hành Kaggle, GPU |
| `docs/17_PLAN4_SWV.md`, `docs/18_PLAN5_TWO_LORA.md` | kế hoạch 4 (W unit), 5 (linker / phán xử) |

## 3. Bản đồ mã nguồn

- **Luồng nộp** `src/predict_flow.py` + `configs/flow_v2_final.json` (chế độ H): W1 `prepare_lrl` (bind) → `prepare_units` (`lrl_gen.py --mode unit`,
  S viết hàm cửa hàng; `units_file` để dùng lại) → W2 `prepare_link` (`link.py`, WL1; `links_file`) → S `agent.py --flow hybrid` → W3 `judge_phase`
  (`judge.py`, WJR1) → `merge_sub` (**lỗi gộp abstain, sửa trước**).
- **Chế độ H**: `src/lrl.py` (`hidden_names`, `strip_hidden`), biến `ZC_HIDE_API`.
- **Sinh dữ liệu / huấn luyện v2**: `src/lrl_gen.py` (`unit`, `unitq`, `rowgen`, `force_json`), `training/{tg1_job,contrast_job,full_job,sft}.py`,
  `training/data/{wt_build,wt_from_programs,s2_build,s3_build,uq_filter}.py`.
- **Đánh giá trên Kaggle**: `training/soup_job.py` (`--runner predict` = đường container; `--sets` qua `eval_flow.py` cho tập trong `tests/splits/`).
- **Vận hành**: `automation/launch_many.py <runs dir> automation/plans_v2/plan_<x>.json` (kiểm quota), `run_kernel.py`, `watch_done.py` +
  `Process/tools/v2/watch_loop.sh`, `resume.py`, `reattach.py`, `quota.py`, `archive_v2.py`, `md2pdf.py`.
- **Phân tích offline**: `Process/tools/score_sets.py`, `Process/tools/v2/*` (chấm, gọi hàm, biến thể, gộp).
- **Docker**: `docker/` (`fetch_assets.sh` → `build.sh` → `run_public_test.sh`; `assets.env`: `ADAPTER_DS`, `W_ADAPTER_DS=nhatlong1103/zc-w-v2`).

## 4. Việc tiếp theo (chi tiết PDF 3 §5)

1. Sửa lỗi gộp container (phán xử bỏ trả lời bị S ghi đè) + test CPU + xác nhận offline trên `Process/runs/v2/h0a`.
2. Thu + chấm + lưu kết quả 7 kernel còn chạy lúc bàn giao; ghi `handoff/06` §5.
3. S″: `uq_filter` → `s3_build` (s3f) → `plan_s3f`; `plan_es3ar` khi s3a xong; `plan_phut` (bộ test đã ALL PASS 15/11 04:40).
4. Chọn cấu hình nộp (điểm container chế độ H cao nhất) → `configs/flow_v2_final.json` → Docker + 1 lần chạy bố cục image trên Kaggle, trước
   **16/11 04:00 UTC**.
