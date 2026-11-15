# PDF 3 — Handoff (DataMate, 15/11/2026 04:30 UTC)

*Mục tiêu, quyết định, tiến độ, vị trí, việc tiếp theo. Kết quả khoa học + bằng chứng: PDF 1 (§9 = kế hoạch v2). Lỗi / cách tránh / kỹ thuật:
PDF 2 (§10 = v2). Môi trường: `environment_summary.pdf`. Quy trình: `research_workflow.pdf`. Trạng thái sống: `handoff/STATUS.md`.*

## 1. Mục tiêu

Zalo AI Challenge 2026, track DataMate: hỏi đáp tiếng Việt trên dữ liệu bảng bẩn của cửa hàng nhỏ. PA = (đúng − sai) / N, bỏ trả lời = 0. Mô hình
≤ 15B tham số, offline, ≤ 6 h trên 2×RTX 5090, nộp Docker. **Đóng băng container 16/11 04:00 UTC**; nộp 16/11 23:59 GMT+7 (user build Docker).
Mục tiêu user: public cao nhất bằng hệ **chế độ H** (không hàm ngành viết tay), mọi cải tiến là năng lực mô hình tự học / tổng quát.

## 2. Quyết định còn hiệu lực (mới nhất trước)

| Ngày (UTC) | Quyết định |
|---|---|
| 15/11 04:00 | **Tạm dừng**: cho các tiến trình đang chạy chạy xong, **không khởi tạo tiến trình mới**; bàn giao sang khung chat mới rồi mới chạy tiếp. Routine mỗi giờ `trig_015hi1s1WHshqorFZC7DWY9j` **đã tắt** (xoá khi xong dự án) |
| 15/11 01:15 | Dừng mọi held-out; thử nhiều giải pháp trên public, chọn điểm cao nhất (vẫn chế độ H); cần thêm acc thì xin user; thử có hệ thống, không spam |
| 14/11 17:00 | Duyệt kế hoạch v2 (`handoff/04`, `05`), tự quyết kỹ thuật; **bản nộp chế độ H** (`hide_api: industry`); không hard-code / rule-based / heuristic thủ công, kể cả "có vẻ tổng quát"; sửa nguyên nhân gốc; không đổi tổng quát lấy điểm |
| 14/11 02:30 | GPU: mỗi acc 2 phiên × 4 L4, FSDP 4 GPU hoặc mỗi GPU một biến thể; không tiết kiệm acc; kiểm quota trước khi chạy |
| 13/11 | Ngừng luật viết tay; luật nếu có phải do mô hình viết |
| 08/11 | Cấm hard code đáp án; không train trên public / private; đáp án public chỉ để chấm |

## 3. Tiến độ (public 1000 câu; *container* = file Docker ghi, *ensemble* = file phán xử)

| Hệ | ensemble / container | Ghi chú |
|---|---|---|
| P (luật viết tay, chỉ dự phòng) | 0.994 | `final_submission/submission.jsonl`; vi phạm quyết định 13/11 |
| Kế hoạch 5 có `industry.py` (D1) | 0.952 / 0.951 | vi phạm quyết định 14/11 (chế độ H) |
| **Mốc chế độ H = cấu hình nộp hiện tại** `configs/flow_v2_final.json` (= `flow_p5_final_h`) | **0.863 / 0.860** (H0a2: 0.863 / 0.859) | S = V3A × 0.70; W1 bind, units do S viết, W2 linker WL1, W3 phán xử WJR1 |
| 14 biến thể v2 đã đo (F, S′, WT-b/s, k24, phán xử…) | 0.830–0.861 | đều ≤ mốc; bảng PDF 1 §9.1, `Process/scores_v2.csv`, `handoff/06` §5 |

**Đang chạy lúc bàn giao (để chạy xong; output kernel còn trên Kaggle, tải bằng `kaggle kernels output <id> -p <dir>`):**

| Lần chạy | Kernel | Nội dung | Dự kiến xong (UTC) |
|---|---|---|---|
| phu16 | `caotienminh1/zc-phu16-1` | S viết hàm 16 mẫu, đồng thuận ≥ 3 (`flow_hu16`) | ~05:30 |
| phnou | `mucnokieu/zc-phnou-1` | không hàm (đối chứng, `flow_hnou`) | ~05:30 |
| s3a | `duylnng/zc-s3a-1` | huấn luyện S″-A (adapter `OUT/S3A`) | ~06:00 |
| uqchka / uqchkb | `lelynhan/zc-uqchka-1`, `zc-uqchkb-1` | S giải lại 1,336 câu UQ (kiểm ngược) | ~04:45 |
| pwtbw1r, pbase, pwtbv2 | `huyenco/…`, `anderson1202/…`, `phuongnehe/…` | biến thể WT-b (dự kiến kém mốc) | ~04:30–05:00 |

Kết quả nào đã về trước khi đóng phiên được ghi ở `handoff/06` §5 và `Process/scores_v2.csv`; phần còn lại: tải + chấm theo §5 bước 2.

## 4. Vị trí

| Thứ | Ở đâu |
|---|---|
| Luồng nộp | `src/predict_flow.py` + `configs/flow_v2_final.json`; Docker `docker/` (`predict.sh` mặc định `flow_v2_final.json`) |
| Chế độ H | `src/lrl.py` (`hidden_names`, `strip_hidden`), `ZC_HIDE_API` xuất bởi `predict_flow` / `eval_flow` / `hybrid`; `tests/test_mode_h.py` |
| Bộ viết hàm (units) | `src/lrl_gen.py --mode unit` (`unit_prompt`, `select_units`, `force_json` = budget forcing), `src/units.py` |
| Sinh dữ liệu v2 | `src/lrl_gen.py --mode unitq / rowgen`, `training/tg1_job.py`, `training/contrast_job.py`, `training/data/{wt_build,wt_from_programs,s2_build,s3_build,uq_filter}.py` |
| Cấu hình v2 | `configs/flow_*_h.json`, `flow_hu16/hut/hnou`, `flow_h_reuse` (+ `configs/reuse/h0a_{units,links}.jsonl`), `flow_uqchk` (+ `reuse/uq_units.jsonl`) |
| Job / plan Kaggle | `automation/jobs/*.jobs`, `automation/plans_v2/*.json`; sổ kernel `Process/kaggle/kernels_v2.csv` |
| Log / kết quả v2 | `Process/runs/v2/<run>/` (solver_logs, submission, cards_unit, links, phiếu phán xử — gzip); điểm `Process/scores_v2.csv` |
| Dữ liệu huấn luyện v2 | `Process/train_data/v2/{wt2,s3,s3a,contrast,tg0,wtprog_bal}` (gzip; giải nén vào `training/data/out/`) |
| Công cụ phân tích / vận hành v2 | `Process/tools/v2/` (score_run.sh, unit_use*.py, var_acc*.py, pool.py, watch_loop.sh, when_ready.sh, fake_*.py) |
| Adapter | S: `<acc>/zc-ad3` (V3A, WL1, WJR1, WTB, WTS, S2S1, S2S2) trên ~9 acc, `nhatlong1103/zc-final-adapter` (V3A × 0.70, sha256 `src/adapter_hashes.json`); **W cho Docker: `nhatlong1103/zc-w-v2` (WL1, WJR1; sha256 `src/adapter_hashes_w.json`, đã tải lại kiểm OK)**; Git LFS `Process/adapters/` (WL1, WJR1, WJB1) |
| Kế hoạch / nhật ký v2 | `handoff/04_PROPOSED_PLAN.md`, `05_RUN_PLAN.md` (§6 nhật ký sự cố), `06_PUBLIC_GRID.md`, `07_DATA_AUG_PLAN.md` |

## 5. Việc tiếp theo (theo thứ tự)

1. **Sửa lỗi gộp container (đòn bẩy chắc chắn +0.003–0.013, không phải heuristic):** `src/predict_flow.py` dòng ~390 gộp
   `[jv, solver]` bằng `merge_sub.merge`, mà `merge_sub.read` bỏ dòng abstain → câu phán xử quyết định bỏ bị lấy đáp án S. Sửa: file phán xử là
   nguồn quyết định cho mọi câu nó có (kể cả abstain; ví dụ thêm tham số `keep_abstain` cho nguồn đầu), S chỉ lấp câu phán xử không có. Thêm
   test CPU, chạy `tests/run_all.sh` (timeout 2700, ~35 phút), rồi xác nhận offline: file container mới = `judge/ensemble/submission.jsonl`
   trên `Process/runs/v2/h0a/`.
2. **Thu kết quả các lần chạy còn lại** (§3): `python3 automation/resume.py` hoặc `kaggle kernels output <id>`; chấm
   `Process/tools/v2/score_run.sh <run>` (sửa `$S`); lưu `python3 automation/archive_v2.py <runs dir> <run>`; ghi `handoff/06` §5.
   So ghép cặp với H0a / H0a2 (đoạn mã trong `handoff/06` hoặc `Process/tools/v2/`). Nhớ: so **cùng loại file** (container với container).
3. **S″ (data augmentation, `handoff/07` §8)** — đã chuẩn bị, chưa đẩy:
   - khi uqchk xong: `python3 training/data/uq_filter.py <uqchk_a sub>,<uqchk_b sub> training/data/out/s3/uq training/data/out/s3/uqf`, rồi
     `python3 training/data/s3_build.py training/data/out/s3f --unitq training/data/out/s3/uqf --units training/data/out/s3/units_train.jsonl --name s3f.jsonl --u2 1500 --replay 600`
     → đẩy `automation/plans_v2/plan_s3f.json` (duylnng phiên 2);
   - khi s3a xong: đẩy `plan_es3ar.json` (đo S″-A trên hàm H0a, cùng acc duylnng); so ghép cặp với H0a.
4. **phut** (S viết hàm có suy nghĩ + budget forcing, `plan_phut.json`, caotienminh1): mã đã có, test `test_force_json` PASS; bộ test đầy đủ
   phải PASS trước khi đẩy.
5. **Chọn cấu hình nộp** = cấu hình chế độ H có điểm container cao nhất (sau bước 1), cập nhật `configs/flow_v2_final.json`; nếu S đổi (S″), thêm
   adapter vào `nhatlong1103/zc-final-adapter` (soup.json: V3A 0.70 + S3A 1.0) và sửa `src/adapter_hashes.json`.
6. **Docker trước 16/11 04:00 UTC:** `docker/fetch_assets.sh` (mô hình + `ADAPTER_DS` + `W_ADAPTER_DS=nhatlong1103/zc-w-v2`) → `build.sh` →
   `run_public_test.sh`; một lần chạy Kaggle bố cục image (`soup_job --runner predict` với `ZC_ADAPTER_ROOT` = thư mục adapter W) để có điểm
   container của đúng cấu hình nộp. 2×5090 ước nhanh hơn 4×L4 ~3× → còn dư thời gian trong 6 h.
7. Trước đóng băng: cập nhật STATUS, PDF 1–3; xoá routine `trig_015hi1s1WHshqorFZC7DWY9j`.

**Token Kaggle** (mất khi đổi container, xin lại user): 19 acc ở `handoff/environment_summary.pdf` §2; quota làm mới 21/11 00:00 UTC.
