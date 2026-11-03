# READING_GUIDE — cách đọc repository và việc tiếp theo

Cho Agent (hoặc người) nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm.

## 1. Đọc trong 15 phút (bắt buộc)

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | Nguyên tắc user đã chốt, playbook GPU, cổng kiểm tra trước khi chạy |
| 2 | `handoff/STATUS.md` | Trạng thái sống: điểm portal, kernel đang chạy, việc tiếp theo |
| 3 | `handoff/03_HANDOFF.pdf` | Mục tiêu, quyết định, vị trí mọi thứ, các bước tiếp theo |
| 4 | `handoff/environment_summary.pdf` | Container, Kaggle (máy, acc, quota), máy chấm / Docker |

## 2. Khi cần hiểu sâu

| File | Khi nào đọc |
|---|---|
| `handoff/01_SCIENTIFIC_FINDINGS.pdf` | Trước khi đề xuất hướng mới: cái gì đã hiệu quả, cái gì thất bại, bằng chứng |
| `handoff/02_ENGINEERING_OPS_LESSONS.pdf` | Trước khi sửa code / chạy Kaggle: lỗi đã gặp và cách tránh |
| `docs/03_WORKFLOW.md` (PDF: `docs/RESEARCH_WORKFLOW.pdf`) | Quy trình làm việc; §5 vòng lặp hiện tại |
| `docs/04_DECISION_V3.md` | Vì sao kiến trúc v3, các phương án đã bỏ |
| `docs/00_TASK.md`, `docs/ZAIC_REQUIREMENTS.md` | Đề bài, luật chấm, yêu cầu user |
| `docs/02_KAGGLE_OPS.md`, `docs/05_GPU_OPTIMIZATION.md` | Vận hành Kaggle, tối ưu GPU |
| `EXPERIMENT_LOG.md` | Số liệu từng bản (entry 0–13); entry 13 = phân tích điểm public |
| `data/raw/public_test/public_test/rulebook.md` | Định nghĩa nghiệp vụ của public (khác train ở mục ngành) |

Các file `handoff/*.md` là nguồn của PDF cùng tên (sửa `.md`, rồi `python3 automation/md2pdf.py <in.md> <out.pdf>`).

## 3. Bản đồ mã nguồn (đọc theo luồng dữ liệu)

1. `src/ingest.py` — đọc file thô (CSV / Excel / SQLite) → bảng thô + card mô tả.
2. `src/agent.py` — CLI (`build_parser`), prompt, engine vLLM; `src/pipeline.py` — StepLLM (budget forcing), ánh xạ
   theo DN, chọn ánh xạ, bỏ phiếu theo câu, dự phòng ánh xạ.
3. `src/canon.py` — chạy chương trình ánh xạ an toàn, `secondary_checks` (kiểm tra nội dung), `warning_penalty`.
4. `src/dm.py` — toolkit `Shop` (mọi định nghĩa rulebook), `finalize`, `CANON_DOC` (mô tả bảng chuẩn cho prompt).
5. `src/qparse.py` — luật đọc câu → chương trình 1 dòng; `src/qnorm.py` — làm sạch câu.
6. `src/vote.py` — bỏ phiếu nhiều lần chạy (+ tie-break theo ánh xạ); `src/report.py`, `src/score.py` — chấm.
7. `automation/` — Kaggle (`fanout.py` là cửa vào), `automation/jobs/` — job chuẩn + công cụ phân tích log.
8. `tests/` — `run_all.sh` (bộ CPU), `eval_lib.py` (oracle), `test_plausibility.py`, `test_mapping_fixes.py`.
9. `docker/` — bài nộp cuối.

## 4. Việc tiếp theo (tóm tắt; chi tiết ở `handoff/03_HANDOFF.pdf` §5)

1. Xin token Kaggle → `~/.kaggle/tok_<user>` (chmod 600); tải dữ liệu `nhatlong1103/zc-2026`; ingest train + public.
2. Tải output **v21** (huyenco / mucnokieu) và **v22** (thnliu / duylnng) đang chạy trên Kaggle.
3. Chấm train; tạo file nộp v21 / v22 với `automation/map_penalties.py` + `src/vote.py`; so từng câu với v19
   (portal 0.687); push + gửi user.
4. Ổn định ánh xạ giữa các lần chạy (gộp ứng viên) — đòn bẩy public lớn nhất.
5. Tiếp tục tìm lỗi chung không cần nhãn (`automation/jobs/`), sửa bằng quy tắc tổng quát, giữ oracle 2967 / 1 / 32.
6. Build + đo Docker trên 2×RTX 5090 trước hạn 16/11/2026.

## 5. Ba điều không được làm

- Không fix theo id DN / câu, không gán nhãn tay đáp án public, không train trên test.
- Không đưa token vào code, commit, prompt hay báo cáo; không đụng kernel không phải `zc-*`.
- Không đẩy kernel khi preflight chưa `PREFLIGHT OK`; không `pkill -f` theo mẫu có trong chính lệnh.
