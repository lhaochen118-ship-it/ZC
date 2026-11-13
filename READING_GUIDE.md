# READING_GUIDE — cách đọc repository và việc tiếp theo (cập nhật 13/11/2026)

Cho Agent nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm.

## 0. Mười phút đầu

1. **Xin user token Kaggle** (container mới không còn `~/.kaggle/tok_*`); lưu `~/.kaggle/tok_<username>` (chmod 600).
2. `pip install kaggle==2.2.5 xgrammar duckdb markdown xhtml2pdf`; dựng dữ liệu / cache theo `handoff/ENVIRONMENT_SUMMARY.md` §1 và
   `Process/README.md` (giải nén `Process/train_data/p3/*.gz` vào `training/data/out/{v3a,v3bm,all30,lrl0}/`).
3. **Tạo routine tự tiếp tục mỗi giờ** (`create_trigger`, vào chính phiên mới) với lời nhắc: "Đọc handoff/STATUS.md (bảng Tiến độ P3) và
   docs/16_PLAN3_LEARNED_RULES.md; làm tiếp gói WP3 chưa xong đầu tiên; kiểm quota trước khi chạy; cập nhật STATUS, commit + push".
   (Routine của phiên cũ đã tắt.) Hẹn thêm một lần chạy lúc 14/11 00:06 UTC cho đợt quota mới.
4. Đọc `handoff/STATUS.md` → kernel đang chạy (§2), bảng **Tiến độ P3** (§3c), việc tiếp theo (§4).

## 1. Bắt buộc

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | quy định user (hybrid, quyết định 12/11 và **13/11: kế hoạch 3 "thực hiện hết"**, tốc độ trước, quota, 12 h, `final_submission`, cổng kiểm tra) |
| 2 | `handoff/STATUS.md` | trạng thái sống + bước tiếp theo |
| 3 | `handoff/03_HANDOFF.pdf` (nguồn `.md`) | mục tiêu, quyết định, tiến độ, vị trí, các bước |
| 4 | `docs/16_PLAN3_LEARNED_RULES.md` | **kế hoạch hiện tại**: lớp luật do mô hình học, dữ liệu v3, gói WP3-00…10, quy tắc đổi bản nộp §5.4, nhật ký §9 |
| 5 | `docs/15_AUDIT.md` | kiểm tra rò rỉ / chia dữ liệu / độ phủ train (vì sao số đo của P lạc quan) |

## 2. Khi cần hiểu sâu

| File | Khi nào |
|---|---|
| `handoff/01_SCIENTIFIC_FINDINGS.pdf` | trước khi đề xuất hướng mới: cái gì hiệu quả / thất bại, bằng chứng |
| `handoff/02_ENGINEERING_OPS_LESSONS.pdf` | trước khi sửa code / chạy Kaggle (§7: chuỗi checkpoint, kernel nhiều job, think cap, đo theo DN) |
| `handoff/environment_summary.pdf` | máy, acc, quota, Docker |
| `docs/RESEARCH_WORKFLOW.pdf` (nguồn `docs/03_WORKFLOW.md`) | quy trình, §8 vòng học luật |
| `docs/14_PURE_MODEL_PLAN.md` | hướng thuần Q0 → QS10 và kết luận |
| `docs/12_PLAN2.md`, `docs/10_ERROR_ANALYSIS.md`, `docs/11_LITERATURE.md` | kế hoạch 2 (hướng R / M, quy tắc giữ / bỏ §4), lỗi, tài liệu |
| `docs/00_TASK.md`, `docs/ZAIC_REQUIREMENTS.md` | đề bài, luật chấm, lịch |
| `Process/README.md` | những gì đã lưu (log, dữ liệu, kernel, công cụ) và cách khôi phục |

PDF dựng lại từ `.md`: `python3 automation/md2pdf.py <in.md> <out.pdf>`.

## 3. Bản đồ mã nguồn

- **Luồng nộp P**: `src/predict_flow.py` → `src/agent.py --flow hybrid` (cờ: `build_parser`; cấu hình `configs/flow_final.json`) →
  `src/hybrid.py` (gợi ý → prompt → mẫu → precheck → sandbox → cờ → vòng 2 → phiếu) trên `src/pipeline.py`.
- **Luồng thuần Q / Q2**: `configs/flow_pure.json`, `configs/flow_pure_lrl.json` (`--pure 1`, `--lrl`); prompt `src/shopctx.py`,
  `src/prompting.py` (khối `# RULES`, `<spec>`).
- **Lớp luật (kế hoạch 3)**: `src/lrl.py` (thẻ, khớp, check), `src/lrl_gen.py` (mô hình viết thẻ: compile / aliases / translit / induce /
  newterm), `tests/lrl_accept.py` (quét tín hiệu sạch, A/B), thẻ `src/refs/lrl/`, container `predict_flow.prepare_lrl`.
- **Tính toán**: `src/dm.py`, `src/industry.py` (toolkit), `src/automap.py` / `src/ingest.py` (schema), `src/precheck.py`, `src/tools.py`.
- **Huấn luyện**: `training/full_job.py` (FSDP, `ZC_SESSION_S` + `--resume-from auto`), `training/sft.py`, `training/soup_job.py`
  (gộp + đánh giá); dữ liệu `training/data/*` (kế hoạch 3: `spec_from_program`, `dp_style`, `dc_cards`, `dr_newterm`, `v3_mix`,
  `lrl_cases`, `lrl_collect`, `build_v3b.sh`).
- **Vận hành**: `automation/launch_many.py` (plan → kernel, kiểm quota), `chain_next.py` (nối checkpoint qua acc), `watch_done.py`
  (watcher nền), `resume.py`, `quota.py`, `archive_runs.py`, `update_final_submission.py`; job `automation/jobs/`, plan `Process/kaggle/plans/`.
- **Đánh giá**: `src/eval_flow.py` (tập `tests/splits/`; rulebook theo DN), `src/para_judge.py` (lọc paraphrase), `src/score.py`,
  `Process/tools/score_sets.py`, `Process/tools/cmp_pure_p.py`; test CPU `bash tests/run_all.sh`.

## 4. Việc tiếp theo

Theo `handoff/STATUS.md` §4 (tóm tắt: tải v3a → đánh giá Q / Q2 → A/B thẻ nội dung → đợt 14/11 00:00: v3b + WR30 + vòng học luật 1 →
v4 → quyết định §5.4 ~15/11 → container → user build Docker; hạn 16/11 23:59 GMT+7; sau đó H2).
