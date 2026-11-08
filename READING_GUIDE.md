# READING_GUIDE — cách đọc repository và việc tiếp theo

Cho Agent (hoặc người) nhận dự án ở khung chat mới. Đọc theo thứ tự, dừng khi đủ cho việc đang làm.

## 1. Bắt buộc (≈ 15 phút)

| # | File | Để biết |
|---|---|---|
| 1 | `CLAUDE.md` | Nguyên tắc user đã chốt, cổng kiểm tra trước khi chạy, playbook GPU |
| 2 | `handoff/STATUS.md` | Trạng thái sống, quyết định mới nhất (08/11: chuyển sang LLM-first) |
| 3 | `handoff/03_HANDOFF.pdf` | Mục tiêu, quyết định, vị trí code / dữ liệu / kết quả, **các bước tiếp theo** |
| 4 | `handoff/environment_summary.pdf` | Container, Kaggle (máy, acc, quota), máy chấm / Docker |

## 2. Khi cần hiểu sâu

| File | Khi nào đọc |
|---|---|
| `handoff/01_SCIENTIFIC_FINDINGS.pdf` | Trước khi đề xuất hướng mới: cái gì hiệu quả, cái gì thất bại, bằng chứng |
| `handoff/02_ENGINEERING_OPS_LESSONS.pdf` | Trước khi sửa code / chạy Kaggle: lỗi đã gặp và cách tránh |
| `docs/03_WORKFLOW.md` (PDF: `docs/RESEARCH_WORKFLOW.pdf`) | Quy trình làm việc; §5 vòng lặp LLM-first |
| `docs/00_TASK.md`, `docs/ZAIC_REQUIREMENTS.md` (gốc `ZAIC.docx`) | Đề bài, luật chấm, yêu cầu user |
| `docs/04_DECISION_V3.md` | Thiết kế đường LLM (ánh xạ theo DN + toolkit) — nền của LLM-first |
| `docs/02_KAGGLE_OPS.md`, `docs/05_GPU_OPTIMIZATION.md` | Vận hành Kaggle, tối ưu GPU |
| `EXPERIMENT_LOG.md` | Số liệu từng bản: entry 0–19 đường LLM, 20–27 đường luật + paraphrase |
| `tests/README.md` | Bộ test CPU và lệnh đánh giá |

Các `handoff/*.md` và `docs/03_WORKFLOW.md` là nguồn của PDF cùng tên: sửa `.md` rồi
`python3 automation/md2pdf.py <in.md> <out.pdf>` (cần `pip install markdown xhtml2pdf`).

## 3. Bản đồ mã nguồn

**Đường LLM (giữ, là nền của luồng mới)**
1. `src/ingest.py` — đọc file thô (CSV / Excel / SQLite) → bảng thô + card mô tả; cache theo DN.
2. `src/agent.py` — CLI (`build_parser`), prompt, engine vLLM; `src/pipeline.py` — ánh xạ theo DN, sinh K chương trình,
   bỏ phiếu, sửa lỗi, work stealing; `src/fewshot.py` — truy hồi ví dụ train đã kiểm.
3. `src/canon.py` — chạy chương trình ánh xạ an toàn, `secondary_checks`, `warning_penalty`; `src/consensus.py` — đồng
   thuận ánh xạ giữa các lần chạy.
4. `src/dm.py`, `src/industry.py` — toolkit nghiệp vụ (mọi định nghĩa rulebook), `CANON_DOC` cho prompt.
5. `src/vote.py` — bỏ phiếu nhiều lần chạy + ngưỡng tin cậy; `src/score.py`, `src/report.py` — chấm.

**Đường luật (không trả lời thẳng nữa; `automap`, `qnorm`, nhận diện ý định giữ làm ngữ cảnh cho Agent)**
`src/rules_first.py` (stage 1), `src/automap.py` (ánh xạ theo luật), `src/qnorm.py` (làm sạch câu), `src/qparse.py`
(mẫu câu → chương trình), `src/canon_llm.py` (stage 1.5), `src/predict_flow.py` (luồng nộp hiện tại), `src/merge_sub.py`.

**Đánh giá tổng quát hoá**: `src/paraphrase_llm.py`, `src/perturb.py`, `src/stress_flow.py`, `tests/eval_split.py`,
`tests/eval_all.sh`.

**Vận hành**: `src/run_kaggle.py` (chạy trong kernel), `automation/fanout.py` (cửa vào Kaggle), `automation/preflight.py`,
`automation/run_kernel.py` (`--attach`), `automation/jobs/` (job chuẩn), `docker/` (bài nộp), `submissions/` (v28 LLM,
v30 luật), `goldkey/devset.py` (37 câu đã kiểm, chỉ chấm).

## 4. Việc tiếp theo (chi tiết `handoff/03_HANDOFF.pdf` §5)

1. Xin token Kaggle; tải dữ liệu + ingest; nhận `public_gt.jsonl` (`score_1.jsonl`) từ user nếu mất.
2. Đo nền LLM-only trên paraphrase held-out (chưa từng đo).
3. Viết luồng hybrid mới theo `handoff/03_HANDOFF.md` §2.1 (Agent sinh code mọi câu, luật chỉ làm ngữ cảnh); test, preflight, chạy Kaggle; giữ khi held-out ≥ v34 (0.602) và
   thời gian ≤ 6 h trên 2×5090.
4. File nộp + Docker; user build và nộp trước 16/11/2026 23:59 GMT+7.

## 5. Không được làm

- Không luật sinh thẳng code / đáp án mà không qua mô hình (user, 08/11). Luật tổng quát làm ngữ cảnh / gợi ý (ý định,
  schema, làm sạch câu, toolkit, kiểm tra) thì **được** — `handoff/03_HANDOFF.md` §2.1.
- Không fix theo id DN / câu; không đưa đáp án public (kể cả `goldkey/devset.py`) vào `src/` hay prompt; không train trên test.
- Không đưa token vào code, commit, prompt hay báo cáo; không đụng kernel không phải `zc-*`.
- Không đẩy kernel khi chưa `PREFLIGHT OK`; không `pkill -f` theo mẫu có trong chính lệnh.
